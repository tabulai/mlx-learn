// Shared primitives of the native radius engine (_radius_cpu_native).
//
// Contract (see radius_contract.py): a training row x is a neighbor of q iff
// the exact real S(q, x) = sum_k (q_k - x_k)^2 of the float32 coordinates is
// <= R2. Engines compute the canonical float64 value
//
//     s = 0; for k in 0..d-1: t = double(q_k) - double(x_k); s = s + t * t
//
// strictly sequentially (no contraction, no reassociation; independent pairs
// may be interleaved), classify it against outward-rounded thresholds computed
// in Python, and resolve the uncertain band with exact integer arithmetic
// (S * 2**298 <= floor(R2 * 2**298)). Reported distances are
// sqrt(min(s, R2)).
//
// Only the buffer protocol is used (no NumPy C API). All compute runs with the
// GIL released; worker threads come from GCD, pull task ids from one atomic
// counter, clear FPCR.FZ (and the other non-IEEE FPCR controls) for the
// duration of the work, and capture exceptions per task.
#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace mlx_radius {

// ---------------------------------------------------------------------------
// mode bits (mirrored in radius_cpu.py)
// ---------------------------------------------------------------------------

constexpr int NEED_DISTANCE = 1;
constexpr int SORT = 2;
constexpr int INTEGER_EXACT = 4;
constexpr int COUNT_ONLY = 8;
constexpr int TWO_PASS = 16;
// Test hook: the last task of the call throws, to exercise error propagation.
constexpr int FAIL_TEST = 1 << 20;
// Disable the float32 pre-screen and box tests (tests and A/B runs; same output).
constexpr int NO_SCREEN = 1 << 21;
// A/B switches for the tree: float64 box tests / unscreened leaves (same output).
constexpr int BOX64_TEST = 1 << 22;
constexpr int LEAF_PLAIN_TEST = 1 << 23;
constexpr int KNOWN_MODES = NEED_DISTANCE | SORT | INTEGER_EXACT | COUNT_ONLY | TWO_PASS |
                            FAIL_TEST | NO_SCREEN | BOX64_TEST | LEAF_PLAIN_TEST;

// Thrown for invalid inputs detected inside parallel work; becomes ValueError.
struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------

enum class DType { f32, f64, i32, i64, u8 };

inline const char* dtype_name(DType type) {
    switch (type) {
        case DType::f32: return "float32";
        case DType::f64: return "float64";
        case DType::i32: return "int32";
        case DType::i64: return "int64";
        case DType::u8: return "uint8";
    }
    return "?";
}

inline Py_ssize_t dtype_size(DType type) {
    switch (type) {
        case DType::f32: case DType::i32: return 4;
        case DType::f64: case DType::i64: return 8;
        case DType::u8: return 1;
    }
    return 0;
}

struct Buffer {
    Py_buffer view{};
    bool acquired = false;
    DType type = DType::f32;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        if (acquired) PyBuffer_Release(&view);
    }
    template <class T>
    T* data() const { return static_cast<T*>(view.buf); }
    Py_ssize_t count() const { return acquired ? view.len / view.itemsize : 0; }
    Py_ssize_t dim(int axis) const { return view.shape[axis]; }
};

inline bool format_is(const Py_buffer& view, DType type) {
    const char* format = view.format;
    if (!format) return false;
    // Native order is little-endian on every supported target.
    if (*format == '@' || *format == '=' || *format == '<') ++format;
    if (format[0] == '\0' || format[1] != '\0') return false;
    const char c = format[0];
    switch (type) {
        case DType::f32: return c == 'f' && view.itemsize == 4;
        case DType::f64: return c == 'd' && view.itemsize == 8;
        case DType::i32: return (c == 'i' || c == 'l') && view.itemsize == 4;
        case DType::i64: return (c == 'q' || c == 'l') && view.itemsize == 8;
        case DType::u8: return (c == 'B' || c == '?') && view.itemsize == 1;
    }
    return false;
}

// Acquire a C-contiguous, aligned buffer of the given dtype. ndim > 0 requires
// that many dimensions; ndim == 0 accepts any shape (outputs checked by size).
inline bool get_buffer(PyObject* object, Buffer& buffer, bool writable, const char* name,
                       DType type, int ndim) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, &buffer.view, flags) < 0) return false;
    buffer.acquired = true;
    buffer.type = type;
    const Py_buffer& v = buffer.view;
    const Py_ssize_t itemsize = dtype_size(type);
    bool ok = format_is(v, type) && v.itemsize == itemsize && v.len >= 0 &&
              PyBuffer_IsContiguous(&v, 'C') &&
              (reinterpret_cast<std::uintptr_t>(v.buf) % std::uintptr_t(itemsize)) == 0 &&
              (ndim == 0 || v.ndim == ndim) && (v.len == 0 || v.buf);
    if (ok && ndim > 0) {
        Py_ssize_t total = itemsize;
        for (int axis = 0; axis < ndim; ++axis) {
            if (!v.shape || v.shape[axis] < 0) { ok = false; break; }
            if (v.shape[axis] && total > PY_SSIZE_T_MAX / v.shape[axis]) { ok = false; break; }
            total *= v.shape[axis];
        }
        ok = ok && total == v.len;
    }
    if (!ok) {
        if (ndim > 0)
            PyErr_Format(PyExc_ValueError, "%s must be an aligned C-contiguous %d-d %s array%s",
                         name, ndim, dtype_name(type), writable ? " (writable)" : "");
        else
            PyErr_Format(PyExc_ValueError, "%s must be an aligned C-contiguous %s array%s", name,
                         dtype_name(type), writable ? " (writable)" : "");
        return false;
    }
    return true;
}

// An index array that may be int32 or int64.
inline bool get_index_buffer(PyObject* object, Buffer& buffer, bool writable, const char* name) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, &buffer.view, flags) < 0) return false;
    buffer.acquired = true;
    const Py_buffer& v = buffer.view;
    DType type = format_is(v, DType::i64) ? DType::i64 : format_is(v, DType::i32) ? DType::i32 : DType::u8;
    bool ok = type != DType::u8 && v.ndim == 1 && PyBuffer_IsContiguous(&v, 'C') &&
              (reinterpret_cast<std::uintptr_t>(v.buf) % std::uintptr_t(v.itemsize)) == 0 &&
              (v.len == 0 || v.buf);
    if (!ok) {
        PyErr_Format(PyExc_ValueError, "%s must be an aligned C-contiguous 1-d int32 or int64 array",
                     name);
        return false;
    }
    buffer.type = type;
    return true;
}

inline std::int64_t index_at(const Buffer& buffer, Py_ssize_t i) {
    return buffer.type == DType::i64 ? buffer.data<const std::int64_t>()[i]
                                     : std::int64_t(buffer.data<const std::int32_t>()[i]);
}

inline bool overlaps(const Buffer& a, const Buffer& b) {
    if (!a.acquired || !b.acquired || !a.view.len || !b.view.len) return false;
    auto start_a = reinterpret_cast<std::uintptr_t>(a.view.buf);
    auto start_b = reinterpret_cast<std::uintptr_t>(b.view.buf);
    return start_a < start_b + std::uintptr_t(b.view.len) &&
           start_b < start_a + std::uintptr_t(a.view.len);
}

// ---------------------------------------------------------------------------
// exact arithmetic (copied from _neighbors_cpu_native.cpp, which stays untouched)
// ---------------------------------------------------------------------------

// Ten 64-bit limbs: S * 2**298 needs at most 587 bits for any d < 2**31.
using Key = std::array<std::uint64_t, 10>;

inline std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline std::uint64_t double_bits(double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Float32 x is exactly signed significand * 2**shift in units of 2**-149.
struct Term {
    std::uint64_t significand;
    unsigned shift;
    bool negative;
};

inline Term decode(float value) {
    auto bits = float_bits(value);
    unsigned exponent = (bits >> 23) & 255U;
    return {static_cast<std::uint64_t>((bits & 0x7fffffU) | (exponent ? 0x800000U : 0)),
            exponent ? exponent - 1 : 0, (bits >> 31) != 0};
}

inline void add_shifted(Key& accumulator, std::uint64_t value, unsigned shift) {
    if (!value) return;
    unsigned limb = shift / 64;
    unsigned bit = shift % 64;
    unsigned __int128 carry = static_cast<unsigned __int128>(value) << bit;
    while (carry && limb < accumulator.size()) {
        carry += accumulator[limb];
        accumulator[limb] = static_cast<std::uint64_t>(carry);
        carry >>= 64;
        ++limb;
    }
}

// S(q, x) * 2**298 as an exact integer.
inline Key exact_squared(const float* query, const float* point, Py_ssize_t features) {
    Key positive{}, negative{}, result{};
    for (Py_ssize_t feature = 0; feature < features; ++feature) {
        Term q = decode(query[feature]), x = decode(point[feature]);
        add_shifted(positive, q.significand * q.significand, 2 * q.shift);
        add_shifted(positive, x.significand * x.significand, 2 * x.shift);
        add_shifted(q.negative == x.negative ? negative : positive,
                    q.significand * x.significand, q.shift + x.shift + 1);
    }
    std::uint64_t borrow = 0;
    for (unsigned limb = 0; limb < result.size(); ++limb) {
        unsigned __int128 subtrahend = static_cast<unsigned __int128>(negative[limb]) + borrow;
        result[limb] = positive[limb] - static_cast<std::uint64_t>(subtrahend);
        borrow = static_cast<unsigned __int128>(positive[limb]) < subtrahend;
    }
    return result;
}

inline bool key_le(const Key& left, const Key& right) {
    for (int limb = int(left.size()) - 1; limb >= 0; --limb) {
        if (left[limb] < right[limb]) return true;
        if (left[limb] > right[limb]) return false;
    }
    return true;
}

// floor(R2 * 2**298) for 0 <= R2 < 2**342. Returns false when R2 >= 2**342
// (including +inf): every pair of finite float32 rows is then a member,
// because S < 2**289 for any d < 2**31.
inline bool r2_floor_key(double R2, Key& key) {
    key = Key{};
    if (!(R2 < 0x1p342)) return false;
    if (!(R2 > 0)) return true;
    const std::uint64_t bits = double_bits(R2);
    const int exponent = int((bits >> 52) & 0x7ff);
    std::uint64_t mantissa = bits & ((std::uint64_t(1) << 52) - 1);
    int power;
    if (exponent) {
        mantissa |= std::uint64_t(1) << 52;
        power = exponent - 1075;
    } else {
        power = -1074;
    }
    const int shift = power + 298;
    if (shift >= 0) add_shifted(key, mantissa, unsigned(shift));
    else if (-shift < 64) key[0] = mantissa >> unsigned(-shift);
    return true;
}

// ---------------------------------------------------------------------------
// thresholds and classification
// ---------------------------------------------------------------------------

struct Thr {
    double R2 = 0, f64_in = 0, f64_out = 0, prune_above = 0, bulk_below = 0;
    bool all_members = false;  // R2 >= 2**342 (or +inf)
    Key r2key{};
};

// Parse (R2, f64_in, f64_out, prune_above, bulk_below) and check its ordering.
inline bool parse_thresholds(PyObject* object, Thr& thr) {
    if (!PyArg_ParseTuple(object, "ddddd;thresholds must be a tuple of 5 floats", &thr.R2,
                          &thr.f64_in, &thr.f64_out, &thr.prune_above, &thr.bulk_below))
        return false;
    const bool ordered = thr.R2 >= 0 && thr.bulk_below >= 0 && thr.bulk_below <= thr.f64_in &&
                         thr.f64_in <= thr.f64_out && thr.f64_out <= thr.prune_above &&
                         !std::isnan(thr.prune_above);
    if (!ordered) {
        PyErr_SetString(PyExc_ValueError,
                        "thresholds must satisfy 0 <= bulk_below <= f64_in <= f64_out <= "
                        "prune_above and R2 >= 0");
        return false;
    }
    thr.all_members = !r2_floor_key(thr.R2, thr.r2key);
    return true;
}

// Exact S(q, x) <= R2.
__attribute__((noinline)) inline bool exact_member(const float* q, const float* x,
                                                   Py_ssize_t d, const Thr& thr) {
    if (thr.all_members) return true;
    return key_le(exact_squared(q, x, d), thr.r2key);
}

// Member iff s <= f64_in; non-member iff s > f64_out (or s is NaN); exact
// resolution in between. With integer-exact data f64_in == f64_out == R2.
inline bool classify(double s, const Thr& thr, const float* q, const float* x, Py_ssize_t d) {
    if (s <= thr.f64_in) return true;
    if (!(s <= thr.f64_out)) return false;
    return exact_member(q, x, d, thr);
}

// ---------------------------------------------------------------------------
// canonical float64 squared distances
// ---------------------------------------------------------------------------

// D > 0 fixes the dimension at compile time; D == 0 uses the runtime d.
template <int D>
inline double canonical_d2(const double* q, const float* x, Py_ssize_t d) {
    const Py_ssize_t dims = D > 0 ? D : d;
    double s = 0.0;
    for (Py_ssize_t k = 0; k < dims; ++k) {
        const double t = q[k] - double(x[k]);
        s = s + t * t;
    }
    return s;
}

// Four independent pairs, each with its own strictly sequential chain.
template <int D>
inline void canonical_d2_x4(const double* q, const float* x0, const float* x1, const float* x2,
                            const float* x3, Py_ssize_t d, double* out) {
    const Py_ssize_t dims = D > 0 ? D : d;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (Py_ssize_t k = 0; k < dims; ++k) {
        const double v = q[k];
        const double t0 = v - double(x0[k]);
        const double t1 = v - double(x1[k]);
        const double t2 = v - double(x2[k]);
        const double t3 = v - double(x3[k]);
        s0 = s0 + t0 * t0;
        s1 = s1 + t1 * t1;
        s2 = s2 + t2 * t2;
        s3 = s3 + t3 * t3;
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

// ---------------------------------------------------------------------------
// FPCR and parallel_for
// ---------------------------------------------------------------------------

inline std::uint64_t read_fpcr() {
#if defined(__aarch64__)
    std::uint64_t value;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(value) : : "memory");
    return value;
#else
    return 0;
#endif
}

inline void write_fpcr(std::uint64_t value) {
#if defined(__aarch64__)
    __asm__ __volatile__("msr fpcr, %0" : : "r"(value) : "memory");
#else
    (void)value;
#endif
}

// FZ (24), RMode (22-23), AH (1) and FIZ (0): IEEE behavior needs all clear.
constexpr std::uint64_t FPCR_NON_IEEE = (std::uint64_t(1) << 24) | (std::uint64_t(3) << 22) |
                                        (std::uint64_t(1) << 1) | std::uint64_t(1);

struct FpcrScope {
    std::uint64_t saved;
    bool changed;
    FpcrScope() : saved(read_fpcr()) {
        const std::uint64_t wanted = saved & ~FPCR_NON_IEEE;
        changed = wanted != saved;
        if (changed) write_fpcr(wanted);
    }
    ~FpcrScope() {
        if (changed) write_fpcr(saved);
    }
    FpcrScope(const FpcrScope&) = delete;
    FpcrScope& operator=(const FpcrScope&) = delete;
};

struct ParallelState {
    std::atomic<std::size_t> next{0};
    std::size_t n_tasks = 0;
    void (*call)(void*, std::size_t) = nullptr;
    void* function = nullptr;
    std::atomic<bool> failed{false};
    std::mutex mutex;
    std::exception_ptr error;
};

inline void parallel_worker(void* context, std::size_t) {
    auto* state = static_cast<ParallelState*>(context);
    FpcrScope fpcr;
    for (;;) {
        if (state->failed.load(std::memory_order_relaxed)) return;
        const std::size_t task = state->next.fetch_add(1, std::memory_order_relaxed);
        if (task >= state->n_tasks) return;
        try {
            state->call(state->function, task);
        } catch (...) {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (!state->error) state->error = std::current_exception();
            state->failed.store(true, std::memory_order_relaxed);
            return;
        }
    }
}

// Below this estimated work a call runs on the calling thread.
constexpr double SERIAL_NS = 50e3;

// Run fn(task) for task in [0, n_tasks). Tasks must write disjoint data. The
// first exception thrown by any task is rethrown after every worker stopped.
template <class F>
void parallel_for(std::size_t n_tasks, int threads, double estimated_ns, F&& fn) {
    if (n_tasks == 0) return;
    const std::size_t workers =
        std::min<std::size_t>(std::size_t(threads < 1 ? 1 : threads), n_tasks);
    if (workers <= 1 || estimated_ns < SERIAL_NS) {
        FpcrScope fpcr;
        for (std::size_t task = 0; task < n_tasks; ++task) fn(task);
        return;
    }
    using Function = std::remove_reference_t<F>;
    ParallelState state;
    state.n_tasks = n_tasks;
    state.function = const_cast<void*>(static_cast<const void*>(std::addressof(fn)));
    state.call = [](void* function, std::size_t task) { (*static_cast<Function*>(function))(task); };
#if defined(__APPLE__)
    dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &state,
                     parallel_worker);
#else
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (std::size_t w = 1; w < workers; ++w) pool.emplace_back(parallel_worker, &state, w);
    parallel_worker(&state, 0);
    for (auto& thread : pool) thread.join();
#endif
    if (state.error) std::rethrow_exception(state.error);
}

inline int clamp_threads(Py_ssize_t threads) {
    if (threads < 1) return 1;
    if (threads > 1024) return 1024;
    return int(threads);
}

// ---------------------------------------------------------------------------
// uninitialized storage
// ---------------------------------------------------------------------------

// std::vector whose resize() leaves trivial elements uninitialized, so large
// buffers are first touched (and faulted in) by the parallel workers that
// fill them instead of by a serial zero fill.
template <class T>
struct UninitAllocator {
    using value_type = T;
    UninitAllocator() = default;
    template <class U>
    UninitAllocator(const UninitAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) { return std::allocator<T>{}.allocate(n); }
    void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
    template <class U>
    bool operator==(const UninitAllocator<U>&) const noexcept { return true; }
    template <class U>
    bool operator!=(const UninitAllocator<U>&) const noexcept { return false; }
};

template <class T>
using RawVector = std::vector<T, UninitAllocator<T>>;

// ---------------------------------------------------------------------------
// results
// ---------------------------------------------------------------------------

struct TaskOut {
    std::vector<std::int32_t> cols;
    std::vector<double> d2;
};

// Rows live in per-task buffers: row r is tasks[row_task[r]] at
// [row_off[r], row_off[r] + row_cnt[r]). Results own all their data; the
// keep_alive references are released in the capsule destructor.
struct Result {
    Py_ssize_t n_query = 0;
    Py_ssize_t n_train = 0;
    int mode = 0;  // NEED_DISTANCE / SORT / COUNT_ONLY as computed
    double R2 = 0;
    std::int64_t nnz = 0;
    std::vector<TaskOut> tasks;
    RawVector<std::int32_t> row_task;  // every row is written by exactly one task
    RawVector<std::int64_t> row_off;
    RawVector<std::int64_t> row_cnt;
    std::vector<PyObject*> keep_alive;
};

constexpr const char* RESULT_CAPSULE = "mlxlearn._radius_cpu_native.Result";
constexpr const char* TREE_CAPSULE = "mlxlearn._radius_cpu_native.Tree";

inline void result_destructor(PyObject* capsule) {
    auto* result = static_cast<Result*>(PyCapsule_GetPointer(capsule, RESULT_CAPSULE));
    if (!result) {
        PyErr_Clear();
        return;
    }
    for (PyObject* object : result->keep_alive) Py_XDECREF(object);
    delete result;
}

inline PyObject* wrap_result(std::unique_ptr<Result> result) {
    PyObject* capsule = PyCapsule_New(result.get(), RESULT_CAPSULE, result_destructor);
    if (!capsule) {
        for (PyObject* object : result->keep_alive) Py_XDECREF(object);
        return nullptr;
    }
    result.release();
    return capsule;
}

inline Result* unwrap_result(PyObject* capsule) {
    return static_cast<Result*>(PyCapsule_GetPointer(capsule, RESULT_CAPSULE));
}

// Convert a captured C++ exception into a Python exception; returns nullptr.
inline PyObject* raise_exception(const std::exception_ptr& error) {
    try {
        std::rethrow_exception(error);
    } catch (const InputError& exc) {
        PyErr_SetString(PyExc_ValueError, exc.what());
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
    } catch (const std::length_error&) {
        PyErr_NoMemory();
    } catch (const std::exception& exc) {
        PyErr_SetString(PyExc_RuntimeError, exc.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown native radius error");
    }
    return nullptr;
}

// Order one row's entries: ascending column, or ascending (distance, column)
// with distance = sqrt(min(d2, R2)) as returned to the caller.
struct SortScratch {
    std::vector<double> query;  // per-task float64 copy of the current query
    std::vector<std::uint64_t> keys;
    std::vector<double> values;
    std::vector<std::uint32_t> word_offsets;
    struct Item {
        std::uint64_t distance;
        std::int32_t col;
        double d2;
    };
    std::vector<Item> items;
};

inline void sort_row(std::int32_t* cols, double* d2, std::size_t count, bool by_distance,
                     double R2, SortScratch& scratch, std::size_t unique_columns = 0) {
    if (count <= 1) return;
    if (!by_distance) {
        bool ascending = true;
        for (std::size_t i = 1; i < count && ascending; ++i) ascending = cols[i - 1] < cols[i];
        if (ascending) return;
        // Dense tree rows contain unique training indices. Enumerate their
        // set bits in index order; a word prefix and a popcount give the
        // sorted position of each payload without comparing or moving pairs.
        const std::size_t word_count = (unique_columns + 63) / 64;
        if (word_count && count >= 128 && word_count <= count) {
            auto& words = scratch.keys;
            words.assign(word_count, 0);
            for (std::size_t i = 0; i < count; ++i)
                words[cols[i] >> 6] |= std::uint64_t(1) << (cols[i] & 63);
            if (d2) {
                auto& offsets = scratch.word_offsets;
                offsets.resize(word_count);
                std::uint32_t offset = 0;
                for (std::size_t w = 0; w < word_count; ++w) {
                    offsets[w] = offset;
                    offset += __builtin_popcountll(words[w]);
                }
                auto& values = scratch.values;
                values.resize(count);
                for (std::size_t i = 0; i < count; ++i) {
                    const std::size_t w = std::size_t(cols[i]) >> 6;
                    const auto bit = std::uint64_t(1) << (cols[i] & 63);
                    values[offsets[w] + __builtin_popcountll(words[w] & (bit - 1))] = d2[i];
                }
                std::memcpy(d2, values.data(), count * sizeof(double));
            }
            std::size_t i = 0;
            for (std::size_t w = 0; w < word_count; ++w)
                for (auto mask = words[w]; mask; mask &= mask - 1)
                    cols[i++] = std::int32_t(w * 64 + __builtin_ctzll(mask));
            return;
        }
        if (!d2) {
            std::sort(cols, cols + count);
            return;
        }
        if (count <= 24) {
            for (std::size_t i = 1; i < count; ++i) {
                const std::int32_t col = cols[i];
                const double value = d2[i];
                std::size_t j = i;
                while (j && cols[j - 1] > col) {
                    cols[j] = cols[j - 1];
                    d2[j] = d2[j - 1];
                    --j;
                }
                cols[j] = col;
                d2[j] = value;
            }
            return;
        }
        auto& keys = scratch.keys;
        auto& values = scratch.values;
        keys.resize(count);
        values.resize(count);
        for (std::size_t i = 0; i < count; ++i)
            keys[i] = (std::uint64_t(std::uint32_t(cols[i])) << 32) | std::uint64_t(i);
        std::sort(keys.begin(), keys.end());
        for (std::size_t i = 0; i < count; ++i) {
            values[i] = d2[keys[i] & 0xffffffffU];
            cols[i] = std::int32_t(keys[i] >> 32);
        }
        std::memcpy(d2, values.data(), count * sizeof(double));
        return;
    }
    auto& items = scratch.items;
    items.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double distance = std::sqrt(std::min(d2[i], R2));
        items[i] = {double_bits(distance), cols[i], d2[i]};
    }
    std::sort(items.begin(), items.end(), [](const SortScratch::Item& a, const SortScratch::Item& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.col < b.col);
    });
    for (std::size_t i = 0; i < count; ++i) {
        cols[i] = items[i].col;
        d2[i] = items[i].d2;
    }
}

// Collects one row's members into a task buffer (or only counts them).
struct RowSink {
    TaskOut* out = nullptr;
    bool keep_d2 = false;
    bool count_only = false;
    std::int64_t count = 0;

    inline void add(std::int32_t col, double s) {
        ++count;
        if (count_only) return;
        out->cols.push_back(col);
        if (keep_d2) out->d2.push_back(s);
    }
};

// Scan contiguous rows [begin, end) of `points`, classify each against q and
// emit members. The emitted column is perm[p] when perm is given, else p.
template <int D>
inline void scan_classify(const double* q, const float* qf, const float* points,
                          const std::int32_t* perm, Py_ssize_t begin, Py_ssize_t end,
                          Py_ssize_t d, const Thr& thr, std::int64_t self, RowSink& sink) {
    Py_ssize_t p = begin;
    double s[4];
    for (; p + 4 <= end; p += 4) {
        const float* x = points + p * d;
        canonical_d2_x4<D>(q, x, x + d, x + 2 * d, x + 3 * d, d, s);
        for (int j = 0; j < 4; ++j) {
            if (s[j] <= thr.f64_in ||
                (s[j] <= thr.f64_out && exact_member(qf, x + j * d, d, thr))) {
                const std::int64_t col = perm ? perm[p + j] : std::int64_t(p + j);
                if (col != self) sink.add(std::int32_t(col), s[j]);
            }
        }
    }
    for (; p < end; ++p) {
        const float* x = points + p * d;
        const double value = canonical_d2<D>(q, x, d);
        if (classify(value, thr, qf, x, d)) {
            const std::int64_t col = perm ? perm[p] : std::int64_t(p);
            if (col != self) sink.add(std::int32_t(col), value);
        }
    }
}

// Emit every row of [begin, end) as a member (bulk inclusion), computing the
// canonical d2 only when it is kept. self_pos is the position of the excluded
// row in `points` (or -1).
template <int D>
inline void scan_bulk(const double* q, const float* points, const std::int32_t* perm,
                      Py_ssize_t begin, Py_ssize_t end, Py_ssize_t d, std::int64_t self,
                      std::int64_t self_pos, RowSink& sink) {
    if (sink.count_only) {
        sink.count += (end - begin) - (self_pos >= begin && self_pos < end ? 1 : 0);
        return;
    }
    if (!sink.keep_d2) {
        for (Py_ssize_t p = begin; p < end; ++p) {
            const std::int64_t col = perm ? perm[p] : std::int64_t(p);
            if (col != self) sink.add(std::int32_t(col), 0.0);
        }
        return;
    }
    Py_ssize_t p = begin;
    double s[4];
    for (; p + 4 <= end; p += 4) {
        const float* x = points + p * d;
        canonical_d2_x4<D>(q, x, x + d, x + 2 * d, x + 3 * d, d, s);
        for (int j = 0; j < 4; ++j) {
            const std::int64_t col = perm ? perm[p + j] : std::int64_t(p + j);
            if (col != self) sink.add(std::int32_t(col), s[j]);
        }
    }
    for (; p < end; ++p) {
        const std::int64_t col = perm ? perm[p] : std::int64_t(p);
        if (col != self) sink.add(std::int32_t(col), canonical_d2<D>(q, points + p * d, d));
    }
}

// Task t covers positions [bounds[t], bounds[t + 1]) of the processing order.
inline std::vector<std::int64_t> uniform_bounds(std::int64_t count, std::int64_t per_task) {
    if (per_task < 1) per_task = 1;
    std::vector<std::int64_t> bounds;
    bounds.reserve(std::size_t(count / per_task + 2));
    for (std::int64_t start = 0; start < count; start += per_task) bounds.push_back(start);
    bounds.push_back(count);
    return bounds;
}

// ---------------------------------------------------------------------------
// float32 pre-screen
// ---------------------------------------------------------------------------

// A float32 value s32 of S, summed in any order with round-to-nearest and
// gradual underflow (FPCR.FZ is cleared in every worker), satisfies
// |s32 - S| <= g * S + A with g = (d + 6) u / (1 - (d + 6) u), u = 2**-24,
// and A = (d + 8) * 2**-149: every term passes through at most d + 5
// roundings (difference, square or fused accumulation, partial and
// horizontal sums), all terms are nonnegative, and each operation adds at most
// 2**-150 of absolute underflow error. An overflow to +inf needs S > 2**127,
// so with R2 <= 2**120 the test s32 > out (out >= R2 (1 + 2g) + A) proves
// S > R2 and s32 <= in (in <= R2 (1 - 2g) - A) proves S <= R2. The screen
// only skips work: anything not certain goes through the canonical float64
// value and the exact classification.
struct Screen32 {
    bool on = false;
    float out = 0.0f;  // reject iff s32 > out
    float in = -1.0f;  // certain member iff s32 <= in
};

inline Screen32 make_screen32(double R2, Py_ssize_t d) {
    Screen32 screen;
    if (!(R2 <= 0x1p120) || d > (1 << 20)) return screen;
    FpcrScope fpcr;
    const double u = 0x1p-24;
    const double terms = double(d) + 6.0;
    const double g = terms * u / (1.0 - terms * u);
    const double A = (double(d) + 8.0) * 0x1p-149;
    const double inf = std::numeric_limits<double>::infinity();
    double out = std::nextafter(std::nextafter(R2 * (1.0 + 2.0 * g) + A, inf), inf);
    float out32 = float(out);
    if (double(out32) < out) out32 = std::nextafter(out32, std::numeric_limits<float>::infinity());
    double in = std::nextafter(std::nextafter(R2 * (1.0 - 2.0 * g) - A, -inf), -inf);
    float in32 = -1.0f;
    if (in >= 0.0) {
        in32 = float(in);
        if (double(in32) > in) in32 = std::nextafter(in32, -std::numeric_limits<float>::infinity());
    }
    screen.on = true;
    screen.out = out32;
    screen.in = in32;
    return screen;
}

// Float32 screening values of two rows against q (features vectorized).
inline void screen_d2_x2(const float* q, const float* x0, const float* x1, Py_ssize_t d, float& s0,
                         float& s1) {
    Py_ssize_t k = 0;
    float r0 = 0.0f, r1 = 0.0f;
#if defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0.0f), b0 = a0, a1 = a0, b1 = a0;
    for (; k + 8 <= d; k += 8) {
        const float32x4_t qa = vld1q_f32(q + k), qb = vld1q_f32(q + k + 4);
        float32x4_t t = vsubq_f32(qa, vld1q_f32(x0 + k));
        a0 = vfmaq_f32(a0, t, t);
        t = vsubq_f32(qb, vld1q_f32(x0 + k + 4));
        b0 = vfmaq_f32(b0, t, t);
        t = vsubq_f32(qa, vld1q_f32(x1 + k));
        a1 = vfmaq_f32(a1, t, t);
        t = vsubq_f32(qb, vld1q_f32(x1 + k + 4));
        b1 = vfmaq_f32(b1, t, t);
    }
    for (; k + 4 <= d; k += 4) {
        const float32x4_t qa = vld1q_f32(q + k);
        float32x4_t t = vsubq_f32(qa, vld1q_f32(x0 + k));
        a0 = vfmaq_f32(a0, t, t);
        t = vsubq_f32(qa, vld1q_f32(x1 + k));
        a1 = vfmaq_f32(a1, t, t);
    }
    r0 = vaddvq_f32(vaddq_f32(a0, b0));
    r1 = vaddvq_f32(vaddq_f32(a1, b1));
#endif
    for (; k < d; ++k) {
        const float t0 = q[k] - x0[k];
        const float t1 = q[k] - x1[k];
        r0 = r0 + t0 * t0;
        r1 = r1 + t1 * t1;
    }
    s0 = r0;
    s1 = r1;
}

inline float screen_d2(const float* q, const float* x, Py_ssize_t d) {
    float s0, s1;
    screen_d2_x2(q, x, x, d, s0, s1);
    return s0;
}

// Classify contiguous rows [begin, end) with the float32 screen in front of
// the canonical float64 value. Same output as scan_classify.
template <int D>
inline void scan_screened(const double* q, const float* qf, const float* points,
                          const std::int32_t* perm, Py_ssize_t begin, Py_ssize_t end,
                          Py_ssize_t d, const Thr& thr, const Screen32& screen, std::int64_t self,
                          RowSink& sink) {
    const bool certify = !sink.keep_d2;
    auto handle = [&](Py_ssize_t p, float s32) {
        const std::int64_t col = perm ? perm[p] : std::int64_t(p);
        if (col == self || !(s32 <= screen.out)) return;
        if (certify && s32 <= screen.in) {
            sink.add(std::int32_t(col), 0.0);
            return;
        }
        const float* x = points + p * d;
        const double value = canonical_d2<D>(q, x, d);
        if (classify(value, thr, qf, x, d)) sink.add(std::int32_t(col), value);
    };
    Py_ssize_t p = begin;
    const float cut = screen.out;
#if defined(__aarch64__)
    if constexpr (D >= 1 && D <= 4) {
        const float32x4_t limit = vdupq_n_f32(cut);
        float32x4_t qv[4];
        for (int k = 0; k < D; ++k) qv[k] = vdupq_n_f32(qf[k]);
        for (; p + 4 <= end; p += 4) {
            const float* base = points + p * D;
            float32x4_t s;
            if constexpr (D == 1) {
                const float32x4_t t = vsubq_f32(qv[0], vld1q_f32(base));
                s = vmulq_f32(t, t);
            } else if constexpr (D == 2) {
                const float32x4x2_t v = vld2q_f32(base);
                const float32x4_t t0 = vsubq_f32(qv[0], v.val[0]);
                const float32x4_t t1 = vsubq_f32(qv[1], v.val[1]);
                s = vfmaq_f32(vmulq_f32(t0, t0), t1, t1);
            } else if constexpr (D == 3) {
                const float32x4x3_t v = vld3q_f32(base);
                const float32x4_t t0 = vsubq_f32(qv[0], v.val[0]);
                const float32x4_t t1 = vsubq_f32(qv[1], v.val[1]);
                const float32x4_t t2 = vsubq_f32(qv[2], v.val[2]);
                s = vfmaq_f32(vfmaq_f32(vmulq_f32(t0, t0), t1, t1), t2, t2);
            } else {
                const float32x4x4_t v = vld4q_f32(base);
                const float32x4_t t0 = vsubq_f32(qv[0], v.val[0]);
                const float32x4_t t1 = vsubq_f32(qv[1], v.val[1]);
                const float32x4_t t2 = vsubq_f32(qv[2], v.val[2]);
                const float32x4_t t3 = vsubq_f32(qv[3], v.val[3]);
                s = vfmaq_f32(vfmaq_f32(vfmaq_f32(vmulq_f32(t0, t0), t1, t1), t2, t2), t3, t3);
            }
            if (vmaxvq_u32(vcleq_f32(s, limit)) == 0) continue;
            float lanes[4];
            vst1q_f32(lanes, s);
            for (int j = 0; j < 4; ++j)
                if (lanes[j] <= cut) handle(p + j, lanes[j]);
        }
    } else {
        // With wide vectors and a selective radius, the first cache line
        // often certifies a miss. Probe eight rows before paying for a
        // separate prefix screen. Only each row's own bound can reject it.
        if constexpr (D == 0) {
            constexpr Py_ssize_t prefix = 16;
            if (!perm && d >= 64 && end - begin >= 128) {
                int rejected = 0;
                for (int j = 0; j < 8; ++j) {
                    const auto row = begin + j * ((end - begin) / 8);
                    rejected += screen_d2(qf, points + row * d, prefix) > cut;
                }
                if (rejected >= 7) {
                    for (; p + 2 <= end; p += 2) {
                        const float* x = points + p * d;
                        float a, b;
                        screen_d2_x2(qf, x, x + d, prefix, a, b);
                        if (a <= cut) handle(p, screen_d2(qf, x, d));
                        if (b <= cut) handle(p + 1, screen_d2(qf, x + d, d));
                    }
                }
            }
        }
        for (; p + 2 <= end; p += 2) {
            float s0, s1;
            screen_d2_x2(qf, points + p * d, points + (p + 1) * d, d, s0, s1);
            if (s0 <= cut) handle(p, s0);
            if (s1 <= cut) handle(p + 1, s1);
        }
    }
#endif
    for (; p < end; ++p) {
        const float s32 = screen_d2(qf, points + p * d, d);
        if (s32 <= cut) handle(p, s32);
    }
}

// The leaf/brute scan: screened when the float32 screen is valid.
template <int D>
inline void scan_rows(const double* q, const float* qf, const float* points, const std::int32_t* perm,
                      Py_ssize_t begin, Py_ssize_t end, Py_ssize_t d, const Thr& thr,
                      const Screen32& screen, std::int64_t self, RowSink& sink) {
    if (screen.on)
        scan_screened<D>(q, qf, points, perm, begin, end, d, thr, screen, self, sink);
    else
        scan_classify<D>(q, qf, points, perm, begin, end, d, thr, self, sink);
}

// Run `fn(row, sink, scratch)` for every query row, in tasks that cover
// consecutive entries of `order` (the identity when empty), and assemble a
// Result. fn appends the row's members through the sink; rows are then sorted
// in their task buffer while still in cache.
template <class RowFn>
std::unique_ptr<Result> run_rows(Py_ssize_t n_query, Py_ssize_t n_train, const Thr& thr,
                                 int mode, const std::vector<std::int64_t>& order,
                                 const std::vector<std::int64_t>& bounds, int threads,
                                 double estimated_ns, RowFn&& fn, std::size_t unique_columns = 0) {
    auto result = std::make_unique<Result>();
    result->n_query = n_query;
    result->n_train = n_train;
    result->R2 = thr.R2;
    result->mode = mode & (NEED_DISTANCE | SORT | COUNT_ONLY);
    const bool count_only = (mode & COUNT_ONLY) != 0;
    const bool keep_d2 = !count_only && (mode & NEED_DISTANCE);
    const bool by_distance = keep_d2 && (mode & SORT);
    const std::size_t n_tasks = bounds.empty() ? 0 : bounds.size() - 1;
    result->tasks.resize(n_tasks);
    result->row_task.resize(std::size_t(n_query));
    result->row_off.resize(std::size_t(n_query));
    result->row_cnt.resize(std::size_t(n_query));
    const bool fail = (mode & FAIL_TEST) != 0;
    Result* raw = result.get();
    parallel_for(n_tasks, threads, estimated_ns, [&](std::size_t task) {
        if (fail && task + 1 == n_tasks) throw std::runtime_error("injected failure in a native radius task");
        TaskOut& out = raw->tasks[task];
        SortScratch scratch;
        RowSink sink;
        sink.out = &out;
        sink.keep_d2 = keep_d2;
        sink.count_only = count_only;
        const std::size_t first = std::size_t(bounds[task]);
        const std::size_t last = std::size_t(bounds[task + 1]);
        for (std::size_t i = first; i < last; ++i) {
            const std::int64_t row = order.empty() ? std::int64_t(i) : order[i];
            const std::size_t begin = out.cols.size();
            sink.count = 0;
            fn(row, sink, scratch);
            if (!count_only) {
                const std::size_t count = out.cols.size() - begin;
                sort_row(out.cols.data() + begin, keep_d2 ? out.d2.data() + begin : nullptr, count,
                         by_distance, thr.R2, scratch, unique_columns);
            }
            raw->row_task[row] = std::int32_t(task);
            raw->row_off[row] = std::int64_t(begin);
            raw->row_cnt[row] = sink.count;
        }
        // Bound the doubling slack of finished buffers (<= ~15 B per pair
        // kept), so a result plus its CSR copy stays under 32 B per pair.
        if (out.cols.capacity() > out.cols.size() + out.cols.size() / 4 + 1024) {
            out.cols.shrink_to_fit();
            out.d2.shrink_to_fit();
        }
    });
    std::int64_t nnz = 0;
    for (auto count : result->row_cnt) nnz += count;
    result->nnz = nnz;
    return result;
}

// ---------------------------------------------------------------------------
// entry points implemented in _radius_tree.cpp
// ---------------------------------------------------------------------------

PyObject* py_build_tree(PyObject*, PyObject* args);
PyObject* py_tree_info(PyObject*, PyObject* args);
PyObject* py_tree_export(PyObject*, PyObject* args);
PyObject* py_tree_query(PyObject*, PyObject* args);
PyObject* py_tree_fill(PyObject*, PyObject* args);
PyObject* py_tree_pilot(PyObject*, PyObject* args);

}  // namespace mlx_radius
