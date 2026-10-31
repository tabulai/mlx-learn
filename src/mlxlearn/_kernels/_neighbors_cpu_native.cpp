// Exact original-coordinate CPU nearest neighbors. No NumPy C API is used.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {

using Key = std::array<std::uint64_t, 10>;

struct Buffer {
    Py_buffer view{};
    bool acquired = false;
    ~Buffer() { if (acquired) PyBuffer_Release(&view); }
};

bool get_buffer(PyObject* object, Buffer& buffer, bool writable, const char* name,
                char format, Py_ssize_t itemsize, std::size_t alignment) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, &buffer.view, flags) < 0) return false;
    buffer.acquired = true;
    const auto& v = buffer.view;
    bool correct_format = v.format && v.format[0] == format && v.format[1] == '\0';
    if (format == 'q' && v.format && v.format[0] == 'l' && v.format[1] == '\0')
        correct_format = sizeof(long) == 8;
    if (!correct_format || v.itemsize != itemsize || v.ndim != 2 || !v.shape ||
        v.shape[0] < 0 || v.shape[1] < 0 || v.len < 0 ||
        !PyBuffer_IsContiguous(&v, 'C') ||
        (reinterpret_cast<std::uintptr_t>(v.buf) % alignment) != 0) {
        PyErr_Format(PyExc_ValueError, "%s must be an aligned C-contiguous native %s matrix",
                     name, format == 'f' ? "float32" : format == 'd' ? "float64" : "int64");
        return false;
    }
    if (v.shape[1] > PY_SSIZE_T_MAX / itemsize ||
        (v.shape[1] && v.shape[0] > PY_SSIZE_T_MAX / (v.shape[1] * itemsize)) ||
        v.len != v.shape[0] * v.shape[1] * itemsize || (v.len && !v.buf)) {
        PyErr_Format(PyExc_ValueError, "%s has inconsistent buffer dimensions", name);
        return false;
    }
    auto address = reinterpret_cast<std::uintptr_t>(v.buf);
    if (static_cast<std::uint64_t>(v.len) > std::numeric_limits<std::uintptr_t>::max() - address) {
        PyErr_Format(PyExc_ValueError, "%s has an invalid address span", name);
        return false;
    }
    return true;
}

bool overlaps(const Buffer& a, const Buffer& b) {
    if (!a.acquired || !b.acquired || !a.view.len || !b.view.len) return false;
    auto start_a = reinterpret_cast<std::uintptr_t>(a.view.buf);
    auto start_b = reinterpret_cast<std::uintptr_t>(b.view.buf);
    return start_a < start_b + b.view.len && start_b < start_a + a.view.len;
}

std::uint32_t bits_of(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool finite_buffer(const float* values, Py_ssize_t count, bool& subnormal, bool& screen) {
    std::uint32_t invalid = 0, tiny = 0, outside = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        auto bits = bits_of(values[i]) & 0x7fffffffU;
        invalid |= bits >= 0x7f800000U;
        tiny |= bits && bits < 0x00800000U;
        outside |= bits && (bits < (95U << 23) || bits > (167U << 23));
    }
    subnormal |= tiny != 0;
    screen &= outside == 0;
    return invalid == 0;
}

// Float32 x is exactly signed significand * 2**shift in units of 2**-149.
struct Term { std::uint64_t significand; unsigned shift; bool negative; };
Term decode(float value) {
    auto bits = bits_of(value);
    unsigned exponent = (bits >> 23) & 255U;
    return {static_cast<std::uint64_t>((bits & 0x7fffffU) | (exponent ? 0x800000U : 0)),
            exponent ? exponent - 1 : 0, (bits >> 31) != 0};
}

void add_shifted(Key& accumulator, std::uint64_t value, unsigned shift) {
    if (!value) return;
    unsigned limb = shift / 64;
    unsigned bit = shift % 64;
    unsigned __int128 carry = static_cast<unsigned __int128>(value) << bit;
    while (carry) {
        carry += accumulator[limb];
        accumulator[limb] = static_cast<std::uint64_t>(carry);
        carry >>= 64;
        ++limb;
    }
}

Key exact_squared(const float* query, const float* point, Py_ssize_t features) {
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

double rounded_squared(const Key& key) {
    int highest_limb = 9;
    while (highest_limb >= 0 && key[highest_limb] == 0) --highest_limb;
    if (highest_limb < 0) return 0;
    unsigned bits = unsigned(highest_limb) * 64 +
                    64 - unsigned(__builtin_clzll(key[highest_limb]));
    if (bits <= 53) return std::ldexp(static_cast<double>(key[0]), -298);

    unsigned shift = bits - 53;
    unsigned limb = shift / 64, bit = shift % 64;
    std::uint64_t significand = key[limb] >> bit;
    if (bit && limb + 1 < key.size()) significand |= key[limb + 1] << (64 - bit);
    unsigned guard_position = shift - 1;
    bool guard = (key[guard_position / 64] >> (guard_position % 64)) & 1;
    bool sticky = false;
    for (unsigned i = 0; i < guard_position / 64; ++i) sticky |= key[i] != 0;
    unsigned tail = guard_position % 64;
    if (tail) sticky |= (key[guard_position / 64] & ((std::uint64_t(1) << tail) - 1)) != 0;
    if (guard && (sticky || (significand & 1))) ++significand;

    // A carry can produce exactly 2**53, still exactly representable. The
    // complete squared-distance range is normal float64, so this power-of-two
    // scaling introduces no further rounding, overflow, or underflow.
    return std::ldexp(static_cast<double>(significand), int(shift) - 298);
}

// The highest single-coordinate squared bit is below 2**556. Even INT32_MAX
// features fit below 2**587; ten limbs leave over 50 spare bits in either sum.
struct Neighbor {
    double squared, lower, upper;
    std::int64_t index;
    bool exact_ready;
    Key exact;
};

void copy_neighbor(Neighbor& destination, const Neighbor& source) {
    destination.squared = source.squared;
    destination.lower = source.lower;
    destination.upper = source.upper;
    destination.index = source.index;
    destination.exact_ready = source.exact_ready;
    if (source.exact_ready) destination.exact = source.exact;
}

int compare(Neighbor& left, Neighbor& right, const float* query,
            const float* train, Py_ssize_t features) {
    if (left.upper < right.lower) return -1;
    if (left.lower > right.upper) return 1;
    // All nonzero original float32 differences have normal float64 squares.
    if (left.squared == 0 && right.squared == 0)
        return left.index < right.index ? -1 : left.index > right.index ? 1 : 0;
    if (!left.exact_ready) {
        left.exact = exact_squared(query, train + left.index * features, features);
        left.exact_ready = true;
    }
    if (!right.exact_ready) {
        right.exact = exact_squared(query, train + right.index * features, features);
        right.exact_ready = true;
    }
    for (int limb = 9; limb >= 0; --limb) {
        if (left.exact[limb] < right.exact[limb]) return -1;
        if (left.exact[limb] > right.exact[limb]) return 1;
    }
    return left.index < right.index ? -1 : left.index > right.index ? 1 : 0;
}

double exact_conversion(float value) {
    auto bits = bits_of(value);
    auto magnitude = bits & 0x7fffffffU;
    if (magnitude && magnitude < 0x00800000U) {
        double result = std::ldexp(static_cast<double>(magnitude), -149);
        return bits >> 31 ? -result : result;
    }
    return static_cast<double>(value);
}

double squared_distance(const float* query, const float* point, Py_ssize_t features,
                        bool subnormal) {
    if (subnormal) {
        double sum = 0;
        for (Py_ssize_t j = 0; j < features; ++j) {
            double delta = exact_conversion(query[j]) - exact_conversion(point[j]);
            sum += delta * delta;
        }
        return sum;
    }
    Py_ssize_t feature = 0;
#if defined(__aarch64__)
    float64x2_t sum_low = vdupq_n_f64(0), sum_high = vdupq_n_f64(0);
    for (; feature + 4 <= features; feature += 4) {
        float32x4_t q = vld1q_f32(query + feature), x = vld1q_f32(point + feature);
        float64x2_t low = vsubq_f64(vcvt_f64_f32(vget_low_f32(q)), vcvt_f64_f32(vget_low_f32(x)));
        float64x2_t high = vsubq_f64(vcvt_f64_f32(vget_high_f32(q)), vcvt_f64_f32(vget_high_f32(x)));
        sum_low = vaddq_f64(sum_low, vmulq_f64(low, low));
        sum_high = vaddq_f64(sum_high, vmulq_f64(high, high));
    }
    double sum = vaddvq_f64(vaddq_f64(sum_low, sum_high));
#else
    double a = 0, b = 0, c = 0, d = 0;
    for (; feature + 4 <= features; feature += 4) {
        double x0 = double(query[feature]) - point[feature];
        double x1 = double(query[feature + 1]) - point[feature + 1];
        double x2 = double(query[feature + 2]) - point[feature + 2];
        double x3 = double(query[feature + 3]) - point[feature + 3];
        a += x0 * x0; b += x1 * x1; c += x2 * x2; d += x3 * x3;
    }
    double sum = (a + b) + (c + d);
#endif
    for (; feature < features; ++feature) {
        double delta = double(query[feature]) - point[feature];
        sum += delta * delta;
    }
    return sum;
}

double screened_squared_distance(const float* query, const float* point,
                                 Py_ssize_t features, double cutoff) {
    // Called only for D<=64 and coordinates zero or in [2**-32, 2**40].
    // Nonzero differences are >=2**-56, their squares >=2**-112, and the
    // complete sum <=2**88. Thus every nonzero arithmetic intermediate is
    // normal float32 and none overflows. The usual relative error bound applies.
    Py_ssize_t feature = 0;
#if defined(__aarch64__)
    float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
    float32x4_t c = vdupq_n_f32(0), d = vdupq_n_f32(0);
    if (features >= 16) {
        float32x4_t x0 = vsubq_f32(vld1q_f32(query), vld1q_f32(point));
        float32x4_t x1 = vsubq_f32(vld1q_f32(query + 4), vld1q_f32(point + 4));
        a = vmulq_f32(x0, x0);
        b = vmulq_f32(x1, x1);
        if (double(vaddvq_f32(vaddq_f32(a, b))) > cutoff) return -1;
        feature = 8;
    }
    for (; feature + 16 <= features; feature += 16) {
        float32x4_t x0 = vsubq_f32(vld1q_f32(query + feature), vld1q_f32(point + feature));
        float32x4_t x1 = vsubq_f32(vld1q_f32(query + feature + 4), vld1q_f32(point + feature + 4));
        float32x4_t x2 = vsubq_f32(vld1q_f32(query + feature + 8), vld1q_f32(point + feature + 8));
        float32x4_t x3 = vsubq_f32(vld1q_f32(query + feature + 12), vld1q_f32(point + feature + 12));
        a = vaddq_f32(a, vmulq_f32(x0, x0));
        b = vaddq_f32(b, vmulq_f32(x1, x1));
        c = vaddq_f32(c, vmulq_f32(x2, x2));
        d = vaddq_f32(d, vmulq_f32(x3, x3));
    }
    for (; feature + 4 <= features; feature += 4) {
        float32x4_t delta = vsubq_f32(vld1q_f32(query + feature), vld1q_f32(point + feature));
        a = vaddq_f32(a, vmulq_f32(delta, delta));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
#else
    float sum = 0;
    if (features >= 16) {
        for (; feature < 8; ++feature) {
            float delta = query[feature] - point[feature];
            sum += delta * delta;
        }
        if (double(sum) > cutoff) return -1;
    }
#endif
    for (; feature < features; ++feature) {
        float delta = query[feature] - point[feature];
        sum += delta * delta;
    }
    return sum;
}

struct QueryRows {
    const float* x;
    const float* q;
    double* out_distances;
    std::int64_t* out_indices;
    Py_ssize_t n, features, k, offset;
    bool exclude_self, screen, subnormal;
    double error;
};

template <bool FastCutoff>
void query_rows(const QueryRows& data, Neighbor* best, Py_ssize_t begin, Py_ssize_t end) {
    const auto& [x, q, out_distances, out_indices, n, features, k, offset,
                 exclude_self, screen, subnormal, error] = data;
    const double cutoff_factor = 1 + 2 * error;
    for (Py_ssize_t row = begin; row < end; ++row) {
        const float* current_query = q + row * features;
        Py_ssize_t count = 0;
        double cutoff = std::numeric_limits<double>::infinity();
        for (Py_ssize_t reference = 0; reference < n; ++reference) {
            if (exclude_self && reference == offset + row) continue;
            Neighbor candidate;
            candidate.squared = screen
                ? screened_squared_distance(current_query, x + reference * features, features, cutoff)
                : squared_distance(current_query, x + reference * features, features, subnormal);
            if (candidate.squared < 0) continue;
            candidate.lower = candidate.squared * (1 - error);
            candidate.upper = candidate.squared * (1 + error);
            candidate.index = reference;
            candidate.exact_ready = false;
            if (count == k && compare(candidate, best[k - 1], current_query, x, features) >= 0) continue;
            Py_ssize_t position = count < k ? count : k - 1;
            while (position && compare(candidate, best[position - 1], current_query, x, features) < 0) {
                copy_neighbor(best[position], best[position - 1]);
                --position;
            }
            copy_neighbor(best[position], candidate);
            if (count < k) ++count;
            if (screen && count == k) {
                // A partial squared distance has the same conservative
                // relative error envelope as a full sum. Rounding outward
                // ensures partial-sum rejection cannot lose a boundary tie.
                if constexpr (FastCutoff) {
                    // For the FP32 screen, 2**-18 <= e <= 65*2**-19.
                    // (1+2e)*(1-e) = 1+e-2e*e exceeds one by >2**-19,
                    // far beyond the FP64 multiplication's rounding error.
                    // The factor is exactly representable; nonzero screened
                    // sums lie in [2**-112,2**88], so this looser bound has
                    // neither FP64 underflow nor overflow. Zero stays zero.
                    cutoff = best[k - 1].upper * cutoff_factor;
                } else {
                    cutoff = std::nextafter(best[k - 1].upper / (1 - error),
                                            std::numeric_limits<double>::infinity());
                }
            }
        }
        for (Py_ssize_t neighbor = 0; neighbor < k; ++neighbor) {
            out_indices[row * k + neighbor] = best[neighbor].index;
            if (out_distances) {
                // Equal exact keys must also have equal reported distances
                // and downstream distance weights, independent of feature
                // order or the approximate reduction's grouping.
                double squared = best[neighbor].exact_ready ? rounded_squared(best[neighbor].exact)
                    : screen ? squared_distance(current_query, x + best[neighbor].index * features, features, false)
                             : best[neighbor].squared;
                out_distances[row * k + neighbor] = std::sqrt(squared);
            }
        }
    }
}

template <bool Configurable>
PyObject* query_impl(PyObject* args) {
    PyObject *train_object, *query_object, *distance_object, *index_object;
    Py_ssize_t k, offset;
    int exclude_self;
    if (!PyArg_ParseTuple(args, "OOnpnOO:query", &train_object, &query_object, &k,
                          &exclude_self, &offset, &distance_object, &index_object)) return nullptr;
    Buffer train, queries, distances, indices;
    if (!get_buffer(train_object, train, false, "train", 'f', 4, alignof(float)) ||
        !get_buffer(query_object, queries, false, "queries", 'f', 4, alignof(float)) ||
        (distance_object != Py_None && !get_buffer(distance_object, distances, true, "distances", 'd', 8, alignof(double))) ||
        !get_buffer(index_object, indices, true, "indices", 'q', 8, alignof(std::int64_t))) return nullptr;
    Py_ssize_t n = train.view.shape[0], features = train.view.shape[1], m = queries.view.shape[0];
    if (n <= 0 || features <= 0 || features > INT32_MAX || queries.view.shape[1] != features ||
        k <= 0 || k > n - exclude_self || offset < 0 ||
        (exclude_self && (offset > n || m > n - offset)) ||
        indices.view.shape[0] != m || indices.view.shape[1] != k ||
        (distances.acquired && (distances.view.shape[0] != m || distances.view.shape[1] != k))) {
        PyErr_SetString(PyExc_ValueError, "invalid nearest-neighbor dimensions, k, or self offset");
        return nullptr;
    }
    if (overlaps(indices, train) || overlaps(indices, queries) || overlaps(distances, train) ||
        overlaps(distances, queries) || overlaps(indices, distances)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap inputs or each other");
        return nullptr;
    }
    auto* x = static_cast<const float*>(train.view.buf);
    auto* q = static_cast<const float*>(queries.view.buf);
    auto* out_distances = distances.acquired ? static_cast<double*>(distances.view.buf) : nullptr;
    auto* out_indices = static_cast<std::int64_t*>(indices.view.buf);
    bool subnormal = false;
    bool screen = features <= 64;
    if (!finite_buffer(x, n * features, subnormal, screen) ||
        !finite_buffer(q, m * features, subnormal, screen)) {
        PyErr_SetString(PyExc_ValueError, "training and query coordinates must be finite");
        return nullptr;
    }
    try {
        std::vector<Neighbor> best(static_cast<std::size_t>(k));
        const double epsilon = screen ? std::numeric_limits<float>::epsilon()
                                      : std::numeric_limits<double>::epsilon();
        const double error = 16.0 * (double(features) + 1) * epsilon;
        Py_BEGIN_ALLOW_THREADS
        QueryRows data{x, q, out_distances, out_indices, n, features, k, offset,
                       bool(exclude_self), screen, subnormal, error};
        query_rows<Configurable>(data, best.data(), 0, m);
        Py_END_ALLOW_THREADS
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
}

PyObject* query(PyObject*, PyObject* args) { return query_impl<false>(args); }
PyObject* query_concentrated(PyObject*, PyObject* args) { return query_impl<true>(args); }

#if defined(__aarch64__)
uint32x4_t bounded_integer_mask(float32x4_t value, bool nonnegative) {
    uint32x4_t magnitude = vandq_u32(vreinterpretq_u32_f32(value), vdupq_n_u32(0x7fffffffU));
    uint32x4_t integral_magnitude = vorrq_u32(vceqq_u32(magnitude, vdupq_n_u32(0)),
                                             vcgeq_u32(magnitude, vdupq_n_u32(0x3f800000U)));
    uint32x4_t mask = vandq_u32(vceqq_f32(value, vrndq_f32(value)),
                               vcleq_f32(vabsq_f32(value), vdupq_n_f32(4194304.0f)));
    // Raw-bit rejection keeps subnormals from being accepted as integer
    // zero even if the caller's FPCR flushes float32 subnormals.
    mask = vandq_u32(mask, integral_magnitude);
    if (nonnegative) mask = vandq_u32(mask, vcgeq_f32(value, vdupq_n_f32(0)));
    return mask;
}
#endif

bool bounded_integer_buffer(const float* values, Py_ssize_t count, bool nonnegative) {
    // A certified coordinate matrix has each norm and absolute dot product
    // <=2**22. Checking these integer intermediates makes the fused expression
    // exact even when this private entry point is called independently.
    Py_ssize_t i = 0;
#if defined(__aarch64__)
    uint32x4_t valid0 = vdupq_n_u32(~std::uint32_t(0));
    uint32x4_t valid1 = valid0, valid2 = valid0, valid3 = valid0;
    for (; i + 16 <= count; i += 16) {
        valid0 = vandq_u32(valid0, bounded_integer_mask(vld1q_f32(values + i), nonnegative));
        valid1 = vandq_u32(valid1, bounded_integer_mask(vld1q_f32(values + i + 4), nonnegative));
        valid2 = vandq_u32(valid2, bounded_integer_mask(vld1q_f32(values + i + 8), nonnegative));
        valid3 = vandq_u32(valid3, bounded_integer_mask(vld1q_f32(values + i + 12), nonnegative));
    }
    uint32x4_t valid = vandq_u32(vandq_u32(valid0, valid1), vandq_u32(valid2, valid3));
    for (; i + 4 <= count; i += 4) {
        valid = vandq_u32(valid, bounded_integer_mask(vld1q_f32(values + i), nonnegative));
    }
    if (!vminvq_u32(valid)) return false;
#endif
    for (; i < count; ++i) {
        float value = values[i];
        auto magnitude = bits_of(value) & 0x7fffffffU;
        if ((magnitude && magnitude < 0x3f800000U) ||
            !(std::abs(value) <= 4194304.0f) || value != std::trunc(value) ||
            (nonnegative && value < 0)) return false;
    }
    return true;
}

struct IntegerNeighbor { float squared; std::int64_t index; };

struct IntegerTopFive {
    float s0 = std::numeric_limits<float>::infinity(), s1 = s0, s2 = s0, s3 = s0, s4 = s0;
    std::int64_t i0 = -1, i1 = -1, i2 = -1, i3 = -1, i4 = -1;

    void insert(float squared, std::int64_t index) {
        // Incoming original IDs increase strictly. Equal values therefore go
        // after existing values, including at the final retained boundary.
        if (squared >= s4) return;
        if (squared >= s3) { s4 = squared; i4 = index; return; }
        s4 = s3; i4 = i3;
        if (squared >= s2) { s3 = squared; i3 = index; return; }
        s3 = s2; i3 = i2;
        if (squared >= s1) { s2 = squared; i2 = index; return; }
        s2 = s1; i2 = i1;
        if (squared >= s0) { s1 = squared; i1 = index; return; }
        s1 = s0; i1 = i0;
        s0 = squared; i0 = index;
    }

#if defined(__aarch64__)
    void insert_vector(float32x4_t values, Py_ssize_t reference, Py_ssize_t excluded) {
        if (vminvq_f32(values) >= s4) return;
        float lanes[4];
        vst1q_f32(lanes, values);
        for (unsigned lane = 0; lane < 4; ++lane) {
            Py_ssize_t index = reference + lane;
            if (index != excluded) insert(lanes[lane], index);
        }
    }
#endif

    void write(double* squared, std::int64_t* indices) const {
        indices[0] = i0; indices[1] = i1; indices[2] = i2; indices[3] = i3; indices[4] = i4;
        if (squared) {
            squared[0] = s0; squared[1] = s1; squared[2] = s2; squared[3] = s3; squared[4] = s4;
        }
    }
};

enum class IntegerSelectionStatus { valid, invalid_input, negative_squared };

IntegerSelectionStatus select_integer_five(const float* dot, const float* xn, const float* qn,
                         Py_ssize_t m, Py_ssize_t n, bool exclude_self, Py_ssize_t offset,
                         IntegerTopFive* results) {
    bool negative_squared = false, invalid_input = false;
#if defined(__aarch64__)
    uint32x4_t valid0 = vdupq_n_u32(~std::uint32_t(0));
    uint32x4_t valid1 = valid0, valid2 = valid0, valid3 = valid0;
#endif
    for (Py_ssize_t row = 0; row < m; ++row) {
        IntegerTopFive best;
        const float* current = dot + row * n;
        Py_ssize_t reference = 0;
        const Py_ssize_t excluded = exclude_self ? offset + row : -1;
#if defined(__aarch64__)
        const float32x4_t query_norm = vdupq_n_f32(qn[row]);
        for (; reference + 16 <= n; reference += 16) {
            float32x4_t dots0 = vld1q_f32(current + reference);
            float32x4_t dots1 = vld1q_f32(current + reference + 4);
            float32x4_t dots2 = vld1q_f32(current + reference + 8);
            float32x4_t dots3 = vld1q_f32(current + reference + 12);
            uint32x4_t mask0 = bounded_integer_mask(dots0, false);
            uint32x4_t mask1 = bounded_integer_mask(dots1, false);
            uint32x4_t mask2 = bounded_integer_mask(dots2, false);
            uint32x4_t mask3 = bounded_integer_mask(dots3, false);
            valid0 = vandq_u32(valid0, mask0); valid1 = vandq_u32(valid1, mask1);
            valid2 = vandq_u32(valid2, mask2); valid3 = vandq_u32(valid3, mask3);
            // Invalid lanes still participate in the global validation mask,
            // including excluded IDs and blocks that cannot enter top-k. Mask
            // them to zero before arithmetic; results are private until every
            // dot is certified. No malformed input can expose partial output.
            dots0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dots0), mask0));
            dots1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dots1), mask1));
            dots2 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dots2), mask2));
            dots3 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dots3), mask3));
            float32x4_t values0 = vaddq_f32(vaddq_f32(
                vmulq_n_f32(dots0, -2),
                vld1q_f32(xn + reference)), query_norm);
            float32x4_t values1 = vaddq_f32(vaddq_f32(
                vmulq_n_f32(dots1, -2),
                vld1q_f32(xn + reference + 4)), query_norm);
            float32x4_t values2 = vaddq_f32(vaddq_f32(
                vmulq_n_f32(dots2, -2),
                vld1q_f32(xn + reference + 8)), query_norm);
            float32x4_t values3 = vaddq_f32(vaddq_f32(
                vmulq_n_f32(dots3, -2),
                vld1q_f32(xn + reference + 12)), query_norm);
            float minimum = vminvq_f32(vminq_f32(vminq_f32(values0, values1),
                                                vminq_f32(values2, values3)));
            if (minimum < 0) { negative_squared = true; continue; }
            if (minimum >= best.s4) continue;
            best.insert_vector(values0, reference, excluded);
            best.insert_vector(values1, reference + 4, excluded);
            best.insert_vector(values2, reference + 8, excluded);
            best.insert_vector(values3, reference + 12, excluded);
        }
        for (; reference + 4 <= n; reference += 4) {
            float32x4_t dots = vld1q_f32(current + reference);
            uint32x4_t mask = bounded_integer_mask(dots, false);
            valid0 = vandq_u32(valid0, mask);
            dots = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dots), mask));
            float32x4_t values = vaddq_f32(vaddq_f32(
                vmulq_n_f32(dots, -2),
                vld1q_f32(xn + reference)), query_norm);
            if (vminvq_f32(values) < 0) { negative_squared = true; continue; }
            best.insert_vector(values, reference, excluded);
        }
#endif
        for (; reference < n; ++reference) {
            if (!bounded_integer_buffer(current + reference, 1, false)) {
                invalid_input = true;
                continue;
            }
            float squared = (-2 * current[reference] + xn[reference]) + qn[row];
            if (squared < 0) { negative_squared = true; continue; }
            if (reference != excluded) best.insert(squared, reference);
        }
        results[row] = best;
    }
#if defined(__aarch64__)
    invalid_input = invalid_input || !vminvq_u32(
        vandq_u32(vandq_u32(valid0, valid1), vandq_u32(valid2, valid3)));
#endif
    // Preserve the full prepass's error priority when an early negative score
    // is followed by a malformed dot. Staged outputs are committed only after
    // both checks, also leaving outputs intact for inconsistent Gram scores.
    return invalid_input ? IntegerSelectionStatus::invalid_input
        : negative_squared ? IntegerSelectionStatus::negative_squared
        : IntegerSelectionStatus::valid;
}

void insert_integer_neighbor(std::vector<IntegerNeighbor>& best, Py_ssize_t& count,
                             Py_ssize_t k, float squared, Py_ssize_t index) {
    // References arrive in ascending original-index order. A later equal
    // distance therefore cannot replace the existing kth point.
    if (count == k && squared >= best[k - 1].squared) return;
    Py_ssize_t position = count < k ? count : k - 1;
    while (position && squared < best[position - 1].squared) {
        best[position] = best[position - 1];
        --position;
    }
    best[position] = {squared, index};
    if (count < k) ++count;
}

#if defined(__aarch64__)
void insert_integer_vector(float32x4_t values, std::vector<IntegerNeighbor>& best,
                           Py_ssize_t& count, Py_ssize_t k, Py_ssize_t reference,
                           Py_ssize_t excluded) {
    if (count == k && vminvq_f32(values) >= best[k - 1].squared) return;
    float lanes[4];
    vst1q_f32(lanes, values);
    for (unsigned lane = 0; lane < 4; ++lane) {
        Py_ssize_t index = reference + lane;
        if (index != excluded) insert_integer_neighbor(best, count, k, lanes[lane], index);
    }
}
#endif

PyObject* select_integer_gram(PyObject*, PyObject* args) {
    PyObject *dot_object, *train_norm_object, *query_norm_object, *squared_object, *index_object;
    Py_ssize_t k, offset;
    int exclude_self;
    if (!PyArg_ParseTuple(args, "OOOnpnOO:select_integer_gram", &dot_object, &train_norm_object,
                          &query_norm_object, &k, &exclude_self, &offset, &squared_object,
                          &index_object)) return nullptr;
    Buffer dots, train_norms, query_norms, squared, indices;
    if (!get_buffer(dot_object, dots, false, "dots", 'f', 4, alignof(float)) ||
        !get_buffer(train_norm_object, train_norms, false, "train_norms", 'f', 4, alignof(float)) ||
        !get_buffer(query_norm_object, query_norms, false, "query_norms", 'f', 4, alignof(float)) ||
        (squared_object != Py_None && !get_buffer(squared_object, squared, true, "squared", 'd', 8, alignof(double))) ||
        !get_buffer(index_object, indices, true, "indices", 'q', 8, alignof(std::int64_t))) return nullptr;
    Py_ssize_t m = dots.view.shape[0], n = dots.view.shape[1];
    if (n <= 0 || k <= 0 || k > n - exclude_self || offset < 0 ||
        (exclude_self && (offset > n || m > n - offset)) ||
        train_norms.view.shape[0] != 1 || train_norms.view.shape[1] != n ||
        query_norms.view.shape[0] != m || query_norms.view.shape[1] != 1 ||
        indices.view.shape[0] != m || indices.view.shape[1] != k ||
        (squared.acquired && (squared.view.shape[0] != m || squared.view.shape[1] != k))) {
        PyErr_SetString(PyExc_ValueError, "invalid integer Gram dimensions, k, or self offset");
        return nullptr;
    }
    if (overlaps(squared, indices) || overlaps(squared, dots) || overlaps(squared, train_norms) ||
        overlaps(squared, query_norms) || overlaps(indices, dots) ||
        overlaps(indices, train_norms) || overlaps(indices, query_norms)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap inputs or each other");
        return nullptr;
    }
    auto* dot = static_cast<const float*>(dots.view.buf);
    auto* xn = static_cast<const float*>(train_norms.view.buf);
    auto* qn = static_cast<const float*>(query_norms.view.buf);
    if ((k != 5 && !bounded_integer_buffer(dot, m * n, false)) || !bounded_integer_buffer(xn, n, true) ||
        !bounded_integer_buffer(qn, m, true)) {
        PyErr_SetString(PyExc_ValueError, "integer Gram inputs must be integral and bounded by 2**22; norms must be nonnegative");
        return nullptr;
    }
    auto* out_squared = squared.acquired ? static_cast<double*>(squared.view.buf) : nullptr;
    auto* out_indices = static_cast<std::int64_t*>(indices.view.buf);
    bool invalid_squared = false;
    try {
        if (k == 5) {
            std::vector<IntegerTopFive> results(static_cast<std::size_t>(m));
            IntegerSelectionStatus status;
            Py_BEGIN_ALLOW_THREADS
            status = select_integer_five(dot, xn, qn, m, n, bool(exclude_self), offset, results.data());
            if (status == IntegerSelectionStatus::valid) {
                for (Py_ssize_t row = 0; row < m; ++row)
                    results[row].write(out_squared ? out_squared + row * 5 : nullptr, out_indices + row * 5);
            }
            Py_END_ALLOW_THREADS
            if (status == IntegerSelectionStatus::invalid_input) {
                PyErr_SetString(PyExc_ValueError, "integer Gram inputs must be integral and bounded by 2**22; norms must be nonnegative");
                return nullptr;
            }
            invalid_squared = status == IntegerSelectionStatus::negative_squared;
        } else {
            std::vector<IntegerNeighbor> best(static_cast<std::size_t>(k));
            Py_BEGIN_ALLOW_THREADS
            for (Py_ssize_t row = 0; row < m && !invalid_squared; ++row) {
                Py_ssize_t count = 0, reference = 0;
                const float* current = dot + row * n;
    #if defined(__aarch64__)
                const float32x4_t query_norm = vdupq_n_f32(qn[row]);
                const Py_ssize_t excluded = exclude_self ? offset + row : -1;
                for (; reference + 16 <= n; reference += 16) {
                    float32x4_t values0 = vaddq_f32(vaddq_f32(
                        vmulq_n_f32(vld1q_f32(current + reference), -2),
                        vld1q_f32(xn + reference)), query_norm);
                    float32x4_t values1 = vaddq_f32(vaddq_f32(
                        vmulq_n_f32(vld1q_f32(current + reference + 4), -2),
                        vld1q_f32(xn + reference + 4)), query_norm);
                    float32x4_t values2 = vaddq_f32(vaddq_f32(
                        vmulq_n_f32(vld1q_f32(current + reference + 8), -2),
                        vld1q_f32(xn + reference + 8)), query_norm);
                    float32x4_t values3 = vaddq_f32(vaddq_f32(
                        vmulq_n_f32(vld1q_f32(current + reference + 12), -2),
                        vld1q_f32(xn + reference + 12)), query_norm);
                    float minimum = vminvq_f32(vminq_f32(vminq_f32(values0, values1),
                                                        vminq_f32(values2, values3)));
                    if (minimum < 0) { invalid_squared = true; break; }
                    if (count == k && minimum >= best[k - 1].squared) continue;
                    insert_integer_vector(values0, best, count, k, reference, excluded);
                    insert_integer_vector(values1, best, count, k, reference + 4, excluded);
                    insert_integer_vector(values2, best, count, k, reference + 8, excluded);
                    insert_integer_vector(values3, best, count, k, reference + 12, excluded);
                }
                for (; reference + 4 <= n && !invalid_squared; reference += 4) {
                    float32x4_t values = vaddq_f32(
                        vaddq_f32(vmulq_n_f32(vld1q_f32(current + reference), -2),
                                  vld1q_f32(xn + reference)), query_norm);
                    float minimum = vminvq_f32(values);
                    if (minimum < 0) { invalid_squared = true; break; }
                    insert_integer_vector(values, best, count, k, reference, excluded);
                }
    #endif
                for (; reference < n && !invalid_squared; ++reference) {
                    float value = (-2 * current[reference] + xn[reference]) + qn[row];
                    if (value < 0) { invalid_squared = true; break; }
                    if (!exclude_self || reference != offset + row)
                        insert_integer_neighbor(best, count, k, value, reference);
                }
                if (invalid_squared) break;
                for (Py_ssize_t neighbor = 0; neighbor < k; ++neighbor) {
                    out_indices[row * k + neighbor] = best[neighbor].index;
                    if (out_squared) out_squared[row * k + neighbor] = best[neighbor].squared;
                }
            }
            Py_END_ALLOW_THREADS
        }
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        return PyErr_NoMemory();
    }
    if (invalid_squared) {
        PyErr_SetString(PyExc_ValueError, "integer Gram inputs produce a negative squared distance");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* feature_ranges(PyObject*, PyObject* args) {
    PyObject *train_object, *range_object;
    if (!PyArg_ParseTuple(args, "OO:feature_ranges", &train_object, &range_object)) return nullptr;
    Buffer train, ranges;
    if (!get_buffer(train_object, train, false, "train", 'f', 4, alignof(float)) ||
        !get_buffer(range_object, ranges, true, "ranges", 'd', 8, alignof(double))) return nullptr;
    Py_ssize_t n = train.view.shape[0], features = train.view.shape[1];
    if (n <= 0 || features <= 0 || ranges.view.shape[0] != 1 || ranges.view.shape[1] != features) {
        PyErr_SetString(PyExc_ValueError, "invalid feature range dimensions");
        return nullptr;
    }
    if (overlaps(train, ranges)) {
        PyErr_SetString(PyExc_ValueError, "range output must not overlap training storage");
        return nullptr;
    }
    const float* values = static_cast<const float*>(train.view.buf);
    double* output = static_cast<double*>(ranges.view.buf);
    bool invalid = false;
    try {
        std::vector<std::uint32_t> low(static_cast<std::size_t>(features), 0xffffffffU);
        std::vector<std::uint32_t> high(static_cast<std::size_t>(features), 0);
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t row = 0; row < n; ++row) {
            Py_ssize_t feature = 0;
#if defined(__aarch64__)
            uint32x4_t nonfinite = vdupq_n_u32(0);
            for (; feature + 4 <= features; feature += 4) {
                uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(values + row * features + feature));
                uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(0x7fffffffU));
                nonfinite = vorrq_u32(nonfinite, vcgeq_u32(magnitude, vdupq_n_u32(0x7f800000U)));
                uint32x4_t sign_mask = vsubq_u32(vdupq_n_u32(0), vshrq_n_u32(bits, 31));
                uint32x4_t ordered = veorq_u32(bits, vorrq_u32(sign_mask, vdupq_n_u32(0x80000000U)));
                vst1q_u32(low.data() + feature, vminq_u32(vld1q_u32(low.data() + feature), ordered));
                vst1q_u32(high.data() + feature, vmaxq_u32(vld1q_u32(high.data() + feature), ordered));
            }
            invalid |= vmaxvq_u32(nonfinite) != 0;
#endif
            for (; feature < features; ++feature) {
                auto bits = bits_of(values[row * features + feature]);
                invalid |= (bits & 0x7fffffffU) >= 0x7f800000U;
                auto ordered = bits ^ (bits >> 31 ? 0xffffffffU : 0x80000000U);
                if (ordered < low[feature]) low[feature] = ordered;
                if (ordered > high[feature]) high[feature] = ordered;
            }
        }
        if (!invalid) {
            for (Py_ssize_t feature = 0; feature < features; ++feature) {
                // Total-order keys keep original finite float32 extrema,
                // including signed zero and subnormals, without FP32 compares.
                auto low_bits = low[feature] & 0x80000000U ? low[feature] ^ 0x80000000U : ~low[feature];
                auto high_bits = high[feature] & 0x80000000U ? high[feature] ^ 0x80000000U : ~high[feature];
                float minimum, maximum;
                std::memcpy(&minimum, &low_bits, sizeof(minimum));
                std::memcpy(&maximum, &high_bits, sizeof(maximum));
                output[feature] = exact_conversion(maximum) - exact_conversion(minimum);
            }
        }
        Py_END_ALLOW_THREADS
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        return PyErr_NoMemory();
    }
    if (invalid) {
        PyErr_SetString(PyExc_ValueError, "training coordinates must be finite");
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"query", query, METH_VARARGS,
     "query(train, queries, k, exclude_self, offset, distances_or_none, indices) -> None"},
    {"query_concentrated", query_concentrated, METH_VARARGS,
     "query_concentrated(train, queries, k, exclude_self, offset, distances_or_none, indices) -> None"},
    {"select_integer_gram", select_integer_gram, METH_VARARGS,
     "select_integer_gram(dots, train_norms_1xn, query_norms_mx1, k, exclude_self, offset, squared_or_none, indices) -> None"},
    {"feature_ranges", feature_ranges, METH_VARARGS,
     "feature_ranges(train, ranges_1xd_float64) -> None"},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_neighbors_cpu_native",
    "Exact original-float32 CPU neighbors with certified dyadic tie ordering.",
    -1, methods, nullptr, nullptr, nullptr, nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__neighbors_cpu_native() {
    PyObject* result = PyModule_Create(&module);
    if (result && PyModule_AddStringConstant(result, "BUILD_VERSION", "1") < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
