// Native SVC reductions and multiclass aggregation. No NumPy C API is used.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace {

struct Buffer {
    Py_buffer view{};
    bool acquired = false;
    ~Buffer() { if (acquired) PyBuffer_Release(&view); }
};

struct ReleaseGIL {
    PyThreadState* state = PyEval_SaveThread();
    ~ReleaseGIL() { PyEval_RestoreThread(state); }
};

bool get_buffer(PyObject* object, Buffer& buffer, bool writable, const char* name,
                char format, int ndim) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, &buffer.view, flags) < 0) return false;
    buffer.acquired = true;
    const auto& v = buffer.view;
    const char actual = v.format && v.format[0] && v.format[1] == '\0' ? v.format[0] : '\0';
    const bool integer = format == 'q' && (actual == 'q' || (actual == 'l' && sizeof(long) == 8));
    const bool floating = format == '*' && (actual == 'f' || actual == 'd');
    const Py_ssize_t size = actual == 'f' ? 4 : 8;
    if (!(actual == format || integer || floating) || v.itemsize != size ||
        v.ndim != ndim || !v.shape || v.len < 0 || !PyBuffer_IsContiguous(&v, 'C') ||
        reinterpret_cast<std::uintptr_t>(v.buf) % static_cast<std::size_t>(size)) {
        PyErr_Format(PyExc_ValueError, "%s has an unsupported dtype, shape, or alignment", name);
        return false;
    }
    Py_ssize_t length = size;
    for (int axis = 0; axis < ndim; ++axis) {
        if (v.shape[axis] < 0 || (v.shape[axis] && length > PY_SSIZE_T_MAX / v.shape[axis])) {
            PyErr_Format(PyExc_ValueError, "%s has invalid buffer dimensions", name);
            return false;
        }
        length *= v.shape[axis];
    }
    const auto address = reinterpret_cast<std::uintptr_t>(v.buf);
    if (v.len != length || (v.len && !v.buf) ||
        static_cast<std::uint64_t>(v.len) > std::numeric_limits<std::uintptr_t>::max() - address) {
        PyErr_Format(PyExc_ValueError, "%s has an inconsistent buffer span", name);
        return false;
    }
    return true;
}

bool overlaps(const Buffer& a, const Buffer& b) {
    if (!a.acquired || !b.acquired || !a.view.len || !b.view.len) return false;
    const auto start_a = reinterpret_cast<std::uintptr_t>(a.view.buf);
    const auto start_b = reinterpret_cast<std::uintptr_t>(b.view.buf);
    return start_a < start_b + b.view.len && start_b < start_a + a.view.len;
}

bool output_safe(const Buffer& output, const Buffer& a, const Buffer& b = Buffer{},
                 const Buffer& c = Buffer{}) {
    if (overlaps(output, a) || overlaps(output, b) || overlaps(output, c)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap input buffers");
        return false;
    }
    return true;
}

// Neumaier's correction keeps small terms even when a later term is larger
// than the running sum. Preserve nonfinite results without inf-inf corrections.
struct Sum {
    double value = 0.0, correction = 0.0;
    void add(double term) {
        const double total = value + term;
        if (std::isfinite(total)) {
            correction += std::abs(value) >= std::abs(term)
                ? (value - total) + term : (term - total) + value;
        } else {
            correction = 0.0;
        }
        value = total;
    }
    double get() const { return value + correction; }
};

#if defined(__aarch64__)
struct VectorSum {
    float64x2_t value = vdupq_n_f64(0.0), correction = vdupq_n_f64(0.0);
    void add(float64x2_t term) {
        const auto total = vaddq_f64(value, term);
        const auto larger = vcgeq_f64(vabsq_f64(value), vabsq_f64(term));
        const auto error = vbslq_f64(larger,
            vaddq_f64(vsubq_f64(value, total), term),
            vaddq_f64(vsubq_f64(term, total), value));
        correction = vbslq_f64(vcltq_f64(vabsq_f64(total), vdupq_n_f64(INFINITY)),
                               vaddq_f64(correction, error), vdupq_n_f64(0.0));
        value = total;
    }
};

double binary_dot(const float* k, const double* w, Py_ssize_t support) {
    VectorSum parts[4];
    Py_ssize_t s = 0;
    // Eight independent lanes hide the latency of compensated accumulation.
    for (; s + 8 <= support; s += 8) {
        const auto low = vld1q_f32(k + s), high = vld1q_f32(k + s + 4);
        parts[0].add(vmulq_f64(vcvt_f64_f32(vget_low_f32(low)), vld1q_f64(w + s)));
        parts[1].add(vmulq_f64(vcvt_f64_f32(vget_high_f32(low)), vld1q_f64(w + s + 2)));
        parts[2].add(vmulq_f64(vcvt_f64_f32(vget_low_f32(high)), vld1q_f64(w + s + 4)));
        parts[3].add(vmulq_f64(vcvt_f64_f32(vget_high_f32(high)), vld1q_f64(w + s + 6)));
    }
    Sum result;
    for (const auto& part : parts) {
        result.add(vgetq_lane_f64(part.value, 0));
        result.add(vgetq_lane_f64(part.correction, 0));
        result.add(vgetq_lane_f64(part.value, 1));
        result.add(vgetq_lane_f64(part.correction, 1));
    }
    for (; s < support; ++s) result.add(static_cast<double>(k[s]) * w[s]);
    return result.get();
}

void pair_dot(const float* k, const double* w, Py_ssize_t support,
              Py_ssize_t stride, double* out) {
    VectorSum parts[4];
    Py_ssize_t s = 0;
    for (; s + 4 <= support; s += 4) {
        parts[0].add(vmulq_n_f64(vld1q_f64(w + s * stride), static_cast<double>(k[s])));
        parts[1].add(vmulq_n_f64(vld1q_f64(w + (s + 1) * stride), static_cast<double>(k[s + 1])));
        parts[2].add(vmulq_n_f64(vld1q_f64(w + (s + 2) * stride), static_cast<double>(k[s + 2])));
        parts[3].add(vmulq_n_f64(vld1q_f64(w + (s + 3) * stride), static_cast<double>(k[s + 3])));
    }
    Sum left, right;
    for (const auto& part : parts) {
        left.add(vgetq_lane_f64(part.value, 0));
        left.add(vgetq_lane_f64(part.correction, 0));
        right.add(vgetq_lane_f64(part.value, 1));
        right.add(vgetq_lane_f64(part.correction, 1));
    }
    for (; s < support; ++s) {
        left.add(static_cast<double>(k[s]) * w[s * stride]);
        right.add(static_cast<double>(k[s]) * w[s * stride + 1]);
    }
    out[0] = left.get(); out[1] = right.get();
}
#endif

#if defined(__APPLE__)
void tiled_contract(const float* k, const double* w, const double* bias, double* out,
                    Py_ssize_t rows, Py_ssize_t support, Py_ssize_t pairs,
                    double* scratch, Py_ssize_t row_tile, Py_ssize_t support_tile) {
    // At most four MiB is widened, regardless of either Gram dimension. BLAS
    // computes only float64 products/reductions and never receives host Python
    // callbacks. Accumulation order can differ from a single full-size BLAS call.
    for (Py_ssize_t start = 0; start < rows; start += row_tile) {
        const Py_ssize_t count = std::min(row_tile, rows - start);
        for (Py_ssize_t offset = 0; offset < support; offset += support_tile) {
            const Py_ssize_t width = std::min(support_tile, support - offset);
            for (Py_ssize_t r = 0; r < count; ++r)
                for (Py_ssize_t s = 0; s < width; ++s)
                    scratch[r * width + s] = static_cast<double>(k[(start + r) * support + offset + s]);
            const double beta = offset == 0 ? 0.0 : 1.0;
            if (pairs == 1) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(count),
                    static_cast<int>(width), 1.0, scratch, static_cast<int>(width),
                    w + offset, 1, beta, out + start, 1);
            } else {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(count), static_cast<int>(pairs), static_cast<int>(width),
                    1.0, scratch, static_cast<int>(width), w + offset * pairs,
                    static_cast<int>(pairs), beta, out + start * pairs, static_cast<int>(pairs));
            }
        }
        if (bias) {
            for (Py_ssize_t r = 0; r < count; ++r)
                for (Py_ssize_t p = 0; p < pairs; ++p)
                    out[(start + r) * pairs + p] -= bias[p];
        }
    }
}
#endif

PyObject* contract(PyObject*, PyObject* args) {
    PyObject *tile_object, *weight_object, *bias_object, *output_object;
    if (!PyArg_ParseTuple(args, "OOOO", &tile_object, &weight_object, &bias_object, &output_object))
        return nullptr;
    Buffer tile, weights, bias, output;
    if (!get_buffer(tile_object, tile, false, "tile", 'f', 2) ||
        !get_buffer(weight_object, weights, false, "weights", 'd', 2) ||
        !get_buffer(output_object, output, true, "output", 'd', 2) ||
        (bias_object != Py_None && !get_buffer(bias_object, bias, false, "bias", 'd', 1)))
        return nullptr;
    const Py_ssize_t rows = tile.view.shape[0], support = tile.view.shape[1];
    const Py_ssize_t pairs = weights.view.shape[1];
    if (weights.view.shape[0] != support || output.view.shape[0] != rows ||
        output.view.shape[1] != pairs || (bias.acquired && bias.view.shape[0] != pairs)) {
        PyErr_SetString(PyExc_ValueError, "contraction buffer shapes do not agree");
        return nullptr;
    }
    if (!output_safe(output, tile, weights, bias)) return nullptr;
    const auto* k = static_cast<const float*>(tile.view.buf);
    const auto* w = static_cast<const double*>(weights.view.buf);
    const auto* b = bias.acquired ? static_cast<const double*>(bias.view.buf) : nullptr;
    auto* out = static_cast<double*>(output.view.buf);
#if defined(__APPLE__)
    // Small calls avoid BLAS setup and retain compensated scalar/SIMD sums.
    // int-sized leading dimensions are required by the platform BLAS API.
    if (rows >= 8 && support >= 32 && pairs > 0 && pairs <= std::numeric_limits<int>::max()) {
        constexpr Py_ssize_t capacity = (4 << 20) / sizeof(double);
        const Py_ssize_t support_tile = std::min<Py_ssize_t>(support, 16384);
        const Py_ssize_t row_tile = std::min(rows, capacity / support_tile);
        try {
            // Every consumed element is filled by conversion; value-initializing
            // the scratch would add an unnecessary full memory write per call.
            std::unique_ptr<double[]> scratch(new double[static_cast<std::size_t>(row_tile * support_tile)]);
            ReleaseGIL unlocked;
            tiled_contract(k, w, b, out, rows, support, pairs, scratch.get(), row_tile, support_tile);
        } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
          catch (const std::length_error&) { return PyErr_NoMemory(); }
        Py_RETURN_NONE;
    }
#endif
    {
        ReleaseGIL unlocked;
        for (Py_ssize_t r = 0; r < rows; ++r) {
            Py_ssize_t p = 0;
#if defined(__aarch64__)
            if (pairs == 1) {
                out[r] = binary_dot(k + r * support, w, support) - (b ? b[0] : 0.0);
                continue;
            }
            for (; p + 2 <= pairs; p += 2) {
                pair_dot(k + r * support, w + p, support, pairs, out + r * pairs + p);
                if (b) { out[r * pairs + p] -= b[p]; out[r * pairs + p + 1] -= b[p + 1]; }
            }
#endif
            for (; p < pairs; ++p) {
                Sum sum;
                for (Py_ssize_t s = 0; s < support; ++s)
                    sum.add(static_cast<double>(k[r * support + s]) * w[s * pairs + p]);
                out[r * pairs + p] = sum.get() - (b ? b[p] : 0.0);
            }
        }
    }
    Py_RETURN_NONE;
}

template <class T>
double all_variance(const T* x, Py_ssize_t count) {
    // Shift before reduction: large common offsets should not consume the
    // precision needed to resolve gamma on a low-variance design.
    const double origin = static_cast<double>(x[0]);
    Sum shifted;
    for (Py_ssize_t k = 0; k < count; ++k) shifted.add(static_cast<double>(x[k]) - origin);
    const double mean_delta = shifted.get() / static_cast<double>(count);
    Sum squared;
    for (Py_ssize_t k = 0; k < count; ++k) {
        const double delta = (static_cast<double>(x[k]) - origin) - mean_delta;
        squared.add(delta * delta);
    }
    return squared.get() / static_cast<double>(count);
}

PyObject* variance(PyObject*, PyObject* args) {
    PyObject* input_object;
    if (!PyArg_ParseTuple(args, "O", &input_object)) return nullptr;
    Buffer input;
    if (!get_buffer(input_object, input, false, "X", '*', 2)) return nullptr;
    const Py_ssize_t count = input.view.len / input.view.itemsize;
    if (!count) {
        PyErr_SetString(PyExc_ValueError, "variance requires at least one value");
        return nullptr;
    }
    double result;
    {
        ReleaseGIL unlocked;
        result = input.view.itemsize == 4
            ? all_variance(static_cast<const float*>(input.view.buf), count)
            : all_variance(static_cast<const double*>(input.view.buf), count);
    }
    return PyFloat_FromDouble(result);
}

template <class T>
void means(const T* x, Py_ssize_t rows, Py_ssize_t cols, double* out) {
    for (Py_ssize_t c = 0; c < cols; ++c) {
        const double origin = static_cast<double>(x[c]);
        Sum shifted;
        for (Py_ssize_t r = 0; r < rows; ++r)
            shifted.add(static_cast<double>(x[r * cols + c]) - origin);
        out[c] = origin + shifted.get() / static_cast<double>(rows);
    }
}

PyObject* column_mean(PyObject*, PyObject* args) {
    PyObject *input_object, *output_object;
    if (!PyArg_ParseTuple(args, "OO", &input_object, &output_object)) return nullptr;
    Buffer input, output;
    if (!get_buffer(input_object, input, false, "X", '*', 2) ||
        !get_buffer(output_object, output, true, "output", 'd', 1)) return nullptr;
    const Py_ssize_t rows = input.view.shape[0], cols = input.view.shape[1];
    if (!rows || output.view.shape[0] != cols) {
        PyErr_SetString(PyExc_ValueError, "column means require rows and a matching output");
        return nullptr;
    }
    if (!output_safe(output, input)) return nullptr;
    auto* out = static_cast<double*>(output.view.buf);
    {
        ReleaseGIL unlocked;
        if (input.view.itemsize == 4)
            means(static_cast<const float*>(input.view.buf), rows, cols, out);
        else
            means(static_cast<const double*>(input.view.buf), rows, cols, out);
    }
    Py_RETURN_NONE;
}

double integer_power(double base, Py_ssize_t exponent) {
    double value = 1.0;
    while (exponent) {
        if (exponent & 1) value *= base;
        exponent >>= 1;
        if (exponent) base *= base;
    }
    return value;
}

PyObject* diagonal(PyObject*, PyObject* args) {
    PyObject *input_object, *output_object;
    int kernel;
    double gamma, coef0;
    Py_ssize_t degree;
    if (!PyArg_ParseTuple(args, "OiddnO", &input_object, &kernel, &gamma, &coef0, &degree,
                         &output_object)) return nullptr;
    Buffer input, output;
    if (!get_buffer(input_object, input, false, "X", 'f', 2) ||
        !get_buffer(output_object, output, true, "output", 'd', 1)) return nullptr;
    const Py_ssize_t rows = input.view.shape[0], cols = input.view.shape[1];
    if (kernel < 0 || kernel > 3 || degree < 0 || output.view.shape[0] != rows) {
        PyErr_SetString(PyExc_ValueError, "invalid kernel, degree, or diagonal output shape");
        return nullptr;
    }
    if (!output_safe(output, input)) return nullptr;
    const auto* x = static_cast<const float*>(input.view.buf);
    auto* out = static_cast<double*>(output.view.buf);
    {
        ReleaseGIL unlocked;
        for (Py_ssize_t r = 0; r < rows; ++r) {
            if (kernel == 1) { out[r] = 1.0; continue; }
            Sum squares;
            for (Py_ssize_t c = 0; c < cols; ++c) {
                const double value = static_cast<double>(x[r * cols + c]);
                squares.add(value * value);
            }
            const double squared = squares.get();
            const double z = gamma * squared + coef0;
            out[r] = kernel == 0 ? squared : kernel == 3 ? std::tanh(z) : integer_power(z, degree);
        }
    }
    Py_RETURN_NONE;
}

PyObject* aggregate(PyObject*, PyObject* args) {
    PyObject *input_object, *output_object;
    Py_ssize_t classes;
    int mode;
    if (!PyArg_ParseTuple(args, "OniO", &input_object, &classes, &mode, &output_object)) return nullptr;
    Buffer input, output;
    if (!get_buffer(input_object, input, false, "decisions", 'd', 2) ||
        !get_buffer(output_object, output, true, "output", mode == 1 ? 'd' : 'q',
                    mode == 1 ? 2 : 1)) return nullptr;
    const Py_ssize_t rows = input.view.shape[0], pairs = input.view.shape[1];
    if (mode < 0 || mode > 2 || classes < 2 || classes > PY_SSIZE_T_MAX / (classes - 1) ||
        classes * (classes - 1) / 2 != pairs || output.view.shape[0] != rows ||
        (mode == 1 && output.view.shape[1] != classes)) {
        PyErr_SetString(PyExc_ValueError, "invalid aggregation mode, class count, or buffer shapes");
        return nullptr;
    }
    if (!output_safe(output, input)) return nullptr;
    const auto* decisions = static_cast<const double*>(input.view.buf);
    try {
        std::vector<Py_ssize_t> votes(static_cast<std::size_t>(classes));
        std::vector<double> confidence(mode ? static_cast<std::size_t>(classes) : 0);
        ReleaseGIL unlocked;
        for (Py_ssize_t row = 0; row < rows; ++row) {
            std::fill(votes.begin(), votes.end(), 0);
            std::fill(confidence.begin(), confidence.end(), 0.0);
            Py_ssize_t pair = 0;
            for (Py_ssize_t i = 0; i < classes; ++i) {
                for (Py_ssize_t j = i + 1; j < classes; ++j, ++pair) {
                    const double value = decisions[row * pairs + pair];
                    // OVO uses >0, whereas OVR maps decision<0 to class j.
                    ++votes[(mode ? !(value < 0.0) : value > 0.0) ? i : j];
                    if (mode) { confidence[i] += value; confidence[j] -= value; }
                }
            }
            Py_ssize_t winner = 0;
            double largest = -std::numeric_limits<double>::infinity();
            for (Py_ssize_t c = 0; c < classes; ++c) {
                double score = static_cast<double>(votes[c]);
                if (mode) score += confidence[c] / (3.0 * (std::abs(confidence[c]) + 1.0));
                if (mode == 1) static_cast<double*>(output.view.buf)[row * classes + c] = score;
                if (score > largest || (std::isnan(score) && !std::isnan(largest))) {
                    largest = score; winner = c;
                }
            }
            if (mode != 1) static_cast<std::int64_t*>(output.view.buf)[row] = winner;
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
      catch (const std::length_error&) { return PyErr_NoMemory(); }
    Py_RETURN_NONE;
}

bool cast_matches(const double* source, const float* snapshot, Py_ssize_t count) {
    Py_ssize_t i = 0;
#if defined(__aarch64__)
    for (; i + 4 <= count; i += 4) {
        const auto converted = vcombine_f32(vcvt_f32_f64(vld1q_f64(source + i)),
                                            vcvt_f32_f64(vld1q_f64(source + i + 2)));
        const auto same = vceqq_u32(vreinterpretq_u32_f32(converted),
                                    vreinterpretq_u32_f32(vld1q_f32(snapshot + i)));
        if (vminvq_u32(same) != UINT32_MAX) return false;
    }
#endif
    for (; i < count; ++i) {
        const float converted = static_cast<float>(source[i]);
        std::uint32_t actual, expected;
        std::memcpy(&actual, &converted, sizeof(actual));
        std::memcpy(&expected, snapshot + i, sizeof(expected));
        if (actual != expected) return false;
    }
    return true;
}

PyObject* matches_float32(PyObject*, PyObject* args) {
    PyObject *source_object, *snapshot_object;
    if (!PyArg_ParseTuple(args, "OO", &source_object, &snapshot_object)) return nullptr;
    Buffer source, snapshot;
    if (!get_buffer(source_object, source, false, "source", '*', 2) ||
        !get_buffer(snapshot_object, snapshot, false, "snapshot", 'f', 2)) return nullptr;
    if (source.view.shape[0] != snapshot.view.shape[0] ||
        source.view.shape[1] != snapshot.view.shape[1]) Py_RETURN_FALSE;
    bool matches;
    {
        ReleaseGIL unlocked;
        const Py_ssize_t count = snapshot.view.len / sizeof(float);
        matches = source.view.itemsize == sizeof(float)
            ? !count || std::memcmp(source.view.buf, snapshot.view.buf,
                                    static_cast<std::size_t>(snapshot.view.len)) == 0
            : cast_matches(static_cast<const double*>(source.view.buf),
                           static_cast<const float*>(snapshot.view.buf), count);
    }
    return PyBool_FromLong(matches);
}

PyMethodDef methods[] = {
    {"contract", contract, METH_VARARGS, "contract(tile, weights, bias_or_none, output) -> None"},
    {"variance", variance, METH_VARARGS, "variance(X) -> float"},
    {"column_mean", column_mean, METH_VARARGS, "column_mean(X, output) -> None"},
    {"diagonal", diagonal, METH_VARARGS, "diagonal(X, kernel, gamma, coef0, degree, output) -> None"},
    {"aggregate", aggregate, METH_VARARGS, "aggregate(decisions, classes, mode, output) -> None"},
    {"matches_float32", matches_float32, METH_VARARGS,
     "matches_float32(source_float32_or_float64, snapshot_float32) -> bool"},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_svc_cpu_native", "Native SVC arithmetic without NumPy calculations.",
    -1, methods, nullptr, nullptr, nullptr, nullptr
};

}  // namespace

PyMODINIT_FUNC PyInit__svc_cpu_native() { return PyModule_Create(&module); }
