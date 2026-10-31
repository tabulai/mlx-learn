// Exact C-SVC decomposition optimizer. Float64 state, float32 row cache.
// Uses the buffer protocol only; NumPy remains an API container, not a solver.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <list>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();

struct Buffer {
    Py_buffer view{};
    bool acquired = false;
    ~Buffer() { if (acquired) PyBuffer_Release(&view); }
};

bool get_buffer(PyObject* object, Buffer& buffer, bool writable,
                const char* name, char format, int dimensions) {
    int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(object, &buffer.view, flags) < 0) return false;
    buffer.acquired = true;
    const auto& v = buffer.view;
    const Py_ssize_t itemsize = format == 'd' ? sizeof(double) : sizeof(float);
    if (!v.format || v.format[0] != format || v.format[1] != '\0' ||
        v.itemsize != itemsize || v.ndim != dimensions || !v.shape || v.len < 0 ||
        !PyBuffer_IsContiguous(&v, 'C') ||
        (reinterpret_cast<std::uintptr_t>(v.buf) % itemsize) != 0) {
        PyErr_Format(PyExc_ValueError, "%s must be an aligned C-contiguous native float%d %s",
                     name, int(itemsize * 8), dimensions == 1 ? "vector" : "matrix");
        return false;
    }
    Py_ssize_t bytes = itemsize;
    for (int axis = 0; axis < dimensions; ++axis) {
        if (v.shape[axis] < 0 || (v.shape[axis] && bytes > PY_SSIZE_T_MAX / v.shape[axis])) {
            PyErr_Format(PyExc_ValueError, "%s has invalid buffer dimensions", name);
            return false;
        }
        bytes *= v.shape[axis];
    }
    auto address = reinterpret_cast<std::uintptr_t>(v.buf);
    if (bytes != v.len || (bytes && !v.buf) ||
        static_cast<std::uint64_t>(v.len) > std::numeric_limits<std::uintptr_t>::max() - address) {
        PyErr_Format(PyExc_ValueError, "%s has inconsistent buffer dimensions", name);
        return false;
    }
    return true;
}

bool overlaps(const Buffer& a, const Buffer& b) {
    if (!a.acquired || !b.acquired || !a.view.len || !b.view.len) return false;
    auto x = reinterpret_cast<std::uintptr_t>(a.view.buf);
    auto y = reinterpret_cast<std::uintptr_t>(b.view.buf);
    return x < y + b.view.len && y < x + a.view.len;
}

// Restore the GIL before any Python API call or any Buffer destructor runs.
struct AllowThreads {
    PyThreadState* state;
    AllowThreads() : state(PyEval_SaveThread()) {}
    void restore() { if (state) { PyEval_RestoreThread(state); state = nullptr; } }
    void release() { state = PyEval_SaveThread(); }
    ~AllowThreads() { restore(); }
};
struct PythonError {};

struct CacheEntry { Py_ssize_t index; std::vector<float> row; };
class RowCache {
    const float* gram_;
    PyObject* fetch_;
    Py_ssize_t n_, capacity_;
    AllowThreads& threads_;
    std::list<CacheEntry> rows_;
    std::unordered_map<Py_ssize_t, std::list<CacheEntry>::iterator> positions_;
public:
    RowCache(const float* gram, PyObject* fetch, Py_ssize_t n,
             Py_ssize_t capacity, AllowThreads& threads)
        : gram_(gram), fetch_(fetch), n_(n), capacity_(capacity), threads_(threads) {}

    const float* get(Py_ssize_t index) {
        if (gram_) return gram_ + index * n_;
        auto hit = positions_.find(index);
        if (hit != positions_.end()) {
            rows_.splice(rows_.end(), rows_, hit->second);
            return hit->second->row.data();
        }
        // A miss crosses Python exactly once. Copy to owned native memory:
        // a callback is allowed to reuse its scratch array for the next row.
        std::vector<float> owned(static_cast<std::size_t>(n_));
        threads_.restore();
        {
            PyObject* row = PyObject_CallFunction(fetch_, "n", index);
            if (!row) throw PythonError{};
            Buffer buffer;
            bool valid = get_buffer(row, buffer, false, "kernel row", 'f', 1);
            Py_DECREF(row);
            if (!valid) throw PythonError{};
            if (buffer.view.shape[0] != n_) {
                PyErr_SetString(PyExc_ValueError, "kernel row length must match y");
                throw PythonError{};
            }
            std::memcpy(owned.data(), buffer.view.buf, static_cast<std::size_t>(n_) * sizeof(float));
        }
        threads_.release();
        for (float value : owned) {
            if (!std::isfinite(value)) throw std::invalid_argument("kernel row must be finite");
        }
        if (static_cast<Py_ssize_t>(rows_.size()) == capacity_) {
            positions_.erase(rows_.front().index);
            rows_.pop_front();
        }
        rows_.push_back({index, std::move(owned)});
        auto position = std::prev(rows_.end());
        positions_.emplace(index, position);
        return position->row.data();
    }
};

struct Selection {
    Py_ssize_t i = -1;
    double upper = -infinity;
    double lower = -infinity;
    void add(Py_ssize_t k, double yg, bool up, bool low) {
        // Last index wins on a tie, matching LIBSVM's >= scan.
        if (up && -yg >= upper) { upper = -yg; i = k; }
        if (low && yg > lower) lower = yg;
    }
    double gap() const { return i < 0 || lower == -infinity ? 0.0 : upper + lower; }
    bool empty() const { return i < 0 || lower == -infinity; }
};

void status(double alpha, double bound, double y, unsigned char& up, unsigned char& low) {
    const bool upper = alpha >= bound;
    const bool lower = !upper && alpha <= 0.0;
    up = y > 0.0 ? !upper : !lower;
    low = y > 0.0 ? !lower : !upper;
}

struct Result { double rho, violation; Py_ssize_t iterations; bool converged; };
Result solve(const double* y, const double* bounds, const double* diagonal,
             double* alpha, Py_ssize_t n, const float* gram, PyObject* fetch,
             double tolerance, Py_ssize_t max_iter, Py_ssize_t capacity, double tau,
             bool shrinking, AllowThreads& threads) {
    std::vector<double> yg(static_cast<std::size_t>(n));
    std::vector<unsigned char> up(static_cast<std::size_t>(n)), low(static_cast<std::size_t>(n));
    Selection selected;
    for (Py_ssize_t k = 0; k < n; ++k) {
        if ((y[k] != 1.0 && y[k] != -1.0) || !std::isfinite(bounds[k]) || bounds[k] < 0.0 ||
            !std::isfinite(diagonal[k]))
            throw std::invalid_argument("labels must be +/-1, bounds finite and nonnegative, diagonal finite");
    }
    if (gram) {
        for (Py_ssize_t k = 0; k < n * n; ++k)
            if (!std::isfinite(gram[k])) throw std::invalid_argument("Gram matrix must be finite");
    }
    // Validate before writing any caller output.
    for (Py_ssize_t k = 0; k < n; ++k) {
        alpha[k] = 0.0;
        yg[k] = -y[k];
        status(alpha[k], bounds[k], y[k], up[k], low[k]);
        selected.add(k, yg[k], up[k], low[k]);
    }
    RowCache cache(gram, fetch, n, std::min(capacity, std::max(n, Py_ssize_t{2})), threads);
    // Only active coordinates participate in iterative scans and updates.
    // Stable original-index order preserves the last-index tie convention even
    // after compaction. Bound coordinates can be screened provisionally; a full
    // gradient reconstruction and KKT check are mandatory before acceptance.
    std::vector<Py_ssize_t> active(static_cast<std::size_t>(n));
    for (Py_ssize_t k = 0; k < n; ++k) active[k] = k;
    Py_ssize_t iterations = 0;
    Py_ssize_t verified_iteration = -1;
    const Py_ssize_t shrink_period = std::min(n, Py_ssize_t{1000});
    Py_ssize_t until_shrink = shrink_period;
    bool unshrunk = false;
    bool converged = false;
    auto restore = [&]() {
        // Rebuild from the actual cached kernel rows, not the analytic diagonal
        // or an assumed symmetric Gram. This is the same row orientation used
        // by incremental updates. The first-class precision remains float64.
        for (Py_ssize_t k = 0; k < n; ++k) yg[k] = -y[k];
        for (Py_ssize_t t = 0; t < n; ++t) {
            if (alpha[t] == 0.0) continue;
            const float* row = cache.get(t);
            const double coefficient = alpha[t] * y[t];
            for (Py_ssize_t k = 0; k < n; ++k)
                yg[k] += coefficient * static_cast<double>(row[k]);
        }
        active.resize(static_cast<std::size_t>(n));
        selected = {};
        for (Py_ssize_t k = 0; k < n; ++k) {
            if (!std::isfinite(yg[k])) throw std::invalid_argument("nonfinite reconstructed SMO gradient");
            active[k] = k;
            selected.add(k, yg[k], up[k], low[k]);
        }
        verified_iteration = iterations;
        // Once the full problem is restored, finish on the full set. Repeated
        // shrinking near the stopping threshold can cycle without progress.
        unshrunk = true;
    };
    while (iterations < max_iter) {
        if (shrinking && active.size() < static_cast<std::size_t>(n) &&
            (selected.empty() || selected.gap() <= 10.0 * tolerance)) restore();
        if (selected.empty() || selected.gap() < tolerance) {
            // A small active-set gap is insufficient. Even an all-active
            // incremental gradient may have drifted during a long linear fit.
            if (shrinking && verified_iteration != iterations) restore();
            if (selected.empty() || selected.gap() < tolerance) { converged = true; break; }
        }
        if (shrinking && !unshrunk && until_shrink == 0) {
            until_shrink = shrink_period;
            std::size_t kept = 0;
            for (const Py_ssize_t k : active) {
                const bool upper = alpha[k] >= bounds[k];
                const bool lower = !upper && alpha[k] <= 0.0;
                // A bound point can only move in one direction. Screen it
                // when its gradient lies strictly beyond the opposite active
                // KKT extreme. Free points are never removed.
                const bool remove = upper
                    ? (y[k] > 0.0 ? -yg[k] > selected.upper : yg[k] > selected.lower)
                    : lower && (y[k] > 0.0 ? yg[k] > selected.lower : -yg[k] > selected.upper);
                if (!remove) active[kept++] = k;
            }
            active.resize(kept);
            selected = {};
            for (const Py_ssize_t k : active) selected.add(k, yg[k], up[k], low[k]);
            // A degenerate set is never accepted without a full check.
            if (selected.empty()) { restore(); continue; }
        }
        const Py_ssize_t i = selected.i;
        const float* row_i = cache.get(i);
        Py_ssize_t j = -1;
        double best = infinity;
        // Both lower-set membership and second-order selection are resolved
        // here, without temporary gradient, curvature or gain vectors.
        for (const Py_ssize_t k : active) {
            if (!low[k]) continue;
            const double difference = selected.upper + yg[k];
            if (difference <= 0.0) continue;
            double quadratic = diagonal[i] + diagonal[k] - 2.0 * static_cast<double>(row_i[k]);
            if (quadratic <= 0.0) quadratic = tau;
            const double gain = -(difference * difference) / quadratic;
            if (gain <= best) { best = gain; j = k; }
        }
        if (j < 0) {
            if (shrinking && verified_iteration != iterations) { restore(); continue; }
            converged = true;
            break;
        }
        const float* row_j = cache.get(j);
        const double old_i = alpha[i], old_j = alpha[j];
        const double C_i = bounds[i], C_j = bounds[j];
        const double grad_i = y[i] * yg[i], grad_j = y[j] * yg[j];
        double quadratic = diagonal[i] + diagonal[j] - 2.0 * static_cast<double>(row_i[j]);
        if (quadratic <= 0.0) quadratic = tau;
        double new_i, new_j;
        if (y[i] != y[j]) {
            const double delta = (-grad_i - grad_j) / quadratic;
            const double difference = old_i - old_j;
            new_i = old_i + delta;
            new_j = old_j + delta;
            if (difference > 0.0) {
                if (new_j < 0.0) { new_j = 0.0; new_i = difference; }
            } else {
                if (new_i < 0.0) { new_i = 0.0; new_j = -difference; }
            }
            if (difference > C_i - C_j) {
                if (new_i > C_i) { new_i = C_i; new_j = C_i - difference; }
            } else {
                if (new_j > C_j) { new_j = C_j; new_i = C_j + difference; }
            }
        } else {
            const double delta = (grad_i - grad_j) / quadratic;
            const double total = old_i + old_j;
            new_i = old_i - delta;
            new_j = old_j + delta;
            if (total > C_i) {
                if (new_i > C_i) { new_i = C_i; new_j = total - C_i; }
            } else {
                if (new_j < 0.0) { new_j = 0.0; new_i = total; }
            }
            if (total > C_j) {
                if (new_j > C_j) { new_j = C_j; new_i = total - C_j; }
            } else {
                if (new_i < 0.0) { new_i = 0.0; new_j = total; }
            }
        }
        const double delta_i = (new_i - old_i) * y[i];
        const double delta_j = (new_j - old_j) * y[j];
        alpha[i] = new_i;
        alpha[j] = new_j;
        status(new_i, C_i, y[i], up[i], low[i]);
        status(new_j, C_j, y[j], up[j], low[j]);
        selected = {};
        for (const Py_ssize_t k : active) {
            // Preserve two separately rounded double additions. The build
            // disables contraction/reassociation; float32 state is not enough
            // for near-singular linear kernels or strict KKT tolerances.
            double value = yg[k] + delta_i * static_cast<double>(row_i[k]);
            value += delta_j * static_cast<double>(row_j[k]);
            if (!std::isfinite(value)) throw std::invalid_argument("nonfinite SMO gradient");
            yg[k] = value;
            // Fuse the next first-variable selection and KKT bounds into the
            // gradient pass. The final gap is therefore current even at budget.
            selected.add(k, value, up[k], low[k]);
        }
        ++iterations;
        if (until_shrink > 0) --until_shrink;
        // Long fits remain interruptible, with no Python work in their steps.
        if (iterations % 4096 == 0) {
            threads.restore();
            if (PyErr_CheckSignals() < 0) throw PythonError{};
            threads.release();
        }
    }
    // Budget exhaustion also returns a full, current KKT gap and bias; it
    // must not expose stale gradients from coordinates that were screened.
    if (shrinking && verified_iteration != iterations) restore();
    double sum = 0.0, ub = infinity, lb = -infinity;
    Py_ssize_t n_free = 0;
    for (Py_ssize_t k = 0; k < n; ++k) {
        const bool upper = alpha[k] >= bounds[k];
        const bool lower = !upper && alpha[k] <= 0.0;
        if (!upper && !lower) { ++n_free; sum += yg[k]; }
        else if ((upper && y[k] < 0.0) || (lower && y[k] > 0.0)) ub = std::min(ub, yg[k]);
        else lb = std::max(lb, yg[k]);
    }
    return {n_free ? sum / n_free : (ub + lb) / 2.0, selected.gap(), iterations, converged};
}

PyObject* py_solve(PyObject*, PyObject* args) {
    PyObject *y_object, *bounds_object, *diag_object, *alpha_object, *gram_object, *fetch;
    double tolerance, tau;
    Py_ssize_t max_iter, capacity;
    int shrinking = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOdnnd|p:solve", &y_object, &bounds_object, &diag_object,
                          &alpha_object, &gram_object, &fetch, &tolerance, &max_iter, &capacity, &tau, &shrinking))
        return nullptr;
    if (!std::isfinite(tolerance) || tolerance <= 0.0 || !std::isfinite(tau) || tau <= 0.0 ||
        max_iter < 0 || capacity < 2) {
        PyErr_SetString(PyExc_ValueError, "tol and tau must be positive and finite; max_iter >= 0; cache_rows >= 2");
        return nullptr;
    }
    Buffer labels, bounds, diagonal, alpha, gram;
    if (!get_buffer(y_object, labels, false, "y", 'd', 1) ||
        !get_buffer(bounds_object, bounds, false, "C", 'd', 1) ||
        !get_buffer(diag_object, diagonal, false, "k_diag", 'd', 1) ||
        !get_buffer(alpha_object, alpha, true, "alpha", 'd', 1)) return nullptr;
    const Py_ssize_t n = labels.view.shape[0];
    if (n <= 0 || bounds.view.shape[0] != n || diagonal.view.shape[0] != n || alpha.view.shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "y, C, k_diag and alpha must have the same nonzero length");
        return nullptr;
    }
    if (gram_object != Py_None) {
        if (!get_buffer(gram_object, gram, false, "gram", 'f', 2)) return nullptr;
        if (gram.view.shape[0] != n || gram.view.shape[1] != n) {
            PyErr_SetString(PyExc_ValueError, "gram must have shape (n_samples, n_samples)");
            return nullptr;
        }
    } else if (!PyCallable_Check(fetch)) {
        PyErr_SetString(PyExc_TypeError, "fetch_row must be callable when gram is None");
        return nullptr;
    }
    if (overlaps(alpha, labels) || overlaps(alpha, bounds) || overlaps(alpha, diagonal) || overlaps(alpha, gram)) {
        PyErr_SetString(PyExc_ValueError, "alpha output must not overlap inputs");
        return nullptr;
    }
    try {
        Result result;
        {
            AllowThreads threads;
            result = solve(static_cast<const double*>(labels.view.buf),
                           static_cast<const double*>(bounds.view.buf),
                           static_cast<const double*>(diagonal.view.buf),
                           static_cast<double*>(alpha.view.buf), n,
                           gram.acquired ? static_cast<const float*>(gram.view.buf) : nullptr,
                           fetch, tolerance, max_iter, capacity, tau, shrinking != 0, threads);
        }
        return Py_BuildValue("dnOd", result.rho, result.iterations,
                             result.converged ? Py_True : Py_False, result.violation);
    } catch (const PythonError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"solve", py_solve, METH_VARARGS, "Solve exact C-SVC with float64 state and a float32 Gram or row source."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_smo_native", nullptr, -1, methods};
}  // namespace

PyMODINIT_FUNC PyInit__smo_native() { return PyModule_Create(&module); }
