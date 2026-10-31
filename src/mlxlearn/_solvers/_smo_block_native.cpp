// Hierarchical exact C-SVC optimization. This implementation is independent of
// external implementations; it uses the published working-set decomposition idea.
// Local arithmetic and the complete gradient stay in double precision.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
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
bool get_buffer(PyObject* object, Buffer& b, bool writable, const char* name,
                char format, int ndim) {
    if (PyObject_GetBuffer(object, &b.view, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                          (writable ? PyBUF_WRITABLE : 0)) < 0) return false;
    b.acquired = true;
    const auto& v = b.view;
    const Py_ssize_t itemsize = format == 'd' ? sizeof(double) : sizeof(float);
    if (!v.format || v.format[0] != format || v.format[1] != '\0' || v.itemsize != itemsize ||
        v.ndim != ndim || !v.shape || v.len < 0 || !PyBuffer_IsContiguous(&v, 'C') ||
        reinterpret_cast<std::uintptr_t>(v.buf) % itemsize) {
        PyErr_Format(PyExc_ValueError, "%s must be an aligned C-contiguous native float%d %s",
                     name, int(itemsize * 8), ndim == 1 ? "vector" : "matrix");
        return false;
    }
    Py_ssize_t bytes = itemsize;
    for (int axis = 0; axis < ndim; ++axis) {
        if (v.shape[axis] < 0 || (v.shape[axis] && bytes > PY_SSIZE_T_MAX / v.shape[axis])) {
            PyErr_Format(PyExc_ValueError, "%s has invalid buffer dimensions", name);
            return false;
        }
        bytes *= v.shape[axis];
    }
    if (bytes != v.len || (bytes && !v.buf) ||
        static_cast<std::uint64_t>(bytes) > std::numeric_limits<std::uintptr_t>::max() -
            reinterpret_cast<std::uintptr_t>(v.buf)) {
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
struct AllowThreads {
    PyThreadState* state = PyEval_SaveThread();
    void restore() { if (state) { PyEval_RestoreThread(state); state = nullptr; } }
    void release() { state = PyEval_SaveThread(); }
    ~AllowThreads() { restore(); }
};
struct PythonError {};

// A single owned W-by-n row cache. Retained working-set rows survive an outer
// iteration; all missing rows are requested in one callback. The callback's
// temporary M-by-n result is released immediately after copying (M <= W).
class Rows {
    const float* gram_;
    PyObject* fetch_;
    Py_ssize_t n_, capacity_;
    AllowThreads& threads_;
    std::vector<float> data_;
    std::vector<Py_ssize_t> ids_;
    std::unordered_map<Py_ssize_t, Py_ssize_t> lookup_;
public:
    Rows(const float* gram, PyObject* fetch, Py_ssize_t n, Py_ssize_t capacity, AllowThreads& threads)
        : gram_(gram), fetch_(fetch), n_(n), capacity_(capacity), threads_(threads) {
        if (!gram) {
            data_.resize(static_cast<std::size_t>(capacity) * n);
            ids_.assign(static_cast<std::size_t>(capacity), -1);
        }
    }
    void prepare(const std::vector<Py_ssize_t>& requested) {
        if (gram_) return;
        if (requested.size() > static_cast<std::size_t>(capacity_))
            throw std::logic_error("working set exceeds row cache");
        std::vector<unsigned char> keep(static_cast<std::size_t>(capacity_), 0);
        std::vector<Py_ssize_t> missing;
        for (auto id : requested) {
            auto hit = lookup_.find(id);
            if (hit == lookup_.end()) missing.push_back(id);
            else keep[hit->second] = 1;
        }
        if (missing.empty()) return;
        std::vector<Py_ssize_t> free_slots;
        for (Py_ssize_t slot = 0; slot < capacity_; ++slot) {
            if (!keep[slot]) {
                if (ids_[slot] != -1) lookup_.erase(ids_[slot]);
                ids_[slot] = -1;
                free_slots.push_back(slot);
            }
        }
        threads_.restore();
        {
            PyObject* list = PyList_New(static_cast<Py_ssize_t>(missing.size()));
            if (!list) throw PythonError{};
            for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(missing.size()); ++i) {
                PyObject* item = PyLong_FromSsize_t(missing[i]);
                if (!item) { Py_DECREF(list); throw PythonError{}; }
                PyList_SET_ITEM(list, i, item);
            }
            PyObject* fetched = PyObject_CallFunctionObjArgs(fetch_, list, nullptr);
            Py_DECREF(list);
            if (!fetched) throw PythonError{};
            Buffer buffer;
            bool valid = get_buffer(fetched, buffer, false, "kernel rows", 'f', 2);
            Py_DECREF(fetched);
            if (!valid) throw PythonError{};
            if (buffer.view.shape[0] != static_cast<Py_ssize_t>(missing.size()) || buffer.view.shape[1] != n_) {
                PyErr_SetString(PyExc_ValueError, "kernel rows must have shape (len(indices), n_samples)");
                throw PythonError{};
            }
            const float* source = static_cast<const float*>(buffer.view.buf);
            for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(missing.size()); ++i) {
                std::memcpy(data_.data() + free_slots[i] * n_, source + i * n_,
                            static_cast<std::size_t>(n_) * sizeof(float));
            }
        }
        threads_.release();
        for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(missing.size()); ++i) {
            const auto slot = free_slots[i];
            const float* data = data_.data() + slot * n_;
            for (Py_ssize_t k = 0; k < n_; ++k)
                if (!std::isfinite(data[k])) throw std::invalid_argument("kernel rows must be finite");
            ids_[slot] = missing[i];
            lookup_[missing[i]] = slot;
        }
    }
    const float* get(Py_ssize_t id) const {
        if (gram_) return gram_ + id * n_;
        auto hit = lookup_.find(id);
        if (hit == lookup_.end()) throw std::logic_error("unprepared kernel row");
        return data_.data() + hit->second * n_;
    }
};

bool up(double a, double c, double y) {
    // A fixed [0,0] coordinate has no feasible direction and contributes to
    // neither set; the estimator already removes these rows before solving.
    return c > 0 && (y > 0 ? a < c : a > 0);
}
bool low(double a, double c, double y) { return c > 0 && (y > 0 ? a > 0 : a < c); }
struct Extremes {
    double high = -infinity, bottom = -infinity;
    Py_ssize_t i = -1, j = -1;
    void add(Py_ssize_t id, double value, double a, double c, double y) {
        if (up(a, c, y) && (-value > high || (-value == high && id > i))) { high = -value; i = id; }
        if (low(a, c, y) && (value > bottom || (value == bottom && id > j))) { bottom = value; j = id; }
    }
    double gap() const { return i < 0 || j < 0 ? 0 : high + bottom; }
};
Extremes extremes(const double* alpha, const double* C, const double* y,
                  const std::vector<double>& g) {
    Extremes e;
    for (Py_ssize_t k = 0; k < static_cast<Py_ssize_t>(g.size()); ++k) e.add(k, g[k], alpha[k], C[k], y[k]);
    return e;
}

// Analytic minimization of the two-variable feasible segment. Explicit box
// branches preserve equality and exact bound values, including unequal C_i/C_j.
bool pair_step(double& ai, double& aj, double ci, double cj, double yi, double yj,
               double gi, double gj, double quadratic, double tau) {
    if (quadratic <= 0) quadratic = tau;
    const double old_i = ai, old_j = aj;
    const double gradient_i = yi * gi, gradient_j = yj * gj;
    if (yi != yj) {
        const double delta = (-gradient_i - gradient_j) / quadratic;
        const double difference = old_i - old_j;
        ai = old_i + delta; aj = old_j + delta;
        if (difference > 0) {
            if (aj < 0) { aj = 0; ai = difference; }
        } else if (ai < 0) { ai = 0; aj = -difference; }
        if (difference > ci - cj) {
            if (ai > ci) { ai = ci; aj = ci - difference; }
        } else if (aj > cj) { aj = cj; ai = cj + difference; }
    } else {
        const double delta = (gradient_i - gradient_j) / quadratic;
        const double total = old_i + old_j;
        ai = old_i - delta; aj = old_j + delta;
        if (total > ci) {
            if (ai > ci) { ai = ci; aj = total - ci; }
        } else if (aj < 0) { aj = 0; ai = total; }
        if (total > cj) {
            if (aj > cj) { aj = cj; ai = total - cj; }
        } else if (ai < 0) { ai = 0; aj = total; }
    }
    if (!std::isfinite(ai) || !std::isfinite(aj)) throw std::invalid_argument("nonfinite SMO coefficient");
    return ai != old_i || aj != old_j;
}

// At most W indices are sorted. Partial sorting the two feasible extrema lists
// avoids a full O(n log n) ranking at every global step.
std::vector<Py_ssize_t> choose_set(const double* alpha, const double* C, const double* y,
                                  const std::vector<double>& g, const Extremes& e, Py_ssize_t width,
                                  const std::vector<Py_ssize_t>& previous,
                                  const std::vector<double>& previous_delta) {
    std::vector<Py_ssize_t> result;
    result.reserve(static_cast<std::size_t>(width));
    std::vector<unsigned char> selected(g.size(), 0);
    auto add = [&](Py_ssize_t id) {
        if (id >= 0 && !selected[id] && result.size() < static_cast<std::size_t>(width)) {
            selected[id] = 1; result.push_back(id);
        }
    };
    // Always include BOTH global extrema. A local solve alone is not a global
    // stopping test; these inclusions guarantee a genuinely violating pair.
    add(e.i); add(e.j);
    std::vector<Py_ssize_t> retain(previous.size());
    std::iota(retain.begin(), retain.end(), 0);
    std::sort(retain.begin(), retain.end(), [&](auto a, auto b) {
        bool free_a = alpha[previous[a]] > 0 && alpha[previous[a]] < C[previous[a]];
        bool free_b = alpha[previous[b]] > 0 && alpha[previous[b]] < C[previous[b]];
        if (free_a != free_b) return free_a;
        double da = std::abs(previous_delta[a]), db = std::abs(previous_delta[b]);
        return da != db ? da > db : previous[a] > previous[b];
    });
    for (auto position : retain) {
        if (result.size() >= static_cast<std::size_t>(std::max(Py_ssize_t{2}, width / 2))) break;
        auto id = previous[position];
        if (C[id] > 0) add(id);
    }
    std::vector<Py_ssize_t> upper, lower;
    for (Py_ssize_t k = 0; k < static_cast<Py_ssize_t>(g.size()); ++k) {
        if (selected[k]) continue;
        if (up(alpha[k], C[k], y[k])) upper.push_back(k);
        if (low(alpha[k], C[k], y[k])) lower.push_back(k);
    }
    auto ascending = [&](auto a, auto b) { return g[a] != g[b] ? g[a] < g[b] : a > b; };
    auto descending = [&](auto a, auto b) { return g[a] != g[b] ? g[a] > g[b] : a > b; };
    const auto ku = std::min(static_cast<std::size_t>(width), upper.size());
    const auto kl = std::min(static_cast<std::size_t>(width), lower.size());
    std::partial_sort(upper.begin(), upper.begin() + ku, upper.end(), ascending);
    std::partial_sort(lower.begin(), lower.begin() + kl, lower.end(), descending);
    for (std::size_t k = 0; (k < ku || k < kl) && result.size() < static_cast<std::size_t>(width); ++k) {
        if (k < ku) add(upper[k]);
        if (k < kl) add(lower[k]);
    }
    return result;
}

void add_rows(std::vector<double>& gradient, Rows& rows,
              const std::vector<Py_ssize_t>& ids, const std::vector<double>& coefficients) {
    const Py_ssize_t n = static_cast<Py_ssize_t>(gradient.size());
    std::vector<const float*> pointers;
    std::vector<double> nonzero;
    for (std::size_t k = 0; k < ids.size(); ++k) {
        if (coefficients[k] != 0) { pointers.push_back(rows.get(ids[k])); nonzero.push_back(coefficients[k]); }
    }
    // Keep one short gradient tile hot across the entire batch of changed
    // coefficients, then write it once. The inner contiguous loop vectorizes.
    constexpr Py_ssize_t tile = 64;
    for (Py_ssize_t start = 0; start < n; start += tile) {
        double values[tile];
        const auto count = std::min(tile, n - start);
        for (Py_ssize_t t = 0; t < count; ++t) values[t] = gradient[start + t];
        for (std::size_t k = 0; k < pointers.size(); ++k) {
            const float* row = pointers[k] + start;
            const double scale = nonzero[k];
            for (Py_ssize_t t = 0; t < count; ++t) values[t] += scale * static_cast<double>(row[t]);
        }
        for (Py_ssize_t t = 0; t < count; ++t) {
            if (!std::isfinite(values[t])) throw std::invalid_argument("nonfinite SMO gradient");
            gradient[start + t] = values[t];
        }
    }
}

void reconstruct(std::vector<double>& g, const double* alpha, const double* y,
                 Rows& rows, Py_ssize_t width) {
    for (Py_ssize_t k = 0; k < static_cast<Py_ssize_t>(g.size()); ++k) g[k] = -y[k];
    std::vector<Py_ssize_t> ids;
    std::vector<double> weights;
    for (Py_ssize_t k = 0; k < static_cast<Py_ssize_t>(g.size()); ++k) {
        if (alpha[k] == 0) continue;
        ids.push_back(k); weights.push_back(alpha[k] * y[k]);
        if (ids.size() == static_cast<std::size_t>(width)) {
            rows.prepare(ids); add_rows(g, rows, ids, weights); ids.clear(); weights.clear();
        }
    }
    if (!ids.empty()) { rows.prepare(ids); add_rows(g, rows, ids, weights); }
}

double calculate_rho(const double* alpha, const double* C, const double* y,
                     const std::vector<double>& g) {
    double sum = 0, ub = infinity, lb = -infinity;
    Py_ssize_t free_count = 0;
    for (Py_ssize_t k = 0; k < static_cast<Py_ssize_t>(g.size()); ++k) {
        if (C[k] == 0) continue;
        const bool upper = alpha[k] >= C[k], lower = !upper && alpha[k] <= 0;
        if (!upper && !lower) { ++free_count; sum += g[k]; }
        else if ((upper && y[k] < 0) || (lower && y[k] > 0)) ub = std::min(ub, g[k]);
        else lb = std::max(lb, g[k]);
    }
    return free_count ? sum / free_count : (ub + lb) / 2;
}

struct Result { double rho, gap; Py_ssize_t n_iter; bool converged; };
Result optimize(const double* y, const double* C, const double* diag, double* alpha,
                Py_ssize_t n, const float* gram, PyObject* fetch, double tol,
                Py_ssize_t budget, Py_ssize_t width, double tau, AllowThreads& threads) {
    bool positive = false, negative = false;
    for (Py_ssize_t k = 0; k < n; ++k) {
        if ((y[k] != 1 && y[k] != -1) || !std::isfinite(C[k]) || C[k] < 0 || !std::isfinite(diag[k]))
            throw std::invalid_argument("labels must be +/-1, bounds finite and nonnegative, diagonal finite");
        positive |= y[k] > 0 && C[k] > 0; negative |= y[k] < 0 && C[k] > 0;
    }
    if (!positive || !negative) throw std::invalid_argument("both labels must have a positive bound");
    if (gram)
        for (Py_ssize_t k = 0; k < n * n; ++k)
            if (!std::isfinite(gram[k])) throw std::invalid_argument("Gram matrix must be finite");
    std::vector<double> g(static_cast<std::size_t>(n));
    for (Py_ssize_t k = 0; k < n; ++k) { alpha[k] = 0; g[k] = -y[k]; }
    Rows rows(gram, fetch, n, width, threads);
    std::vector<Py_ssize_t> previous;
    std::vector<double> previous_delta;
    Py_ssize_t iterations = 0, outer_steps = 0;
    bool converged = false, force_full_pair = false, reconstructed = false;
    while (iterations < budget) {
        const Extremes global = extremes(alpha, C, y, g);
        if (global.gap() < tol) {
            // A fresh full-gradient reconstruction is the convergence certificate.
            // No subset residual or stalled-progress heuristic can return success.
            reconstruct(g, alpha, y, rows, width);
            reconstructed = true;
            if (extremes(alpha, C, y, g).gap() < tol) { converged = true; break; }
            previous.clear(); previous_delta.clear();
            force_full_pair = true;
            continue;
        }
        std::vector<Py_ssize_t> ws;
        if (force_full_pair) {
            // Deterministic full second-order selection repairs a stalled local
            // solve, without resetting alpha or weakening the accuracy target.
            rows.prepare({global.i});
            const float* row_i = rows.get(global.i);
            double best = infinity;
            Py_ssize_t j = -1;
            for (Py_ssize_t k = 0; k < n; ++k) {
                if (!low(alpha[k], C[k], y[k])) continue;
                const double difference = global.high + g[k];
                if (difference <= 0) continue;
                double q = diag[global.i] + diag[k] - 2.0 * static_cast<double>(row_i[k]);
                if (q <= 0) q = tau;
                double gain = -(difference * difference) / q;
                if (gain <= best) { best = gain; j = k; }
            }
            if (j < 0) break;
            ws = {global.i, j};
        } else {
            ws = choose_set(alpha, C, y, g, global, width, previous, previous_delta);
        }
        rows.prepare(ws);
        const Py_ssize_t m = static_cast<Py_ssize_t>(ws.size());
        std::vector<float> local_kernel(static_cast<std::size_t>(m) * m);
        std::vector<double> local_alpha(static_cast<std::size_t>(m)), local_g(static_cast<std::size_t>(m));
        std::vector<double> local_C(static_cast<std::size_t>(m)), local_y(static_cast<std::size_t>(m)), local_diag(static_cast<std::size_t>(m));
        for (Py_ssize_t i = 0; i < m; ++i) {
            auto id = ws[i]; local_alpha[i] = alpha[id]; local_g[i] = g[id];
            local_C[i] = C[id]; local_y[i] = y[id]; local_diag[i] = diag[id];
            const float* row = rows.get(id);
            for (Py_ssize_t j = 0; j < m; ++j) local_kernel[i * m + j] = row[ws[j]];
        }
        const double inner_tol = force_full_pair ? tol : std::max(tol, global.gap() * 0.1);
        const Py_ssize_t inner_budget = force_full_pair ? 1 : std::min(budget - iterations, std::max(Py_ssize_t{100}, m * 100));
        Py_ssize_t inner = 0;
        for (; inner < inner_budget && iterations < budget; ++inner) {
            const auto e = extremes(local_alpha.data(), local_C.data(), local_y.data(), local_g);
            if (e.gap() < inner_tol) break;
            const auto i = e.i;
            if (i < 0) break;
            Py_ssize_t j = -1;
            double best = infinity;
            for (Py_ssize_t k = 0; k < m; ++k) {
                if (!low(local_alpha[k], local_C[k], local_y[k])) continue;
                const double difference = e.high + local_g[k];
                if (difference <= 0) continue;
                double q = local_diag[i] + local_diag[k] - 2.0 * static_cast<double>(local_kernel[i * m + k]);
                if (q <= 0) q = tau;
                double gain = -(difference * difference) / q;
                if (gain <= best) { best = gain; j = k; }
            }
            if (j < 0) break;
            const double old_i = local_alpha[i], old_j = local_alpha[j];
            const double q = local_diag[i] + local_diag[j] - 2.0 * static_cast<double>(local_kernel[i * m + j]);
            bool changed = pair_step(local_alpha[i], local_alpha[j], local_C[i], local_C[j], local_y[i], local_y[j],
                                     local_g[i], local_g[j], q, tau);
            ++iterations;
            if (!changed) break;
            const double di = (local_alpha[i] - old_i) * local_y[i], dj = (local_alpha[j] - old_j) * local_y[j];
            for (Py_ssize_t k = 0; k < m; ++k) {
                local_g[k] += di * static_cast<double>(local_kernel[i * m + k]);
                local_g[k] += dj * static_cast<double>(local_kernel[j * m + k]);
                if (!std::isfinite(local_g[k])) throw std::invalid_argument("nonfinite SMO gradient");
            }
        }
        previous_delta.resize(static_cast<std::size_t>(m));
        bool any_change = false;
        for (Py_ssize_t k = 0; k < m; ++k) {
            previous_delta[k] = (local_alpha[k] - alpha[ws[k]]) * y[ws[k]];
            any_change |= previous_delta[k] != 0;
            alpha[ws[k]] = local_alpha[k];
        }
        if (any_change) {
            add_rows(g, rows, ws, previous_delta);
            reconstructed = false;
            force_full_pair = false;
        } else {
            if (force_full_pair) break;  // Numerical stall: never claim convergence.
            force_full_pair = true;
        }
        previous = std::move(ws);
        ++outer_steps;
        if (outer_steps % 16 == 0) {
            threads.restore();
            if (PyErr_CheckSignals() < 0) throw PythonError{};
            threads.release();
        }
    }
    if (!reconstructed) reconstruct(g, alpha, y, rows, width);
    return {calculate_rho(alpha, C, y, g), extremes(alpha, C, y, g).gap(), iterations, converged};
}

PyObject* py_solve(PyObject*, PyObject* args) {
    PyObject *yo, *co, *dio, *ao, *go, *fetch;
    double tol, tau;
    Py_ssize_t budget, width;
    if (!PyArg_ParseTuple(args, "OOOOOOdnnd:solve", &yo, &co, &dio, &ao, &go, &fetch, &tol, &budget, &width, &tau))
        return nullptr;
    if (!std::isfinite(tol) || tol <= 0 || !std::isfinite(tau) || tau <= 0 || budget < 0 || width < 2) {
        PyErr_SetString(PyExc_ValueError, "tol and tau must be positive and finite; max_iter >= 0; working_set_size >= 2");
        return nullptr;
    }
    Buffer y, C, diag, alpha, gram;
    if (!get_buffer(yo, y, false, "y", 'd', 1) || !get_buffer(co, C, false, "C", 'd', 1) ||
        !get_buffer(dio, diag, false, "k_diag", 'd', 1) || !get_buffer(ao, alpha, true, "alpha", 'd', 1)) return nullptr;
    const auto n = y.view.shape[0];
    if (n < 2 || C.view.shape[0] != n || diag.view.shape[0] != n || alpha.view.shape[0] != n) {
        PyErr_SetString(PyExc_ValueError, "y, C, k_diag and alpha must have the same length of at least two"); return nullptr;
    }
    width = std::min(width, n);
    if (width > PY_SSIZE_T_MAX / n / static_cast<Py_ssize_t>(sizeof(float))) {
        PyErr_SetString(PyExc_OverflowError, "working-set buffer is too large"); return nullptr;
    }
    if (go != Py_None) {
        if (!get_buffer(go, gram, false, "gram", 'f', 2)) return nullptr;
        if (gram.view.shape[0] != n || gram.view.shape[1] != n) {
            PyErr_SetString(PyExc_ValueError, "gram must have shape (n_samples, n_samples)"); return nullptr;
        }
    } else if (!PyCallable_Check(fetch)) {
        PyErr_SetString(PyExc_TypeError, "fetch_rows must be callable when gram is None"); return nullptr;
    }
    if (overlaps(alpha, y) || overlaps(alpha, C) || overlaps(alpha, diag) || overlaps(alpha, gram)) {
        PyErr_SetString(PyExc_ValueError, "alpha output must not overlap inputs"); return nullptr;
    }
    try {
        Result result;
        {
            AllowThreads threads;
            result = optimize(static_cast<const double*>(y.view.buf), static_cast<const double*>(C.view.buf),
                              static_cast<const double*>(diag.view.buf), static_cast<double*>(alpha.view.buf), n,
                              gram.acquired ? static_cast<const float*>(gram.view.buf) : nullptr,
                              fetch, tol, budget, width, tau, threads);
        }
        return Py_BuildValue("dnOd", result.rho, result.n_iter, result.converged ? Py_True : Py_False, result.gap);
    } catch (const PythonError&) { return nullptr; }
    catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::exception& error) { PyErr_SetString(PyExc_ValueError, error.what()); return nullptr; }
}
PyMethodDef methods[] = {
    {"solve", py_solve, METH_VARARGS, "Exact hierarchical C-SVC solver with batched kernel-row requests."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_smo_block_native", nullptr, -1, methods};
}  // namespace
PyMODINIT_FUNC PyInit__smo_block_native() { return PyModule_Create(&module); }
