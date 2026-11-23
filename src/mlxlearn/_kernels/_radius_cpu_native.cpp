// Exact multithreaded CPU radius neighbors: candidate finisher, brute force,
// result capsules and fused vote/mean reductions. The KD-tree lives in
// _radius_tree.cpp; shared primitives in _radius_common.h. No NumPy C API.
#include "_radius_common.h"

namespace mlx_radius {
namespace {

// ---------------------------------------------------------------------------
// finish_pairs: classify candidate pairs from any screen
// ---------------------------------------------------------------------------

struct PairCall {
    const float* train;
    const float* queries;
    Py_ssize_t n, m, d;
    std::int64_t k;
    const Thr* thr;
    std::int64_t q_offset;
    bool exclude_self;
    int mode;
    int threads;
    bool presorted;
};

// Group pair ids by row: row r owns order[row_begin[r] .. row_begin[r + 1])
// (order stays empty for presorted input, where the ids are the positions).
// Validates every index; invalid input raises InputError from the tasks.
template <class Index>
void group_by_row(const PairCall& call, const Index* rows, const Index* cols,
                  RawVector<std::int64_t>& row_begin, RawVector<std::int64_t>& order) {
    const std::int64_t k = call.k, m = call.m, n = call.n;
    const int threads = call.threads;
    row_begin.resize(std::size_t(m + 1));  // every entry is written below
    const std::int64_t chunk = std::max<std::int64_t>(1 << 16, (k + 4 * threads - 1) / (4 * threads));
    const std::size_t n_chunks = k ? std::size_t((k + chunk - 1) / chunk) : 0;
    if (call.presorted) {
        parallel_for(n_chunks, threads, double(k) * 1.5, [&](std::size_t c) {
            const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(k, i0 + chunk);
            std::int64_t previous = i0 ? std::int64_t(rows[i0 - 1]) : 0;
            for (std::int64_t i = i0; i < i1; ++i) {
                const std::int64_t r = rows[i], col = cols[i];
                if (r < 0 || r >= m || col < 0 || col >= n)
                    throw InputError("candidate pair index out of range");
                if (r < previous) throw InputError("rows must be non-decreasing when presorted=True");
                previous = r;
            }
        });
        const std::int64_t block = 4096;
        const std::size_t blocks = std::size_t((m + block - 1) / block);
        parallel_for(blocks, threads, double(m) * 40.0, [&](std::size_t b) {
            const std::int64_t r0 = std::int64_t(b) * block, r1 = std::min(m, r0 + block);
            const Index* position = rows;
            for (std::int64_t r = r0; r < r1; ++r) {
                position = std::lower_bound(position, rows + k, Index(r));
                row_begin[r] = position - rows;
            }
        });
        row_begin[m] = k;
        return;
    }
    // Stable counting sort: blocks of 4096 rows, then rows inside each block.
    const std::int64_t n_blocks = (m + 4095) >> 12;
    std::vector<std::int64_t> hist(n_chunks * std::size_t(n_blocks), 0);
    parallel_for(n_chunks, threads, double(k) * 2.0, [&](std::size_t c) {
        std::int64_t* counts = hist.data() + c * std::size_t(n_blocks);
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(k, i0 + chunk);
        for (std::int64_t i = i0; i < i1; ++i) {
            const std::int64_t r = rows[i], col = cols[i];
            if (r < 0 || r >= m || col < 0 || col >= n)
                throw InputError("candidate pair index out of range");
            ++counts[r >> 12];
        }
    });
    std::vector<std::int64_t> block_start(std::size_t(n_blocks + 1), 0);
    std::int64_t running = 0;
    for (std::int64_t b = 0; b < n_blocks; ++b) {
        block_start[b] = running;
        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::int64_t count = hist[c * std::size_t(n_blocks) + std::size_t(b)];
            hist[c * std::size_t(n_blocks) + std::size_t(b)] = running;
            running += count;
        }
    }
    block_start[n_blocks] = running;
    RawVector<std::int64_t> staged(static_cast<std::size_t>(k));
    parallel_for(n_chunks, threads, double(k) * 2.0, [&](std::size_t c) {
        std::int64_t* cursor = hist.data() + c * std::size_t(n_blocks);
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(k, i0 + chunk);
        for (std::int64_t i = i0; i < i1; ++i) staged[cursor[std::int64_t(rows[i]) >> 12]++] = i;
    });
    order.resize(std::size_t(k));
    parallel_for(std::size_t(n_blocks), threads, double(k) * 3.0, [&](std::size_t b) {
        const std::int64_t r0 = std::int64_t(b) << 12, r1 = std::min(m, r0 + 4096);
        std::vector<std::int64_t> cursor(std::size_t(r1 - r0), 0);
        const std::int64_t s0 = block_start[b], s1 = block_start[b + 1];
        for (std::int64_t i = s0; i < s1; ++i) ++cursor[std::int64_t(rows[staged[i]]) - r0];
        std::int64_t start = s0;
        for (std::int64_t r = r0; r < r1; ++r) {
            row_begin[r] = start;
            const std::int64_t count = cursor[r - r0];
            cursor[r - r0] = start;
            start += count;
        }
        for (std::int64_t i = s0; i < s1; ++i) {
            const std::int64_t p = staged[i];
            order[cursor[std::int64_t(rows[p]) - r0]++] = p;
        }
    });
    row_begin[m] = k;
}

// Tasks of consecutive rows with balanced candidate counts.
std::vector<std::int64_t> pair_bounds(const RawVector<std::int64_t>& row_begin, std::int64_t m,
                                      std::int64_t target_pairs, std::int64_t max_rows) {
    std::vector<std::int64_t> bounds{0};
    std::int64_t start = 0;
    for (std::int64_t r = 0; r < m; ++r) {
        if (row_begin[r + 1] - row_begin[start] >= target_pairs || r + 1 - start >= max_rows) {
            bounds.push_back(r + 1);
            start = r + 1;
        }
    }
    if (bounds.back() != m) bounds.push_back(m);
    return bounds;
}

template <int D, class Index>
std::unique_ptr<Result> finish_pairs_impl(const PairCall& call, const Index* rows, const Index* cols) {
    RawVector<std::int64_t> row_begin, order;
    group_by_row(call, rows, cols, row_begin, order);
    const Py_ssize_t d = call.d;
    const Thr& thr = *call.thr;
    // For index-only high-dimensional pairs a certified float32 result avoids
    // the sequential float64 chain. Distance outputs always need that chain.
    const bool screen_indices = d >= 16 && !(call.mode & (NEED_DISTANCE | NO_SCREEN));
    const Screen32 screen = screen_indices ? make_screen32(thr.R2, d) : Screen32{};
    const std::int64_t target = std::max<std::int64_t>(2048, call.k / (8 * std::int64_t(call.threads)));
    const auto bounds = pair_bounds(row_begin, call.m, target, 4096);
    const double estimate = double(call.k) * (20.0 + 2.0 * double(d)) + double(call.m) * 50.0;
    const std::vector<std::int64_t> identity;
    return run_rows(
        call.m, call.n, thr, call.mode, identity, bounds, call.threads, estimate,
        [&](std::int64_t row, RowSink& sink, SortScratch& scratch) {
            const float* qf = call.queries + row * d;
            double fixed[D > 0 ? D : 1];
            double* q = fixed;
            if (D == 0) {
                scratch.query.resize(std::size_t(d));
                q = scratch.query.data();
            }
            for (Py_ssize_t k = 0; k < d; ++k) q[k] = double(qf[k]);
            const std::int64_t self = call.exclude_self ? call.q_offset + row : -1;
            const std::int64_t b = row_begin[row], e = row_begin[row + 1];
            auto pair = [&](std::int64_t i) { return order.empty() ? i : order[i]; };
            std::int64_t i = b;
            if (screen.on) {
                auto finish = [&](std::int64_t col, float s32) {
                    if (col == self || !(s32 <= screen.out)) return;
                    if (s32 > screen.in) {
                        const float* x = call.train + col * d;
                        if (!classify(canonical_d2<D>(q, x, d), thr, qf, x, d)) return;
                    }
                    sink.add(std::int32_t(col), 0.0);
                };
                for (; i + 2 <= e; i += 2) {
                    const std::int64_t c0 = cols[pair(i)], c1 = cols[pair(i + 1)];
                    float s0, s1;
                    screen_d2_x2(qf, call.train + c0 * d, call.train + c1 * d, d, s0, s1);
                    finish(c0, s0);
                    finish(c1, s1);
                }
                if (i < e) {
                    const std::int64_t col = cols[pair(i)];
                    finish(col, screen_d2(qf, call.train + col * d, d));
                }
                return;
            }
            double s[4];
            for (; i + 4 <= e; i += 4) {
                const std::int64_t c0 = cols[pair(i)], c1 = cols[pair(i + 1)];
                const std::int64_t c2 = cols[pair(i + 2)], c3 = cols[pair(i + 3)];
                const float* x0 = call.train + c0 * d;
                const float* x1 = call.train + c1 * d;
                const float* x2 = call.train + c2 * d;
                const float* x3 = call.train + c3 * d;
                canonical_d2_x4<D>(q, x0, x1, x2, x3, d, s);
                const std::int64_t c[4] = {c0, c1, c2, c3};
                const float* x[4] = {x0, x1, x2, x3};
                for (int j = 0; j < 4; ++j)
                    if (c[j] != self && classify(s[j], thr, qf, x[j], d)) sink.add(std::int32_t(c[j]), s[j]);
            }
            for (; i < e; ++i) {
                const std::int64_t col = cols[pair(i)];
                if (col == self) continue;
                const float* x = call.train + col * d;
                const double value = canonical_d2<D>(q, x, d);
                if (classify(value, thr, qf, x, d)) sink.add(std::int32_t(col), value);
            }
        });
}

// ---------------------------------------------------------------------------
// brute force
// ---------------------------------------------------------------------------

struct BruteCall {
    const float* train;
    const float* queries;
    Py_ssize_t n, m, d;
    const Thr* thr;
    std::int64_t q_offset;
    bool exclude_self;
    int mode;
    int threads;
};

template <int D>
std::unique_ptr<Result> brute_impl(const BruteCall& call) {
    const Py_ssize_t n = call.n, m = call.m, d = call.d;
    const Thr& thr = *call.thr;
    const double work = double(n) * double(m) * double(d);
    const int threads = work > double(1 << 20) ? call.threads : 1;
    const double estimate = work * 0.5 + double(m) * 100.0;
    const Screen32 screen = (call.mode & NO_SCREEN) ? Screen32{} : make_screen32(thr.R2, d);
    if (m >= 4 * std::int64_t(threads) || threads == 1 || n < 2048) {
        const std::int64_t per_task = std::max<std::int64_t>(1, m / (8 * std::int64_t(threads)));
        const auto bounds = uniform_bounds(m, per_task);
        const std::vector<std::int64_t> identity;
        return run_rows(m, n, thr, call.mode, identity, bounds, threads, estimate,
                        [&](std::int64_t row, RowSink& sink, SortScratch& scratch) {
                            const float* qf = call.queries + row * d;
                            double fixed[D > 0 ? D : 1];
                            double* q = fixed;
                            if (D == 0) {
                                scratch.query.resize(std::size_t(d));
                                q = scratch.query.data();
                            }
                            for (Py_ssize_t k = 0; k < d; ++k) q[k] = double(qf[k]);
                            const std::int64_t self = call.exclude_self ? call.q_offset + row : -1;
                            scan_rows<D>(q, qf, call.train, nullptr, 0, n, d, thr, screen, self, sink);
                        });
    }
    // Few queries: split the training rows into blocks, then concatenate
    // every row's block segments in block (= ascending column) order.
    const bool count_only = (call.mode & COUNT_ONLY) != 0;
    const bool keep_d2 = !count_only && (call.mode & NEED_DISTANCE);
    const bool by_distance = keep_d2 && (call.mode & SORT);
    const std::int64_t n_blocks = std::clamp<std::int64_t>((n + 255) / 256, 1, 4 * std::int64_t(threads));
    std::vector<TaskOut> outs(static_cast<std::size_t>(n_blocks));
    std::vector<std::int64_t> seg_off(static_cast<std::size_t>(n_blocks * m)), seg_cnt(static_cast<std::size_t>(n_blocks * m));
    const bool fail = (call.mode & FAIL_TEST) != 0;
    parallel_for(std::size_t(n_blocks), threads, estimate, [&](std::size_t block) {
        if (fail && std::int64_t(block) + 1 == n_blocks) throw std::runtime_error("injected failure in a native radius task");
        const std::int64_t x0 = n * std::int64_t(block) / n_blocks;
        const std::int64_t x1 = n * std::int64_t(block + 1) / n_blocks;
        TaskOut& out = outs[block];
        RowSink sink;
        sink.out = &out;
        sink.keep_d2 = keep_d2;
        sink.count_only = count_only;
        std::vector<double> q(static_cast<std::size_t>(d));
        for (std::int64_t row = 0; row < m; ++row) {
            const float* qf = call.queries + row * d;
            for (Py_ssize_t k = 0; k < d; ++k) q[k] = double(qf[k]);
            const std::int64_t self = call.exclude_self ? call.q_offset + row : -1;
            const std::size_t begin = out.cols.size();
            sink.count = 0;
            scan_rows<D>(q.data(), qf, call.train, nullptr, x0, x1, d, thr, screen, self, sink);
            seg_off[std::size_t(block * m + row)] = std::int64_t(begin);
            seg_cnt[std::size_t(block * m + row)] = sink.count;
        }
    });
    auto result = std::make_unique<Result>();
    result->n_query = m;
    result->n_train = n;
    result->R2 = thr.R2;
    result->mode = call.mode & (NEED_DISTANCE | SORT | COUNT_ONLY);
    result->tasks.resize(1);
    result->row_task.assign(std::size_t(m), 0);
    result->row_off.resize(std::size_t(m));
    result->row_cnt.resize(std::size_t(m));
    std::int64_t total = 0;
    for (std::size_t i = 0; i < seg_cnt.size(); ++i) total += seg_cnt[i];
    result->nnz = total;
    TaskOut& merged = result->tasks[0];
    if (!count_only) {
        merged.cols.reserve(std::size_t(total));
        if (keep_d2) merged.d2.reserve(std::size_t(total));
    }
    SortScratch scratch;
    FpcrScope fpcr;
    for (std::int64_t row = 0; row < m; ++row) {
        const std::size_t begin = merged.cols.size();
        std::int64_t count = 0;
        for (std::int64_t block = 0; block < n_blocks; ++block) {
            const std::int64_t off = seg_off[std::size_t(block * m + row)];
            const std::int64_t cnt = seg_cnt[std::size_t(block * m + row)];
            count += cnt;
            if (count_only || !cnt) continue;
            const TaskOut& out = outs[std::size_t(block)];
            merged.cols.insert(merged.cols.end(), out.cols.begin() + off, out.cols.begin() + off + cnt);
            if (keep_d2) merged.d2.insert(merged.d2.end(), out.d2.begin() + off, out.d2.begin() + off + cnt);
        }
        if (!count_only && by_distance)
            sort_row(merged.cols.data() + begin, merged.d2.data() + begin, std::size_t(count), true,
                     thr.R2, scratch);
        result->row_off[row] = std::int64_t(begin);
        result->row_cnt[row] = count;
    }
    return result;
}

// ---------------------------------------------------------------------------
// result consumers
// ---------------------------------------------------------------------------

struct RowView {
    const std::int32_t* cols;
    const double* d2;
    std::int64_t count;
};

inline RowView row_view(const Result& result, std::int64_t row) {
    const TaskOut& out = result.tasks[std::size_t(result.row_task[row])];
    const std::int64_t off = result.row_off[row];
    return {out.cols.data() + off, out.d2.empty() ? nullptr : out.d2.data() + off, result.row_cnt[row]};
}

std::vector<std::int64_t> row_blocks(const Result& result, int threads) {
    const std::int64_t m = result.n_query;
    const std::int64_t target = std::max<std::int64_t>(1 << 14, result.nnz / (8 * std::int64_t(threads) + 1));
    std::vector<std::int64_t> bounds{0};
    std::int64_t pairs = 0, rows = 0;
    for (std::int64_t r = 0; r < m; ++r) {
        pairs += result.row_cnt[r];
        ++rows;
        if (pairs >= target || rows >= 8192) {
            bounds.push_back(r + 1);
            pairs = rows = 0;
        }
    }
    if (bounds.back() != m) bounds.push_back(m);
    return bounds;
}

void fill_csr(const Result& result, std::int64_t* indptr, std::int64_t* indices, double* distances,
              int threads) {
    const std::int64_t m = result.n_query;
    indptr[0] = 0;
    for (std::int64_t r = 0; r < m; ++r) indptr[r + 1] = indptr[r] + result.row_cnt[r];
    const auto bounds = row_blocks(result, threads);
    const double R2 = result.R2;
    parallel_for(bounds.size() - 1, threads, double(result.nnz) * 1.5 + double(m) * 4.0,
                 [&](std::size_t block) {
                     for (std::int64_t r = bounds[block]; r < bounds[block + 1]; ++r) {
                         const RowView row = row_view(result, r);
                         std::int64_t* out = indices + indptr[r];
                         for (std::int64_t j = 0; j < row.count; ++j) out[j] = row.cols[j];
                         if (distances) {
                             double* dist = distances + indptr[r];
                             for (std::int64_t j = 0; j < row.count; ++j)
                                 dist[j] = std::sqrt(std::min(row.d2[j], R2));
                         }
                     }
                 });
}

// Votes and means accumulate each row in stored (ascending column) order,
// exactly like np.bincount over a CSR in radius_reduce.
template <class Code>
void fill_votes(const Result& result, const Code* codes, std::int64_t n_classes, bool weighted,
                double* votes, std::int64_t* counts, int threads) {
    const auto bounds = row_blocks(result, threads);
    const double R2 = result.R2;
    parallel_for(bounds.size() - 1, threads,
                 double(result.nnz) * 3.0 + double(result.n_query) * double(n_classes),
                 [&](std::size_t block) {
                     for (std::int64_t r = bounds[block]; r < bounds[block + 1]; ++r) {
                         const RowView row = row_view(result, r);
                         double* v = votes + r * n_classes;
                         std::fill(v, v + n_classes, 0.0);
                         counts[r] = row.count;
                         bool zero_row = false;
                         if (weighted)
                             for (std::int64_t j = 0; j < row.count && !zero_row; ++j)
                                 zero_row = std::sqrt(std::min(row.d2[j], R2)) == 0.0;
                         for (std::int64_t j = 0; j < row.count; ++j) {
                             const std::int64_t code = codes[row.cols[j]];
                             if (code < 0 || code >= n_classes) throw InputError("class code out of range");
                             if (!weighted) {
                                 v[code] += 1.0;
                                 continue;
                             }
                             const double dist = std::sqrt(std::min(row.d2[j], R2));
                             const double w = zero_row ? (dist == 0.0 ? 1.0 : 0.0) : 1.0 / dist;
                             v[code] += w;
                         }
                     }
                 });
}

void fill_mean(const Result& result, const double* targets, bool weighted, double* pred,
               std::int64_t* counts, int threads) {
    const auto bounds = row_blocks(result, threads);
    const double R2 = result.R2;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    parallel_for(bounds.size() - 1, threads, double(result.nnz) * 3.0 + double(result.n_query),
                 [&](std::size_t block) {
                     for (std::int64_t r = bounds[block]; r < bounds[block + 1]; ++r) {
                         const RowView row = row_view(result, r);
                         counts[r] = row.count;
                         if (!row.count) {
                             pred[r] = nan;
                             continue;
                         }
                         if (!weighted) {
                             double numerator = 0.0;
                             for (std::int64_t j = 0; j < row.count; ++j) numerator += targets[row.cols[j]];
                             pred[r] = numerator / double(row.count);
                             continue;
                         }
                         bool zero_row = false;
                         for (std::int64_t j = 0; j < row.count && !zero_row; ++j)
                             zero_row = std::sqrt(std::min(row.d2[j], R2)) == 0.0;
                         double numerator = 0.0, denominator = 0.0;
                         for (std::int64_t j = 0; j < row.count; ++j) {
                             const double dist = std::sqrt(std::min(row.d2[j], R2));
                             const double w = zero_row ? (dist == 0.0 ? 1.0 : 0.0) : 1.0 / dist;
                             const double product = targets[row.cols[j]] * w;
                             numerator += product;
                             denominator += w;
                         }
                         pred[r] = numerator / denominator;
                     }
                 });
}

// ---------------------------------------------------------------------------
// dispatch helpers
// ---------------------------------------------------------------------------

template <template <int> class Op, class... Args>
auto dispatch_dimension(Py_ssize_t d, Args&&... args) {
    switch (d) {
        case 1: return Op<1>::run(std::forward<Args>(args)...);
        case 2: return Op<2>::run(std::forward<Args>(args)...);
        case 3: return Op<3>::run(std::forward<Args>(args)...);
        case 4: return Op<4>::run(std::forward<Args>(args)...);
        case 5: return Op<5>::run(std::forward<Args>(args)...);
        case 6: return Op<6>::run(std::forward<Args>(args)...);
        case 7: return Op<7>::run(std::forward<Args>(args)...);
        case 8: return Op<8>::run(std::forward<Args>(args)...);
        case 10: return Op<10>::run(std::forward<Args>(args)...);
        case 12: return Op<12>::run(std::forward<Args>(args)...);
        case 16: return Op<16>::run(std::forward<Args>(args)...);
        default: return Op<0>::run(std::forward<Args>(args)...);
    }
}

template <int D>
struct FinishOp {
    template <class Index>
    static std::unique_ptr<Result> run(const PairCall& call, const Index* rows, const Index* cols) {
        return finish_pairs_impl<D, Index>(call, rows, cols);
    }
};

template <int D>
struct BruteOp {
    static std::unique_ptr<Result> run(const BruteCall& call) { return brute_impl<D>(call); }
};

bool check_mode(int mode) {
    if (mode & ~KNOWN_MODES) {
        PyErr_SetString(PyExc_ValueError, "unknown mode bits");
        return false;
    }
    if ((mode & SORT) && !(mode & NEED_DISTANCE)) {
        PyErr_SetString(PyExc_ValueError, "SORT requires NEED_DISTANCE");
        return false;
    }
    return true;
}

bool check_points(const Buffer& train, const Buffer& queries) {
    if (train.dim(1) != queries.dim(1) || train.dim(1) < 1) {
        PyErr_SetString(PyExc_ValueError, "train and queries must have the same number (>= 1) of features");
        return false;
    }
    if (train.dim(0) > Py_ssize_t(std::numeric_limits<std::int32_t>::max())) {
        PyErr_SetString(PyExc_ValueError, "at most 2**31 - 1 training rows are supported");
        return false;
    }
    return true;
}

Result* get_result(PyObject* object) {
    Result* result = unwrap_result(object);
    if (!result) return nullptr;
    return result;
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

PyObject* py_finish_pairs(PyObject*, PyObject* args) {
    PyObject *train_object, *query_object, *rows_object, *cols_object, *thr_object;
    Py_ssize_t q_offset, threads;
    int exclude_self, mode, presorted;
    if (!PyArg_ParseTuple(args, "OOOOOnpinp:finish_pairs", &train_object, &query_object,
                          &rows_object, &cols_object, &thr_object, &q_offset, &exclude_self, &mode,
                          &threads, &presorted))
        return nullptr;
    Buffer train, queries, rows, cols;
    Thr thr;
    if (!get_buffer(train_object, train, false, "train", DType::f32, 2) ||
        !get_buffer(query_object, queries, false, "queries", DType::f32, 2) ||
        !get_index_buffer(rows_object, rows, false, "rows") ||
        !get_index_buffer(cols_object, cols, false, "cols") || !parse_thresholds(thr_object, thr) ||
        !check_mode(mode) || !check_points(train, queries))
        return nullptr;
    if (rows.count() != cols.count() || rows.type != cols.type || q_offset < 0) {
        PyErr_SetString(PyExc_ValueError, "rows and cols must have the same length and dtype; q_offset >= 0");
        return nullptr;
    }
    PairCall call{train.data<const float>(), queries.data<const float>(), train.dim(0), queries.dim(0),
                  train.dim(1), std::int64_t(rows.count()), &thr, q_offset, exclude_self != 0, mode,
                  clamp_threads(threads), presorted != 0};
    std::unique_ptr<Result> result;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try {
        if (rows.type == DType::i64)
            result = dispatch_dimension<FinishOp>(call.d, call, rows.data<const std::int64_t>(),
                                                  cols.data<const std::int64_t>());
        else
            result = dispatch_dimension<FinishOp>(call.d, call, rows.data<const std::int32_t>(),
                                                  cols.data<const std::int32_t>());
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    return wrap_result(std::move(result));
}

// Finish a Metal candidate bitmap. Words and bits follow training order.
PyObject* py_finish_mask(PyObject*, PyObject* args) {
    PyObject *train_object, *query_object, *mask_object, *thr_object;
    Py_ssize_t q_offset, threads;
    int exclude_self, mode;
    if (!PyArg_ParseTuple(args, "OOOOnpin:finish_mask", &train_object, &query_object,
                          &mask_object, &thr_object, &q_offset, &exclude_self, &mode, &threads))
        return nullptr;
    Buffer train, queries, mask;
    Thr thr;
    if (!get_buffer(train_object, train, false, "train", DType::f32, 2) ||
        !get_buffer(query_object, queries, false, "queries", DType::f32, 2) ||
        !get_buffer(mask_object, mask, false, "mask", DType::i32, 2) ||
        !parse_thresholds(thr_object, thr) || !check_mode(mode) || !check_points(train, queries))
        return nullptr;
    const auto n = train.dim(0), d = train.dim(1), m = queries.dim(0), words = mask.dim(1);
    if (q_offset < 0 || q_offset > PY_SSIZE_T_MAX - m ||
        mask.dim(0) != m || words != (n + 31) / 32) {
        PyErr_SetString(PyExc_ValueError, "invalid candidate bitmap shape or query offset");
        return nullptr;
    }
    const float* x = train.data<const float>();
    const float* q = queries.data<const float>();
    const auto* bits = mask.data<const std::uint32_t>();
    std::int64_t total = 0;
    std::unique_ptr<Result> result;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try {
        std::vector<std::int64_t> counts(m);
        const int workers = clamp_threads(threads);
        const auto bounds = uniform_bounds(m, std::max<std::int64_t>(1, m / (4 * workers)));
        result = run_rows(m, n, thr, mode, {}, bounds, workers, double(m) * n,
            [&](std::int64_t row, RowSink& sink, SortScratch& scratch) {
                const float* qf = q + row * d;
                scratch.query.assign(qf, qf + d);
                const auto self = exclude_self ? q_offset + row : -1;
                // Evaluate four candidates at a time, retaining the canonical
                // feature summation order for each candidate.
                std::int32_t cols[4];
                int used = 0;
                std::int64_t candidates = 0;
                for (std::int64_t w = 0; w < words; ++w) {
                    auto word = bits[row * words + w];
                    candidates += __builtin_popcount(word);
                    while (word) {
                        const auto c = std::int32_t(32 * w + __builtin_ctz(word));
                        word &= word - 1;
                        if (c >= n) throw InputError("invalid padding bits in candidate bitmap");
                        if (c == self) continue;
                        cols[used++] = c;
                        if (used == 4) {
                            double s[4];
                            canonical_d2_x4<0>(scratch.query.data(), x + cols[0] * d,
                                x + cols[1] * d, x + cols[2] * d, x + cols[3] * d, d, s);
                            for (int k = 0; k < 4; ++k)
                                if (classify(s[k], thr, qf, x + cols[k] * d, d))
                                    sink.add(cols[k], s[k]);
                            used = 0;
                        }
                    }
                }
                counts[row] = candidates;
                for (int k = 0; k < used; ++k) {
                    const float* point = x + cols[k] * d;
                    const double s = canonical_d2<0>(scratch.query.data(), point, d);
                    if (classify(s, thr, qf, point, d)) sink.add(cols[k], s);
                }
            });
        for (const auto count : counts) total += count;
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    PyObject* capsule = wrap_result(std::move(result));
    if (!capsule) return nullptr;
    return Py_BuildValue("NL", capsule, static_cast<long long>(total));
}

PyObject* py_brute_query(PyObject*, PyObject* args) {
    PyObject *train_object, *query_object, *thr_object;
    Py_ssize_t q_offset, threads;
    int exclude_self, mode;
    if (!PyArg_ParseTuple(args, "OOOnpin:brute_query", &train_object, &query_object, &thr_object,
                          &q_offset, &exclude_self, &mode, &threads))
        return nullptr;
    Buffer train, queries;
    Thr thr;
    if (!get_buffer(train_object, train, false, "train", DType::f32, 2) ||
        !get_buffer(query_object, queries, false, "queries", DType::f32, 2) ||
        !parse_thresholds(thr_object, thr) || !check_mode(mode) || !check_points(train, queries))
        return nullptr;
    if (q_offset < 0) {
        PyErr_SetString(PyExc_ValueError, "q_offset must be >= 0");
        return nullptr;
    }
    BruteCall call{train.data<const float>(), queries.data<const float>(), train.dim(0), queries.dim(0),
                   train.dim(1), &thr, q_offset, exclude_self != 0, mode, clamp_threads(threads)};
    std::unique_ptr<Result> result;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try {
        result = dispatch_dimension<BruteOp>(call.d, call);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    return wrap_result(std::move(result));
}

PyObject* py_result_sizes(PyObject*, PyObject* args) {
    PyObject* object;
    if (!PyArg_ParseTuple(args, "O:result_sizes", &object)) return nullptr;
    Result* result = get_result(object);
    if (!result) return nullptr;
    return Py_BuildValue("(nL)", result->n_query, static_cast<long long>(result->nnz));
}

PyObject* py_result_info(PyObject*, PyObject* args) {
    PyObject* object;
    if (!PyArg_ParseTuple(args, "O:result_info", &object)) return nullptr;
    Result* result = get_result(object);
    if (!result) return nullptr;
    return Py_BuildValue("(nnLid)", result->n_query, result->n_train,
                         static_cast<long long>(result->nnz), result->mode, result->R2);
}

// Bytes owned by the result (task buffers at capacity plus per-row arrays).
PyObject* py_result_memory(PyObject*, PyObject* args) {
    PyObject* object;
    if (!PyArg_ParseTuple(args, "O:result_memory", &object)) return nullptr;
    Result* result = get_result(object);
    if (!result) return nullptr;
    std::size_t bytes = sizeof(Result) + result->tasks.capacity() * sizeof(TaskOut);
    for (const TaskOut& out : result->tasks)
        bytes += out.cols.capacity() * sizeof(std::int32_t) + out.d2.capacity() * sizeof(double);
    bytes += result->row_task.capacity() * sizeof(std::int32_t) +
             (result->row_off.capacity() + result->row_cnt.capacity()) * sizeof(std::int64_t);
    return PyLong_FromSize_t(bytes);
}

PyObject* py_result_counts(PyObject*, PyObject* args) {
    PyObject *object, *counts_object;
    if (!PyArg_ParseTuple(args, "OO:result_counts", &object, &counts_object)) return nullptr;
    Result* result = get_result(object);
    if (!result) return nullptr;
    Buffer counts;
    if (!get_buffer(counts_object, counts, true, "counts", DType::i64, 1)) return nullptr;
    if (counts.count() != result->n_query) {
        PyErr_SetString(PyExc_ValueError, "counts must have one entry per query");
        return nullptr;
    }
    std::copy(result->row_cnt.begin(), result->row_cnt.end(), counts.data<std::int64_t>());
    Py_RETURN_NONE;
}

PyObject* py_result_fill_csr(PyObject*, PyObject* args) {
    PyObject *object, *indptr_object, *indices_object, *distances_object;
    Py_ssize_t threads = 1;
    if (!PyArg_ParseTuple(args, "OOOO|n:result_fill_csr", &object, &indptr_object, &indices_object,
                          &distances_object, &threads))
        return nullptr;
    Result* result = get_result(object);
    if (!result) return nullptr;
    if (result->mode & COUNT_ONLY) {
        PyErr_SetString(PyExc_ValueError, "a count-only result has no columns");
        return nullptr;
    }
    Buffer indptr, indices, distances;
    if (!get_buffer(indptr_object, indptr, true, "indptr", DType::i64, 1) ||
        !get_buffer(indices_object, indices, true, "indices", DType::i64, 1) ||
        (distances_object != Py_None &&
         !get_buffer(distances_object, distances, true, "distances", DType::f64, 1)))
        return nullptr;
    if (indptr.count() != result->n_query + 1 || indices.count() != result->nnz ||
        (distances.acquired && distances.count() != result->nnz)) {
        PyErr_SetString(PyExc_ValueError, "CSR output sizes do not match the result");
        return nullptr;
    }
    if (distances.acquired && !(result->mode & NEED_DISTANCE)) {
        PyErr_SetString(PyExc_ValueError, "the result was computed without distances");
        return nullptr;
    }
    if (overlaps(indptr, indices) || overlaps(indptr, distances) || overlaps(indices, distances)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap");
        return nullptr;
    }
    std::exception_ptr error;
    std::int64_t* p = indptr.data<std::int64_t>();
    std::int64_t* i = indices.data<std::int64_t>();
    double* dist = distances.acquired ? distances.data<double>() : nullptr;
    const int workers = clamp_threads(threads);
    Py_BEGIN_ALLOW_THREADS
    try {
        fill_csr(*result, p, i, dist, workers);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    Py_RETURN_NONE;
}

PyObject* py_result_fill_votes(PyObject*, PyObject* args) {
    PyObject *object, *codes_object, *votes_object, *counts_object;
    Py_ssize_t n_classes, threads = 1;
    int weighted;
    if (!PyArg_ParseTuple(args, "OOnpOO|n:result_fill_votes", &object, &codes_object, &n_classes,
                          &weighted, &votes_object, &counts_object, &threads))
        return nullptr;
    Result* result = get_result(object);
    if (!result) return nullptr;
    Buffer codes, votes, counts;
    if (!get_index_buffer(codes_object, codes, false, "codes") ||
        !get_buffer(votes_object, votes, true, "votes", DType::f64, 0) ||
        !get_buffer(counts_object, counts, true, "counts", DType::i64, 1))
        return nullptr;
    if ((result->mode & COUNT_ONLY) || (weighted && !(result->mode & NEED_DISTANCE))) {
        PyErr_SetString(PyExc_ValueError, "votes need neighbor columns (and distances when weighted)");
        return nullptr;
    }
    // Overflow-safe size check: n_query * n_classes must not wrap.
    Py_ssize_t vote_slots = 0;
    const bool slots_overflow =
        n_classes < 1 || __builtin_mul_overflow(Py_ssize_t(result->n_query), n_classes, &vote_slots);
    if (slots_overflow || codes.count() != result->n_train ||
        votes.count() != vote_slots || counts.count() != result->n_query) {
        PyErr_SetString(PyExc_ValueError, "codes, votes or counts sizes do not match the result");
        return nullptr;
    }
    if (overlaps(votes, counts) || overlaps(votes, codes) || overlaps(counts, codes)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap inputs or each other");
        return nullptr;
    }
    std::exception_ptr error;
    double* v = votes.data<double>();
    std::int64_t* c = counts.data<std::int64_t>();
    const int workers = clamp_threads(threads);
    Py_BEGIN_ALLOW_THREADS
    try {
        if (codes.type == DType::i64)
            fill_votes(*result, codes.data<const std::int64_t>(), n_classes, weighted != 0, v, c, workers);
        else
            fill_votes(*result, codes.data<const std::int32_t>(), n_classes, weighted != 0, v, c, workers);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    Py_RETURN_NONE;
}

PyObject* py_result_fill_mean(PyObject*, PyObject* args) {
    PyObject *object, *targets_object, *pred_object, *counts_object;
    Py_ssize_t threads = 1;
    int weighted;
    if (!PyArg_ParseTuple(args, "OOpOO|n:result_fill_mean", &object, &targets_object, &weighted,
                          &pred_object, &counts_object, &threads))
        return nullptr;
    Result* result = get_result(object);
    if (!result) return nullptr;
    Buffer targets, pred, counts;
    if (!get_buffer(targets_object, targets, false, "targets", DType::f64, 1) ||
        !get_buffer(pred_object, pred, true, "pred", DType::f64, 1) ||
        !get_buffer(counts_object, counts, true, "counts", DType::i64, 1))
        return nullptr;
    if ((result->mode & COUNT_ONLY) || (weighted && !(result->mode & NEED_DISTANCE))) {
        PyErr_SetString(PyExc_ValueError, "means need neighbor columns (and distances when weighted)");
        return nullptr;
    }
    if (targets.count() != result->n_train || pred.count() != result->n_query ||
        counts.count() != result->n_query) {
        PyErr_SetString(PyExc_ValueError, "targets, pred or counts sizes do not match the result");
        return nullptr;
    }
    if (overlaps(pred, counts) || overlaps(pred, targets) || overlaps(counts, targets)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap inputs or each other");
        return nullptr;
    }
    std::exception_ptr error;
    const double* y = targets.data<const double>();
    double* p = pred.data<double>();
    std::int64_t* c = counts.data<std::int64_t>();
    const int workers = clamp_threads(threads);
    Py_BEGIN_ALLOW_THREADS
    try {
        fill_mean(*result, y, weighted != 0, p, c, workers);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    Py_RETURN_NONE;
}

// canonical_distances(train, queries, rows, cols, out, threads): the canonical
// float64 squared distance of every (queries[rows[i]], train[cols[i]]).
template <class Index>
void canonical_pairs(const float* train, const float* queries, Py_ssize_t n, Py_ssize_t m,
                     Py_ssize_t d, const Index* rows, const Index* cols, std::int64_t k, double* out,
                     int threads) {
    const std::int64_t chunk = std::max<std::int64_t>(4096, (k + 8 * threads - 1) / (8 * threads));
    const std::size_t n_chunks = k ? std::size_t((k + chunk - 1) / chunk) : 0;
    parallel_for(n_chunks, threads, double(k) * double(d + 8), [&](std::size_t c) {
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(k, i0 + chunk);
        std::vector<double> q(static_cast<std::size_t>(d));
        for (std::int64_t i = i0; i < i1; ++i) {
            const std::int64_t r = rows[i], col = cols[i];
            if (r < 0 || r >= m || col < 0 || col >= n) throw InputError("pair index out of range");
            const float* qf = queries + r * d;
            for (Py_ssize_t f = 0; f < d; ++f) q[f] = double(qf[f]);
            out[i] = canonical_d2<0>(q.data(), train + col * d, d);
        }
    });
}

template <class Index>
void exact_pairs(const float* train, const float* queries, Py_ssize_t n, Py_ssize_t m, Py_ssize_t d,
                 const Index* rows, const Index* cols, std::int64_t k, const Thr& thr,
                 std::uint8_t* out, int threads) {
    const std::int64_t chunk = std::max<std::int64_t>(1024, (k + 8 * threads - 1) / (8 * threads));
    const std::size_t n_chunks = k ? std::size_t((k + chunk - 1) / chunk) : 0;
    parallel_for(n_chunks, threads, double(k) * double(d) * 40.0, [&](std::size_t c) {
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(k, i0 + chunk);
        for (std::int64_t i = i0; i < i1; ++i) {
            const std::int64_t r = rows[i], col = cols[i];
            if (r < 0 || r >= m || col < 0 || col >= n) throw InputError("pair index out of range");
            out[i] = exact_member(queries + r * d, train + col * d, d, thr) ? 1 : 0;
        }
    });
}

PyObject* py_canonical_distances(PyObject*, PyObject* args) {
    PyObject *train_object, *query_object, *rows_object, *cols_object, *out_object;
    Py_ssize_t threads = 1;
    if (!PyArg_ParseTuple(args, "OOOOO|n:canonical_distances", &train_object, &query_object,
                          &rows_object, &cols_object, &out_object, &threads))
        return nullptr;
    Buffer train, queries, rows, cols, out;
    if (!get_buffer(train_object, train, false, "train", DType::f32, 2) ||
        !get_buffer(query_object, queries, false, "queries", DType::f32, 2) ||
        !get_index_buffer(rows_object, rows, false, "rows") ||
        !get_index_buffer(cols_object, cols, false, "cols") ||
        !get_buffer(out_object, out, true, "out", DType::f64, 1) || !check_points(train, queries))
        return nullptr;
    if (rows.count() != cols.count() || rows.type != cols.type || out.count() != rows.count()) {
        PyErr_SetString(PyExc_ValueError, "rows, cols and out must have the same length (rows/cols one dtype)");
        return nullptr;
    }
    if (overlaps(out, train) || overlaps(out, queries) || overlaps(out, rows) || overlaps(out, cols)) {
        PyErr_SetString(PyExc_ValueError, "out must not overlap the inputs");
        return nullptr;
    }
    std::exception_ptr error;
    const float* x = train.data<const float>();
    const float* q = queries.data<const float>();
    double* o = out.data<double>();
    const std::int64_t k = rows.count();
    const int workers = clamp_threads(threads);
    Py_BEGIN_ALLOW_THREADS
    try {
        if (rows.type == DType::i64)
            canonical_pairs(x, q, train.dim(0), queries.dim(0), train.dim(1),
                            rows.data<const std::int64_t>(), cols.data<const std::int64_t>(), k, o, workers);
        else
            canonical_pairs(x, q, train.dim(0), queries.dim(0), train.dim(1),
                            rows.data<const std::int32_t>(), cols.data<const std::int32_t>(), k, o, workers);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    Py_RETURN_NONE;
}

PyObject* py_exact_le_pairs(PyObject*, PyObject* args) {
    PyObject *train_object, *query_object, *rows_object, *cols_object, *out_object;
    double R2;
    Py_ssize_t threads = 1;
    if (!PyArg_ParseTuple(args, "OOOOdO|n:exact_le_pairs", &train_object, &query_object,
                          &rows_object, &cols_object, &R2, &out_object, &threads))
        return nullptr;
    Buffer train, queries, rows, cols, out;
    if (!get_buffer(train_object, train, false, "train", DType::f32, 2) ||
        !get_buffer(query_object, queries, false, "queries", DType::f32, 2) ||
        !get_index_buffer(rows_object, rows, false, "rows") ||
        !get_index_buffer(cols_object, cols, false, "cols") ||
        !get_buffer(out_object, out, true, "out", DType::u8, 1) || !check_points(train, queries))
        return nullptr;
    if (rows.count() != cols.count() || rows.type != cols.type || out.count() != rows.count() ||
        !(R2 >= 0)) {
        PyErr_SetString(PyExc_ValueError, "rows, cols and out must have the same length; R2 >= 0");
        return nullptr;
    }
    Thr thr;
    thr.R2 = R2;
    thr.all_members = !r2_floor_key(R2, thr.r2key);
    std::exception_ptr error;
    const float* x = train.data<const float>();
    const float* q = queries.data<const float>();
    std::uint8_t* o = out.data<std::uint8_t>();
    const std::int64_t k = rows.count();
    const int workers = clamp_threads(threads);
    Py_BEGIN_ALLOW_THREADS
    try {
        if (rows.type == DType::i64)
            exact_pairs(x, q, train.dim(0), queries.dim(0), train.dim(1),
                        rows.data<const std::int64_t>(), cols.data<const std::int64_t>(), k, thr, o, workers);
        else
            exact_pairs(x, q, train.dim(0), queries.dim(0), train.dim(1),
                        rows.data<const std::int32_t>(), cols.data<const std::int32_t>(), k, thr, o, workers);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    Py_RETURN_NONE;
}

// _fpcr([value]) -> previous FPCR of the calling thread; sets it when given.
// Test helper only.
PyObject* py_fpcr(PyObject*, PyObject* args) {
    long long value = -1;
    if (!PyArg_ParseTuple(args, "|L:_fpcr", &value)) return nullptr;
    const std::uint64_t previous = read_fpcr();
    if (value >= 0) write_fpcr(std::uint64_t(value));
    return PyLong_FromUnsignedLongLong(previous);
}

PyMethodDef methods[] = {
    {"finish_mask", py_finish_mask, METH_VARARGS, "Refine a Metal candidate bitmap in column order."},
    {"build_tree", py_build_tree, METH_VARARGS, "build_tree(train_f32, leaf_size, threads) -> tree"},
    {"tree_info", py_tree_info, METH_VARARGS, "tree_info(tree) -> (n, d, leaf_size, depth, n_nodes)"},
    {"tree_export", py_tree_export, METH_VARARGS,
     "tree_export(tree, perm_i32[n], points_f32[n*d], lo_f32[nodes*d], hi_f32[nodes*d]) -> None"},
    {"tree_query", py_tree_query, METH_VARARGS,
     "tree_query(tree, queries_f32, thresholds, q_offset, exclude_self, mode, threads) -> result"},
    {"tree_fill", py_tree_fill, METH_VARARGS,
     "tree_fill(tree, queries_f32, thresholds, q_offset, exclude_self, mode, threads, indptr, "
     "indices, distances_or_none) -> None (two-pass fill into a counted CSR)"},
    {"tree_pilot", py_tree_pilot, METH_VARARGS,
     "tree_pilot(tree, queries_f32, R2, visit_cap, out_i64[s, 3][, threads]) -> None"},
    {"brute_query", py_brute_query, METH_VARARGS,
     "brute_query(train_f32, queries_f32, thresholds, q_offset, exclude_self, mode, threads) -> result"},
    {"finish_pairs", py_finish_pairs, METH_VARARGS,
     "finish_pairs(train_f32, queries_f32, rows, cols, thresholds, q_offset, exclude_self, mode, "
     "threads, presorted) -> result"},
    {"result_sizes", py_result_sizes, METH_VARARGS, "result_sizes(result) -> (n_query, nnz)"},
    {"result_info", py_result_info, METH_VARARGS,
     "result_info(result) -> (n_query, n_train, nnz, mode, R2)"},
    {"result_counts", py_result_counts, METH_VARARGS, "result_counts(result, counts_i64) -> None"},
    {"result_memory", py_result_memory, METH_VARARGS, "result_memory(result) -> bytes owned"},
    {"result_fill_csr", py_result_fill_csr, METH_VARARGS,
     "result_fill_csr(result, indptr_i64, indices_i64, distances_or_none[, threads]) -> None"},
    {"result_fill_votes", py_result_fill_votes, METH_VARARGS,
     "result_fill_votes(result, codes, n_classes, weighted, votes_f64, counts_i64[, threads]) -> None"},
    {"result_fill_mean", py_result_fill_mean, METH_VARARGS,
     "result_fill_mean(result, targets_f64, weighted, pred_f64, counts_i64[, threads]) -> None"},
    {"canonical_distances", py_canonical_distances, METH_VARARGS,
     "canonical_distances(train_f32, queries_f32, rows, cols, out_f64[, threads]) -> None"},
    {"exact_le_pairs", py_exact_le_pairs, METH_VARARGS,
     "exact_le_pairs(train_f32, queries_f32, rows, cols, R2, out_u8[, threads]) -> None"},
    {"_fpcr", py_fpcr, METH_VARARGS, "_fpcr([value]) -> previous FPCR (test helper)"},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT,
                      "_radius_cpu_native",
                      "Exact multithreaded CPU radius neighbors (KD-tree, brute force, candidate "
                      "finisher, fused reductions).",
                      -1,
                      methods,
                      nullptr,
                      nullptr,
                      nullptr,
                      nullptr};

}  // namespace
}  // namespace mlx_radius

PyMODINIT_FUNC PyInit__radius_cpu_native() {
    PyObject* result = PyModule_Create(&mlx_radius::module);
    if (!result) return nullptr;
    if (PyModule_AddStringConstant(result, "BUILD_VERSION", "1") < 0 ||
        PyModule_AddIntConstant(result, "NEED_DISTANCE", mlx_radius::NEED_DISTANCE) < 0 ||
        PyModule_AddIntConstant(result, "SORT", mlx_radius::SORT) < 0 ||
        PyModule_AddIntConstant(result, "INTEGER_EXACT", mlx_radius::INTEGER_EXACT) < 0 ||
        PyModule_AddIntConstant(result, "COUNT_ONLY", mlx_radius::COUNT_ONLY) < 0 ||
        PyModule_AddIntConstant(result, "TWO_PASS", mlx_radius::TWO_PASS) < 0 ||
        PyModule_AddIntConstant(result, "FAIL_TEST", mlx_radius::FAIL_TEST) < 0) {
        Py_DECREF(result);
        return nullptr;
    }
    return result;
}
