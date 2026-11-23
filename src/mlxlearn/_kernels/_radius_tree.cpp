// Native KD-tree for exact radius queries (part of _radius_cpu_native).
//
// The tree is an implicit balanced heap: node h has children 2h+1 and 2h+2,
// node ranges are fixed by n alone (left child gets floor(count / 2) points),
// and leaves sit at level `depth`, the smallest depth with
// ceil(n / 2**depth) <= leaf_size. Each split uses the dimension with the
// largest float32 extent of the node box (ties go to the lowest dimension) and
// the exact median of the (float32 key, original index) total order, so the
// tree depends only on the data, never on the thread count or on how a split
// was computed. Node boxes are exact float32 minima and maxima, points are
// gathered in tree order and every leaf is ordered by original index.
//
// Queries test node boxes in float64 against the outward-rounded prune and
// bulk thresholds (both certified for the error of a float64 box distance)
// and classify leaf points with the canonical float64 distance plus the exact
// resolver, so membership is exact.
#include "_radius_common.h"

namespace mlx_radius {
namespace {

struct Tree {
    Py_ssize_t n = 0, d = 0;
    int leaf_size = 0, depth = 0;
    std::int64_t n_nodes = 0, first_leaf = 0;
    RawVector<std::int32_t> begin, end;  // per node
    RawVector<float> lo, hi;             // per node, d values each
    RawVector<float> points;             // n * d, tree order
    RawVector<std::int32_t> perm;        // tree position -> original index
    RawVector<std::int32_t> position;    // original index -> tree position
};

void tree_destructor(PyObject* capsule) {
    auto* tree = static_cast<Tree*>(PyCapsule_GetPointer(capsule, TREE_CAPSULE));
    if (!tree) {
        PyErr_Clear();
        return;
    }
    delete tree;
}

Tree* unwrap_tree(PyObject* capsule) {
    return static_cast<Tree*>(PyCapsule_GetPointer(capsule, TREE_CAPSULE));
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

// Total order of (value, index): ordered float bits, then the original index.
inline std::uint64_t split_key(float value, std::int32_t index) {
    std::uint32_t bits = float_bits(value);
    bits ^= (bits >> 31) ? 0xffffffffU : 0x80000000U;
    return (std::uint64_t(bits) << 32) | std::uint32_t(index);
}

int split_dimension(const float* lo, const float* hi, Py_ssize_t d) {
    int best = 0;
    double best_extent = -1.0;
    for (Py_ssize_t k = 0; k < d; ++k) {
        const double extent = double(hi[k]) - double(lo[k]);
        if (extent > best_extent) {
            best_extent = extent;
            best = int(k);
        }
    }
    return best;
}

inline void empty_box(float* lo, float* hi, Py_ssize_t d) {
    for (Py_ssize_t k = 0; k < d; ++k) {
        lo[k] = std::numeric_limits<float>::infinity();
        hi[k] = -std::numeric_limits<float>::infinity();
    }
}

// Merge box (lo2, hi2) into (lo, hi). A strict comparison keeps the earlier
// value among equal ones (+0/-0), matching a sequential scan.
inline void merge_box(float* lo, float* hi, const float* lo2, const float* hi2, Py_ssize_t d) {
    for (Py_ssize_t k = 0; k < d; ++k) {
        if (lo2[k] < lo[k]) lo[k] = lo2[k];
        if (hi2[k] > hi[k]) hi[k] = hi2[k];
    }
}

struct BuildState {
    Tree* tree = nullptr;
    Py_ssize_t d = 0;
    const float* src = nullptr;
    const std::int32_t* src_index = nullptr;
    float* dst = nullptr;
    std::int32_t* dst_index = nullptr;
};

// Stable partition of positions [i0, i1) by key < pivot; left rows go to
// left_pos.., right rows to right_pos..; the boxes of both sides are updated.
template <int D>
void partition_range(const BuildState& s, std::int64_t i0, std::int64_t i1, int dim,
                     std::uint64_t pivot, std::int64_t left_pos, std::int64_t right_pos,
                     float* lo_left, float* hi_left, float* lo_right, float* hi_right) {
    const Py_ssize_t dims = D > 0 ? D : s.d;
    for (std::int64_t i = i0; i < i1; ++i) {
        const float* x = s.src + i * dims;
        const std::int32_t index = s.src_index[i];
        const bool left = split_key(x[dim], index) < pivot;
        const std::int64_t pos = left ? left_pos++ : right_pos++;
        float* y = s.dst + pos * dims;
        float* lo = left ? lo_left : lo_right;
        float* hi = left ? hi_left : hi_right;
        for (Py_ssize_t k = 0; k < dims; ++k) {
            const float v = x[k];
            y[k] = v;
            if (v < lo[k]) lo[k] = v;
            if (v > hi[k]) hi[k] = v;
        }
        s.dst_index[pos] = index;
    }
}

template <int D>
void split_serial(const BuildState& s, std::int64_t h, RawVector<std::uint64_t>& keys) {
    Tree& tree = *s.tree;
    const Py_ssize_t d = s.d;
    const std::int64_t b = tree.begin[h], e = tree.end[h], count = e - b, half = count / 2;
    float* lo_left = tree.lo.data() + (2 * h + 1) * d;
    float* hi_left = tree.hi.data() + (2 * h + 1) * d;
    float* lo_right = tree.lo.data() + (2 * h + 2) * d;
    float* hi_right = tree.hi.data() + (2 * h + 2) * d;
    empty_box(lo_left, hi_left, d);
    empty_box(lo_right, hi_right, d);
    if (count == 0) return;
    const int dim = split_dimension(tree.lo.data() + h * d, tree.hi.data() + h * d, d);
    keys.resize(std::size_t(count));
    for (std::int64_t i = 0; i < count; ++i)
        keys[i] = split_key(s.src[(b + i) * d + dim], s.src_index[b + i]);
    std::nth_element(keys.begin(), keys.begin() + half, keys.end());
    const std::uint64_t pivot = keys[std::size_t(half)];
    partition_range<D>(s, b, e, dim, pivot, b, b + half, lo_left, hi_left, lo_right, hi_right);
}

constexpr int RADIX_BITS = 11;
constexpr std::size_t RADIX_BUCKETS = std::size_t(1) << RADIX_BITS;
constexpr std::int64_t SERIAL_SELECT = 1 << 15;

// The same split as split_serial, computed with all threads: an exact radix
// select of the median key, then a chunked stable partition.
template <int D>
void split_parallel(const BuildState& s, std::int64_t h, int threads,
                    RawVector<std::uint64_t>& keys, RawVector<std::uint64_t>& buffer_a,
                    RawVector<std::uint64_t>& buffer_b) {
    Tree& tree = *s.tree;
    const Py_ssize_t d = s.d;
    const std::int64_t b = tree.begin[h], e = tree.end[h], count = e - b, half = count / 2;
    float* lo_left = tree.lo.data() + (2 * h + 1) * d;
    float* hi_left = tree.hi.data() + (2 * h + 1) * d;
    float* lo_right = tree.lo.data() + (2 * h + 2) * d;
    float* hi_right = tree.hi.data() + (2 * h + 2) * d;
    empty_box(lo_left, hi_left, d);
    empty_box(lo_right, hi_right, d);
    if (count == 0) return;
    const int dim = split_dimension(tree.lo.data() + h * d, tree.hi.data() + h * d, d);

    auto chunking = [&](std::int64_t size, std::int64_t& chunk) {
        chunk = std::max<std::int64_t>(16384, (size + 4 * threads - 1) / (4 * threads));
        return std::size_t((size + chunk - 1) / chunk);
    };
    std::int64_t chunk;
    std::size_t n_chunks = chunking(count, chunk);
    std::vector<std::int64_t> hist(n_chunks * RADIX_BUCKETS);
    unsigned shift = 64 - RADIX_BITS;
    keys.resize(std::size_t(count));
    parallel_for(n_chunks, threads, double(count) * 2.0, [&](std::size_t c) {
        std::int64_t* counts = hist.data() + c * RADIX_BUCKETS;
        std::fill(counts, counts + RADIX_BUCKETS, 0);
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(count, i0 + chunk);
        for (std::int64_t i = i0; i < i1; ++i) {
            const std::uint64_t key = split_key(s.src[(b + i) * d + dim], s.src_index[b + i]);
            keys[i] = key;
            ++counts[key >> shift];
        }
    });

    const std::uint64_t* candidates = keys.data();
    std::int64_t size = count, rank = half;
    std::uint64_t* output = buffer_a.data();
    std::uint64_t* spare = buffer_b.data();
    std::uint64_t pivot = 0;
    bool histogram_ready = true;
    for (;;) {
        if (size <= SERIAL_SELECT) {
            if (candidates != output) std::copy(candidates, candidates + size, output);
            std::nth_element(output, output + rank, output + size);
            pivot = output[rank];
            break;
        }
        if (!histogram_ready) {
            n_chunks = chunking(size, chunk);
            hist.assign(n_chunks * RADIX_BUCKETS, 0);
            parallel_for(n_chunks, threads, double(size), [&](std::size_t c) {
                std::int64_t* counts = hist.data() + c * RADIX_BUCKETS;
                const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(size, i0 + chunk);
                for (std::int64_t i = i0; i < i1; ++i)
                    ++counts[(candidates[i] >> shift) & (RADIX_BUCKETS - 1)];
            });
        }
        std::int64_t below = 0;
        std::size_t bucket = 0;
        for (; bucket < RADIX_BUCKETS; ++bucket) {
            std::int64_t total = 0;
            for (std::size_t c = 0; c < n_chunks; ++c) total += hist[c * RADIX_BUCKETS + bucket];
            if (below + total > rank) break;
            below += total;
        }
        rank -= below;
        std::vector<std::int64_t> offsets(n_chunks);
        std::int64_t next_size = 0;
        for (std::size_t c = 0; c < n_chunks; ++c) {
            offsets[c] = next_size;
            next_size += hist[c * RADIX_BUCKETS + bucket];
        }
        const unsigned current = shift;
        parallel_for(n_chunks, threads, double(size), [&](std::size_t c) {
            std::int64_t out = offsets[c];
            const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(size, i0 + chunk);
            for (std::int64_t i = i0; i < i1; ++i)
                if (((candidates[i] >> current) & (RADIX_BUCKETS - 1)) == bucket)
                    output[out++] = candidates[i];
        });
        candidates = output;
        std::swap(output, spare);
        size = next_size;
        shift = shift >= unsigned(RADIX_BITS) ? shift - RADIX_BITS : 0;
        histogram_ready = false;
    }

    // Chunked stable partition with per-chunk boxes, merged in chunk order.
    n_chunks = chunking(count, chunk);
    std::vector<std::int64_t> lefts(n_chunks);
    parallel_for(n_chunks, threads, double(count), [&](std::size_t c) {
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(count, i0 + chunk);
        std::int64_t left = 0;
        for (std::int64_t i = i0; i < i1; ++i) left += keys[i] < pivot;
        lefts[c] = left;
    });
    std::vector<std::int64_t> left_pos(n_chunks), right_pos(n_chunks);
    std::int64_t left_total = 0, right_total = 0;
    for (std::size_t c = 0; c < n_chunks; ++c) {
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(count, i0 + chunk);
        left_pos[c] = b + left_total;
        right_pos[c] = b + half + right_total;
        left_total += lefts[c];
        right_total += (i1 - i0) - lefts[c];
    }
    if (left_total != half) throw std::runtime_error("radius tree: inconsistent median split");
    std::vector<float> boxes(n_chunks * 4 * std::size_t(d));
    parallel_for(n_chunks, threads, double(count) * double(d + 2), [&](std::size_t c) {
        float* box = boxes.data() + c * 4 * std::size_t(d);
        empty_box(box, box + d, d);
        empty_box(box + 2 * d, box + 3 * d, d);
        const std::int64_t i0 = std::int64_t(c) * chunk, i1 = std::min(count, i0 + chunk);
        partition_range<D>(s, b + i0, b + i1, dim, pivot, left_pos[c], right_pos[c], box, box + d,
                           box + 2 * d, box + 3 * d);
    });
    for (std::size_t c = 0; c < n_chunks; ++c) {
        const float* box = boxes.data() + c * 4 * std::size_t(d);
        merge_box(lo_left, hi_left, box, box + d, d);
        merge_box(lo_right, hi_right, box + 2 * d, box + 3 * d, d);
    }
}

constexpr std::int64_t PARALLEL_SPLIT_MIN = 1 << 16;

template <int D>
std::unique_ptr<Tree> build_tree_impl(const float* X, Py_ssize_t n, Py_ssize_t d, int leaf_size,
                                      int threads) {
    auto tree = std::make_unique<Tree>();
    tree->n = n;
    tree->d = d;
    tree->leaf_size = leaf_size;
    int depth = 0;
    while (((std::int64_t(n) + (std::int64_t(1) << depth) - 1) >> depth) > leaf_size) ++depth;
    tree->depth = depth;
    tree->n_nodes = (std::int64_t(2) << depth) - 1;
    tree->first_leaf = (std::int64_t(1) << depth) - 1;
    const std::int64_t n_nodes = tree->n_nodes;
    tree->begin.resize(std::size_t(n_nodes));
    tree->end.resize(std::size_t(n_nodes));
    tree->begin[0] = 0;
    tree->end[0] = std::int32_t(n);
    for (std::int64_t h = 0; h < tree->first_leaf; ++h) {
        const std::int32_t b = tree->begin[h], e = tree->end[h], mid = b + (e - b) / 2;
        tree->begin[2 * h + 1] = b;
        tree->end[2 * h + 1] = mid;
        tree->begin[2 * h + 2] = mid;
        tree->end[2 * h + 2] = e;
    }
    tree->lo.resize(std::size_t(n_nodes * d));
    tree->hi.resize(std::size_t(n_nodes * d));

    RawVector<float> points_a(static_cast<std::size_t>(n * d)), points_b(static_cast<std::size_t>(n * d));
    RawVector<std::int32_t> index_a(static_cast<std::size_t>(n)), index_b(static_cast<std::size_t>(n));
    const std::int64_t copy_chunk = 1 << 16;
    const std::size_t copy_chunks = std::size_t((n + copy_chunk - 1) / copy_chunk);
    std::vector<float> chunk_boxes(copy_chunks * 2 * std::size_t(d));
    parallel_for(copy_chunks, threads, double(n) * double(d), [&](std::size_t c) {
        const std::int64_t i0 = std::int64_t(c) * copy_chunk;
        const std::int64_t i1 = std::min<std::int64_t>(n, i0 + copy_chunk);
        std::memcpy(points_a.data() + i0 * d, X + i0 * d, std::size_t((i1 - i0) * d) * sizeof(float));
        for (std::int64_t i = i0; i < i1; ++i) index_a[i] = std::int32_t(i);
        float* lo = chunk_boxes.data() + c * 2 * d;
        float* hi = lo + d;
        empty_box(lo, hi, d);
        for (std::int64_t i = i0; i < i1; ++i) {
            const float* x = X + i * d;
            for (Py_ssize_t k = 0; k < d; ++k) {
                if (x[k] < lo[k]) lo[k] = x[k];
                if (x[k] > hi[k]) hi[k] = x[k];
            }
        }
    });
    empty_box(tree->lo.data(), tree->hi.data(), d);
    for (std::size_t c = 0; c < copy_chunks; ++c) {
        const float* lo = chunk_boxes.data() + c * 2 * d;
        merge_box(tree->lo.data(), tree->hi.data(), lo, lo + d, d);
    }

    BuildState state;
    state.tree = tree.get();
    state.d = d;
    float* src = points_a.data();
    float* dst = points_b.data();
    std::int32_t* src_index = index_a.data();
    std::int32_t* dst_index = index_b.data();
    // Top levels split level by level (big nodes with all threads); from the
    // first level with at least 2 * threads nodes on, every node's subtree is
    // one task that splits all remaining levels without per-level dispatch.
    int subtree_level = 0;
    while (subtree_level < depth && (std::int64_t(1) << subtree_level) < 2 * std::int64_t(threads))
        ++subtree_level;
    RawVector<std::uint64_t> keys, buffer_a, buffer_b;
    for (int level = 0; level < subtree_level; ++level) {
        state.src = src;
        state.src_index = src_index;
        state.dst = dst;
        state.dst_index = dst_index;
        const std::int64_t first = (std::int64_t(1) << level) - 1;
        const std::int64_t nodes = std::int64_t(1) << level;
        const std::int64_t node_size = (std::int64_t(n) + nodes - 1) / nodes;
        if (threads > 1 && nodes < threads && node_size >= PARALLEL_SPLIT_MIN) {
            if (buffer_a.size() < std::size_t(node_size)) {
                buffer_a.resize(std::size_t(node_size));
                buffer_b.resize(std::size_t(node_size));
            }
            for (std::int64_t h = first; h < first + nodes; ++h)
                split_parallel<D>(state, h, threads, keys, buffer_a, buffer_b);
        } else {
            parallel_for(std::size_t(nodes), threads, double(n) * double(d + 8), [&](std::size_t g) {
                RawVector<std::uint64_t> scratch;
                split_serial<D>(state, first + std::int64_t(g), scratch);
            });
        }
        std::swap(src, dst);
        std::swap(src_index, dst_index);
    }
    {
        RawVector<std::uint64_t>().swap(keys);
        RawVector<std::uint64_t>().swap(buffer_a);
        RawVector<std::uint64_t>().swap(buffer_b);
    }
    const std::int64_t subtrees = std::int64_t(1) << subtree_level;
    const std::int64_t sub_first = subtrees - 1;
    const int sub_levels = depth - subtree_level;
    parallel_for(sub_levels ? std::size_t(subtrees) : 0, threads, double(n) * double(d + 8) * sub_levels,
                 [&](std::size_t g) {
        BuildState local = state;
        float* s_points = src;
        float* d_points = dst;
        std::int32_t* s_index = src_index;
        std::int32_t* d_index = dst_index;
        const std::int64_t root = sub_first + std::int64_t(g);
        RawVector<std::uint64_t> scratch;
        for (int below = 0; below < sub_levels; ++below) {
            local.src = s_points;
            local.src_index = s_index;
            local.dst = d_points;
            local.dst_index = d_index;
            const std::int64_t count = std::int64_t(1) << below;
            const std::int64_t first = (root + 1) * count - 1;
            for (std::int64_t h = first; h < first + count; ++h) split_serial<D>(local, h, scratch);
            std::swap(s_points, d_points);
            std::swap(s_index, d_index);
        }
    });
    // Starting from buffer A, each level stably partitions into the other
    // buffer. The leaves retain ascending index order without a sort or copy.
    if (depth % 2 == 0) {
        tree->points = std::move(points_a);
        tree->perm = std::move(index_a);
    } else {
        tree->points = std::move(points_b);
        tree->perm = std::move(index_b);
    }
    RawVector<float>().swap(points_a);
    RawVector<float>().swap(points_b);
    RawVector<std::int32_t>().swap(index_a);
    RawVector<std::int32_t>().swap(index_b);
    tree->position.resize(std::size_t(n));
    parallel_for(copy_chunks, threads, double(n) * 2.0, [&](std::size_t c) {
        const std::int64_t i0 = std::int64_t(c) * copy_chunk;
        const std::int64_t i1 = std::min<std::int64_t>(n, i0 + copy_chunk);
        for (std::int64_t p = i0; p < i1; ++p) tree->position[tree->perm[p]] = std::int32_t(p);
    });
    return tree;
}

// ---------------------------------------------------------------------------
// queries
// ---------------------------------------------------------------------------

template <int D>
struct Searcher {
    const Tree& tree;
    const Thr& thr;
    Screen32 screen;
    bool box32_on = false;     // float32 box tests (needs the screen)
    bool leaf_screen = false;  // float32 pre-screen in leaf scans

    Searcher(const Tree& tree_, const Thr& thr_, const Screen32& screen_, int mode = 0)
        : tree(tree_), thr(thr_), screen(screen_) {
        box32_on = screen.on && !(mode & BOX64_TEST);
        leaf_screen = screen.on && (D == 0 || D >= 4) && !(mode & LEAF_PLAIN_TEST);
    }

    // Lower and upper bounds of S over the node box, in float64.
    inline void box(std::int64_t h, const double* q, double& lower, double& upper) const {
        const Py_ssize_t dims = D > 0 ? D : tree.d;
        const float* lo = tree.lo.data() + h * dims;
        const float* hi = tree.hi.data() + h * dims;
        double near = 0.0, far = 0.0;
        for (Py_ssize_t k = 0; k < dims; ++k) {
            const double v = q[k];
            const double below = double(lo[k]) - v;  // > 0 when q is below the box
            const double above = v - double(hi[k]);  // > 0 when q is above the box
            double t = below > above ? below : above;
            if (!(t > 0.0)) t = 0.0;
            near = near + t * t;
            const double a = std::fabs(below), c = std::fabs(above);
            const double u = a > c ? a : c;
            far = far + u * u;
        }
        lower = near;
        upper = far;
    }

    // Float32 bounds of S over the node box. With the screen's certified
    // thresholds, lower > screen.out proves no point of the box is a member and
    // upper <= screen.in proves every point is (see Screen32); the summation
    // order is free because only the error bound matters.
    inline void box32(std::int64_t h, const float* qf, float& lower, float& upper) const {
        const Py_ssize_t dims = D > 0 ? D : tree.d;
        const float* lo = tree.lo.data() + h * dims;
        const float* hi = tree.hi.data() + h * dims;
        Py_ssize_t k = 0;
        float near = 0.0f, far = 0.0f;
#if defined(__aarch64__)
        if (dims >= 4) {
            const float32x4_t zero = vdupq_n_f32(0.0f);
            float32x4_t near4 = zero, far4 = zero;
            for (; k + 4 <= dims; k += 4) {
                const float32x4_t v = vld1q_f32(qf + k);
                const float32x4_t below = vsubq_f32(vld1q_f32(lo + k), v);
                const float32x4_t above = vsubq_f32(v, vld1q_f32(hi + k));
                const float32x4_t t = vmaxq_f32(vmaxq_f32(below, above), zero);
                near4 = vfmaq_f32(near4, t, t);
                const float32x4_t u = vmaxq_f32(vabsq_f32(below), vabsq_f32(above));
                far4 = vfmaq_f32(far4, u, u);
            }
            near = vaddvq_f32(near4);
            far = vaddvq_f32(far4);
        }
#endif
        for (; k < dims; ++k) {
            const float v = qf[k];
            const float below = lo[k] - v;
            const float above = v - hi[k];
            float t = below > above ? below : above;
            if (!(t > 0.0f)) t = 0.0f;
            near = near + t * t;
            const float a = std::fabs(below), c = std::fabs(above);
            const float u = a > c ? a : c;
            far = far + u * u;
        }
        lower = near;
        upper = far;
    }

    enum Decision { PRUNE, BULK, DESCEND };

    inline Decision decide(std::int64_t h, const double* q, const float* qf) const {
        if (box32_on) {
            float lower, upper;
            box32(h, qf, lower, upper);
            if (lower > screen.out) return PRUNE;
            return upper <= screen.in ? BULK : DESCEND;
        }
        double lower, upper;
        box(h, q, lower, upper);
        if (lower > thr.prune_above) return PRUNE;
        return upper <= thr.bulk_below ? BULK : DESCEND;
    }

    inline void include(std::int64_t h, const double* q, std::int64_t self, std::int64_t self_pos,
                        RowSink& sink) const {
        scan_bulk<D>(q, tree.points.data(), tree.perm.data(), tree.begin[h], tree.end[h], tree.d,
                     self, self_pos, sink);
    }

    void search(const double* q, const float* qf, std::int64_t self, std::int64_t self_pos,
                RowSink& sink) const {
        if (tree.n == 0) return;
        const Decision root = decide(0, q, qf);
        if (root == PRUNE) return;
        if (root == BULK) {
            include(0, q, self, self_pos, sink);
            return;
        }
        std::int64_t stack[128];
        int top = 0;
        stack[top++] = 0;
        const std::int64_t first_leaf = tree.first_leaf;
        while (top) {
            const std::int64_t h = stack[--top];
            if (h >= first_leaf) {
                if (leaf_screen)
                    scan_screened<D>(q, qf, tree.points.data(), tree.perm.data(), tree.begin[h],
                                     tree.end[h], tree.d, thr, screen, self, sink);
                else
                    scan_classify<D>(q, qf, tree.points.data(), tree.perm.data(), tree.begin[h],
                                     tree.end[h], tree.d, thr, self, sink);
                continue;
            }
            for (std::int64_t c = 2 * h + 2; c >= 2 * h + 1; --c) {
                const Decision decision = decide(c, q, qf);
                if (decision == PRUNE) continue;
                if (decision == BULK) {
                    include(c, q, self, self_pos, sink);
                    continue;
                }
                stack[top++] = c;
            }
        }
    }

    // Visited nodes, visited points and hits (s <= R2) for routing.
    void pilot(const double* q, double R2, std::int64_t cap, std::int64_t* out) const {
        std::int64_t nodes = 0, points = 0, hits = 0;
        if (tree.n == 0) {
            out[0] = out[1] = out[2] = 0;
            return;
        }
        const Py_ssize_t d = tree.d;
        std::int64_t stack[128];
        int top = 0;
        double lower, upper;
        box(0, q, lower, upper);
        ++nodes;
        if (lower <= R2) {
            if (upper <= R2) {
                points += tree.n;
                hits += tree.n;
            } else {
                stack[top++] = 0;
            }
        }
        while (top && points < cap) {
            const std::int64_t h = stack[--top];
            if (h >= tree.first_leaf) {
                for (std::int64_t p = tree.begin[h]; p < tree.end[h]; ++p) {
                    ++points;
                    hits += canonical_d2<D>(q, tree.points.data() + p * d, d) <= R2;
                }
                continue;
            }
            for (std::int64_t c = 2 * h + 2; c >= 2 * h + 1; --c) {
                box(c, q, lower, upper);
                ++nodes;
                if (lower > R2) continue;
                if (upper <= R2) {
                    const std::int64_t size = tree.end[c] - tree.begin[c];
                    points += size;
                    hits += size;
                    continue;
                }
                stack[top++] = c;
            }
        }
        out[0] = nodes;
        out[1] = points;
        out[2] = hits;
    }
};

struct QueryCall {
    const Tree* tree;
    const float* queries;
    Py_ssize_t m;
    const Thr* thr;
    std::int64_t q_offset;
    bool exclude_self;
    int mode;
    int threads;
};

Screen32 call_screen(const QueryCall& call) {
    if (call.mode & NO_SCREEN) return Screen32{};
    return make_screen32(call.thr->R2, call.tree->d);
}

// Self-queries run in tree order for locality; other queries keep row order.
std::vector<std::int64_t> query_order(const QueryCall& call) {
    std::vector<std::int64_t> order;
    const Tree& tree = *call.tree;
    if (!call.exclude_self || call.m < 2 || tree.n == 0) return order;
    const std::int64_t lo = call.q_offset, hi = call.q_offset + call.m;
    order.reserve(std::size_t(call.m));
    if (std::int64_t(call.m) * 8 >= std::int64_t(tree.n)) {
        for (std::int64_t p = 0; p < tree.n; ++p) {
            const std::int64_t original = tree.perm[p];
            if (original >= lo && original < hi) order.push_back(original - lo);
        }
    } else {
        std::vector<std::pair<std::int32_t, std::int64_t>> keyed;
        keyed.reserve(std::size_t(call.m));
        for (std::int64_t r = 0; r < call.m; ++r) {
            const std::int64_t original = lo + r;
            if (original < tree.n) keyed.emplace_back(tree.position[original], r);
        }
        std::sort(keyed.begin(), keyed.end());
        for (const auto& item : keyed) order.push_back(item.second);
    }
    for (std::int64_t r = std::max<std::int64_t>(0, tree.n - lo); r < call.m; ++r) order.push_back(r);
    return order;
}

std::size_t rows_per_task(Py_ssize_t m, int threads) {
    const std::int64_t target = std::int64_t(m) / (8 * std::int64_t(threads));
    return std::size_t(std::clamp<std::int64_t>(target, 8, 64));
}

double query_estimate(const QueryCall& call) {
    return double(call.m) * (500.0 + 50.0 * double(call.tree->d));
}

template <int D>
std::unique_ptr<Result> tree_query_impl(const QueryCall& call) {
    const Tree& tree = *call.tree;
    const Thr& thr = *call.thr;
    const Py_ssize_t d = tree.d;
    Searcher<D> searcher(tree, thr, call_screen(call), call.mode);
    const std::vector<std::int64_t> order = query_order(call);
    const auto bounds = uniform_bounds(call.m, std::int64_t(rows_per_task(call.m, call.threads)));
    return run_rows(call.m, tree.n, thr, call.mode, order, bounds, call.threads,
                    query_estimate(call),
                    [&](std::int64_t row, RowSink& sink, SortScratch& scratch) {
                        const float* qf = call.queries + row * d;
                        double fixed[D > 0 ? D : 1];
                        double* q = fixed;
                        if (D == 0) {
                            scratch.query.resize(std::size_t(d));
                            q = scratch.query.data();
                        }
                        for (Py_ssize_t k = 0; k < d; ++k) q[k] = double(qf[k]);
                        std::int64_t self = -1, self_pos = -1;
                        if (call.exclude_self) {
                            self = call.q_offset + row;
                            if (self < tree.n) self_pos = tree.position[self];
                        }
                        searcher.search(q, qf, self, self_pos, sink);
                    }, tree.n);
}

// Second pass of TWO_PASS mode: write rows straight into the caller's CSR.
template <int D>
void tree_fill_impl(const QueryCall& call, const std::int64_t* indptr, std::int64_t* indices,
                    double* distances) {
    const Tree& tree = *call.tree;
    const Thr& thr = *call.thr;
    const Py_ssize_t d = tree.d;
    Searcher<D> searcher(tree, thr, call_screen(call), call.mode);
    const std::vector<std::int64_t> order = query_order(call);
    const std::size_t per_task = rows_per_task(call.m, call.threads);
    const std::size_t n_tasks = (std::size_t(call.m) + per_task - 1) / per_task;
    const bool keep_d2 = distances != nullptr || (call.mode & SORT);
    const bool by_distance = (call.mode & SORT) != 0;
    const bool fail = (call.mode & FAIL_TEST) != 0;
    parallel_for(n_tasks, call.threads, query_estimate(call), [&](std::size_t task) {
        if (fail && task + 1 == n_tasks) throw std::runtime_error("injected failure in a native radius task");
        TaskOut out;
        SortScratch scratch;
        RowSink sink;
        sink.out = &out;
        sink.keep_d2 = keep_d2;
        std::vector<double> query(static_cast<std::size_t>(d));
        const std::size_t first = task * per_task;
        const std::size_t last = std::min(first + per_task, std::size_t(call.m));
        for (std::size_t i = first; i < last; ++i) {
            const std::int64_t row = order.empty() ? std::int64_t(i) : order[i];
            const float* qf = call.queries + row * d;
            for (Py_ssize_t k = 0; k < d; ++k) query[k] = double(qf[k]);
            std::int64_t self = -1, self_pos = -1;
            if (call.exclude_self) {
                self = call.q_offset + row;
                if (self < tree.n) self_pos = tree.position[self];
            }
            out.cols.clear();
            out.d2.clear();
            sink.count = 0;
            searcher.search(query.data(), qf, self, self_pos, sink);
            const std::size_t count = out.cols.size();
            if (std::int64_t(count) != indptr[row + 1] - indptr[row])
                throw InputError("indptr does not match the counting pass of this query");
            sort_row(out.cols.data(), keep_d2 ? out.d2.data() : nullptr, count, by_distance, thr.R2,
                     scratch, tree.n);
            std::int64_t* cols = indices + indptr[row];
            for (std::size_t j = 0; j < count; ++j) cols[j] = out.cols[j];
            if (distances) {
                double* dist = distances + indptr[row];
                for (std::size_t j = 0; j < count; ++j) dist[j] = std::sqrt(std::min(out.d2[j], thr.R2));
            }
        }
    });
}

template <int D>
void tree_pilot_impl(const Tree& tree, const float* queries, Py_ssize_t s, double R2,
                     std::int64_t cap, std::int64_t* out, int threads) {
    Thr thr;
    thr.R2 = R2;
    Searcher<D> searcher(tree, thr, Screen32{});
    const Py_ssize_t d = tree.d;
    parallel_for(std::size_t(s), threads, double(s) * double(std::min<std::int64_t>(cap, tree.n)) * 4.0,
                 [&](std::size_t row) {
                     std::vector<double> q(static_cast<std::size_t>(d));
                     for (Py_ssize_t k = 0; k < d; ++k) q[k] = double(queries[row * d + k]);
                     searcher.pilot(q.data(), R2, cap, out + row * 3);
                 });
}

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
struct BuildOp {
    static std::unique_ptr<Tree> run(const float* X, Py_ssize_t n, Py_ssize_t d, int leaf_size,
                                     int threads) {
        return build_tree_impl<D>(X, n, d, leaf_size, threads);
    }
};

template <int D>
struct QueryOp {
    static std::unique_ptr<Result> run(const QueryCall& call) { return tree_query_impl<D>(call); }
};

template <int D>
struct FillOp {
    static int run(const QueryCall& call, const std::int64_t* indptr, std::int64_t* indices,
                   double* distances) {
        tree_fill_impl<D>(call, indptr, indices, distances);
        return 0;
    }
};

template <int D>
struct PilotOp {
    static int run(const Tree& tree, const float* queries, Py_ssize_t s, double R2,
                   std::int64_t cap, std::int64_t* out, int threads) {
        tree_pilot_impl<D>(tree, queries, s, R2, cap, out, threads);
        return 0;
    }
};

bool check_finite(const float* values, std::size_t count) {
    std::uint32_t invalid = 0;
    for (std::size_t i = 0; i < count; ++i) invalid |= (float_bits(values[i]) & 0x7fffffffU) >= 0x7f800000U;
    return invalid == 0;
}

// Parse the common (tree, queries, thresholds, q_offset, exclude_self, mode,
// threads) prefix of tree_query and tree_fill.
bool parse_query_call(PyObject* tree_object, PyObject* query_object, PyObject* thr_object,
                      Py_ssize_t q_offset, int exclude_self, int mode, Py_ssize_t threads,
                      Buffer& queries, Thr& thr, QueryCall& call) {
    Tree* tree = unwrap_tree(tree_object);
    if (!tree) return false;
    if (!parse_thresholds(thr_object, thr)) return false;
    if (!get_buffer(query_object, queries, false, "queries", DType::f32, 2)) return false;
    if (queries.dim(1) != tree->d) {
        PyErr_Format(PyExc_ValueError, "queries have %zd features, the tree has %zd",
                     queries.dim(1), tree->d);
        return false;
    }
    if (q_offset < 0 || (mode & ~KNOWN_MODES)) {
        PyErr_SetString(PyExc_ValueError, "invalid q_offset or mode");
        return false;
    }
    if ((mode & SORT) && !(mode & NEED_DISTANCE)) {
        PyErr_SetString(PyExc_ValueError, "SORT requires NEED_DISTANCE");
        return false;
    }
    call.tree = tree;
    call.queries = queries.data<const float>();
    call.m = queries.dim(0);
    call.thr = &thr;
    call.q_offset = q_offset;
    call.exclude_self = exclude_self != 0;
    call.mode = mode;
    call.threads = clamp_threads(threads);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

PyObject* py_build_tree(PyObject*, PyObject* args) {
    PyObject* train_object;
    Py_ssize_t leaf_size, threads;
    if (!PyArg_ParseTuple(args, "Onn:build_tree", &train_object, &leaf_size, &threads)) return nullptr;
    Buffer train;
    if (!get_buffer(train_object, train, false, "train", DType::f32, 2)) return nullptr;
    const Py_ssize_t n = train.dim(0), d = train.dim(1);
    if (d < 1 || n > Py_ssize_t(std::numeric_limits<std::int32_t>::max()) || leaf_size < 1 ||
        leaf_size > (1 << 20)) {
        PyErr_SetString(PyExc_ValueError, "build_tree needs d >= 1, n < 2**31 and 1 <= leaf_size <= 2**20");
        return nullptr;
    }
    const float* X = train.data<const float>();
    std::unique_ptr<Tree> tree;
    std::exception_ptr error;
    bool finite = true;
    Py_BEGIN_ALLOW_THREADS
    try {
        finite = check_finite(X, std::size_t(n * d));
        if (finite) tree = dispatch_dimension<BuildOp>(d, X, n, d, int(leaf_size), clamp_threads(threads));
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    if (!finite) {
        PyErr_SetString(PyExc_ValueError, "training coordinates must be finite");
        return nullptr;
    }
    PyObject* capsule = PyCapsule_New(tree.get(), TREE_CAPSULE, tree_destructor);
    if (!capsule) return nullptr;
    tree.release();
    return capsule;
}

PyObject* py_tree_info(PyObject*, PyObject* args) {
    PyObject* tree_object;
    if (!PyArg_ParseTuple(args, "O:tree_info", &tree_object)) return nullptr;
    Tree* tree = unwrap_tree(tree_object);
    if (!tree) return nullptr;
    return Py_BuildValue("(nniiL)", tree->n, tree->d, tree->leaf_size, tree->depth,
                         static_cast<long long>(tree->n_nodes));
}

PyObject* py_tree_export(PyObject*, PyObject* args) {
    PyObject *tree_object, *perm_object, *points_object, *lo_object, *hi_object;
    if (!PyArg_ParseTuple(args, "OOOOO:tree_export", &tree_object, &perm_object, &points_object,
                          &lo_object, &hi_object))
        return nullptr;
    Tree* tree = unwrap_tree(tree_object);
    if (!tree) return nullptr;
    Buffer perm, points, lo, hi;
    if (!get_buffer(perm_object, perm, true, "perm", DType::i32, 0) ||
        !get_buffer(points_object, points, true, "points", DType::f32, 0) ||
        !get_buffer(lo_object, lo, true, "lo", DType::f32, 0) ||
        !get_buffer(hi_object, hi, true, "hi", DType::f32, 0))
        return nullptr;
    if (perm.count() != tree->n || points.count() != tree->n * tree->d ||
        lo.count() != tree->n_nodes * tree->d || hi.count() != tree->n_nodes * tree->d) {
        PyErr_SetString(PyExc_ValueError, "tree_export output sizes do not match the tree");
        return nullptr;
    }
    std::copy(tree->perm.begin(), tree->perm.end(), perm.data<std::int32_t>());
    std::copy(tree->points.begin(), tree->points.end(), points.data<float>());
    std::copy(tree->lo.begin(), tree->lo.end(), lo.data<float>());
    std::copy(tree->hi.begin(), tree->hi.end(), hi.data<float>());
    Py_RETURN_NONE;
}

PyObject* py_tree_query(PyObject*, PyObject* args) {
    PyObject *tree_object, *query_object, *thr_object;
    Py_ssize_t q_offset, threads;
    int exclude_self, mode;
    if (!PyArg_ParseTuple(args, "OOOnpin:tree_query", &tree_object, &query_object, &thr_object,
                          &q_offset, &exclude_self, &mode, &threads))
        return nullptr;
    Buffer queries;
    Thr thr;
    QueryCall call{};
    if (!parse_query_call(tree_object, query_object, thr_object, q_offset, exclude_self, mode,
                          threads, queries, thr, call))
        return nullptr;
    std::unique_ptr<Result> result;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try {
        result = dispatch_dimension<QueryOp>(call.tree->d, call);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    return wrap_result(std::move(result));
}

PyObject* py_tree_fill(PyObject*, PyObject* args) {
    PyObject *tree_object, *query_object, *thr_object, *indptr_object, *indices_object,
        *distances_object;
    Py_ssize_t q_offset, threads;
    int exclude_self, mode;
    if (!PyArg_ParseTuple(args, "OOOnpinOOO:tree_fill", &tree_object, &query_object, &thr_object,
                          &q_offset, &exclude_self, &mode, &threads, &indptr_object,
                          &indices_object, &distances_object))
        return nullptr;
    Buffer queries, indptr, indices, distances;
    Thr thr;
    QueryCall call{};
    if (!parse_query_call(tree_object, query_object, thr_object, q_offset, exclude_self, mode,
                          threads, queries, thr, call))
        return nullptr;
    if (!get_buffer(indptr_object, indptr, false, "indptr", DType::i64, 1) ||
        !get_buffer(indices_object, indices, true, "indices", DType::i64, 1) ||
        (distances_object != Py_None &&
         !get_buffer(distances_object, distances, true, "distances", DType::f64, 1)))
        return nullptr;
    const std::int64_t* bounds = indptr.data<const std::int64_t>();
    if (indptr.count() != call.m + 1 || bounds[0] != 0 || bounds[call.m] != indices.count() ||
        (distances.acquired && distances.count() != indices.count())) {
        PyErr_SetString(PyExc_ValueError, "indptr, indices and distances sizes do not agree");
        return nullptr;
    }
    for (Py_ssize_t r = 0; r < call.m; ++r) {
        if (bounds[r + 1] < bounds[r]) {
            PyErr_SetString(PyExc_ValueError, "indptr must be non-decreasing");
            return nullptr;
        }
    }
    if (overlaps(indices, distances) || overlaps(indices, indptr) || overlaps(distances, indptr) ||
        overlaps(indices, queries) || overlaps(distances, queries)) {
        PyErr_SetString(PyExc_ValueError, "output buffers must not overlap inputs or each other");
        return nullptr;
    }
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try {
        dispatch_dimension<FillOp>(call.tree->d, call, bounds, indices.data<std::int64_t>(),
                                   distances.acquired ? distances.data<double>() : nullptr);
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    Py_RETURN_NONE;
}

PyObject* py_tree_pilot(PyObject*, PyObject* args) {
    PyObject *tree_object, *query_object, *out_object;
    double R2;
    Py_ssize_t cap, threads = 1;
    if (!PyArg_ParseTuple(args, "OOdnO|n:tree_pilot", &tree_object, &query_object, &R2, &cap,
                          &out_object, &threads))
        return nullptr;
    Tree* tree = unwrap_tree(tree_object);
    if (!tree) return nullptr;
    Buffer queries, out;
    if (!get_buffer(query_object, queries, false, "queries", DType::f32, 2) ||
        !get_buffer(out_object, out, true, "out", DType::i64, 2))
        return nullptr;
    const Py_ssize_t s = queries.dim(0);
    if (queries.dim(1) != tree->d || out.dim(0) != s || out.dim(1) != 3 || !(R2 >= 0)) {
        PyErr_SetString(PyExc_ValueError, "tree_pilot needs queries[s, d], out[s, 3] and R2 >= 0");
        return nullptr;
    }
    if (overlaps(out, queries)) {
        PyErr_SetString(PyExc_ValueError, "output must not overlap the queries");
        return nullptr;
    }
    std::exception_ptr error;
    const float* q = queries.data<const float>();
    std::int64_t* o = out.data<std::int64_t>();
    Py_BEGIN_ALLOW_THREADS
    try {
        dispatch_dimension<PilotOp>(tree->d, *tree, q, s, R2, std::int64_t(cap < 1 ? 1 : cap), o,
                                    clamp_threads(threads));
    } catch (...) {
        error = std::current_exception();
    }
    Py_END_ALLOW_THREADS
    if (error) return raise_exception(error);
    Py_RETURN_NONE;
}

}  // namespace mlx_radius
