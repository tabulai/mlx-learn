"""Adaptive exact ordering of squared distances on original FP32 coordinates.

Ordinary comparisons use conservative intervals around the compensated FP32
distance. Overlapping intervals use a guarded double-single filter followed by
exact dyadic integers when unresolved, retaining discarded subtraction/product bits. Returned distances
remain FP32; these helpers only choose their corresponding training indices.
"""

PRECISION_HEADER = r"""
#ifndef MLXLEARN_NEIGHBOR_PRECISION
#define MLXLEARN_NEIGHBOR_PRECISION
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// A finite binary32 number is sign*mantissa*2**(shift-149). Subnormals use
// shift=0 without a hidden bit. Decode the bits rather than performing floating
// operations, which are allowed to flush subnormal operands/results to zero.
inline uint precise_mantissa(uint bits) {
    return (bits & 0x007fffffu) | ((bits & 0x7f800000u) ? 0x00800000u : 0u);
}
inline uint precise_shift(uint bits) {
    const uint exponent = (bits >> 23) & 255u;
    return exponent ? exponent - 1u : 0u;
}

// Add/subtract a shifted 48-bit product in a signed two's-complement
// superaccumulator. Nineteen limbs cover every intermediate for positive
// int32 D: four terms per feature have magnitude below 2**557 after scaling
// by 2**298, hence their accumulated magnitude is below 2**588 < 2**607.
template <int Words>
inline void precise_add_product(thread uint* accumulator, uint first, uint second,
                                uint shift, bool negative) {
    const ulong product = ulong(first) * ulong(second);
    if (!product) return;
    const uint word = shift >> 5;
    const uint offset = shift & 31u;
    const ulong low = product << offset;
    const uint chunks[3] = {uint(low), uint(low >> 32),
                            offset ? uint(product >> (64u - offset)) : 0u};
    ulong carry = 0;
    for (uint chunk = 0; chunk < 3; ++chunk) {
        const uint position = word + chunk;
        if (position >= Words) break;
        const ulong term = ulong(chunks[chunk]) + carry;
        const uint previous = accumulator[position];
        if (negative) {
            accumulator[position] = previous - uint(term);
            carry = ulong(previous) < term;
        } else {
            const ulong total = ulong(previous) + term;
            accumulator[position] = uint(total);
            carry = total >> 32;
        }
    }
    for (uint position = word + 3; position < Words && carry; ++position) {
        const uint previous = accumulator[position];
        if (negative) {
            accumulator[position] = previous - 1u;
            carry = previous == 0u;
        } else {
            accumulator[position] = previous + 1u;
            carry = previous == 0xffffffffu;
        }
    }
}

template <int D, int Words, typename TrainPointer, typename QueryPointer>
__attribute__((noinline)) int precise_distance_limbs(int first, int second,
    TrainPointer train, QueryPointer queries, int query, uint base_shift) {
    const size_t qa = size_t(query) * D, aa = size_t(first) * D, ba = size_t(second) * D;
    uint accumulator[Words];
    for (int word = 0; word < Words; ++word) accumulator[word] = 0u;
    #pragma clang loop unroll(disable)
        for (int feature = 0; feature < D; ++feature) {
        const uint qb = as_type<uint>(queries[qa + feature]);
        const uint ab = as_type<uint>(train[aa + feature]);
        const uint bb = as_type<uint>(train[ba + feature]);
        const uint q = precise_mantissa(qb), a = precise_mantissa(ab), b = precise_mantissa(bb);
        const uint qs = precise_shift(qb) - base_shift;
        const uint as = precise_shift(ab) - base_shift, bs = precise_shift(bb) - base_shift;
        // ||q-a||**2 - ||q-b||**2 = a*a - b*b - 2*q*a + 2*q*b.
        precise_add_product<Words>(accumulator, a, a, 2u*as, false);
        precise_add_product<Words>(accumulator, b, b, 2u*bs, true);
        precise_add_product<Words>(accumulator, q, a, qs + as + 1u, ((qb ^ ab) >> 31) == 0u);
        precise_add_product<Words>(accumulator, q, b, qs + bs + 1u, ((qb ^ bb) >> 31) != 0u);
    }
    if (accumulator[Words - 1] & 0x80000000u) return -1;
    uint nonzero = 0u;
    for (int word = 0; word < Words; ++word) nonzero |= accumulator[word];
    return int(nonzero != 0u);
}


inline float2 precise_two_sum(float a, float b) {
    const float sum = a + b;
    const float part = sum - a;
    return float2(sum, (a - (sum - part)) + (b - part));
}

template <int D, typename TrainPointer, typename QueryPointer>
__attribute__((noinline)) int precise_pair_compare(int first, int second,
    TrainPointer train, QueryPointer queries, int query) {
    float2 sums[2] = {float2(0.0f), float2(0.0f)};
    #pragma clang loop unroll(disable)
    for (int feature = 0; feature < D; ++feature) {
        const float q = queries[size_t(query) * D + feature];
        const float points[2] = {train[size_t(first) * D + feature], train[size_t(second) * D + feature]};
        const uint qb = as_type<uint>(q) & 0x7fffffffu;
        if (qb && (qb < 0x30800000u || qb > 0x4e800000u)) return 2;
        for (int item = 0; item < 2; ++item) {
            const uint bits = as_type<uint>(points[item]) & 0x7fffffffu;
            if (bits && (bits < 0x30800000u || bits > 0x4e800000u)) return 2;
            const float2 delta = precise_two_sum(q, -points[item]);
            const float product = delta.x * delta.x;
            const float tail = metal::fma(delta.x, delta.x, -product)
                             + (2.0f * delta.x * delta.y + delta.y * delta.y);
            const float2 added = precise_two_sum(sums[item].x, product);
            sums[item] = precise_two_sum(added.x, (sums[item].y + tail) + added.y);
        }
    }
    const float difference = (sums[0].x - sums[1].x) + (sums[0].y - sums[1].y);
    // Twice-working-precision accumulation: the generous O(D*epsilon^2)
    // envelope also covers the final difference and any flushed residual.
    const float bound = (128.0f * float(D) * 0x1p-46f) * (sums[0].x + sums[1].x)
                      + float(D) * 0x1p-90f;
    if (difference < -bound) return -1;
    if (difference > bound) return 1;
    return 2;  // unresolved, never an assertion of equality
}

template <int D, typename TrainPointer, typename QueryPointer>
__attribute__((noinline)) int precise_distance_compare(int first, int second,
                                    TrainPointer train, QueryPointer queries, int query) {
    const size_t qa = size_t(query) * D;
    const size_t aa = size_t(first) * D;
    const size_t ba = size_t(second) * D;
    if (D <= 4096) {
        const int pair_order = precise_pair_compare<D>(first, second, train, queries, query);
        if (pair_order != 2) return pair_order;
    }
    // Common-offset inputs normally share an exponent. Their exact ordering
    // fits a single signed 64-bit accumulator, without the general limb path.
    // Each feature contributes less than 6*2**48 in absolute value.
    if (D <= 4096) {
        const uint exponent = (as_type<uint>(queries[qa]) >> 23) & 255u;
        long difference = 0;
        bool common_exponent = true;
        #pragma clang loop unroll(disable)
        for (int feature = 0; feature < D; ++feature) {
            const uint qb = as_type<uint>(queries[qa + feature]);
            const uint ab = as_type<uint>(train[aa + feature]);
            const uint bb = as_type<uint>(train[ba + feature]);
            if (ab == bb) continue;
            if (((qb >> 23) & 255u) != exponent ||
                ((ab >> 23) & 255u) != exponent ||
                ((bb >> 23) & 255u) != exponent) {
                common_exponent = false;
                break;
            }
            const long q = (qb >> 31) ? -long(precise_mantissa(qb)) : long(precise_mantissa(qb));
            const long a = (ab >> 31) ? -long(precise_mantissa(ab)) : long(precise_mantissa(ab));
            const long b = (bb >> 31) ? -long(precise_mantissa(bb)) : long(precise_mantissa(bb));
            difference += a*a - b*b - 2*q*a + 2*q*b;
        }
        if (common_exponent) return difference < 0 ? -1 : int(difference > 0);
    }
    // Most real workloads occupy a narrow exponent range. Rebase their exact
    // integers and use four limbs; this is the same exact arithmetic, with a
    // proved smaller accumulator, not a reduced-precision approximation.
    if (D <= 4096) {
        uint low = 253u, high = 0u;
        #pragma clang loop unroll(disable)
        for (int feature = 0; feature < D; ++feature) {
            const uint values[3] = {as_type<uint>(queries[qa + feature]),
                as_type<uint>(train[aa + feature]), as_type<uint>(train[ba + feature])};
            for (int item = 0; item < 3; ++item) {
                if (precise_mantissa(values[item])) {
                    const uint shift = precise_shift(values[item]);
                    low = metal::min(low, shift); high = metal::max(high, shift);
                }
            }
        }
        // Four products per feature, 48 mantissa bits, one cross-term bit:
        // with span<=30 and D<=4096 the absolute sum is below 2^123.
        if (high <= low + 30u)
            return precise_distance_limbs<D, 4>(first, second, train, queries, query, low);
    }
    return precise_distance_limbs<D, 19>(first, second, train, queries, query, 0u);
}

// For D<=4096, telescoping Kahan residuals bounds accumulation by
// 4u*S + (13D+1)u*u*S + 11D*u*u*u*S; products/subtractions add <4u*T.
// The 12u envelope includes endpoint rounding and inversion. Larger D uses
// exact integers rather than this inexpensive filter.
// Restrict the inexpensive filter's dimension/range; all other comparisons
// use exact integers. The absolute floor also covers permitted FP32 flushing.
template <int D>
inline float precise_distance_upper(float value) {
    if (D > 4096 || !(value >= 0.0f && value <= 0x1p110f)) return INFINITY;
    const float relative = 6.0f * 0x1p-23f;
    return value * (1.0f + relative) + float(D) * 0x1p-100f;
}
template <int D>
inline float precise_distance_lower(float value) {
    if (D > 4096 || !(value >= 0.0f && value <= 0x1p110f)) return -INFINITY;
    const float relative = 6.0f * 0x1p-23f;
    return value * (1.0f - relative) - float(D) * 0x1p-100f;
}

template <int D, typename TrainPointer, typename QueryPointer>
inline bool precise_neighbor_precedes(float distance, int index,
                                     float other_distance, int other_index,
                                     TrainPointer train, QueryPointer queries,
                                     int query, int n_train) {
    if (index == other_index || index < 0 || index >= n_train) return false;
    if (other_index < 0 || other_index >= n_train) return true;
    // The overwhelmingly common rejected scan candidate needs only this
    // wide one-sided interval check. These guarded shortcuts are looser than
    // the symmetric endpoints below and cannot resolve an ambiguous pair.
    if (D <= 4096 && other_distance >= float(D) * 0x1p-70f && distance <= 0x1p110f &&
        distance > other_distance * (1.0f + 0x1p-17f)) return false;
    if (D <= 4096 && distance >= float(D) * 0x1p-70f && other_distance <= 0x1p110f &&
        other_distance > distance * (1.0f + 0x1p-17f)) return true;
    if (precise_distance_upper<D>(distance) < precise_distance_lower<D>(other_distance)) return true;
    if (precise_distance_upper<D>(other_distance) < precise_distance_lower<D>(distance)) return false;
    const int comparison = precise_distance_compare<D>(index, other_index, train, queries, query);
    return comparison < 0 || (comparison == 0 && index < other_index);
}

struct PreciseNeighbor { float distance; int index; };

template <int D, typename TrainPointer, typename QueryPointer>
inline PreciseNeighbor precise_simd_min(float head, int index,
                              TrainPointer train, QueryPointer queries,
                              int query, int n_train) {
    const int sentinel = 2147483647;
    const bool valid = index >= 0 && index < n_train;
    const float provisional_distance = metal::simd_min(valid ? head : INFINITY);
    int winner = metal::simd_min(valid && head == provisional_distance ? index : sentinel);
    if (winner == sentinel) return {INFINITY, winner};
    // Any point capable of beating the provisional winner overlaps its upper
    // endpoint. Ordinary well-separated heads retain the SIMD-min fast path.
    const bool possible = valid &&
        precise_distance_lower<D>(head) <= precise_distance_upper<D>(provisional_distance);
    if (!metal::simd_any(possible && index != winner)) return {provisional_distance, winner};
    float winner_distance = provisional_distance;
    for (;;) {
        const bool better = possible && precise_neighbor_precedes<D>(
            head, index, winner_distance, winner, train, queries, query, n_train);
        const int replacement = metal::simd_min(better ? index : sentinel);
        if (replacement == sentinel) return {winner_distance, winner};
        // Each replacement strictly precedes the previous winner. There are
        // at most 32 real heads, so this exact tournament always terminates.
        winner = replacement;
        winner_distance = metal::simd_min(index == winner ? head : INFINITY);
    }
}
#endif
"""
