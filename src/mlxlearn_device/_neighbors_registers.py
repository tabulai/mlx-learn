"""Bounded Metal source generation for two-query direct neighbor scans.

The k=5 specialization uses named scalar candidates and shifts a winning lane's
list after each SIMD reduction. Other k values retain the general array selector.
Both variants share the same compensated arithmetic and optionally load four
features at a time through packed_float4, whose alignment is only one float.
"""

_PREFIX = r"""
    #pragma clang fp contract(off)
    #pragma clang fp reassociate(off)
    const uint lane = thread_index_in_simdgroup;
    const uint task = thread_position_in_grid.x / 32;
    const int n_train = shape[0];
    const int n_query = shape[1];
    const int parts = shape[2];
    const int query_base = int(task / uint(parts)) * 2;
    if (query_base >= n_query) return;
    const int part = int(task % uint(parts));
    const int begin = int((size_t(part) * size_t(n_train)) / size_t(parts));
    const int end = int((size_t(part + 1) * size_t(n_train)) / size_t(parts));
    const int sentinel = 2147483647;
"""


def _compensated_step(query, delta):
    """Emit one original-order Kahan update with contraction disabled above."""
    return f"""{{
        const float delta = {delta};
        const float adjusted = delta * delta - correction{query};
        const float total = sum{query} + adjusted;
        correction{query} = (total - sum{query}) - adjusted;
        sum{query} = total;
    }}"""


def _distance_source(packed):
    lines = ["float sum0 = 0.0f, sum1 = 0.0f;",
             "float correction0 = 0.0f, correction1 = 0.0f;"]
    if packed:
        lines += [
            "for (int feature = 0; feature < D; feature += 4) {",
            "const packed_float4 training = *reinterpret_cast<device const packed_float4*>(",
            "    train + size_t(reference) * D + feature);",
        ]
        for q in range(2):
            lines += [
                f"const packed_float4 query{q} = (!TAIL || query_base + {q} < n_query)",
                "    ? *reinterpret_cast<device const packed_float4*>(",
                f"        queries + size_t(query_base + {q}) * D + feature)",
                "    : packed_float4(0.0f);",
            ]
        # Component order is deliberately x,y,z,w, preserving the scalar scan's
        # per-pair rounding. Query chains are independent of one another.
        for component in "xyzw":
            for q in range(2):
                lines.append(_compensated_step(q, f"query{q}.{component} - training.{component}"))
    else:
        lines += ["for (int feature = 0; feature < D; ++feature) {",
                  "const float training = train[size_t(reference) * D + feature];"]
        for q in range(2):
            lines += [f"if (!TAIL || query_base + {q} < n_query) {{",
                      _compensated_step(q, f"queries[size_t(query_base + {q}) * D + feature] - training"),
                      "}"]
    lines.append("}")
    return "\n".join(lines)


def _candidate_guard(q):
    # Once nonnegative squared terms overflow, the original guarded sum remains
    # +inf. Unguarded compensation can become NaN; repair only after the loop.
    return f"""
    if ((!TAIL || query_base + {q} < n_query) &&
        (!EXCLUDE_SELF || reference != query_base + {q} + shape[3])) {{
        sum{q} = metal::isfinite(sum{q}) ? sum{q} : INFINITY;
    """


def _register_insert(q, screen=False):
    lines = [f"if (precise_neighbor_precedes<D>(sum{q}, reference, d{q}_4, i{q}_4, train, queries, query_base + {q}, n_train)) {{",
             f"d{q}_4 = sum{q}; i{q}_4 = reference;"]
    # Nested scalar insertion is the sorted-array insertion algorithm with its
    # four possible shifts expanded. No dynamic private-array index remains.
    for j in range(3, -1, -1):
        lines += [f"if (precise_neighbor_precedes<D>(sum{q}, reference, d{q}_{j}, i{q}_{j}, train, queries, query_base + {q}, n_train)) {{",
                  f"d{q}_{j+1} = d{q}_{j}; i{q}_{j+1} = i{q}_{j};",
                  f"d{q}_{j} = sum{q}; i{q}_{j} = reference;"]
    lines += ["}" for _ in range(5)]
    source = "\n".join(lines)
    if screen:
        source = source.replace("precise_neighbor_precedes<D>", "neighbor_precedes")
        source = source.replace(f", train, queries, query_base + {q}, n_train", "")
    return source


def _array_insert(q, screen=False):
    source = f"""
    if (precise_neighbor_precedes<D>(sum{q}, reference, distances[{q}][K-1], indices[{q}][K-1],
                                       train, queries, query_base + {q}, n_train)) {{
        int slot = K - 1;
        while (slot > 0 && precise_neighbor_precedes<D>(sum{q}, reference,
                                            distances[{q}][slot-1], indices[{q}][slot-1],
                                            train, queries, query_base + {q}, n_train)) {{
            distances[{q}][slot] = distances[{q}][slot-1];
            indices[{q}][slot] = indices[{q}][slot-1];
            --slot;
        }}
        distances[{q}][slot] = sum{q};
        indices[{q}][slot] = reference;
    }}
    """

    if screen:
        source = source.replace("precise_neighbor_precedes<D>", "neighbor_precedes")
        source = source.replace(f",\n                                       train, queries, query_base + {q}, n_train", "")
        source = source.replace(f",\n                                            train, queries, query_base + {q}, n_train", "")
    return source


def _merge_source(q, registers, screen=False):
    lines = [f"if (!TAIL || query_base + {q} < n_query) {{",
             f"const size_t target = size_t(query_base + {q}) * parts + part;"]
    if not registers:
        lines.append("int cursor = 0;")
    lines.append("for (int slot = 0; slot < K; ++slot) {")
    if registers:
        lines += [f"const float head = d{q}_0; const int index = i{q}_0;"]
    else:
        lines += [f"const float head = cursor < K ? distances[{q}][cursor] : INFINITY;",
                  f"const int index = cursor < K ? indices[{q}][cursor] : sentinel;"]
    lines += [f"const auto winner = precise_simd_min<D>(head, index, train, queries, query_base + {q}, n_train);",
              "const int winner_index = winner.index;",
              "const float winner_distance = winner.distance;",
              "if (lane == uint(slot) % 32) {",
              "    out_distances[target * K + slot] = winner_distance;",
              "    out_indices[target * K + slot] = winner_index;", "}"]
    if registers:
        lines.append("if (index == winner_index) {")
        for j in range(4):
            lines.append(f"d{q}_{j} = d{q}_{j+1}; i{q}_{j} = i{q}_{j+1};")
        lines += [f"d{q}_4 = INFINITY; i{q}_4 = sentinel;", "}"]
    else:
        lines.append("if (index == winner_index && cursor < K) ++cursor;")
    lines.append("}")
    if screen:
        head = f"d{q}_0" if registers else f"(cursor < K ? distances[{q}][cursor] : INFINITY)"
        lines += [f"const float next = metal::simd_min(metal::min({head}, omitted{q}));",
                  "if (lane == 0) out_barriers[target] = next;"]
    lines.append("}")
    source = "\n".join(lines)
    if screen:
        source = source.replace(
            f"const auto winner = precise_simd_min<D>(head, index, train, queries, query_base + {q}, n_train);\n"
            "const int winner_index = winner.index;\nconst float winner_distance = winner.distance;",
            "const float winner_distance = metal::simd_min(head);\n"
            "const int winner_index = metal::simd_min(head == winner_distance ? index : sentinel);",
        )
    return source


def paired_scan_source(*, registers, packed, screen=False):
    """Build one of four bounded source variants; registers requires k=5."""
    lines = [_PREFIX]
    if screen:
        lines.append("float omitted0 = INFINITY, omitted1 = INFINITY;")
    if registers:
        for q in range(2):
            for j in range(5):
                lines.append(f"float d{q}_{j} = INFINITY; int i{q}_{j} = sentinel;")
    else:
        lines += ["float distances[2][K]; int indices[2][K];",
                  "for (int slot = 0; slot < K; ++slot) {",
                  "    distances[0][slot] = INFINITY; distances[1][slot] = INFINITY;",
                  "    indices[0][slot] = sentinel; indices[1][slot] = sentinel;", "}"]
    lines += ["for (int reference = begin + int(lane); reference < end; reference += 32) {",
              _distance_source(packed)]
    for q in range(2):
        lines.append(_candidate_guard(q))
        if screen:
            worst = f"d{q}_4" if registers else f"distances[{q}][K-1]"
            lines.append(f"omitted{q} = metal::min(omitted{q}, metal::max(sum{q}, {worst}));")
        lines += [_register_insert(q, screen) if registers else _array_insert(q, screen), "}"]
    lines.append("}")
    for q in range(2):
        lines.append(_merge_source(q, registers, screen))
    return "\n".join(lines)
