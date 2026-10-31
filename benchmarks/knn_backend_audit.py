"""Runtime-free validation and counts for automatic benchmark routing."""

from collections import Counter


def validate_record_backend(record, current_mode, *, is_sklearn=False):
    """Error requests remain evidence; successful calls require full telemetry."""
    if "error" in record:
        if record.get("audit", {}).get("pass", False):
            raise ValueError("An error request cannot pass its audit")
        return
    query = record.get("query_backend")
    fitted = record.get("fit_backend")
    policy = record.get("execution_policy")
    if record.get("backend") != query:
        raise ValueError("Backend label disagrees with actual query backend")
    if record.get("query_backend_phase") not in {"timed_query", "untimed_fit_audit"}:
        raise ValueError("Missing query timing scope")
    if is_sklearn:
        valid = (policy, fitted, query) == ("fixed", "sklearn", "sklearn")
    elif current_mode == "auto":
        valid = policy == "auto" and fitted in {"cpu", "mlx"} and query in {"cpu", "mlx"}
    elif current_mode == "gpu":
        valid = (policy, fitted, query) == ("fixed", "mlx", "mlx")
    else:
        raise ValueError(f"Unknown current mode: {current_mode}")
    if not valid:
        raise ValueError(f"Unexpected {current_mode} backend telemetry: {(policy, fitted, query)}")


def backend_counts(records):
    """Count requests, including failures; older GPU evidence keeps its label."""
    records = list(records)
    return {
        "request_count": len(records),
        "query_backend_counts": dict(Counter(
            "error" if "error" in r else r.get("query_backend", r.get("backend", "missing"))
            for r in records
        )),
        "fit_backend_counts": dict(Counter(
            "error" if "error" in r else r.get("fit_backend", r.get("backend", "missing"))
            for r in records
        )),
        "execution_policy_counts": dict(Counter(
            "error" if "error" in r else r.get("execution_policy", "fixed") for r in records
        )),
    }
