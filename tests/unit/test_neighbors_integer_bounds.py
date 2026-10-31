"""CPU-only certificate checks independent of MPP execution and lane layout."""

from __future__ import annotations

import ast
import re
import struct
from fractions import Fraction
from pathlib import Path

import pytest

F = Fraction
U = F(1, 2**23)
ETA = F(1, 2**126)


def _float32(text):
    return F.from_float(struct.unpack("f", struct.pack("f", float(text)))[0])


def _production_coefficients():
    path = Path(__file__).resolve().parents[2] / "src/mlxlearn_device/_neighbors_integer.py"
    source = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"_PREPARE", "_SCAN"}:
                source[name] = ast.literal_eval(node.value)
    preparation = re.search(
        r"biases\[row\]=safe\?([0-9.]+)f\*quant_norm-([0-9.]+)f\*measured",
        source["_PREPARE"],
    )
    query = re.search(r"qb=safe\?([0-9.]+)f\*quant_norm-([0-9.]+)f\*measured", source["_SCAN"])
    cross = re.search(r"qfactor=\(2\.0f\*([0-9.]+)f\)\*qs", source["_SCAN"])
    tiny = re.search(r"const float lower=.*-([0-9.e-]+)f;", source["_SCAN"])
    assert preparation and query and cross and tiny
    assert preparation.groups() == query.groups(), "training/query certificate coefficients differ"
    return (_float32(cross[1]), _float32(preparation[1]),
            _float32(preparation[2]), _float32(tiny[1]))


def _certificate_margin(a, c, beta, tiny):
    # Shared compact certificate: original Kahan >=(1-512u)*true distance
    # -4096eta. Measured error covers any32-term positive reduction order.
    b, measured, zeta = 1-512*U, F(99998, 100000), F(1, 1024)
    proved_h = b*F(63, 64)
    proved_error = 126*b*(1+zeta)/measured
    proved_norm = 126*b*(1+1/zeta)*U**2
    proved_absolute = (4096+16384*proved_error)*ETA
    rho = 1+U
    norm_reserve = a-c-c*(rho**7-1)-a*(rho**5-1)
    error_penalty = beta*(2-rho**5)
    return (proved_h-a, error_penalty-proved_error,
            norm_reserve-proved_norm, (1-U)*tiny-proved_absolute-1000*ETA)


def test_production_bias_is_below_original_distance_envelope():
    # Positive margins prove domination for all nonnegative H,E,N, rather
    # than checking a few sampled coordinate pairs or duplicating a kernel.
    assert all(margin > 0 for margin in _certificate_margin(*_production_coefficients()))


@pytest.mark.parametrize("missing_margin", ["norm", "quantization"])
def test_certificate_check_detects_unsafe_coefficient_tightening(missing_margin):
    a, c, beta, tiny = _production_coefficients()
    if missing_margin == "norm":
        c = a
    else:
        beta = F(126)
    assert any(margin <= 0 for margin in _certificate_margin(a, c, beta, tiny))


def test_integer_products_and_scaled_reconstruction_have_range_headroom():
    largest_dot = 32*127**2
    assert largest_dot < 2**19 < 2**24 < 2**31
    # Int32->FP32 conversion is exact; arbitrary scaling may round, but
    # normal nonzero products and norm/cross terms stay far from overflow.
    assert F(1, 2**64) > ETA
    assert largest_dot*2**65 < 2**85 < 2**127


def test_simd_error_reduction_fits_the_shared_scalar_envelope():
    # Two subtraction factors, square, arbitrary31-add reduction, optional
    # Eq+Et add, and an explicit Young allowance for subtraction FTZ.
    lower = (1-U)**35*(1-F(1, 2**24))
    assert lower > F(99998, 100000)
