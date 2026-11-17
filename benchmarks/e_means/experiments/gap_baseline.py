"""Load the immutable pre-investigation BF16 kernel from its source archive.

The extracted copy is verified against the frozen archive on every import.
Experiments use this module rather than whichever production version is current.
"""

import hashlib
import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "benchmarks/e_means/reference/metal-v2-source.zip"
SOURCE_NAME = "src/mlxlearn/_kernels/e_means_bf16.py"
with zipfile.ZipFile(ARCHIVE) as archive:
    SOURCE_BYTES = archive.read(SOURCE_NAME)
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()
EXPECTED_SOURCE_SHA256 = "cbe903b6972ab2fcca06b940005a5d3ae4e700f9595460036f14880625f3e4ea"
if SOURCE_SHA256 != EXPECTED_SOURCE_SHA256:
    raise RuntimeError("BF16 baseline archive is not the recorded pre-investigation version")
EXTRACTED = ROOT / "benchmarks/e_means/results/gap_investigation/source/metal_v2_e_means_bf16.py"
EXTRACTED.parent.mkdir(parents=True, exist_ok=True)
if not EXTRACTED.exists():
    EXTRACTED.write_bytes(SOURCE_BYTES)
if hashlib.sha256(EXTRACTED.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise RuntimeError("Frozen BF16 baseline copy differs from source archive")
spec = importlib.util.spec_from_file_location("emeans_gap_frozen_bf16", EXTRACTED)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)
