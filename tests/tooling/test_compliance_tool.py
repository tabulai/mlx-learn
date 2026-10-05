"""tools/compliance.py: output contract, license gate, layering gate."""

from __future__ import annotations

import ast
import json
import textwrap
from email.message import Message

import pytest

from ._load import REPO, load_tool

compliance = load_tool("compliance")


# ------------------------------------------------------------------------------ CLI


def test_json_mode_prints_one_parseable_document(capsys):
    status = compliance.main(["--json", "layout", "layering"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)  # "Extra data" here was the bug
    assert set(payload) == {"layout", "layering"}
    assert status == 0 and payload["layout"]["ok"]
    assert "compliance passed" in captured.err


def test_a_bare_run_selects_every_check(capsys):
    # argparse's nargs="*" plus choices rejects an empty run on Python <= 3.11.
    compliance.main(["--json"])
    assert set(json.loads(capsys.readouterr().out)) == set(compliance.CHECKS)


def test_an_unknown_check_is_a_usage_error_naming_the_choices(capsys):
    with pytest.raises(SystemExit) as info:
        compliance.main(["bogus"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "unknown check(s): bogus" in err and "layering" in err and "[]" not in err


def test_every_branding_exemption_names_a_file_that_exists():
    stale = sorted(p for p in compliance.BRANDING_EXEMPT if not (REPO / p).exists())
    assert not stale, f"stale BRANDING_EXEMPT entries: {stale}"


# ------------------------------------------------------------------------- licenses


@pytest.mark.parametrize(
    ("expression", "allowed"),
    [
        ("MIT", True),
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", True),  # numpy
        ("GPL-3.0-only OR MIT", True),  # the licensee may choose MIT
        ("(MIT OR GPL-3.0-only) AND Apache-2.0", True),
        ("MIT AND GPL-3.0-only", False),  # both apply
        ("GPL-3.0-only OR MIT-incompatible", False),  # substring "mit" used to pass
        ("Apache-2.0 WITH LLVM-exception", True),
        ("GPL-2.0-only WITH Classpath-exception-2.0", False),
        ("mit", True),  # SPDX ids match case-insensitively
        ("MIT OR GPL-3.0-only AND LGPL-2.1-only", True),  # AND binds tighter
        ("GPL-3.0-only AND MIT OR LGPL-2.1-only", False),
    ],
)
def test_spdx_expressions_are_parsed_not_substring_matched(expression, allowed):
    assert compliance.spdx_allows(expression) is allowed


@pytest.mark.parametrize(
    "expression", ["", "MIT AND", "(MIT", "MIT)", "MIT OR OR BSD-3-Clause", "Proprietary; unlicensed"]
)
def test_a_malformed_expression_is_an_error_not_a_pass(expression):
    with pytest.raises(compliance.LicenseExpressionError):
        compliance.spdx_allows(expression)


def _meta(expression=None, license=None, classifiers=()):
    meta = Message()
    if expression is not None:
        meta["License-Expression"] = expression
    if license is not None:
        meta["License"] = license
    for classifier in classifiers:
        meta["Classifier"] = classifier
    return meta


@pytest.mark.parametrize(
    ("meta", "allowed"),
    [
        (_meta(expression="BSD-3-Clause"), True),
        (_meta(expression="GPL-3.0-only OR MIT-incompatible"), False),
        (_meta(license="Proprietary; unlicensed"), False),  # used to pass on "unlicense"
        (_meta(license="MIT"), True),  # mlx
        (_meta(license="BSD 3-Clause", classifiers=["License :: OSI Approved :: BSD License"]), True),
        # scipy: a 47 KB license text that names bundled GPL runtime libraries. The
        # classifier decides; the prose is never scanned.
        (
            _meta(
                license="Copyright (c) SciPy Developers.\n...GPL-3.0-or-later WITH GCC-exception...",
                classifiers=["License :: OSI Approved :: BSD License"],
            ),
            True,
        ),
        (_meta(classifiers=["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"]), False),
        (_meta(license="A very long free-text license\nspanning lines"), False),
        (_meta(), False),
    ],
)
def test_license_verdict_reads_the_most_precise_source(meta, allowed):
    acceptable, _source = compliance.license_verdict("somepkg", meta)
    assert acceptable is allowed


def test_the_declared_dependencies_come_from_pyproject():
    declared = compliance.declared_runtime_dependencies()
    names = {spec.split(">")[0].split("<")[0].split("=")[0].strip() for spec in declared}
    assert names == {"numpy", "scikit-learn", "mlx"}


def test_the_license_gate_walks_the_transitive_closure():
    pytest.importorskip("packaging")
    closure, missing = compliance.runtime_closure(["scikit-learn"])
    assert not missing
    # scikit-learn's own dependencies, none of which the old gate looked at.
    assert {"scikit-learn", "scipy", "joblib", "threadpoolctl"} <= set(closure)
    # Extras are not followed unless requested: none of sklearn's docs extras.
    assert "sphinx" not in closure


# ------------------------------------------------------------------------- layering


def _violations(source: str, *, core: bool = True) -> list[str]:
    tree = ast.parse(textwrap.dedent(source))
    return [what for _, what in compliance.layering_violations(tree, numerical_core=core)]


@pytest.mark.parametrize(
    "source",
    [
        "import sklearn",
        "from sklearn.utils import check_array",  # public, and still a layering bug
        "from sklearn import svm",
        "from .._common.config import get_config",
        "from mlxlearn._common.config import config_context",
        "from .._common import config\nconfig.get_config()",
        "from ..neighbors import KNeighborsClassifier",
        "from .. import svm",
        "import mlxlearn.patching",
    ],
)
def test_the_numerical_core_may_not_reach_upward(source):
    assert _violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "from .._common.config import get_tuning",
        "from .._common.device import use_device",
        "from ..exceptions import ConvergenceError",
        "import numpy as np\nimport mlx.core as mx",
        "import mlx.core as mx\nkey = mx.random.key(0)",
    ],
)
def test_what_the_numerical_core_may_import(source):
    assert _violations(source) == []


def test_estimators_may_import_sklearn_and_read_the_configuration():
    assert _violations("import sklearn\nfrom .._common.config import get_config", core=False) == []


@pytest.mark.parametrize(
    "source",
    [
        "import mlx.core as mx\nmx.random.seed(0)",
        "import mlx.core\nmlx.core.random.seed(0)",
        "import numpy as np\nnp.random.seed(0)",
        "from mlx.core.random import seed",
        "import random\nrandom.seed(1)",
    ],
)
def test_global_rng_seeding_is_banned_everywhere(source):
    assert _violations(source, core=False)


def test_the_layering_gate_passes_on_this_tree():
    result = compliance.check_layering(compliance.tracked_files())
    assert result.ok, result.report()
    assert result.checked > 0
