# Third-party notices

mlxlearn is distributed under the Apache License 2.0 (see [`LICENSE`](LICENSE)).

This file records notices for third-party material that mlxlearn depends on, is derived
from, or is tested against. It is a compliance artifact: the CI job `compliance` fails if
a source file carries an upstream copyright header that is not accounted for here.

> **mlxlearn is not officially associated with scikit-learn or PROBABL, nor with Apple.**

---

## 1. Copied or derived source

### MLX Steel fragment layout and vector loads

`src/mlxlearn/_kernels/e_means_bf16.py` derives SIMD matrix fragment coordinate
mapping and contiguous vector-load techniques from MLX 0.32.4,
`mlx/backend/metal/kernels/steel/gemm/mma.h` and `loader.h`. The retained upstream
header is `Copyright © 2024 Apple Inc.`. The implementation was modified on
2026-11-14 for native BF16 fragments, fused centroid selection, ragged loads,
bounded row batching, and Python dispatch. The experimental predecessor is
`benchmarks/e_means/experimental_kernels.py`.

The following immutable benchmark records embed snapshots of that experimental
source, including its retained `Copyright © 2024 Apple Inc.` notice and source
SHA-256. Those embedded source copies are covered by this same attribution and
MIT license:

- `benchmarks/e_means/results/optimization/bf16-dispatch-sweep.json`
- `benchmarks/e_means/results/optimization/bf16-tile-sweep3.json`
- `benchmarks/e_means/results/optimization/bf16-tile-sweep4.json`

The next investigation retains the same attributed source in its verified
baseline archive/extraction and generated-kernel experiment records:

- `benchmarks/e_means/reference/metal-v2-source.zip`
- `benchmarks/e_means/results/gap_investigation/source/metal_v2_e_means_bf16.py`
- `benchmarks/e_means/results/gap_investigation/roofline.json`
- `benchmarks/e_means/results/gap_investigation/schedule.json`
- `benchmarks/e_means/results/gap_investigation/tiles.json`
- `benchmarks/e_means/results/gap_investigation/lowdim.json`
- `benchmarks/e_means/results/gap_investigation/coalescing.json`

`benchmarks/e_means/experiments/gap_assignment.py` and
`benchmarks/e_means/experiments/gap_coalescing.py` generate modified variants
of the attributed Metal source. Their modification notices identify the
2026-11-14 scheduling, tiling, and partial-result layout experiments. Raw JSON
records remain immutable, including the source snapshots and hashes captured
when each experiment ran; the frozen baseline loader verifies the extracted
source against the archive before reuse.

These source files are covered by the MIT license reproduced below. Historical
attestations for version 0.1.0a2 predate this addition.

```
MIT License

Copyright © 2024 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
```

If a future change introduces derived code, it must be added to this section with:

1. the upstream project, version, and license,
2. the upstream file path,
3. the retained upstream copyright header, verbatim, in the derived file, and
4. a modification line in the derived file stating what was changed and when.

---

## 2. API compatibility target

### scikit-learn

- Homepage: <https://scikit-learn.org>
- License: BSD 3-Clause
- Copyright: The scikit-learn developers

mlxlearn implements estimators that are drop-in compatible with scikit-learn estimators.
Parameter names, attribute names, method signatures, and documented semantics deliberately
mirror scikit-learn's public API, and mlxlearn's test suite asserts numerical parity
against scikit-learn's implementations. Docstrings describe mlxlearn's own behavior and
its deviations; they are not reproductions of scikit-learn's documentation.

Interface compatibility of this kind does not by itself create a derivative work, but the
project takes the conservative position that scikit-learn is credited prominently and that
any future incorporation of scikit-learn source — which would carry BSD-3-Clause
obligations under scikit-learn's `COPYING` independently of any other question — is
recorded in section 1 above.

```
BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 3. Runtime dependencies

| Project | License | Role |
|---|---|---|
| [MLX](https://github.com/ml-explore/mlx) | MIT | Array framework and Apple-silicon compute backend |
| [NumPy](https://numpy.org) | BSD-3-Clause | Public array boundary |
| [scikit-learn](https://scikit-learn.org) | BSD-3-Clause | API target, fallback implementation, parity oracle |

### MLX

```
MIT License

Copyright (c) 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### NumPy

```
Copyright (c) 2005-2024, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## 4. Trademarks

- *scikit-learn* and the scikit-learn logo are trademarks owned by INRIA and exclusively
  licensed to PROBABL. mlxlearn does not use the marks, or abbreviations of them, in its
  project name, logo, or domain. Uses of the words *scikit-learn* and *sklearn* in this
  project are nominative: they identify the library mlxlearn interoperates with.
- *Apple*, *Apple silicon*, *Metal*, and *MLX* are used to identify Apple technologies.
  mlxlearn is not affiliated with, endorsed by, or sponsored by Apple Inc.

---

## 5. Algorithmic references

The e-means implementation follows Bessa et al., *e-means: outperforming
k-means (in one-tenth the epochs)*, arXiv:2611.09501v1 (2026), and the MIT-licensed
Faiss reference at revision `ebab8f784c1d177129602269d5ad966c525fbdcb`,
copyright Meta Platforms, Inc. and affiliates. The estimator and Metal kernels
were written for mlxlearn. Benchmarks load the upstream code from a separate
checkout which retains its license, use upstream dataset split fingerprints,
and credit the original datasets. Faiss is an optional benchmark dependency.
See [the benchmark guide](benchmarks/e_means/README.md).

The October 31, 2026 SVC optimization study inspected scikit-learn-intelex
commit `f50ab692e6296b573efaaa3d05304a8eb3fbcf8d` and oneDAL tag `2026.2.0`.
Their native working-set decomposition and persistent-model patterns informed
independently written mlxlearn implementations; no upstream source code was
incorporated and neither project is a runtime dependency. Algorithmic credit
also belongs to Wen, Shi, Li, He and Chen, *ThunderSVM: A Fast SVM Library on
GPUs and CPUs*, JMLR 19(21), 2018. See the
[source study and precise references](docs/provenance.md).

---

## 6. Review status

This file has **not** yet been reviewed by a lawyer.
