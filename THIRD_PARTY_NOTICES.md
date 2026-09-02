# Third-party notices

mlxlearn is distributed under the Apache License 2.0 (see [`LICENSE`](LICENSE)).

This file records notices for third-party material that mlxlearn depends on, is derived
from, or is tested against. It is a compliance artifact: the CI job `compliance` fails if
a source file carries an upstream copyright header that is not accounted for here.

> **mlxlearn is not officially associated with scikit-learn or PROBABL, nor with Apple.**

---

## 1. Copied or derived source

**None as of 0.1.0a2.**

No source file in `src/mlxlearn/` was copied from another project. Every module was newly
written for this repository; see [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) and
[`phase0/attestation.md`](phase0/attestation.md).

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

## 5. Review status

This file has **not** yet been reviewed by a lawyer.
