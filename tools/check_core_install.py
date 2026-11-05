# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Check an installed tack-core with no tack-vis, as PyPI users get it:
``python tools/check_core_install.py`` from outside the source tree.

tack is the installed package (not a source tree on the path), a kernel runs on
the CPU, and ``tack.dataset`` says that tack-vis is not installed and how to
install it from the repository."""

import sys

import numpy as np

import tack

assert "site-packages" in tack.__file__, f"tack comes from {tack.__file__}, not an install"
tack.set_default_device("cpu")


@tack.kernel
def double(x, out):
    for i in range(x.shape[0]):
        out[i] = 2.0 * x[i]


x = tack.arange(8, dtype=tack.f32)
out = tack.empty_like(x)
double(x, out)
assert np.array_equal(out.numpy(), 2 * np.arange(8, dtype=np.float32)), out.numpy()

try:
    tack.dataset  # noqa: B018  (the attribute is the check)
except AttributeError as error:
    message = str(error)
    assert "tack-vis" in message and "git+https://github.com/Kitware/tack" in message, message
else:
    sys.exit("tack.dataset imported: tack-vis is installed, so this checks nothing")

print(f"OK: tack-core {tack.__version__} alone runs a kernel on the CPU, and "
      "tack.dataset names tack-vis's install")
