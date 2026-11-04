# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Field arguments as text a registered resolver reads."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset.algorithms import _common


@tack.func
def _u(x):
    return x[0] + 2.0 * x[1] - x[2]


@pytest.fixture
def twice():
    def resolve(data, text):
        if not text.startswith("twice "):
            return None
        u = data.fields[text[len("twice "):]]
        values = tack.array(2.0 * u.values.numpy())
        return td.Field(u.space, values, u.unit)

    td.register_field_resolver(resolve)
    yield resolve
    _common._RESOLVERS.remove(resolve)


def test_algorithms_take_text_a_resolver_reads(backend, twice):
    data = td.uniform_grid((5, 4, 3), origin=(0.1, 0.2, 0.3), spacing=(0.3, 0.25, 0.4))
    data.fields["u"] = td.interpolate(data, _u, td.H1(data))
    doubled = td.Field(data.fields["u"].space, tack.array(2.0 * data.fields["u"].values.numpy()))
    a = td.contour(data, "twice u", 1.5)
    b = td.contour(data, doubled, 1.5)
    assert a.num_points == b.num_points > 0
    np.testing.assert_allclose(td.gradients(data, "twice u").values.numpy(),
                               td.gradients(data, doubled).values.numpy())
    with pytest.raises(KeyError):
        td.contour(data, "missing", 1.0)
