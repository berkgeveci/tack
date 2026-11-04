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


def test_reductions_take_text_a_resolver_reads(backend, twice):
    data = td.uniform_grid((5, 4, 3), origin=(0.1, 0.2, 0.3), spacing=(0.3, 0.25, 0.4))
    data.fields["u"] = td.interpolate(data, _u, td.H1(data))
    data.fields["2u"] = td.Field(data.fields["u"].space, tack.array(2.0 * data.fields["u"].values.numpy()))
    by_text = td.statistics(data, "twice u")
    stored = td.statistics(data, "2u")
    assert (by_text.n, by_text.min, by_text.max) == (stored.n, stored.min, stored.max)
    np.testing.assert_allclose(by_text.mean, stored.mean)
    assert td.entropy(data, "twice u") == td.entropy(data, "2u")
    a = td.integrate(data, ["twice u"])
    b = td.integrate(data, {"doubled": "2u", "field": data.fields["2u"]})
    np.testing.assert_allclose(a.values["twice u"], b.values["doubled"])
    np.testing.assert_allclose(b.values["field"], b.values["doubled"])
    with pytest.raises(TypeError, match="names each Field"):
        td.integrate(data, [data.fields["2u"]])
    pieces = td.PartitionedDataSet([data])
    np.testing.assert_allclose(td.statistics(pieces, "twice u").mean, stored.mean)
    np.testing.assert_allclose(td.integrate(pieces, ["twice u"]).values["twice u"],
                               a.values["twice u"])

