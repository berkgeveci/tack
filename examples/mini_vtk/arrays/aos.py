"""Array-of-Structures array types."""

import numpy as np
import pgc


@pgc.data_oriented
class AOSArray:
    """Single-component array backed by a pgc.field."""

    def __init__(self, data):
        self.data = data

    @pgc.func
    def get_value(self, i):
        return self.data[i]

    @pgc.func
    def set_value(self, i, val):
        self.data[i] = val


@pgc.data_oriented
class AOSTupleArray:
    """Multi-component AOS: [x0,y0,z0, x1,y1,z1, ...].

    Memory layout: num_tuples * num_components elements in a single field.
    """

    def __init__(self, data, num_tuples, num_components):
        self.data = data
        self.num_tuples = num_tuples
        self.num_components = num_components

    @pgc.func
    def get_value(self, i, c):
        return self.data[i * self.num_components + c]

    @pgc.func
    def set_value(self, i, c, val):
        self.data[i * self.num_components + c] = val


def make_aos_array(np_array):
    """Create an AOSArray from a 1D numpy array."""
    f = pgc.field(dtype=pgc.f32, shape=np_array.shape)
    f.from_numpy(np_array.astype(np.float32))
    return AOSArray(f)


def make_aos_tuple_array(np_array_2d):
    """Create an AOSTupleArray from a (N, C) numpy array."""
    n, c = np_array_2d.shape
    flat = np_array_2d.astype(np.float32).ravel()
    f = pgc.field(dtype=pgc.f32, shape=(len(flat),))
    f.from_numpy(flat)
    return AOSTupleArray(f, n, c)
