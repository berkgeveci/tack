"""Constant (implicit) array types -- O(1) memory."""

import pgc


@pgc.data_oriented
class ConstantArray:
    """Single-component constant: every element returns the same value."""

    def __init__(self, value):
        self.value = value

    @pgc.func
    def get_value(self, i):
        return self.value


@pgc.data_oriented
class ConstantTupleArray3:
    """3-component constant: every element returns (v0, v1, v2)."""

    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

    @pgc.func
    def get_value(self, i, c):
        result = self.v0
        if c == 1:
            result = self.v1
        if c == 2:
            result = self.v2
        return result
