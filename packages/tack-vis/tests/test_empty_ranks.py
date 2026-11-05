# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Every algorithm over pieces with MPI ranks that hold none of them.

A rank with no pieces is common: four files read on eight ranks, a reader
that gives rank 0 nothing, a hierarchy whose fine boxes are on a few ranks.
Each algorithm that communicates must take part from such a rank, deciding
nothing from its own pieces alone that the others decide differently, or the
others wait for it forever. ``_empty_ranks_mpi.py`` runs each on 3 ranks with
four pieces (boxes, trees) all on rank 0, on ranks 1 and 2, on ranks 0 and 1,
and on none, and holds the answers to the same pieces in one process.
"""

import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).with_name("_empty_ranks_mpi.py")


@pytest.mark.parametrize("layout", ["one", "first-empty", "last-empty", "none"])
def test_algorithms_over_pieces_take_ranks_without_pieces(mpiexec, layout):
    run = mpiexec(3, SCRIPT, layout)
    assert run.returncode == 0 and "ok" in run.stdout.split(), run.stdout + run.stderr[-3000:]
