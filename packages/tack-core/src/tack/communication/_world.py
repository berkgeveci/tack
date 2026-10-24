"""The transport between blocks: what moves messages and gathers values across ranks."""

import math

import numpy as np

import tack
from tack.lang.array import _empty_for_mpi

# Every payload of an exchange travels under one tag on the world's own
# communicator: MPI keeps the order of messages between two ranks on one tag,
# and the receiver posts its receives in the order the header lists them.
_TAG = 7


class World:
    """The processes blocks live in, and how messages move between them.

    ``World()`` is this process alone (rank 0 of 1): its blocks may be on any
    of its devices, and a message reaches its destination as copies of its
    arrays on the destination block's device -- the arrays themselves when
    that device can already address them.

    ``World(comm)`` spans the ranks of ``comm``, an mpi4py communicator (or
    anything with its ``Dup``, ``Get_rank``, ``Get_size``, ``allgather``,
    ``alltoall``, ``Isend`` and ``Irecv``). Tack communicates on a duplicate,
    so its messages never meet the application's; make one ``World`` for a
    run. Messages between ranks are staged through host memory, or with
    ``gpu_aware=True`` passed to MPI as the arrays themselves (through DLPack),
    for an MPI that reads and writes device memory. Where a backend says the
    MPI may not take an array's memory as it is (HIP's stream-ordered pool,
    which MVAPICH-Plus's UCX takes for host memory), the array goes as a device
    copy in memory the MPI does take, and arrives in such memory. Tack imports
    no MPI module: the caller makes the communicator.
    """

    def __init__(self, comm=None, *, gpu_aware=False):
        self.gpu_aware = bool(gpu_aware)
        if comm is None:
            if gpu_aware:
                raise ValueError("gpu_aware applies to messages between ranks; "
                                 "World() is one process")
            self._comm = None
            self.rank, self.size = 0, 1
        else:
            self._comm = comm.Dup()
            self.rank, self.size = self._comm.Get_rank(), self._comm.Get_size()

    def __repr__(self):
        kind = "" if self._comm is None else ", MPI" + (", GPU-aware" if self.gpu_aware else "")
        return f"World(rank={self.rank}, size={self.size}{kind})"

    def _allgather(self, value):
        """Every rank's ``value``, by rank."""
        return [value] if self._comm is None else self._comm.allgather(value)

    def _exchange(self, messages, rank_of, device_of):
        """Deliver ``messages``, ``(source, destination, {name: Array})`` triples from
        this rank's blocks, to the ranks holding their destinations (``rank_of(gid)``),
        and return those addressed to this rank's blocks, each array on its
        destination's device (``device_of(gid)``, a `tack.Device`)."""
        delivered, remote = [], {}
        for source, destination, message in messages:
            rank = rank_of(destination)
            if rank == self.rank:
                delivered.append((source, destination, {
                    name: values.to_device(device_of(destination))
                    for name, values in message.items()}))
            else:
                remote.setdefault(rank, []).append((source, destination, message))
        if self._comm is None:
            return delivered

        # What each rank sends every other: per message its blocks and, per
        # array, its name, element type, shape and NumPy shape.
        comm = self._comm
        headers = comm.alltoall([
            [(source, destination, [(name, values.dtype, values.shape, values._host_shape)
                                    for name, values in message.items()])
             for source, destination, message in remote.get(rank, ())]
            for rank in range(self.size)])
        requests, held = [], []
        for rank, sent in remote.items():
            arrays = [values for _, _, message in sent for values in message.values()]
            if self.gpu_aware:
                for values in arrays:
                    direct = _direct(values)
                    held.append(direct)
                    requests.append(comm.Isend(direct, dest=rank, tag=_TAG))
            else:
                staged = _staged(arrays)
                held.append(staged)
                requests.append(comm.Isend(staged, dest=rank, tag=_TAG))
        arriving = []
        for rank, listed in enumerate(headers):
            if rank == self.rank or not listed:
                continue
            if self.gpu_aware:
                for source, destination, entries in listed:
                    message = {}
                    for name, dtype, shape, _ in entries:
                        values = _empty_for_mpi(shape, dtype, device_of(destination))
                        requests.append(comm.Irecv(values, source=rank, tag=_TAG))
                        message[name] = values
                    delivered.append((source, destination, message))
            else:
                total = sum(_nbytes(dtype, host_shape) for _, _, entries in listed
                            for _, dtype, _, host_shape in entries)
                staged = np.empty(total, np.uint8)
                requests.append(comm.Irecv(staged, source=rank, tag=_TAG))
                arriving.append((listed, staged))
        for request in requests:
            request.Wait()
        for listed, staged in arriving:
            offset = 0
            for source, destination, entries in listed:
                message = {}
                for name, dtype, _, host_shape in entries:
                    size = _nbytes(dtype, host_shape)
                    host = staged[offset:offset + size].view(_scalar(dtype).numpy_dtype)
                    message[name] = tack.array(host.reshape(host_shape), dtype,
                                               device=device_of(destination))
                    offset += size
                delivered.append((source, destination, message))
        return delivered


def _direct(values):
    """``values`` as memory a GPU-aware MPI reads directly: itself, or a device copy
    in ``Backend.allocate_mpi_array``'s memory (HIP's pool is not such memory)."""
    if values._backend.mpi_direct(values._buffer):
        return values
    copy = _empty_for_mpi(values.shape, values.dtype, values.device)
    tack.copyto(copy, values)
    return copy


def _scalar(dtype):
    """The scalar type of an element type, or the type itself."""
    return getattr(dtype, "scalar_dtype", dtype)


def _nbytes(dtype, host_shape):
    return math.prod(host_shape) * _scalar(dtype).numpy_dtype.itemsize


def _staged(arrays):
    """The bytes of ``arrays``, one after another, in one host buffer."""
    if not arrays:
        return np.empty(0, np.uint8)
    return np.concatenate([np.ascontiguousarray(values.numpy()).reshape(-1).view(np.uint8)
                           for values in arrays])
