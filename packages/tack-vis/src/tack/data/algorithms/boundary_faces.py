"""``boundary_faces``: the faces with one side, kept in ``data.sets[name]``; returns their ids."""



__all__ = ['boundary_faces']


def boundary_faces(data):
    """The ids of the faces with one side: the boundary, a field of the topology's
    ``id_dtype``. Put it in ``data.sets`` to keep it under a name."""
    return data.topology.faces().boundary()
