"""``boundary_faces``: the faces with one side, kept in ``data.sets[name]``; returns their ids."""



__all__ = ['boundary_faces']


def boundary_faces(data, name="boundary"):
    """The faces with one side, kept in ``data.sets[name]``; returns their ids."""
    data.sets[name] = data.topology.faces().boundary()
    return data.sets[name]
