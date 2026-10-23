"""45 -- Flow: gradients, vorticity, particles and streamlines.

A velocity field on a uniform grid -- a rigid rotation about the z axis with an
axial drift, v = (-y, x, w) -- whose answers are known exactly, so every
result is checked:

- gradients and flow_quantities: the divergence is 0, the vorticity (0, 0, 2)
  and the Q-criterion 1 in every cell;
- advect: seeds on a ring stay on it (the field is linear, so interpolating
  it in the cells is exact) and rise by w per unit of time;
- streamlines: each seed's path, a helix, as a polyline;
- perot: the velocity rebuilt on the cells from its components normal to
  the faces, as C-grid ocean and atmosphere models store it.

Usage:
  uv run python packages/tack-vis/examples/45_flow.py
  uv run python packages/tack-vis/examples/45_flow.py --arch cuda --size 64 --seeds 1000
  uv run python packages/tack-vis/examples/45_flow.py --output results/flow
"""

import argparse
import os

import numpy as np

import tack
import tack.dataset as td

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--size', type=int, default=33, help='points per axis (default 33)')
_parser.add_argument('--seeds', type=int, default=64, help='seeds on the ring (default 64)')
_parser.add_argument('--output', help='directory to write VTK files into (needs VTK)')
_args = _parser.parse_args()
tack.set_default_device(_args.arch)

W = 0.2                                     # the axial drift


# ================================================================
# THE GRID AND THE VELOCITY
# ================================================================

n = _args.size
data = td.uniform_grid((n, n, n), origin=(-1.0, -1.0, -1.0), spacing=(2.0 / (n - 1),) * 3)


@tack.kernel
def swirl(positions, velocity, w: tack.f32):
    for p in range(velocity.shape[0]):
        x = positions.get(p)                 # the grid computes point p's position
        velocity[p] = tack.vec3(-x[1], x[0], w)


velocity = tack.empty((data.num_points,), tack.vec3())
swirl(data.geometry.values, velocity, W)
data.fields["velocity"] = td.Field(td.H1(data), velocity)
print(f"{n}^3 points, {data.num_cells} cells; v = (-y, x, {W})")


# ================================================================
# GRADIENTS, DIVERGENCE, VORTICITY, Q
# ================================================================

gradient = td.gradients(data, "velocity")    # 9 values per cell, d(v_i)/d(x_j) at 3i + j
q = td.flow_quantities(data, gradient)
divergence = q.divergence.values.numpy()
vorticity = q.vorticity.values.numpy()
q_criterion = q.q_criterion.values.numpy()
print(f"divergence in [{divergence.min():.2e}, {divergence.max():.2e}], vorticity "
      f"{vorticity.mean(axis=0).round(4)}, Q-criterion {q_criterion.mean():.4f}")
assert np.allclose(divergence, 0.0, atol=1e-4)
assert np.allclose(vorticity, [0.0, 0.0, 2.0], atol=1e-4)
assert np.allclose(q_criterion, 1.0, atol=1e-4)  # 1/2 (|rotation|^2 - |strain|^2)


# ================================================================
# PARTICLES ON A RING
# ================================================================

radius, z0 = 0.5, -0.8
angles = np.linspace(0, 2 * np.pi, _args.seeds, endpoint=False)
seeds = np.stack([radius * np.cos(angles), radius * np.sin(angles),
                  np.full_like(angles, z0)], axis=1)
step, steps = 0.01, 300
moved = td.advect(data, "velocity", seeds, step, steps)
where = moved.positions()
time = moved.fields["time"].values.numpy()
status = moved.fields["status"].values.numpy()
r = np.hypot(where[:, 0], where[:, 1])
print(f"{_args.seeds} particles after {steps} steps of {step}: radius "
      f"{r.min():.6f} to {r.max():.6f} (started at {radius}), z rose by "
      f"{(where[:, 2] - z0).mean():.4f} in time {time.mean():.2f}")
assert np.all(status & td.ParticleStatus.SUCCESS)
assert np.all(status & td.ParticleStatus.TOOK_ANY_STEPS)
assert np.allclose(r, radius, atol=1e-4)
assert np.allclose(where[:, 2] - z0, W * time, atol=1e-4)

# Their paths: a polyline per seed, the seed first.
paths = td.streamlines(data, "velocity", seeds, step, steps)
on_path = paths.positions()
print(f"streamlines: {paths.num_cells} polylines through {paths.num_points} points, "
      f"each a helix at radius {radius}")
assert paths.num_cells == _args.seeds
assert np.allclose(np.hypot(on_path[:, 0], on_path[:, 1]), radius, atol=1e-4)


# ================================================================
# FROM FACE COMPONENTS BACK TO CELLS (PEROT)
# ================================================================

normals, _ = td.face_geometry(data)
centers = td.face_centers(data).values.numpy()
at_faces = np.stack([-centers[:, 1], centers[:, 0], np.full(len(centers), W)], axis=1)
normal_component = (normals.values.numpy() * at_faces).sum(axis=1).astype(np.float32)
flux = td.Field(td.Values(data, "faces"), tack.array(normal_component))
rebuilt = td.perot(data, flux).values.numpy()
cell_centers = td.cell_centers(data).values.numpy()
expected = np.stack([-cell_centers[:, 1], cell_centers[:, 0],
                     np.full(len(cell_centers), W)], axis=1)
print(f"perot: the cell velocities rebuilt from {len(centers)} face components, "
      f"largest error {np.abs(rebuilt - expected).max():.2e}")
assert np.allclose(rebuilt, expected, atol=1e-4)


# ================================================================
# TO VTK (with --output)
# ================================================================

if _args.output:
    from vtkmodules.vtkIOXML import vtkXMLImageDataWriter, vtkXMLUnstructuredGridWriter

    from tack.interop.vtk import dataset_to_vtk

    os.makedirs(_args.output, exist_ok=True)
    data.fields["vorticity"] = q.vorticity
    for stem, piece, kind in (("grid", data, vtkXMLImageDataWriter),
                              ("streamlines", paths, vtkXMLUnstructuredGridWriter)):
        writer = kind()
        writer.file_name = os.path.join(_args.output, f"{stem}.{writer.default_file_extension}")
        writer(dataset_to_vtk(piece))               # a writer called on its input writes it
        print(f"wrote {writer.file_name}")
