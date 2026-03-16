"""05 — Multi-dimensional parallel iteration with pgc.ndrange.

Instead of computing linear indices manually, use pgc.ndrange(w, h) to
iterate over a 2D grid in parallel.  Each (i, j) pair maps to a separate
GPU thread.

Usage:
  uv run python examples/05_ndrange.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

width, height = 64, 64
pixels = pgc.field(dtype=pgc.f32, shape=(width, height))


@pgc.kernel
def distance_field(pixels):
    """Compute distance from center for each pixel."""
    for i, j in pgc.ndrange(64, 64):
        dx = float(i) - 32.0
        dy = float(j) - 32.0
        pixels[i, j] = sqrt(dx * dx + dy * dy)


distance_field(pixels)

result = pixels.to_numpy().reshape(64, 64)
# Check center is ~0, corners are ~45
assert result[32, 32] == 0.0
corner_dist = result[0, 0]
assert abs(corner_dist - np.sqrt(32**2 + 32**2)) < 0.01
print(f"Center distance: {result[32, 32]:.2f} (expected 0)")
print(f"Corner distance: {corner_dist:.2f} (expected {np.sqrt(32**2+32**2):.2f})")

# --- Example 2: 2D Gaussian blur (3x3 kernel) ---

src = pgc.field(dtype=pgc.f32, shape=(64, 64))
dst = pgc.field(dtype=pgc.f32, shape=(64, 64))

# Create a simple image: bright spot at center
img = np.zeros((64, 64), dtype=np.float32)
img[32, 32] = 100.0
src.from_numpy(img)


@pgc.kernel
def blur_3x3(src, dst):
    """Simple 3x3 box blur, skipping borders."""
    for i, j in pgc.ndrange(64, 64):
        if i > 0:
            if i < 63:
                if j > 0:
                    if j < 63:
                        s = 0.0
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                s = s + src[i + di, j + dj]
                        dst[i, j] = s / 9.0


blur_3x3(src, dst)
blurred = dst.to_numpy().reshape(64, 64)
print(f"\nBlur: center pixel {blurred[32,32]:.2f} (was 100.00)")
print(f"Blur: neighbor    {blurred[32,33]:.2f} (was 0.00)")
print("2D ndrange + stencil: OK")
