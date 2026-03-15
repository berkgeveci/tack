"""12 — Mandelbrot set fractal.

Classic GPU compute example: compute the Mandelbrot set on a 2D grid.
Each pixel is independent, making this embarrassingly parallel.

Optionally saves the result as a PNG image (requires matplotlib).

Usage:
  uv run python examples/12_mandelbrot.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

WIDTH = 800
HEIGHT = 600
MAX_ITER = 200


@pgc.kernel
def mandelbrot(pixels, max_iter, width, height):
    """Compute escape-time for each pixel in the Mandelbrot set."""
    for idx in range(width * height):
        px = idx % width
        py = idx // width

        # Map pixel to complex plane: [-2.5, 1.0] x [-1.2, 1.2]
        x0 = -2.5 + float(px) * 3.5 / float(width)
        y0 = -1.2 + float(py) * 2.4 / float(height)

        x = 0.0
        y = 0.0
        iteration = 0

        while iteration < max_iter:
            x2 = x * x
            y2 = y * y
            if x2 + y2 > 4.0:
                break
            y = 2.0 * x * y + y0
            x = x2 - y2 + x0
            iteration = iteration + 1

        # Smooth coloring: use fractional escape count
        if iteration < max_iter:
            log_zn = log(x * x + y * y) / 2.0
            nu = log(log_zn / log(2.0)) / log(2.0)
            pixels[idx] = float(iteration) + 1.0 - nu
        else:
            pixels[idx] = float(max_iter)


pixels = pgc.field(dtype=pgc.f32, shape=(WIDTH * HEIGHT,))
mandelbrot(pixels, MAX_ITER, WIDTH, HEIGHT)

data = pixels.to_numpy().reshape(HEIGHT, WIDTH)

# Statistics
in_set = np.sum(data >= MAX_ITER)
escaped = np.sum(data < MAX_ITER)
print(f"Mandelbrot {WIDTH}x{HEIGHT}, max_iter={MAX_ITER}")
print(f"  In set:  {in_set:,} pixels")
print(f"  Escaped: {escaped:,} pixels")

# Try to save image
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(data, cmap="inferno", extent=[-2.5, 1.0, -1.2, 1.2], aspect="equal")
    ax.set_title("Mandelbrot Set (PGC)")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    plt.savefig("mandelbrot.png", dpi=150, bbox_inches="tight")
    print("  Saved: mandelbrot.png")
except ImportError:
    print("  (install matplotlib to save image)")
