"""19 — Image processing kernels.

Demonstrates common image processing operations as PGC kernels:
  1. Brightness/contrast adjustment
  2. Sobel edge detection
  3. Gaussian blur (separable)

Operates on synthetic images stored as 2D float fields.

Usage:
  uv run python examples/19_image_processing.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

W, H = 256, 256


# --- Create a synthetic test image ---
def make_test_image():
    """Concentric rings + gradient."""
    img = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            dx = j - W / 2
            dy = i - H / 2
            r = np.sqrt(dx**2 + dy**2)
            img[i, j] = 0.5 + 0.5 * np.sin(r * 0.3) * np.exp(-r * 0.005)
    return img


src = pgc.field(dtype=pgc.f32, shape=(H, W))
dst = pgc.field(dtype=pgc.f32, shape=(H, W))

test_img = make_test_image()
src.from_numpy(test_img)


# --- 1. Brightness and contrast ---

@pgc.kernel
def adjust_brightness_contrast(src, dst, brightness, contrast, w, h):
    """Apply brightness and contrast: out = contrast * (in - 0.5) + 0.5 + brightness"""
    for i, j in pgc.ndrange(h, w):
        v = contrast * (src[i, j] - 0.5) + 0.5 + brightness
        v = max(0.0, min(1.0, v))
        dst[i, j] = v


adjust_brightness_contrast(src, dst, 0.1, 1.5, W, H)
print("1. Brightness/contrast: OK")
print(f"   Input  range: [{test_img.min():.3f}, {test_img.max():.3f}]")
result = dst.to_numpy().reshape(H, W)
print(f"   Output range: [{result.min():.3f}, {result.max():.3f}]")


# --- 2. Sobel edge detection ---

@pgc.kernel
def sobel(src, dst, w, h):
    """Sobel edge detection (gradient magnitude)."""
    for i, j in pgc.ndrange(h, w):
        if i > 0:
            if i < h - 1:
                if j > 0:
                    if j < w - 1:
                        # Horizontal gradient (Gx)
                        gx = (
                            -1.0 * src[i - 1, j - 1] + 1.0 * src[i - 1, j + 1] +
                            -2.0 * src[i, j - 1]     + 2.0 * src[i, j + 1] +
                            -1.0 * src[i + 1, j - 1] + 1.0 * src[i + 1, j + 1]
                        )
                        # Vertical gradient (Gy)
                        gy = (
                            -1.0 * src[i - 1, j - 1] - 2.0 * src[i - 1, j] - 1.0 * src[i - 1, j + 1] +
                             1.0 * src[i + 1, j - 1] + 2.0 * src[i + 1, j] + 1.0 * src[i + 1, j + 1]
                        )
                        dst[i, j] = sqrt(gx * gx + gy * gy)


src.from_numpy(test_img)
dst.fill(0.0)
sobel(src, dst, W, H)
edges = dst.to_numpy().reshape(H, W)
print(f"\n2. Sobel edge detection: OK")
print(f"   Edge strength: max={edges.max():.3f}, mean={edges.mean():.3f}")


# --- 3. Gaussian blur (box blur approximation) ---

tmp = pgc.field(dtype=pgc.f32, shape=(H, W))


@pgc.kernel
def blur_horizontal(src, dst, w, h):
    """Horizontal pass of a 5-tap box filter."""
    for i, j in pgc.ndrange(h, w):
        if j >= 2:
            if j < w - 2:
                dst[i, j] = (
                    src[i, j - 2] + src[i, j - 1] + src[i, j] +
                    src[i, j + 1] + src[i, j + 2]
                ) / 5.0
            else:
                dst[i, j] = src[i, j]
        else:
            dst[i, j] = src[i, j]


@pgc.kernel
def blur_vertical(src, dst, w, h):
    """Vertical pass of a 5-tap box filter."""
    for i, j in pgc.ndrange(h, w):
        if i >= 2:
            if i < h - 2:
                dst[i, j] = (
                    src[i - 2, j] + src[i - 1, j] + src[i, j] +
                    src[i + 1, j] + src[i + 2, j]
                ) / 5.0
            else:
                dst[i, j] = src[i, j]
        else:
            dst[i, j] = src[i, j]


src.from_numpy(test_img)
blur_horizontal(src, tmp, W, H)
blur_vertical(tmp, dst, W, H)
blurred = dst.to_numpy().reshape(H, W)
print(f"\n3. Separable blur (5-tap): OK")
print(f"   Input  variance: {test_img.var():.6f}")
print(f"   Output variance: {blurred.var():.6f} (should be smaller)")

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(test_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(result, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Brightness/Contrast")
    axes[1, 0].imshow(edges, cmap="gray")
    axes[1, 0].set_title("Sobel Edges")
    axes[1, 1].imshow(blurred, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Gaussian Blur")
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle("PGC Image Processing")
    plt.savefig("image_processing.png", dpi=150, bbox_inches="tight")
    print("  Saved: image_processing.png")
except ImportError:
    print("  (install matplotlib to save image)")
