"""Example: inspecting generated kernel code.

tack.inspect() shows the generated code for a kernel without executing it.
Useful for debugging, learning, and understanding what the compiler produces.

Usage:
    uv run python packages/tack-core/examples/41_inspect.py --arch cpu
    uv run python packages/tack-core/examples/41_inspect.py --arch metal
"""

import argparse
import tack

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
args = parser.parse_args()
tack.init(arch=args.arch)

# A simple kernel to inspect
@tack.kernel
def saxpy(x, y, out, a):
    for i in range(len(x)):
        out[i] = a * x[i] + y[i]

n = 1024
x = tack.field(dtype=tack.f32, shape=(n,))
y = tack.field(dtype=tack.f32, shape=(n,))
out = tack.field(dtype=tack.f32, shape=(n,))

# Print the Tack IR
print("=== Tack IR ===")
print(tack.inspect(saxpy, x, y, out, 2.0, mode="ir"))

# Print the backend-specific source code
print(f"\n=== {args.arch.upper()} source ===")
print(tack.inspect(saxpy, x, y, out, 2.0, mode="source"))


# Data-oriented template example
@tack.data_oriented
class Particles:
    GRAVITY = -9.81  # class-level constant (baked into code)

    def __init__(self, n):
        self.pos = tack.field(dtype=tack.f32, shape=(n,))
        self.vel = tack.field(dtype=tack.f32, shape=(n,))
        self.dt = 0.01  # instance scalar (runtime parameter)

    @tack.func
    def step(self, i):
        self.vel[i] = self.vel[i] + self.GRAVITY * self.dt
        self.pos[i] = self.pos[i] + self.vel[i] * self.dt


@tack.kernel
def update(p):
    for i in range(len(p.pos)):
        p.step(i)


particles = Particles(n)

print("\n=== Template IR ===")
print(tack.inspect(update, particles, mode="ir"))

print(f"\n=== Template {args.arch.upper()} source ===")
print(tack.inspect(update, particles, mode="source"))
