"""Example: inspecting generated kernel code.

pgc.inspect() shows the generated code for a kernel without executing it.
Useful for debugging, learning, and understanding what the compiler produces.

Usage:
    uv run python packages/pgc-core/examples/41_inspect.py --arch cpu
    uv run python packages/pgc-core/examples/41_inspect.py --arch metal
"""

import argparse
import pgc

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
args = parser.parse_args()
pgc.init(arch=args.arch)

# A simple kernel to inspect
@pgc.kernel
def saxpy(x, y, out, a):
    for i in range(len(x)):
        out[i] = a * x[i] + y[i]

n = 1024
x = pgc.field(dtype=pgc.f32, shape=(n,))
y = pgc.field(dtype=pgc.f32, shape=(n,))
out = pgc.field(dtype=pgc.f32, shape=(n,))

# Print the PGC IR
print("=== PGC IR ===")
print(pgc.inspect(saxpy, x, y, out, 2.0, mode="ir"))

# Print the backend-specific source code
print(f"\n=== {args.arch.upper()} source ===")
print(pgc.inspect(saxpy, x, y, out, 2.0, mode="source"))


# Data-oriented template example
@pgc.data_oriented
class Particles:
    GRAVITY = -9.81  # class-level constant (baked into code)

    def __init__(self, n):
        self.pos = pgc.field(dtype=pgc.f32, shape=(n,))
        self.vel = pgc.field(dtype=pgc.f32, shape=(n,))
        self.dt = 0.01  # instance scalar (runtime parameter)

    @pgc.func
    def step(self, i):
        self.vel[i] = self.vel[i] + self.GRAVITY * self.dt
        self.pos[i] = self.pos[i] + self.vel[i] * self.dt


@pgc.kernel
def update(p):
    for i in range(len(p.pos)):
        p.step(i)


particles = Particles(n)

print("\n=== Template IR ===")
print(pgc.inspect(update, particles, mode="ir"))

print(f"\n=== Template {args.arch.upper()} source ===")
print(pgc.inspect(update, particles, mode="source"))
