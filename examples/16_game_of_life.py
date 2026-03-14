"""16 — Conway's Game of Life.

A cellular automaton on a 2D grid.  Each cell is alive (1) or dead (0).
Rules applied simultaneously to all cells:
  - A live cell with 2 or 3 neighbors survives
  - A dead cell with exactly 3 neighbors becomes alive
  - All other cells die

Uses 2D field indexing and ndrange for parallel update.

Usage:
  uv run python examples/16_game_of_life.py
"""

import numpy as np
import pgc

pgc.init(arch=pgc.cpu)

N = 64
STEPS = 100

alive = pgc.field(dtype=pgc.i32, shape=(N, N))
count = pgc.field(dtype=pgc.i32, shape=(N, N))
new_alive = pgc.field(dtype=pgc.i32, shape=(N, N))


@pgc.kernel
def count_neighbors(alive, count, n):
    """Count live neighbors for each cell (no wrapping)."""
    for i, j in pgc.ndrange(n, n):
        c = 0
        # Check all 8 neighbors with boundary checks
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di != 0:
                    ni = i + di
                    nj = j + dj
                    if ni >= 0:
                        if ni < n:
                            if nj >= 0:
                                if nj < n:
                                    c = c + alive[ni, nj]
                else:
                    if dj != 0:
                        ni = i + di
                        nj = j + dj
                        if ni >= 0:
                            if ni < n:
                                if nj >= 0:
                                    if nj < n:
                                        c = c + alive[ni, nj]
        count[i, j] = c


@pgc.kernel
def apply_rules(alive, count, new_alive, n):
    """Apply Conway's rules to produce the next generation."""
    for i, j in pgc.ndrange(n, n):
        c = count[i, j]
        a = alive[i, j]
        if a == 1:
            # Survival: 2 or 3 neighbors
            if c == 2:
                new_alive[i, j] = 1
            else:
                if c == 3:
                    new_alive[i, j] = 1
                else:
                    new_alive[i, j] = 0
        else:
            # Birth: exactly 3 neighbors
            if c == 3:
                new_alive[i, j] = 1
            else:
                new_alive[i, j] = 0


@pgc.kernel
def copy_grid(src, dst, n):
    for i, j in pgc.ndrange(n, n):
        dst[i, j] = src[i, j]


# Initialize with random state
np.random.seed(42)
init_state = (np.random.rand(N, N) > 0.7).astype(np.int32)
alive.from_numpy(init_state)

initial_pop = init_state.sum()
print(f"Game of Life: {N}x{N} grid, {STEPS} generations")
print(f"  Initial population: {initial_pop}")

for step in range(STEPS):
    count_neighbors(alive, count, N)
    apply_rules(alive, count, new_alive, N)
    copy_grid(new_alive, alive, N)

    if (step + 1) % 20 == 0:
        pop = alive.to_numpy().reshape(N, N).sum()
        print(f"  Generation {step+1:>4d}: population = {pop}")

final = alive.to_numpy().reshape(N, N)
final_pop = final.sum()
print(f"\n  Final population: {final_pop}")

# Display the final state as ASCII art (top-left 20x40 corner)
print("\n  Final state (top-left 20x40):")
for i in range(min(20, N)):
    row = "  "
    for j in range(min(40, N)):
        row += "#" if final[N - 1 - i, j] else "."
    print(row)
