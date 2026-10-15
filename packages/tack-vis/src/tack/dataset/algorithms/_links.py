"""Averaging per-corner values over each point's cell corners, through the
point links: ``gradients(at="points")`` and ``to_points``."""

import tack


@tack.kernel
def _average_links(offsets, entries, contributions, out):
    for p in range(out.shape[0]):
        begin = offsets[p]
        end = offsets[p + 1]
        if end > begin:
            total = contributions[entries[begin]]
            for i in range(begin + 1, end):
                total += contributions[entries[i]]
            out[p] = total / (end - begin)
