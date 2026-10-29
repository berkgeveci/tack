"""Write the test files' durations the root conftest's ``--shard`` balances by.

Reads pytest's ``--durations=0`` report on stdin and writes ``test-durations.json``
at the repository root: each test file's seconds (calls, setups and teardowns
summed, to a tenth). Refresh it when shares come out uneven::

    uv run pytest -n auto -q --durations=0 | python tools/test_durations.py
"""

import json
import re
import sys
from pathlib import Path

_LINE = re.compile(r"^(\d+(?:\.\d+)?)s (?:call|setup|teardown)\s+(\S+?)::")


def durations(lines):
    """Seconds per test file from a ``--durations`` report's lines."""
    out = {}
    for line in lines:
        match = _LINE.match(line)
        if match:
            out[match.group(2)] = out.get(match.group(2), 0.0) + float(match.group(1))
    return {path: round(seconds, 1) for path, seconds in sorted(out.items())}


def main():
    target = Path(__file__).resolve().parent.parent / "test-durations.json"
    target.write_text(json.dumps(durations(sys.stdin), indent=1) + "\n")


if __name__ == "__main__":
    main()
