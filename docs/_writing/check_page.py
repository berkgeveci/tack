"""Run the examples of the pages given: every Python block marked with
``<!-- tack-check: name -->`` on the line before its fence, each in a fresh
process, on the CPU; check every snippet a page includes exists; and flag the
set opening formulas the style guide's first rule retired.

    uv run python docs/_writing/check_page.py docs/data/index.md docs/data/fields.md
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Openings written to a template (the style guide, rule 1).
FORMULAS = re.compile(r"This page is for|After it,? you can|After this page,? you can")


def check(page):
    text = page.read_text(encoding="utf-8")
    failures = 0
    for match in FORMULAS.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        print(f"{page}:{line}: a set opening formula ({match.group(0)!r}); see the style guide, rule 1")
        failures += 1
    for path, section in re.findall(r'--8<-- "([^":]+)(?::([^"]+))?"', text):
        file = ROOT / path
        if not file.exists():
            print(f"{page}: snippet file missing: {path}")
            failures += 1
        elif section and f"[start:{section}]" not in file.read_text(encoding="utf-8"):
            print(f"{page}: snippet section missing: {path}:{section}")
            failures += 1
    blocks = re.findall(r"<!-- tack-check: ([a-z0-9-]+) -->\s*```python\n(.*?)```", text, flags=re.S)
    with tempfile.TemporaryDirectory(prefix="tack-page-check-") as temporary:
        for name, code in blocks:
            script = Path(temporary) / f"{name}.py"
            script.write_text(code, encoding="utf-8")
            result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                                    timeout=120, cwd=temporary)
            status = "ok" if result.returncode == 0 else "FAILED"
            print(f"{page}: {name}: {status}")
            if result.returncode:
                print(result.stderr[-3000:])
                failures += 1
    return failures, len(blocks)


if __name__ == "__main__":
    total = ran = 0
    for arg in sys.argv[1:]:
        failed, count = check(Path(arg))
        total += failed
        ran += count
    print(f"{ran} examples run, {total} failures")
    sys.exit(1 if total else 0)
