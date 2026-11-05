# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VTK's test data, for the examples that read real files (50-53).

The files are VTK's own (its ``Testing/Data``), named in ``vtk_data.sha512`` by
their SHA-512, as VTK's ExternalData names them. An example asks for a file or a
directory with ``path("OpenFOAM/cavity/cavity.foam")``; the files come from

- ``TACK_VTK_DATA``, a VTK build's ``ExternalData`` directory, where it has them;
- else ``TACK_VTK_DATA_CACHE`` (by default ``~/.cache/tack/vtk-data``), where they
  are fetched once from the servers VTK's ExternalData uses and checked against
  their hashes.

Without the files and without a network, the example stops, saying it needs
VTK's test data.
"""

import hashlib
import os
import pathlib
import urllib.request

MANIFEST = pathlib.Path(__file__).with_name("vtk_data.sha512")
SOURCES = ("https://data.kitware.com/api/v1/file/hashsum/sha512/{digest}/download",
           "https://www.vtk.org/files/ExternalData/SHA512/{digest}")


def _manifest():
    """``{path under Testing/Data: sha512}``."""
    with open(MANIFEST) as f:
        return {name: digest for digest, name in (line.split() for line in f
                                                  if line.strip())}


def _roots():
    """Where to look for the files, in order: VTK's build, then the cache."""
    roots = []
    if os.environ.get("TACK_VTK_DATA"):
        roots.append(pathlib.Path(os.environ["TACK_VTK_DATA"]) / "Testing" / "Data")
    cache = os.environ.get("TACK_VTK_DATA_CACHE",
                           os.path.join("~", ".cache", "tack", "vtk-data"))
    roots.append(pathlib.Path(cache).expanduser())
    return roots


def _sha512(path):
    digest = hashlib.sha512()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(name, digest, target):
    partial = target.with_name(target.name + ".part")
    problem = None
    for source in SOURCES:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(source.format(digest=digest), partial)
        except OSError as error:
            problem = error
            continue
        if _sha512(partial) == digest:
            os.replace(partial, target)
            return
        problem = "the file fetched has another hash"
    partial.unlink(missing_ok=True)
    raise SystemExit(f"needs VTK's test data: could not fetch {name} ({problem}); "
                     "set TACK_VTK_DATA to a VTK build's ExternalData directory")


def path(name):
    """The local path of ``name``, a file or a directory under VTK's ``Testing/Data``,
    with every file of the manifest in it there."""
    files = _manifest()
    wanted = {n: d for n, d in files.items()
              if n == name or n.startswith(name.rstrip("/") + "/")}
    if not wanted:
        raise KeyError(f"{name} is not in {MANIFEST.name}")
    roots = _roots()
    for root in roots:
        if all((root / n).is_file() for n in wanted):
            return str(root / name)
    cache = roots[-1]
    for n, digest in wanted.items():
        if not (cache / n).is_file():
            _fetch(n, digest, cache / n)
    return str(cache / name)
