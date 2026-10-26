"""A Griffe extension that puts tack-vis's subpackages (``tack.dataset``,
``tack.interop``) under tack-core's ``tack``, as ``pkgutil.extend_path``
does at run time.

Griffe merges namespace portions only when no portion has an ``__init__``
that runs (or one spelled ``__path__ = __import__("pkgutil").extend_path(...)``,
which it then treats as empty, losing ``tack.empty`` and the rest). So it loads
tack-core's ``tack`` as a regular package and stops, and ``::: tack.dataset...``
finds nothing. This loads every subpackage of the other ``tack`` directories on
the search paths into it. ``_load_module`` is Griffe's own (private) loader of
a module under a parent; ``mkdocs build --strict`` fails if it ever goes.
"""

from __future__ import annotations

from typing import Any

import griffe


class TackNamespace(griffe.Extension):
    def on_package(self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any) -> None:
        if pkg.name != "tack":
            return
        own = pkg.filepath.parent.resolve()
        for search_path in loader.finder.search_paths:
            portion = (search_path / "tack").resolve()
            if portion == own or not portion.is_dir():
                continue
            for directory in sorted(portion.iterdir()):
                init = directory / "__init__.py"
                if init.exists() and directory.name not in pkg.members:
                    module = loader._load_module(directory.name, init, parent=pkg)
                    pkg.set_member(directory.name, module)
