"""Tack's disk cache of compiled kernels.

A kernel's first call compiles it, and most of that time is Tack's own
front end. The cache keeps lowered and compiled kernels on disk so later
processes start warm. It is on by default, in a per-user directory.

Settings, read when the first kernel compiles:

- ``TACK_CACHE_DIR``: where entries are kept. Defaults to
  ``~/Library/Caches/tack`` (macOS), ``$XDG_CACHE_HOME/tack`` or
  ``~/.cache/tack`` (Linux), ``%LOCALAPPDATA%\\tack\\Cache`` (Windows). On a
  supercomputer, a directory of your own on a scratch file system.
- ``TACK_CACHE``: ``write`` (look up and store; the default), ``read`` (look
  up only), ``bundle`` (only the directory's bundle file; the default under
  MPI) or ``off``.

``configure(directory=..., mode=...)`` sets either from Python instead.

For MPI jobs: run once with ``TACK_CACHE=write`` (one rank, or a small job)
to fill the directory, run ``python -m tack.cache pack`` to gather it into
one bundle file, then run the job. Each rank reads the bundle once instead
of looking up a file per kernel, and nothing is written during the run.
``load_bundle(comm=comm)`` has rank 0 read it and broadcast it instead.

Command line: ``python -m tack.cache info | pack | clear [--all]``.
"""

from tack.runtime.disk_cache import (
    MODES,
    clear,
    configure,
    default_directory,
    describe,
    load_bundle,
    pack,
    reset_stats,
    settings,
    stats,
)

__all__ = ["MODES", "clear", "configure", "default_directory", "describe", "load_bundle",
           "pack", "reset_stats", "settings", "stats"]


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m tack.cache",
                                     description="Tack's disk cache of compiled kernels.")
    parser.add_argument("--dir", help="the cache directory (default: TACK_CACHE_DIR or the "
                                      "platform's cache directory)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="show the settings and what the directory holds")
    commands.add_parser("pack", help="gather this Tack version's entries into one bundle "
                                     "file, for MPI jobs")
    clear_parser = commands.add_parser("clear", help="delete this Tack version's entries")
    clear_parser.add_argument("--all", action="store_true",
                              help="delete the entries of every Tack version")
    args = parser.parse_args(argv)

    if args.command == "info":
        print(json.dumps(describe(args.dir), indent=2))
    elif args.command == "pack":
        path, count = pack(args.dir)
        print(f"{count} entries in {path}")
    elif args.command == "clear":
        print(f"removed {clear(args.dir, all_versions=args.all)} files")


if __name__ == "__main__":
    main()
