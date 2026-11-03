# Brief for an agent drafting pages of Tack's documentation

You are drafting some pages of Tack's new documentation site, a first draft
for Berk Geveci (Tack's author, a visualization expert) to review, then
Jaswant, Jeff Lee and Utkarsh. Other agents are drafting the other pages at
the same time.

## Read first

1. `docs/_writing/style-guide.md`: the rules every page follows. Follow them.
2. `docs/_writing/terms.md`: one word for one thing, with VTK equivalents.
3. `/Users/berk.geveci/Work/Obsidian/Work/Tack/Docs site - requirements.md`:
   why the site is being rewritten, the readers, the four layers.
4. `/Users/berk.geveci/Work/Obsidian/Work/Tack/Docs site - skeleton.md`: every
   page's kind, summary and **From** (the current pages its content comes
   from). Your pages' entries are your briefs.

## Where you work

- The repository is `/Users/berk.geveci/Work/tack/tack-docs` (branch
  `docs/site-draft`). Run Python with `uv run --no-sync python` there.
- Write **only** your pages (listed in your task) and, if you need them, new
  example files `docs/examples/<your prefix>_*.py`. Do not edit `mkdocs.yml`,
  other pages, existing example files, source code or tests. Do not commit,
  stash, reset or check out anything.
- The current pages (`docs/users-guide/`, `docs/vis-guide/`, `docs/index.md`,
  ...) are your sources: keep every correct fact that belongs on your page,
  rewritten to the style guide. They stay on disk until the review.
- The code is the truth: check what you write against the docstrings and the
  code under `packages/` (and `CLAUDE.md`, the architecture notes). If a source
  page says something the code contradicts, follow the code and report it.

## The site's pages (link to these, relative to your page)

Front: `index.md` (What Tack is), `use-cases.md`, `install.md`.
Core: `core/index.md`, `how-programs-run.md`, `arrays-and-types.md`,
`kernels.md`, `control-flow-and-math.md`, `device-functions.md`,
`vectors-and-matrices.md`, `device-classes.md`, `reductions-scans-sorts.md`,
`memory-and-correctness.md`, `choose-a-device.md`, `share-arrays.md`,
`workgroups-atomics-textures.md`, `asyncio.md`, `debug-and-time.md`,
`kernel-cache.md`, `examples.md` (all in `core/`); the tutorials stay at
`users-guide/tutorials/{heat,dye,physarum,contours,mpm,nbody}.md`.
Communication: `communication/index.md`, `blocks-and-worlds.md`,
`exchange-and-plans.md`, `routing.md`, `sort-and-reduce.md`, `mpi.md`.
Data: `data/index.md`, `topology-and-geometry.md`, `fields.md`,
`units-coordinates-time.md`, `cells-and-meshes.md`, `polyhedra.md`,
`higher-order.md`, `particles.md`, `amr-and-octrees.md`, `pieces.md`,
`storage-and-layouts.md`, `build-meshes-and-fields.md`,
`connect-simulation-data.md`, `devices.md`.
Visualization: `visualization/index.md`, `extract.md`, `derive.md`,
`sample-and-trace.md`, `reduce.md`, `points-and-particles.md`,
`adaptive-data.md`, `finite-volume.md`, `over-pieces.md`, `over-time.md`,
`views.md`, `write-a-mesh-algorithm.md`, `write-a-filter.md`, `vtk.md`,
`mfem-warp-and-others.md`, `examples.md`; galleries stay at
`vis-guide/gallery.md` and `vis-guide/mfem-gallery.md`.
Reference (stays for now): `contracts/runtime-api.md`, `reference/api.md`,
`design/array-api.md`, `reference/language-contract.md`,
`reference/backends.md`, `contracts/backend-capabilities.md`,
`vis-guide/reference/dataset-api.md`, `vis-guide/reference/data-model.md`,
`vis-guide/16-cells-and-bases.md`, `vis-guide/reference/units.md`,
`vis-guide/reference/algorithms.md` (algorithms, with mkdocstrings anchors such
as `#tack.dataset.algorithms.contour.contour`), `vis-guide/reference/interop.md`,
`reference/glossary.md`, `performance.md`.
Design notes (addendum, link as "why" only): `design/*.md`, `contracts/index.md`,
`vis-guide/design/*.md`.

## The layers' order

A page uses only its own layer and the layers below (Core < Communication <
Data < Visualization). It may mention a layer above as "see also".
Terminology weans readers gently: the first Data pages pair familiar terms
with Tack's ("point data (an `H1` field of order 1)"); later Data pages put
Tack's first ("an `H1` field (point data)"); Visualization pages use Tack's
terms, linking the glossary on first use in the page.

## Examples

Every code block runs. Reuse an existing example through a snippet
(`--8<-- "docs/examples/file.py:section"`), or write a self-contained block
marked `<!-- tack-check: name -->` on the line before its fence (it imports
`tack`, calls `tack.set_default_device("cpu")`, and asserts something). Check
your pages with

    uv run --no-sync python docs/_writing/check_page.py docs/<your pages>

until it reports no failures. Keep each example under ten seconds on the CPU.
Run any new example file directly too.

## When you finish

Report, briefly: each page with its word count; what you left out of the
sources and why (internals, history, things for the reference or the design
notes); facts in the old pages that the code contradicts; open questions for
Berk. Do not paste the pages into your report.
