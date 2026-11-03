# Tack documentation: style guide

Every page of the site is written to these rules. They come from readers'
feedback (2026-11-03): pages that "sound like unraveling a puzzle", "sentence
fragments and choppy flow", "no big picture introductions or diagrams",
terminology that loses a visualization expert, and information spread over
many pages.

## The site's shape

Tack is four layers, documented bottom-up: **Core** (GPU computing in
Python), **Communication** (distributed parallelism), **Data** (the data
model), **Visualization** (analysis and algorithms). A page uses only what its
own layer and the layers below explain; it may say "see also" about a layer
above, never depend on it. Each subject has one home page; elsewhere it gets one
sentence and a link.

## Every page

1. **Start with the big picture.** The first paragraph says what the page is
   for, who it is for, and what the reader can do after it. No page opens with
   a caveat, a history or a list of other pages.
2. **Show early.** Within the first screen, a short example the reader can run.
3. **Then explain, in order**: the idea, why it matters, how to use it, the
   details. Edge cases, precision, history and internals go last, in a
   "Details" section, or to the reference or design notes.
4. **End with where to go next**: one to three links, said why.

## Sentences

- Write full sentences in paragraphs. A sentence has one main idea.
- No chains of clauses joined by colons, semicolons and dashes. Split them.
- No parentheses inside parentheses; at most one parenthesis a sentence.
- No stacked qualifications ("only where X, except on Y, unless Z"). Say the
  rule, then the exceptions in their own sentences, or in Details.
- Active voice, present tense; "you" for instructions.
- Prefer the common word: *use*, not *take*; *makes*, not *gives* when you
  mean creates; *point*, not *vertex*, unless the distinction matters.
- Numbers only where a reader decides something with them.

Bad:

> ``field`` is a scalar field with a basis, or its name: ``H1`` of order 1
> (point data), or ``L2`` (DG, an order per cell too).

Good:

> `field` is the scalar field to contour, or its name. It is usually point
> data (an `H1` field of order 1). Discontinuous fields (`L2`) work too.

Bad:

> A dataset's positions are `origin + value` (`DataSet.origin`, three doubles
> on the host), so far coordinates keep their precision in f32.

Good:

> A dataset stores its positions relative to an origin. The origin is three
> double-precision numbers; the positions themselves are stored in the
> dataset's precision, often single. This keeps coordinates far from zero
> accurate in single precision.

## Terms

- Use the term list (`docs/_writing/terms.md`). One word for one thing.
- **Familiar terms first, then Tack's, then Tack's alone.** The first pages of
  the Data layer say "point data (an `H1` field of order 1)". Later pages of
  the Data layer say "an `H1` field (point data)". The Visualization layer
  says `H1`, linking the glossary on first use in a page.
- Define a term before using it, on every page a reader may land on from a
  search; at most about three new terms a section.
- Do not use internal names in user pages: private functions (`_name`), PR
  numbers, branch names, test names. Design notes may.

## Code

- Every example runs. Use an existing example file through a snippet
  (`--8<-- "docs/examples/file.py:section"`), or write a self-contained block
  (it imports `tack`, calls `tack.set_default_device("cpu")`, and asserts
  something) and put `<!-- tack-check: short-name -->` on the line before its
  fence, so `docs/examples/validate.py` runs it. Keep each under ten seconds on
  a laptop CPU.
- Show the output when it teaches something (as a comment or a short block).
- Name variables for what they are (`surface`, not `out2`).

## Structure and format

- Headings say what the section does ("Cut a dataset with a plane"), in
  sentence case.
- Diagrams where structure is the point, in Mermaid (` ```mermaid `).
- Tables for comparisons and lists of options; not for prose.
- Admonitions (`!!! note`) sparingly: at most two a page.
- Link to the API reference for full signatures; do not copy them.
- Length: a how-to or an explanation reads in 10 to 15 minutes (1,500 to
  3,000 words). Split a longer page.

## Honesty

Claims of speed or capability are measured and linked to where they were
measured (`performance.md`, a gallery, a dataset page). Limits are said plainly,
in one place, without apology.
