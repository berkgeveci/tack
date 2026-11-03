### Changed

- The documentation site is rewritten and reorganized by Tack's four layers,
  built bottom-up: Core (GPU computing in Python), Communication (distributed
  parallelism), Data (the data model) and Visualization (analysis and
  algorithms), after a front page (What Tack is, use cases, install). Each
  layer opens with a start page and builds without depending on the layers
  above. The reference is gathered by layer with a new glossary that gives VTK,
  ParaView and Viskores counterparts, and the design notes form an addendum.
  Every example on the new pages runs in the docs check. The writing kit
  (style guide, term list, page checker) is in `docs/_writing/`.
