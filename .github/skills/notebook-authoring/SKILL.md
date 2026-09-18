---
name: notebook-authoring
description: 'Conventions for creating, editing, and reviewing Jupyter notebooks in the Rap corpus project: numbered English Markdown sections, focused code cells, and concise English comments. Use when working on any project notebook.'
---

# Rap Notebook Authoring

## When to Use
- Creating a new analysis notebook.
- Adding, editing, or reorganizing cells in an existing notebook.
- Reviewing a notebook for project conventions.

## Structure
- Begin with a `#` title, a concise objective, and a numbered outline.
- Use numbered `##` sections and, when necessary, numbered `###` subsections.
- Write reader-facing Markdown, output labels, and code comments in English.
- Precede each logical code block with Markdown that explains its analytical purpose.
- Do not place two unrelated code cells together without an intervening explanatory Markdown cell.

## Code Comment Style
- A comment states a convention, rationale, or non-obvious choice; it does not narrate the next line.
- Use at most one short comment for each notable choice. Avoid multi-paragraph docstrings for simple work.
- Example: `# Preserve lyric-line boundaries so syntax never crosses a verse.`

## Code Cell Scope
- A code cell performs one coherent operation: loading, annotation, aggregation, one figure, or one export.
- Keep long-running computations observable with `tqdm`, explicit batches, throughput, ETA, and persisted resumable outputs.
- Prefer several small documented cells to a monolithic cell that mixes unrelated analyses.

## Validation Before Delivery
- Each code cell has an appropriate Markdown explanation before it.
- Heading numbering is continuous and matches the outline.
- Run a focused cell or equivalent command after every substantive edit.
- Validate the full execution path before considering a long-running notebook complete.

## Exports
See `export-organization` for file locations, scoped naming, and Git-size rules.
