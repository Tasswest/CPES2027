---
name: export-organization
description: 'Conventions for generated artifacts in the Rap corpus project: notebooks at the repository root, figures in images/, and structured outputs in export/. Use when saving a figure, exporting a CSV/table/array, adding an output, or setting up a notebook.'
---

# Rap Project Export Organization

## When to use
- A notebook produces a figure that must be saved to disk.
- A notebook exports a table, report, manifest, array, or annotation file.
- A new notebook or script needs output folders.

## Organization Rules
- **Scripts and notebooks** (`.py`, `.ipynb`) remain at the repository root.
- **Images** (figures, diagrams, and saved views) go in `images/`.
- **Structured outputs** (CSV, JSON, text reports, Parquet, and NumPy arrays) go in `export/`.
- Create these folders in the configuration cell with `Path('images').mkdir(exist_ok=True)` and `Path('export').mkdir(exist_ok=True)`.
- Large raw corpora, full token-level CoNLL-U exports, and resumable shard directories remain local. Keep small summaries, manifests, and reports versionable.

## Convention de code
Définir les chemins une seule fois en haut du notebook/script, puis les réutiliser :

```python
from pathlib import Path

images_dir = Path("images")
export_dir = Path("export")
images_dir.mkdir(exist_ok=True)
export_dir.mkdir(exist_ok=True)
```

Puis pour chaque figure :

```python
fig.savefig(images_dir / "0X_descriptive_name.png", bbox_inches="tight")
plt.show()
```

Et pour chaque export tabulaire :

```python
output_path = export_dir / "descriptive_name.csv"
summary_df.to_csv(output_path, index=False)
```

## Naming Rules
- Use a numeric section prefix and a descriptive ASCII snake_case name, for example `02_pos_word_clouds_rank_full_10to4000w.png`.
- Include a run label when results can differ by sample, filter, model, or scope.
- Call `plt.show()` after `fig.savefig(...)` to retain inline output.

## Validation
- Generated `.png`, `.csv`, `.json`, `.txt`, and `.npy` files are not written at the repository root.
- `images/` and `export/` are created by the notebook rather than assumed to exist.
- Large local artifacts are covered by `.gitignore` before staging changes.
