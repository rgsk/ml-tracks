# jupytext pairing — one notebook, two files

Demo of the workflow proposed for walkthroughs: keep a clean, git-friendly
`.py` as the source of truth, paired with an `.ipynb` that carries the same
cells **plus rendered outputs** (prints, plots) for reading.

## The two files here

| file | role | commit? |
|------|------|---------|
| `jupytext_demo.py` | source of truth — clean `py:percent` text, readable git diffs | **yes** |
| `jupytext_demo.ipynb` | paired view with inline outputs, for reading on GitHub / in the editor | optional (see below) |

The pairing is declared once, in the `.py` header:

```
# jupytext:
#   formats: ipynb,py:percent
```

## Cell syntax (in the .py)

- `# %%` — start a **code** cell
- `# %% [markdown]` — start a **markdown** cell (prose lines are `#`-prefixed)

## Everyday commands

```bash
# after editing EITHER file, propagate the change to its pair:
uv run jupytext --sync example/jupytext_demo.py

# (re)generate the notebook AND run every cell so outputs are baked in:
uv run jupytext --to ipynb --execute example/jupytext_demo.py

# one-time: pair an existing plain notebook to a .py
uv run jupytext --set-formats ipynb,py:percent some_notebook.ipynb
```

You can also just edit the `.ipynb` in Jupyter/VS Code as normal — on save (with
the jupytext extension) or on the next `--sync`, the `.py` updates to match.

## Which file does git track?

Two sane policies:

1. **Commit both** — the `.py` gives clean diffs in review; the `.ipynb` lets the
   figures render on GitHub without a kernel. Cost: notebook JSON still bloats
   history. Good for walkthroughs meant to be *read*.
2. **Commit only the `.py`** (add `*.ipynb` to `.gitignore`) — pristine history;
   regenerate the notebook on demand with `--execute`. Good if nobody reads the
   rendered version straight from the repo.

For the CNN-style walkthroughs (meant to be read top-to-bottom with figures),
policy **1** matches the intent.
