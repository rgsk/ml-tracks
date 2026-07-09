# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: pytorch-practice (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Jupytext demo — one notebook, two files
#
# This `.py` is the **source of truth** that git tracks (clean, readable diffs).
# It is *paired* with `jupytext_demo.ipynb`, which holds the same cells **plus
# rendered outputs** (plots, prints) for reading in the browser.
#
# - `# %% [markdown]` starts a markdown cell.
# - `# %%` starts a code cell.
# - Edit *either* file; `jupytext --sync` propagates the change to the other.

# %%
print('hello world changed again')

# %%
import numpy as np
import torch

x = torch.arange(12)
x

# %% [markdown]
# The line below is exactly the kind of thing a walkthrough wants: run code, see
# the shape/stride *right underneath it*, no separate figures pipeline.

# %%
y = x.view(3, 4)
print(f"{y.shape=}, {y.stride()=}")
y

# %% [markdown]
# ## A figure appears inline
#
# In the `.py` this is just code. In the paired `.ipynb`, the plot is rendered
# and committed as output — so it's readable on GitHub without running a kernel.

# %%
import matplotlib.pyplot as plt

t = np.linspace(0, 2 * np.pi, 200)
plt.figure(figsize=(4, 2.5))
plt.plot(t, np.sin(t), label="sin")
plt.plot(t, np.cos(t), label="cos")
plt.legend()
plt.title("inline output, no PNG pipeline")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Edited in the .py, synced to the .ipynb
#
# This cell was added by editing `jupytext_demo.py` and running
# `jupytext --sync jupytext_demo.py`. The change flows into the notebook.

# %% [markdown]
# ## LaTeX in a markdown cell
#
# Inline math renders like $y = Wx + b$ inside a sentence.
#
# Display math sits on its own line:
#
# $$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
#
# And a multi-line aligned block:
#
# $$
# \begin{aligned}
# \text{conv out size} &= \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1 \\
# \text{params}        &= k^2 \cdot C_\text{in} \cdot C_\text{out}
# \end{aligned}
# $$
