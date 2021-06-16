# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Create a Markdown cell
#
# `# %% [markdown]`
#
# `# Add some text that will be rendered as markdown text.`

# %% [markdown]
# ## Create a code cell
#
# `# %%`

# %%
import torch
print(torch.__version__)

# %% [markdown]
# ## Add any Python codes
# Easy integration with Python ecosystem libraries component.
#
# For example create a simple plot with `matplotlib`.
#
# From: https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html

# %%
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
# %%
