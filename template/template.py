# %% [markdown]
# ## Create a Markdown cell
#
# NOTE: that the fist line shall be empty, so we recognise where to enter generated header
#
# `# %% [markdown]`
#
#  the content of single cell shall be connected with `# ` at each line, so for example:
#  `# Add some text that will be rendered as markdown text.`

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
# For example create a simple plot with `matplotlib` with an image:
#
# ![test image](test.png)
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

ax.set(xlabel='time (s)', ylabel='voltage (mV)', title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
