from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np

"""Fig. 2.4-1 Exact and approximate solutions at x = 0.

"""

N = 20
x = (np.arange(N + 1) / N) ** 1.2
t = np.linspace(0, 0.01)

fig, ax = subplots()
ax.plot(t, np.sqrt(4 * t / np.pi), marker="None", linestyle="dashed")
ax.set_xlim(0, 0.012)
ax.set_ylim(0, 0.12)
ax.set_aspect(0.1)
fig.savefig(Path(__file__).with_suffix(".png"))
