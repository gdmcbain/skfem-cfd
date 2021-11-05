from pathlib import Path
from typing import Iterator, Tuple

from matplotlib.pyplot import subplots
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu

import skfem
from skfem.models.poisson import mass, laplace, unit_load

"""Fig. 2.4-1 Exact and approximate solutions at x = 0.

"""

N = 20
x = (np.arange(N + 1) / N) ** 1.2
t = np.linspace(0, 0.01)

mesh = skfem.MeshLine(x).with_boundaries(
    {"origin": lambda x: x[0] == 0.0, "far": lambda s: s[0] == x.max()}
)
basis = skfem.CellBasis(mesh, skfem.ElementLineP1())

origin_basis = skfem.FacetBasis(mesh, basis.elem, facets=mesh.boundaries["origin"])
heating = unit_load.assemble(origin_basis)

M = mass.assemble(basis)
L = laplace.assemble(basis)
D = basis.get_dofs(mesh.boundaries["far"])

dt = t.max() / 2 ** 6
theta = 0.5  # Crank-Nicolson
L0, M0 = skfem.penalize(L, M, D=D)
A = M0 + theta * L0 * dt
B = M0 - (1 - theta) * L0 * dt
backsolve = splu(A.T).solve


def evolve(time: float, u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
    while time < t.max():
        yield time, u
        time, u = time + dt, backsolve(heating * dt + B @ u)


trajectory = [(time, temperature[0]) for time, temperature in evolve(0, basis.zeros())]

def exact(t: float) -> float:
    return np.sqrt(4 * t / np.pi)

    
fig, ax = subplots()
ax.set_title("Fig. 2.4-1 Exact and approximate solutions at x = 0.")
ax.plot(t, exact(t), marker="None", linestyle="dashed", label="exact")
t0 = 0.004
ax.text(t0, exact(t0), "$T(0,t)=\sqrt{4t/\pi}$", horizontalalignment="right")

ax.plot(*np.array(trajectory).T, label="skfem", color="k")
t1 = 0.002
ax.text(t1, exact(t1) - 5e-3, "$T_1(t)$ for $N = 21$ and $x_i = \{i/N\}^{1/2}$")
ax.set_xlim(0, 0.012)
ax.set_ylim(0, 0.12)
ax.set_aspect(0.1)
ax.set_xlabel("t")
fig.savefig(Path(__file__).with_suffix(".png"))
