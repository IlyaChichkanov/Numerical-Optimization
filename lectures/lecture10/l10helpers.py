"""Вспомогательный код демонстраций лекции 10.

Задача «парк»: HOUSE_OUT = (2, 1), HOUSE_IN = (0.5, 0.3); park_solution(a) -> (x*, mu*); park_tau(a, tau) -> (x, mu) при mu*h = tau.
Портфель: PF_R, PF_SIGMA; portfolio_active(rho, A) -> (x, lam1, lam_rho, mu) при x_i = 0 для i из A;
        portfolio_bounded(rho) -> (A, x, lam1, lam_rho, mu) с x >= 0 перебором активных множеств.
Рисовалка: draw_park(ax). Нужен CasADi (--group casadi).

Файл генерируется скриптом scripts/build_lecture10.py из scripts/build_demo10.py — не правьте руками.
"""

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

from itertools import combinations

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import brentq

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


# ---------------------------------------------------------------- методы (те же, что в demo10 / l10helpers.py)
# задача «парк»
HOUSE_OUT = np.array([2.0, 1.0])       # дом снаружи парка
HOUSE_IN = np.array([0.5, 0.3])        # дом внутри парка


def park_solution(a):
    """min 1/2 |x - a|^2 при h(x) = 1 - |x|^2 >= 0. Возвращает (x*, mu*).
    Внутри (|a| <= 1): x* = a, mu = 0. Снаружи: x* = a/|a|, mu = (|a| - 1)/2 — из x - a = -2 mu x."""
    a = np.asarray(a, float)
    na = np.linalg.norm(a)
    if na <= 1:
        return a.copy(), 0.0
    return a / na, (na - 1) / 2


def park_tau(a, tau):
    """Точка, где вместо mu*h = 0 стоит mu*h = tau: x = a/(1 + 2 mu), mu*(1 - |x|^2) = tau."""
    a = np.asarray(a, float)
    na = np.linalg.norm(a)
    phi = lambda mu: mu * (1 - na**2 / (1 + 2 * mu) ** 2) - tau
    mu = brentq(phi, max((na - 1) / 2, 0) + 1e-12, 1e3)
    return a / (1 + 2 * mu), mu


# портфель (данные лекций 3 и 8)
PF_R = np.array([0.04, 0.08, 0.11, 0.15])
PF_VOL = np.array([0.05, 0.12, 0.18, 0.30])
PF_CORR = np.array([[1, 0.2, 0.1, 0.0], [0.2, 1, 0.5, 0.3], [0.1, 0.5, 1, 0.4], [0.0, 0.3, 0.4, 1.0]])
PF_SIGMA = np.outer(PF_VOL, PF_VOL) * PF_CORR


def portfolio_active(rho, active):
    """Портфель с равенствами 1^T x = 1, r^T x = rho и границами x_i = 0 для i из active (активное множество).
    Система ККТ: Sigma x - lam1 1 - lam_rho r - sum mu_i e_i = 0 и ограничения. Возвращает (x, lam1, lam_rho, mu)."""
    active = list(active)
    k = len(active)
    one = np.ones(4)
    E = np.eye(4)[active].reshape(k, 4)
    K = np.block([[PF_SIGMA, -one[:, None], -PF_R[:, None], -E.T],
                  [one[None, :], np.zeros((1, 2 + k))],
                  [PF_R[None, :], np.zeros((1, 2 + k))],
                  [E, np.zeros((k, 2 + k))]])
    rhs = np.concatenate([np.zeros(4), [1.0, rho], np.zeros(k)])
    z = np.linalg.solve(K, rhs)
    return z[:4], z[4], z[5], z[6:]


def portfolio_bounded(rho):
    """Портфель с x >= 0 перебором активных множеств: годится то, где x >= 0 и mu >= 0 (задача выпуклая, ответ один)."""
    found = []
    for k in range(3):                       # 2 равенства + не больше 2 границ: иначе ограничений больше, чем переменных
        for A in combinations(range(4), k):
            x, l1, lr, mu = portfolio_active(rho, A)
            if x.min() >= -1e-12 and (mu.min() if k else 0) >= -1e-12:
                found.append((A, x, l1, lr, mu))
    assert len(found) == 1, (rho, [f[0] for f in found])
    return found[0]


def draw_park(ax):
    """Парк — круг радиуса 1 с центром в начале координат."""
    ax.add_patch(Circle((0, 0), 1, facecolor=AQUA, alpha=0.18, edgecolor=AQUA, lw=2))
    ax.set_aspect("equal")
