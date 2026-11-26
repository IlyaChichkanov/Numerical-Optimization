"""Вспомогательный код демонстраций лекции 10.

Задача «такси»: TAXI_T, TAXI_Y, TAXI_A; taxi_f(x), taxi_grad(x); taxi_guess(A) -> (x, mu) при ограничениях из A как равенствах;
        taxi_tau(tau) -> (x, mu) при mu_i*h_i = tau. Ближайшая точка круга (упражнение 10.1): park_solution(a) -> (x*, mu*).
Портфель: PF_R, PF_SIGMA; portfolio_active(rho, A) -> (x, lam1, lam_rho, mu) при x_i = 0 для i из A;
        portfolio_bounded(rho) -> (A, x, lam1, lam_rho, mu) с x >= 0 перебором активных множеств.
Рисовалка: draw_quadrant(ax). Нужен CasADi (--group casadi).

Файл генерируется скриптом scripts/build_lecture10.py из scripts/build_demo10.py — не правьте руками.
"""

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

from itertools import combinations

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


# ---------------------------------------------------------------- методы (те же, что в demo10 / l10helpers.py)
# задача «такси»: цена поездки y ~ a t + b, a >= 0, b >= 0 (t — км, y — сотни рублей)
TAXI_T = np.array([1.0, 2.0, 3.0, 4.0])
TAXI_Y = np.array([0.6, 1.7, 3.0, 4.1])
TAXI_A = np.column_stack([TAXI_T, np.ones(4)])          # матрица МНК: строка (t_i, 1)


def taxi_f(x):
    """Половина суммы квадратов отклонений модели a t + b от данных; x = (a, b)."""
    return 0.5 * np.sum((TAXI_A @ x - TAXI_Y) ** 2)


def taxi_grad(x):
    """grad f = A^T A x - A^T y = (30a + 10b - 29.4, 10a + 4b - 9.4)."""
    return TAXI_A.T @ (TAXI_A @ x - TAXI_Y)


def taxi_guess(active):
    """Догадка: ограничения из active (0 — a >= 0, 1 — b >= 0) — равенства, остальные выброшены.
    Возвращает (x, mu): x решает систему, mu = grad f(x) (градиенты ограничений — e_1 и e_2)."""
    G, c = TAXI_A.T @ TAXI_A, TAXI_A.T @ TAXI_Y
    free = [i for i in range(2) if i not in active]
    x = np.zeros(2)
    if free:
        x[free] = np.linalg.solve(G[np.ix_(free, free)], c[free])
    return x, taxi_grad(x)


def taxi_tau(tau):
    """Точка, где вместо mu_i h_i = 0 стоит mu_i h_i = tau: grad f = (mu_a, mu_b), mu_a a = mu_b b = tau, a, b > 0.
    Из второй строки b(a) — положительный корень 4b^2 + (10a - 9.4) b - tau = 0; a — корень первой строки."""
    def b_of(a):
        q = 10 * a - 9.4
        return (-q + np.sqrt(q * q + 16 * tau)) / 8
    a = brentq(lambda a: 30 * a + 10 * b_of(a) - 29.4 - tau / a, 1e-9, 5)
    x = np.array([a, b_of(a)])
    return x, tau / x


# ближайшая точка круга — упражнение 10.1
def park_solution(a):
    """min 1/2 |x - a|^2 при h(x) = 1 - |x|^2 >= 0. Возвращает (x*, mu*).
    Внутри (|a| <= 1): x* = a, mu = 0. Снаружи: x* = a/|a|, mu = (|a| - 1)/2 — из x - a = -2 mu x."""
    a = np.asarray(a, float)
    na = np.linalg.norm(a)
    if na <= 1:
        return a.copy(), 0.0
    return a / na, (na - 1) / 2


# портфель: четыре актива, учебные данные
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


def draw_quadrant(ax, a_max=1.6, b_max=2.6):
    """Допустимая область задачи «такси»: четверть a >= 0, b >= 0."""
    ax.fill_between([0, a_max], 0, b_max, color=AQUA, alpha=0.15)
    ax.axhline(0, color=AQUA, lw=2)
    ax.axvline(0, color=AQUA, lw=2)
