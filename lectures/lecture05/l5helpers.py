"""Вспомогательный код демонстраций лекции 5.

Методы: gd(f, grad, x0, alpha=None, tol, maxit, c) — градиентный спуск (лекция 4);
        newton(grad, hess, x0, tol, maxit) — чистый Ньютон;
        newton_damped(f, grad, hess, x0, tol, maxit, c) — Ньютон с backtracking по Армихо;
        bfgs(f, grad, x0, tol, maxit, c, Q=None) — BFGS, H_0 = I; Q — точный шаг на квадратичной;
        gauss_newton(resid, jac, x0, tol, maxit, damped) — Гаусс–Ньютон для 1/2 ||r||^2.
Задачи: rosen, rosen_grad, rosen_hess, X0_ROSEN; make_chain(N) -> (energy, grad_E, H, b, v0, unpack);
        sine_data() -> (t, y, starts); sine_problem(t, y) -> (resid, jac, f, grad, hess); X0_SINE.
Рисовальщики (принимают ax=None, возвращают ax): plot_rosen_path(paths, labels, ax, ...),
        plot_errors(errs, labels, ax, logx).

Файл генерируется скриптом build_lecture05.py из build_demo05.py — не правьте руками.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares

BLUE, ORANGE, AQUA, RED, VIOLET, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7", "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


# ---------------------------------------------------------------- методы
def gd(f, grad, x0, alpha=None, tol=1e-6, maxit=100_000, c=1e-4):
    """Градиентный спуск (лекция 4): alpha=None — backtracking по Армихо с константой c,
    иначе постоянный шаг. Возвращает (x, число итераций, путь (k+1, n))."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        if alpha is None:
            a, fx = 1.0, f(x)
            while f(x - a * g) > fx - c * a * (g @ g):
                a /= 2
        else:
            a = alpha
        x = x - a * g; path.append(x.copy())
    return x, maxit, np.array(path)


def newton(grad, hess, x0, tol=1e-10, maxit=50):
    """Чистый метод Ньютона: x <- x - H^{-1} g, без линейного поиска. Возвращает (x, итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        x = x - np.linalg.solve(hess(x), g); path.append(x.copy())
    return x, maxit, np.array(path)


def newton_damped(f, grad, hess, x0, tol=1e-10, maxit=200, c=1e-4):
    """Демпфированный Ньютон: направление Ньютона, длина шага — backtracking по Армихо.
    Если гессиан не положительно определён и направление не является спуском, матрица
    регуляризуется: H + (|lambda_min| + 1e-3) I. Возвращает (x, итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        H = hess(x)
        p = -np.linalg.solve(H, g)
        if g @ p >= 0:                                   # не спуск: сдвигаем собственные числа
            lam_min = np.linalg.eigvalsh(H)[0]
            p = -np.linalg.solve(H + (abs(lam_min) + 1e-3) * np.eye(len(x)), g)
        a, fx = 1.0, f(x)
        while f(x + a * p) > fx + c * a * (g @ p):
            a /= 2
        x = x + a * p; path.append(x.copy())
    return x, maxit, np.array(path)


def bfgs(f, grad, x0, tol=1e-6, maxit=10_000, c=1e-4, Q=None):
    """BFGS с H_0 = I: p = -H g, обновление обратного гессиана по паре (s, y).
    Длина шага: backtracking по Армихо (Q=None) или точный шаг на квадратичной
    1/2 x^T Q x + b^T x (передать Q). Возвращает (x, итераций, путь, число вычислений f)."""
    x = np.asarray(x0, float).copy(); n = len(x)
    H = np.eye(n); path = [x.copy()]; g = grad(x); nfev = 1
    for k in range(maxit):
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path), nfev
        p = -H @ g
        if Q is None:
            a, fx = 1.0, f(x)
            while f(x + a * p) > fx + c * a * (g @ p):
                a /= 2; nfev += 1
            nfev += 1
        else:
            a = -(g @ p) / (p @ Q @ p)
        s = a * p; x_new = x + s; g_new = grad(x_new); y = g_new - g
        if s @ y > 1e-12:                                # условие кривизны: иначе H не обновляем
            rho = 1.0 / (s @ y); I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, g = x_new, g_new; path.append(x.copy())
    return x, maxit, np.array(path), nfev


def gauss_newton(resid, jac, x0, tol=1e-6, maxit=100, damped=False, c=1e-4):
    """Гаусс–Ньютон для f = 1/2 ||r(x)||^2: шаг — решение линейного МНК min ||J p + r||
    (np.linalg.lstsq). damped=True — backtracking по Армихо. Возвращает (x, итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        r = resid(x); J = jac(x); g = J.T @ r
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        p = np.linalg.lstsq(J, -r, rcond=None)[0]
        a = 1.0
        if damped:
            fx = 0.5 * (r @ r)
            while 0.5 * np.sum(resid(x + a * p) ** 2) > fx + c * a * (g @ p) and a > 1e-10:
                a /= 2
        x = x + a * p; path.append(x.copy())
    return x, maxit, np.array(path)


# ---------------------------------------------------------------- задачи
def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def rosen_hess(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200.0]])


X0_ROSEN = np.array([-1.2, 1.0])


def make_chain(N, D=70.0, g=9.81):
    """Цепь без пола (лекция 1, §7.3): E(v) = 1/2 v^T H v + b^T v + const, v = (y_1..y_N, z_1..z_N).
    Возвращает (energy, grad_E, H, b, v0, unpack); unpack(v) -> (y, z) с закреплёнными концами."""
    m = 4.0 / N
    K = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    H = np.kron(np.eye(2), D * K)
    b = np.zeros(2 * N)
    b[0] -= D * (-2.0); b[N - 1] -= D * 2.0
    b[N] -= D * 1.0; b[2 * N - 1] -= D * 1.0
    b[N:] += m * g
    v0 = np.r_[np.linspace(-2, 2, N + 2)[1:-1], np.ones(N)]       # старт: прямая между концами

    def unpack(v):
        return np.r_[-2.0, v[:N], 2.0], np.r_[1.0, v[N:], 1.0]

    def energy(v):
        y, z = unpack(v)
        return 0.5 * D * np.sum(np.diff(y) ** 2 + np.diff(z) ** 2) + m * g * np.sum(z[1:-1])

    def grad_E(v):
        return H @ v + b
    return energy, grad_E, H, b, v0, unpack


def sine_data():
    """Данные задачи 2(б) ДЗ 2: y = 2 sin(3t + 0.7) + шум, 60 точек, тот же генератор и порядок вызовов.
    Возвращает (t, y, starts) — starts: 20 стартов ДЗ 2 в [-5, 5]^3."""
    data_rng = np.random.default_rng(2026)
    N, omega = 60, 3.0
    t = np.sort(data_rng.uniform(0, 2 * np.pi, N))
    y = 2.0 * np.sin(3.0 * t + 0.7) + 0.2 * data_rng.standard_normal(N)
    starts = data_rng.uniform(-5, 5, (20, 3))
    return t, y, starts


def sine_problem(t, y):
    """Модель x1 sin(x2 t + x3). Возвращает (resid, jac, f, grad, hess): невязки, якобиан невязок,
    f = 1/2 ||r||^2, градиент J^T r и полный гессиан J^T J + sum r_i hess(r_i)."""
    def resid(x):
        return x[0] * np.sin(x[1] * t + x[2]) - y

    def jac(x):
        s, c = np.sin(x[1] * t + x[2]), np.cos(x[1] * t + x[2])
        return np.c_[s, x[0] * t * c, x[0] * c]

    def f(x):
        r = resid(x); return 0.5 * (r @ r)

    def grad(x):
        return jac(x).T @ resid(x)

    def hess(x):
        J, r = jac(x), resid(x)
        s, c = np.sin(x[1] * t + x[2]), np.cos(x[1] * t + x[2])
        S = np.zeros((3, 3))                            # сумма r_i * hess(r_i) — то, что отбрасывает Гаусс–Ньютон
        S[0, 1] = S[1, 0] = r @ (t * c); S[0, 2] = S[2, 0] = r @ c
        S[1, 1] = r @ (-x[0] * t ** 2 * s); S[1, 2] = S[2, 1] = r @ (-x[0] * t * s); S[2, 2] = r @ (-x[0] * s)
        return J.T @ J + S
    return resid, jac, f, grad, hess


X0_SINE = np.array([1.0, 3.0, 0.0])


# ---------------------------------------------------------------- рисовальщики: принимают ax=None, возвращают ax
def plot_rosen_path(paths, labels=None, ax=None, xlim=(-2.2, 2.2), ylim=(-3.5, 3.2), every=1):
    """Линии уровня Розенброка и пути методов. paths — список массивов (k+1, 2); labels — подписи;
    every — прореживание длинных путей (каждая every-я точка). Возвращает ax."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    ax.contour(X, Y, (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2, levels=np.logspace(-1, 3.3, 14), colors=[GRAY], linewidths=0.7)
    colors = [ORANGE, VIOLET, AQUA, BLUE, RED]
    for i, p in enumerate(paths):
        pp = p if len(p) <= 60 else np.r_[p[:30], p[30::every]]
        lab = None if labels is None else labels[i]
        ax.plot(pp[:, 0], pp[:, 1], "-o", ms=3 if len(p) > 60 else 4.5, lw=1.3, color=colors[i % len(colors)], label=lab)
    ax.plot(1, 1, "*", ms=13, color=RED, mec="white", zorder=7)
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$"); ax.grid(False)
    if labels is not None:
        ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_errors(errs, labels=None, ax=None, logx=False):
    """График log10 ошибки по итерациям для нескольких методов. errs — список массивов ||e_k||.
    logx=True — логарифмическая ось итераций (когда методы отличаются на порядки). Возвращает ax."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = [ORANGE, VIOLET, AQUA, BLUE, RED]
    for i, e in enumerate(errs):
        k = np.arange(1, len(e) + 1) if logx else np.arange(len(e))
        lab = None if labels is None else labels[i]
        ax.plot(k, np.log10(np.asarray(e) + 1e-17), "-o" if len(e) < 60 else "-", ms=3.5, lw=1.4, color=colors[i % len(colors)], label=lab)
    if logx:
        ax.set_xscale("log")
    ax.set(xlabel="итерация $k$" + (" (лог. шкала)" if logx else ""), ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$")
    if labels is not None:
        ax.legend(fontsize=8)
    return ax
