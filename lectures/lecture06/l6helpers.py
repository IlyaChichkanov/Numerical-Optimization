"""Вспомогательный код демонстраций лекции 6.

Линейный поиск: armijo(f, x, fx, g, p, c1, alpha) — backtracking; wolfe(f, grad, x, p, fx, g, c1, c2) —
        сильные условия Вольфе.
Методы: newton_ls(f, grad, hess, x0, shift="ascent"|"always") — Ньютон с Армихо и сдвигом tau I;
        bfgs(f, grad, x0, line="armijo"|"wolfe"|"exact", Q, scale_H0) — BFGS (Л5) с выбором поиска;
        trust_region(f, grad, hess, x0, step="exact"|"dogleg"|"cauchy") — доверительная область;
        cauchy_point, dogleg, tr_exact — шаги подзадачи; cg(Av, b, x0) — линейные сопряжённые градиенты;
        newton_cg(f, grad, hessp, x0) — неточный Ньютон; gd, newton — из Л4–Л5.
Задачи: rosen, rosen_grad, rosen_hess, X0_ROSEN; him, him_grad, him_hess (Химмельблау);
        make_chain(N) -> (energy, grad_E, H, b, v0, unpack); gd_chain_count(N); f_arctan.

Файл генерируется скриптом scripts/build_lecture06.py из scripts/build_demo06.py — не правьте руками.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen as rosen_nd, rosen_der, rosen_hess_prod

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


# ---------------------------------------------------------------- методы (те же, что в demo06 / l6helpers.py)
def armijo(f, x, fx, g, p, c1=1e-4, alpha=1.0, maxit=60):
    """Backtracking по Армихо: делим alpha пополам, пока f(x + alpha p) > f(x) + c1 alpha g^T p.
    Возвращает (alpha, f(x + alpha p), число вычислений f)."""
    slope = g @ p
    for k in range(1, maxit + 1):
        f_new = f(x + alpha * p)
        if f_new <= fx + c1 * alpha * slope:
            return alpha, f_new, k
        alpha /= 2
    return alpha, f_new, maxit


def wolfe(f, grad, x, p, fx, g, c1=1e-4, c2=0.9, alpha=1.0, maxit=40):
    """Шаг по сильным условиям Вольфе (Nocedal & Wright, алгоритмы 3.5-3.6).
    Фаза 1: alpha удваивается, пока не накроет интервал с подходящим шагом;
    фаза 2 (zoom): интервал делится пополам. Если из-за округления подходящий шаг
    не находится, возвращается лучший найденный шаг с Армихо.
    Возвращает (alpha, f_new, g_new, число вычислений f и grad)."""
    phi0, dphi0 = fx, g @ p
    lo, phi_lo = 0.0, phi0
    nf, hi = 0, None
    for i in range(maxit):
        xa = x + alpha * p; phi = f(xa); ga = grad(xa); dphi = ga @ p; nf += 1
        if phi > phi0 + c1 * alpha * dphi0 or (i > 0 and phi >= phi_lo):
            hi = alpha; break
        if abs(dphi) <= -c2 * dphi0:
            return alpha, phi, ga, nf
        if dphi >= 0:
            hi, lo, phi_lo = lo, alpha, phi; break
        lo, phi_lo = alpha, phi
        alpha *= 2
    else:
        return alpha, phi, ga, nf
    for _ in range(maxit):                         # zoom: lo — лучший шаг с Армихо, hi — другой край
        alpha = 0.5 * (lo + hi)
        xa = x + alpha * p; phi = f(xa); ga = grad(xa); dphi = ga @ p; nf += 1
        if phi > phi0 + c1 * alpha * dphi0 or phi >= phi_lo:
            hi = alpha
        else:
            if abs(dphi) <= -c2 * dphi0:
                return alpha, phi, ga, nf
            if dphi * (hi - lo) >= 0:
                hi = lo
            lo, phi_lo = alpha, phi
        if abs(hi - lo) < 1e-14 * max(1.0, lo):
            break
    xa = x + lo * p
    return lo, f(xa), grad(xa), nf + 1


def newton_ls(f, grad, hess, x0, tol=1e-10, maxit=200, c1=1e-4, shift="ascent"):
    """Ньютон с линейным поиском по Армихо (Л5 §3.4). Сдвиг гессиана H + (|lambda_min| + 1e-3) I:
    shift="ascent" — только если направление Ньютона не спуск (как в Л5 и ДЗ 3);
    shift="always" — всякий раз, когда lambda_min <= 1e-3 («модифицированный Ньютон», как в IPOPT).
    Возвращает (x, итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]; n = len(x)
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        H = hess(x)
        if shift == "always":
            lam_min = np.linalg.eigvalsh(H)[0]
            if lam_min <= 1e-3:
                H = H + (abs(lam_min) + 1e-3) * np.eye(n)
            p = -np.linalg.solve(H, g)
        else:
            p = -np.linalg.solve(H, g)
            if g @ p >= 0:
                lam_min = np.linalg.eigvalsh(H)[0]
                p = -np.linalg.solve(H + (abs(lam_min) + 1e-3) * np.eye(n), g)
        a, _, _ = armijo(f, x, f(x), g, p, c1=c1)
        x = x + a * p; path.append(x.copy())
    return x, maxit, np.array(path)


def bfgs(f, grad, x0, tol=1e-6, maxit=10_000, line="armijo", c1=1e-4, c2=0.9, Q=None, scale_H0=False):
    """BFGS с H_0 = I (Л5 §4.3). Длина шага: line="armijo" — backtracking (как в Л5),
    "wolfe" — сильные условия Вольфе, "exact" — точный шаг на квадратичной 1/2 x^T Q x + b^T x.
    scale_H0=True — после первого шага H_0 <- (s^T y / y^T y) I (Nocedal & Wright, (6.20)).
    Обновление пропускается, если s^T y <= 1e-12 (как в Л5).
    Возвращает (x, итераций, путь, вычислений f, история (s^T y, cos угла между s и y))."""
    x = np.asarray(x0, float).copy(); n = len(x)
    H = np.eye(n); g = grad(x); fx = f(x); nf = 1
    path = [x.copy()]; sy = []
    for k in range(maxit):
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path), nf, np.array(sy)
        p = -H @ g
        if line == "armijo":
            a, fx_new, m = armijo(f, x, fx, g, p, c1=c1); nf += m
            g_new = grad(x + a * p)
        elif line == "wolfe":
            a, fx_new, g_new, m = wolfe(f, grad, x, p, fx, g, c1=c1, c2=c2); nf += m
        else:
            a = -(g @ p) / (p @ Q @ p)
            fx_new = f(x + a * p); g_new = grad(x + a * p); nf += 1
        s = a * p; y = g_new - g
        sy.append((s @ y, (s @ y) / (np.linalg.norm(s) * np.linalg.norm(y) + 1e-300)))
        if k == 0 and scale_H0 and s @ y > 0:
            H = (s @ y) / (y @ y) * np.eye(n)
        if s @ y > 1e-12:
            rho = 1.0 / (s @ y); I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x + s; g = g_new; fx = fx_new
        path.append(x.copy())
    return x, maxit, np.array(path), nf, np.array(sy)


def cauchy_point(g, B, Delta):
    """Точка Коши: минимум модели вдоль -g внутри шара радиуса Delta."""
    gBg = g @ B @ g; gn = np.linalg.norm(g)
    tau = 1.0 if gBg <= 0 else min(gn ** 3 / (Delta * gBg), 1.0)
    return -tau * Delta / gn * g


def dogleg(g, B, Delta):
    """Ломаная Пауэлла (B > 0): от нуля к минимуму модели вдоль -g, затем к точке Ньютона."""
    pB = -np.linalg.solve(B, g)
    if np.linalg.norm(pB) <= Delta:
        return pB
    pU = -(g @ g) / (g @ B @ g) * g
    if np.linalg.norm(pU) >= Delta:
        return -Delta / np.linalg.norm(g) * g
    d = pB - pU
    a, b, c = d @ d, 2 * pU @ d, pU @ pU - Delta ** 2
    return pU + (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a) * d


def tr_exact(g, B, Delta):
    """Точное решение подзадачи min g^T p + 1/2 p^T B p при ||p|| <= Delta (Moré–Sorensen):
    p = -(B + lam I)^{-1} g, lam >= 0, B + lam I >= 0, lam (Delta - ||p||) = 0.
    lam — делением пополам по собственному разложению B (годится для небольших n)."""
    lam, V = np.linalg.eigh(B); gt = V.T @ g

    def p_of(mu):
        return -V @ (gt / (lam + mu))
    if lam[0] > 0 and np.linalg.norm(p_of(0.0)) <= Delta:
        return p_of(0.0)
    lo = max(0.0, -lam[0]); hi = lo + np.linalg.norm(g) / Delta + 1.0
    if np.linalg.norm(p_of(lo + 1e-12 * max(1.0, lo))) < Delta:      # «трудный случай»
        p = p_of(lo + 1e-12 * max(1.0, lo))
        return p + np.sqrt(max(Delta ** 2 - p @ p, 0.0)) * V[:, 0]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(p_of(mid)) > Delta:
            lo = mid
        else:
            hi = mid
    return p_of(hi)


def trust_region(f, grad, hess, x0, step="exact", Delta=1.0, Delta_max=10.0, eta=0.1, tol=1e-6, maxit=20_000):
    """Доверительная область (Nocedal & Wright, алгоритм 4.1): модель m(p) = f + g^T p + 1/2 p^T B p,
    B — гессиан; шаг — step = "exact" | "dogleg" | "cauchy". rho = (f(x) - f(x+p)) / (m(0) - m(p)):
    rho < 1/4 -> Delta/4; rho > 3/4 и шаг на границе -> min(2 Delta, Delta_max); шаг принят при rho > eta.
    Возвращает (x, итераций, путь, история [(x_k, p_k, Delta_k, rho_k, принят)])."""
    x = np.asarray(x0, float).copy(); fx = f(x); path = [x.copy()]; hist = []
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path), hist
        B = hess(x)
        if step == "exact":
            p = tr_exact(g, B, Delta)
        elif step == "dogleg" and np.linalg.eigvalsh(B)[0] > 0:
            p = dogleg(g, B, Delta)
        else:
            p = cauchy_point(g, B, Delta)
        pred = -(g @ p + 0.5 * p @ B @ p)
        f_new = f(x + p)
        rho = (fx - f_new) / pred
        hist.append((x.copy(), p.copy(), Delta, rho, rho > eta))
        if rho < 0.25:
            Delta = 0.25 * Delta
        elif rho > 0.75 and np.linalg.norm(p) >= 0.99 * Delta:
            Delta = min(2 * Delta, Delta_max)
        if rho > eta:
            x = x + p; fx = f_new; path.append(x.copy())
    return x, maxit, np.array(path), hist


def cg(Av, b, x0, tol=1e-6, maxit=None):
    """Линейные сопряжённые градиенты для A x = b (A > 0); A задана только умножением Av(v).
    Останов: ||A x - b|| <= tol. Возвращает (x, итераций, история норм невязки)."""
    x = np.asarray(x0, float).copy(); r = Av(x) - b; d = -r; hist = [np.linalg.norm(r)]
    for k in range(maxit or 10 * len(x)):
        if hist[-1] <= tol:
            return x, k, np.array(hist)
        Ad = Av(d); a = (r @ r) / (d @ Ad)
        x = x + a * d; r_new = r + a * Ad
        beta = (r_new @ r_new) / (r @ r); d = -r_new + beta * d; r = r_new
        hist.append(np.linalg.norm(r))
    return x, maxit, np.array(hist)


def newton_cg(f, grad, hessp, x0, tol=1e-6, maxit=500, c1=1e-4):
    """Неточный Ньютон (Newton-CG): систему H p = -g решаем CG до невязки eta_k ||g||,
    eta_k = min(0.5, sqrt(||g||)); на отрицательной кривизне CG останавливается (если это
    первая итерация — берём p = -g); длина шага — Армихо. hessp(x, v) = H(x) v.
    Возвращает (x, итераций, умножений H v, история (||g_k||, внутренних итераций))."""
    x = np.asarray(x0, float).copy(); nmv = 0; hist = []
    for k in range(maxit):
        g = grad(x); gn = np.linalg.norm(g)
        if gn <= tol:
            hist.append((gn, 0))
            return x, k, nmv, hist
        eta = min(0.5, np.sqrt(gn)); z = np.zeros_like(x); r = g.copy(); d = -r; inner = 0
        for j in range(10 * len(x)):
            Hd = hessp(x, d); nmv += 1; inner += 1; dHd = d @ Hd
            if dHd <= 0:
                if j == 0:
                    z = -g
                break
            a = (r @ r) / dHd; z = z + a * d; r_new = r + a * Hd
            if np.linalg.norm(r_new) <= eta * gn:
                break
            beta = (r_new @ r_new) / (r @ r); d = -r_new + beta * d; r = r_new
        hist.append((gn, inner))
        a, _, _ = armijo(f, x, f(x), g, z, c1=c1)
        x = x + a * z
    return x, maxit, nmv, hist


def gd(f, grad, x0, alpha=None, tol=1e-6, maxit=100_000, c=1e-4):
    """Градиентный спуск (Л4): alpha=None — backtracking по Армихо, иначе постоянный шаг."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        if alpha is None:
            a, _, _ = armijo(f, x, f(x), g, -g, c1=c)
        else:
            a = alpha
        x = x - a * g; path.append(x.copy())
    return x, maxit, np.array(path)


def newton(grad, hess, x0, tol=1e-10, maxit=50):
    """Чистый метод Ньютона (Л5): шаг всегда 1."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        x = x - np.linalg.solve(hess(x), g); path.append(x.copy())
    return x, maxit, np.array(path)


# ---------------------------------------------------------------- задачи
def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def rosen_hess(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200.0]])


X0_ROSEN = np.array([-1.2, 1.0])


def him(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def him_grad(x):
    a, b = x[0] ** 2 + x[1] - 11, x[0] + x[1] ** 2 - 7
    return np.array([4 * x[0] * a + 2 * b, 2 * a + 4 * x[1] * b])


def him_hess(x):
    return np.array([[12 * x[0] ** 2 + 4 * x[1] - 42, 4 * x[0] + 4 * x[1]],
                     [4 * x[0] + 4 * x[1], 4 * x[0] + 12 * x[1] ** 2 - 26]])


def make_chain(N, D=70.0, g=9.81):
    """Цепь без пола (Л1 §7.3): E(v) = 1/2 v^T H v + b^T v + const, v = (y_1..y_N, z_1..z_N)."""
    m = 4.0 / N
    K = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    H = np.kron(np.eye(2), D * K)
    b = np.zeros(2 * N)
    b[0] -= D * (-2.0); b[N - 1] -= D * 2.0
    b[N] -= D * 1.0; b[2 * N - 1] -= D * 1.0
    b[N:] += m * g
    v0 = np.r_[np.linspace(-2, 2, N + 2)[1:-1], np.ones(N)]

    def unpack(v):
        return np.r_[-2.0, v[:N], 2.0], np.r_[1.0, v[N:], 1.0]

    def energy(v):
        y, z = unpack(v)
        return 0.5 * D * np.sum(np.diff(y) ** 2 + np.diff(z) ** 2) + m * g * np.sum(z[1:-1])

    def grad_E(v):
        return H @ v + b
    return energy, grad_E, H, b, v0, unpack


def gd_chain_count(N, tol=1e-6):
    """Число итераций спуска с шагом 1/L на цепи — точно, через собственные числа:
    ||g_k||^2 = sum lam_i^2 e_i^2 (1 - lam_i/L)^(2k). Возвращает (итераций, kappa, функция ||g_k||)."""
    _, _, H, b, v0, _ = make_chain(N)
    lam, V = np.linalg.eigh(H); L = lam[-1]
    e0 = V.T @ (v0 - np.linalg.solve(H, -b))

    def gnorm(k):
        return np.sqrt(np.sum((lam * e0 * (1 - lam / L) ** k) ** 2))
    lo, hi = 0, 1
    while gnorm(hi) > tol:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if gnorm(mid) > tol else (lo, mid)
    return hi, lam[-1] / lam[0], gnorm


def f_arctan(x):
    return x * np.arctan(x) - 0.5 * np.log1p(x * x)


def rosen_contour(ax, xlim=(-2, 2), ylim=(-1, 3), n=400):
    X, Y = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    F = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    ax.contour(X, Y, F, levels=np.geomspace(0.3, 2000, 14), colors=GRAY, linewidths=0.6, alpha=0.7)
    ax.plot(1, 1, "*", color=RED, ms=16, zorder=7)
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$")


def him_contour(ax, lim=6.0, n=400, levels=None):
    X, Y = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    F = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2
    ax.contour(X, Y, F, levels=levels if levels is not None else np.geomspace(0.5, 800, 14),
               colors=GRAY, linewidths=0.6, alpha=0.7)

