"""Иллюстрации к конспекту лекции 6 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture06/scripts/make_figures.py

Скрипт самодостаточен. Палитра и размеры — как в лекциях 4-5 (dpi 100, ширина PNG ≈ ширине
показа 900 px). Методы (armijo, wolfe, newton_ls, bfgs, trust_region, cg, newton_cg) определены
здесь тем же кодом, что в build_demo06.py / l6helpers.py; числа конспекта считаются в скрипте
и проверяются ассертами.

Картинки в порядке появления в конспекте:
  00_hooks            три крючка: Ньютон на арктангенсе, бассейны Химмельблау, цепь N=400
  00_atan_newton      почему чистый Ньютон улетает на арктангенсе: парабола-модель и касательная к f'
  01_review           повторение: пути четырёх методов на Розенброке и биография цепи
  02_armijo_wolfe     phi(alpha) на Розенброке в x_1: допустимые по Армихо и Вольфе шаги
  03_bfgs_wolfe       s^T y у BFGS с Армихо и с Вольфе; цепь: норма градиента и пол округления
  04_saddle_trap      Ньютон с Армихо приходит в седло, доверительная область — в минимум
  05_tr_subproblem    подзадача доверительной области: точка Коши, dogleg, точный шаг
  06_tr_rosenbrock    доверительная область на Розенброке: путь, Delta_k и rho_k
  07_basins_three     Химмельблау: чистый Ньютон / Ньютон с Армихо / доверительная область
  08_cg_ellipse       эллипс kappa=50: спуск 567 против CG 2; цепь N=40: четыре метода
  09_cg_chain         цепь: итерации от N (спуск ~N^2, CG = N/2) и сходимость CG
  10_newton_cg        Розенброк в 1000 переменных: Newton-CG против L-BFGS-B
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image
from scipy.optimize import brentq, minimize, rosen as rosen_nd, rosen_der, rosen_hess_prod

IMG = Path(__file__).resolve().parent.parent / "img"
IMG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7")
GRAY, INK = "#8a8985", "#0b0b0b"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 100,
})

MAX_WIDTH = 920        # px: картинка показывается с width="900", уменьшать её нельзя


def save(name: str) -> None:
    plt.tight_layout()
    path = IMG / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    w, h = Image.open(path).size
    assert 600 <= w <= MAX_WIDTH, f"{name}: ширина {w} px — нужна 600–{MAX_WIDTH}, иначе подписи нечитаемы"
    print(f"   {name}: {w}x{h}")


def arrow(ax, p, q, color, lw=2, ms=14, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=6, **kw))


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


# ---------------------------------------------------------------- Химмельблау на сетке (векторизовано по стартам)
def _him_parts(X, Y):
    a, b = X ** 2 + Y - 11, X + Y ** 2 - 7
    return (a ** 2 + b ** 2, 4 * X * a + 2 * b, 2 * a + 4 * Y * b,
            12 * X ** 2 + 4 * Y - 42, 4 * X + 4 * Y, 4 * X + 12 * Y ** 2 - 26)


def _eig2(hxx, hxy, hyy):
    """Собственные числа l1 <= l2 и единичные векторы v1, v2 симметричных 2x2 поэлементно."""
    tr, dd = 0.5 * (hxx + hyy), np.sqrt(0.25 * (hxx - hyy) ** 2 + hxy ** 2)
    l1, l2 = tr - dd, tr + dd
    vx, vy = hxy, l1 - hxx
    alt = np.hypot(vx, vy) < 1e-12
    vx = np.where(alt, l1 - hyy, vx); vy = np.where(alt, hxy, vy)
    alt = np.hypot(vx, vy) < 1e-12
    vx = np.where(alt, 1.0, vx); vy = np.where(alt, 0.0, vy)
    nv = np.hypot(vx, vy); vx, vy = vx / nv, vy / nv
    return l1, l2, vx, vy, -vy, vx


def basins_pure(X, Y, maxit=30):
    its = np.full(X.shape, -1)
    for k in range(maxit):
        f, gx, gy, hxx, hxy, hyy = _him_parts(X, Y)
        det = hxx * hyy - hxy ** 2
        with np.errstate(all="ignore"):
            dx = (hyy * gx - hxy * gy) / det; dy = (hxx * gy - hxy * gx) / det
        its[(np.hypot(gx, gy) < 1e-8) & (its < 0)] = k
        act = its < 0
        X = np.where(act, X - dx, X); Y = np.where(act, Y - dy, Y)
    return X, Y, its


def basins_ls(X, Y, maxit=300, c1=1e-4):
    """Ньютон + Армихо, сдвиг (|lambda_min| + 1e-3) I только при подъёме — как newton_ls(shift="ascent")."""
    its = np.full(X.shape, -1)
    for k in range(maxit):
        f, gx, gy, hxx, hxy, hyy = _him_parts(X, Y)
        its[(np.hypot(gx, gy) < 1e-8) & (its < 0)] = k
        act = its < 0
        if not act.any():
            break
        det = hxx * hyy - hxy ** 2
        with np.errstate(all="ignore"):
            px = -(hyy * gx - hxy * gy) / det; py = -(hxx * gy - hxy * gx) / det
        bad = ~(gx * px + gy * py < 0)
        l1 = _eig2(hxx, hxy, hyy)[0]
        tau = np.abs(l1) + 1e-3
        det2 = (hxx + tau) * (hyy + tau) - hxy ** 2
        qx = -((hyy + tau) * gx - hxy * gy) / det2; qy = -((hxx + tau) * gy - hxy * gx) / det2
        px = np.where(bad, qx, px); py = np.where(bad, qy, py)
        slope = gx * px + gy * py
        a = np.ones_like(X); need = act.copy()
        for _ in range(60):
            fn = (( (X + a * px) ** 2 + (Y + a * py) - 11) ** 2 + ((X + a * px) + (Y + a * py) ** 2 - 7) ** 2)
            need = need & ~(fn <= f + c1 * a * slope)
            if not need.any():
                break
            a = np.where(need, 0.5 * a, a)
        X = np.where(act, X + a * px, X); Y = np.where(act, Y + a * py, Y)
    return X, Y, its


def basins_tr(X, Y, maxit=200, D0=1.0, Dmax=10.0, eta=0.1):
    """Доверительная область с точным шагом — как trust_region(step="exact")."""
    its = np.full(X.shape, -1); D = np.full(X.shape, D0)
    for k in range(maxit):
        f, gx, gy, hxx, hxy, hyy = _him_parts(X, Y)
        its[(np.hypot(gx, gy) < 1e-8) & (its < 0)] = k
        act = its < 0
        if not act.any():
            break
        l1, l2, v1x, v1y, v2x, v2y = _eig2(hxx, hxy, hyy)
        g1, g2 = v1x * gx + v1y * gy, v2x * gx + v2y * gy

        def pn(mu):
            with np.errstate(all="ignore"):
                return np.hypot(g1 / (l1 + mu), g2 / (l2 + mu))
        inside = (l1 > 0) & (pn(0.0) <= D)
        lo = np.maximum(0.0, -l1); hi = lo + np.hypot(gx, gy) / D + 1.0
        hard = pn(lo + 1e-12 * np.maximum(1.0, lo)) < D
        for _ in range(80):
            mid = 0.5 * (lo + hi); big = pn(mid) > D
            lo = np.where(big, mid, lo); hi = np.where(big, hi, mid)
        mu = np.where(inside, 0.0, hi)
        with np.errstate(all="ignore"):
            c1_, c2_ = -g1 / (l1 + mu), -g2 / (l2 + mu)
            c2h = -g2 / (l2 - l1)
        t = np.sqrt(np.maximum(D ** 2 - c2h ** 2, 0.0))
        c1_ = np.where(hard & ~inside, t, c1_); c2_ = np.where(hard & ~inside, c2h, c2_)
        px, py = c1_ * v1x + c2_ * v2x, c1_ * v1y + c2_ * v2y
        pred = -(gx * px + gy * py + 0.5 * (hxx * px ** 2 + 2 * hxy * px * py + hyy * py ** 2))
        fn = ((X + px) ** 2 + (Y + py) - 11) ** 2 + ((X + px) + (Y + py) ** 2 - 7) ** 2
        with np.errstate(all="ignore"):
            rho = (f - fn) / pred
        rho = np.where(np.isfinite(rho), rho, -1.0)
        pnorm = np.hypot(px, py)
        D = np.where(act & (rho < 0.25), 0.25 * D, D)
        D = np.where(act & (rho > 0.75) & (pnorm >= 0.99 * D), np.minimum(2 * D, Dmax), D)
        acc = act & (rho > eta)
        X = np.where(acc, X + px, X); Y = np.where(acc, Y + py, Y)
    return X, Y, its


def classify(X, Y, its):
    """Тип точки прихода: 0 — минимум, 1 — седло, 2 — максимум, -1 — не сошлось."""
    _, _, _, hxx, hxy, hyy = _him_parts(X, Y)
    l1, l2 = _eig2(hxx, hxy, hyy)[:2]
    kind = np.full(X.shape, -1)
    ok = its >= 0
    kind[ok & (l1 > 0)] = 0; kind[ok & (l1 < 0) & (l2 > 0)] = 1; kind[ok & (l2 < 0)] = 2
    return kind


# 9 стационарных точек Химмельблау: 4 минимума, 4 седла, 1 максимум
HIM_POINTS = None


def him_stationary():
    from scipy.optimize import fsolve
    pts = []
    for a in np.linspace(-5, 5, 21):
        for b in np.linspace(-5, 5, 21):
            p = fsolve(him_grad, [a, b], fprime=him_hess, full_output=True)
            if p[2] == 1 and np.linalg.norm(him_grad(p[0])) < 1e-9:
                if not any(np.allclose(p[0], q, atol=1e-6) for q in pts):
                    pts.append(p[0])
    kinds = []
    for p in pts:
        lam = np.linalg.eigvalsh(him_hess(p))
        kinds.append(0 if lam[0] > 0 else (2 if lam[1] < 0 else 1))
    return np.array(pts), np.array(kinds)


def nearest_point(X, Y, pts):
    d = np.hypot(X[..., None] - pts[:, 0], Y[..., None] - pts[:, 1])
    return np.argmin(d, axis=-1), np.min(d, axis=-1)


# ================================================================ общие прогоны (числа конспекта)
def runs():
    R = {}
    # арктангенс
    x = 2.0; seq = [x]
    for _ in range(5):
        x = x - np.arctan(x) * (1 + x * x); seq.append(x)
    R["atan_pure"] = np.array(seq)
    assert np.allclose(seq[:4], [2, -3.5357, 13.951, -279.34], rtol=1e-3) and abs(seq[5] + 2.339e10) < 1e8, seq
    x = np.array([2.0]); seq_d = [2.0]; alphas = []
    while abs(np.arctan(x[0])) > 1e-10:
        g = np.array([np.arctan(x[0])]); p = -g * (1 + x[0] ** 2)
        a, _, _ = armijo(lambda z: f_arctan(z[0]), x, f_arctan(x[0]), g, p)
        x = x + a * p; seq_d.append(x[0]); alphas.append(a)
    R["atan_damped"] = np.array(seq_d)
    assert len(alphas) == 5 and alphas == [0.5, 1.0, 1.0, 1.0, 1.0], alphas
    assert np.allclose(seq_d[1:4], [-0.7679, 0.2731, -0.01338], rtol=1e-3) and abs(seq_d[4] - 1.6e-6) < 1e-7, seq_d
    # обещание параболы-модели в x0 = 2 против правды; порог старта (шаг переносит ровно в -x)
    x0, x1 = seq[0], seq[1]
    m1 = f_arctan(x0) + np.arctan(x0) * (x1 - x0) + 0.5 / (1 + x0 ** 2) * (x1 - x0) ** 2
    thr = brentq(lambda x: (1 + x * x) * np.arctan(x) - 2 * x, 1.0, 2.0)
    R["atan_promise"] = (f_arctan(x0), m1, f_arctan(x1), f_arctan(seq_d[1]))
    R["atan_thr"] = thr
    assert np.allclose(R["atan_promise"], [1.4096, -1.655, 3.278, 0.2711], rtol=1e-3), R["atan_promise"]
    assert abs(thr - 1.3917) < 1e-4, thr
    ratio = np.abs(seq[3:6]) / np.array(seq[2:5]) ** 2           # |x_{k+1}| / x_k^2 -> pi/2
    assert np.allclose(ratio, [1.435, 1.564, 1.5708], rtol=1e-3), ratio

    # Розенброк: повторение Л5
    xg, kg, pg = gd(rosen, rosen_grad, X0_ROSEN)
    xn, kn, pn = newton(rosen_grad, rosen_hess, X0_ROSEN)
    xd, kd, pd = newton_ls(rosen, rosen_grad, rosen_hess, X0_ROSEN)
    xb, kb, pb, nfb, _ = bfgs(rosen, rosen_grad, X0_ROSEN)
    assert abs(kg - 13756) < 140 and kn == 7 and kd == 22 and (kb, nfb) == (34, 54), (kg, kn, kd, kb, nfb)
    R["rosen_review"] = dict(gd=(kg, pg), newton=(kn, pn), damped=(kd, pd), bfgs=(kb, pb))

    # доверительная область на Розенброке
    xt, kt, pt, ht = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step="exact")
    rej = sum(not h[4] for h in ht)
    assert kt == 25 and rej == 3 and np.allclose(xt, 1, atol=1e-6), (kt, rej)
    xdl, kdl, _, hdl = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step="dogleg")
    assert kdl == 24 and sum(not h[4] for h in hdl) == 3, kdl
    xc, kc, _, _ = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step="cauchy")
    assert abs(kc - 14110) < 150, kc
    R["tr_rosen"] = (kt, pt, ht, rej, kdl, kc)

    # SciPy на Розенброке
    sc = {}
    for m in ["trust-exact", "dogleg", "trust-ncg", "trust-krylov", "CG", "BFGS", "Newton-CG", "L-BFGS-B"]:
        kw = dict(jac=rosen_grad)
        if m not in ("CG", "BFGS", "L-BFGS-B"):
            kw["hess"] = rosen_hess
        opts = {"xtol": 1e-8} if m == "Newton-CG" else {"gtol": 1e-6}
        if m == "L-BFGS-B":
            opts["ftol"] = 0.0                      # иначе останов по относительному уменьшению f при ||g|| ~ 6e-5
        r = minimize(rosen, X0_ROSEN, method=m, options=opts, **kw)
        sc[m] = (r.nit, r.nfev)
    assert sc["trust-exact"][0] == 25 and sc["dogleg"][0] == 24 and sc["trust-ncg"][0] == 29 and sc["trust-krylov"][0] == 38, sc
    assert sc["CG"] == (37, 80) and sc["BFGS"] == (33, 40) and sc["L-BFGS-B"][0] == 37, sc
    assert 80 <= sc["Newton-CG"][0] <= 90, sc                       # 85 на SciPy 1.18
    R["scipy_rosen"] = sc
    return R


# ================================================================ 00 крючки
def fig_hooks(R, B):
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.6))
    ax = axs[0]
    sp = R["atan_pure"]; sd = R["atan_damped"]
    ax.plot(range(len(sp)), sp, "o-", color=RED, label="чистый Ньютон")
    ax.plot(range(len(sd)), sd, "s-", color=BLUE, label="с Армихо")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([-1e10, -1e5, -1, 0, 1, 1e5])
    ax.set_yticklabels(["$-10^{10}$", "$-10^{5}$", "$-1$", "0", "1", "$10^{5}$"])
    ax.set(xlabel="итерация $k$", ylabel="$x_k$", title="арктангенс из $x_0=2$")
    ax.annotate(r"$-2.3\cdot10^{10}$", (5, sp[5]), xytext=(2.2, -1e8), color=RED, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(loc="upper left", fontsize=10)

    ax = axs[1]
    kind, xs = B["pure_kind"], B["xs"]
    cmap = ListedColormap([BLUE, ORANGE, RED])
    ax.imshow(np.where(kind >= 0, kind, np.nan), origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]),
              cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest", alpha=0.85)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Химмельблау: Ньютон", aspect="equal")
    sh = B["pure_shares"]
    ax.text(0.02, 0.02, f"минимум {sh[0]:.0%}\nседло {sh[1]:.0%}\nмаксимум {sh[2]:.0%}", transform=ax.transAxes,
            fontsize=10, va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.grid(False)

    ax = axs[2]
    k_gd, _, gnorm = gd_chain_count(400)
    ks = np.unique(np.geomspace(1, k_gd, 300).astype(int))
    ax.loglog(ks, [gnorm(k) for k in ks], color=ORANGE, label=f"спуск: {k_gd:,}".replace(",", " "))
    _, _, H, b, v0, _ = make_chain(400)
    _, k_cg, hist = cg(lambda v: H @ v, -b, v0)
    ax.loglog(np.arange(1, len(hist) + 1), hist, color=AQUA, label=f"CG: {k_cg}")
    ax.axhline(1e-6, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="итерация $k$", ylabel=r"$\|\nabla E\|$", title="цепь из 400 грузов")
    ax.legend(loc="lower left", fontsize=10)
    save("00_hooks.png")
    return k_gd, k_cg


# ================================================================ 00 почему Ньютон улетает на арктангенсе
def fig_atan_newton(R):
    seq = R["atan_pure"]; x0, x1, x2 = seq[0], seq[1], seq[2]
    f0, m1, f1, f_half = R["atan_promise"]; thr = R["atan_thr"]
    x_half = R["atan_damped"][1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), gridspec_kw=dict(width_ratios=[1, 1.1]))

    ax = axes[0]                                       # f и парабола-модель в x0
    t = np.linspace(-5, 4, 600)
    ax.plot(t, f_arctan(t), color=INK, lw=2.2, label="$f(x)$")
    par = f0 + np.arctan(x0) * (t - x0) + 0.5 / (1 + x0 ** 2) * (t - x0) ** 2
    ax.plot(t, par, color=RED, lw=1.6, label="парабола-модель в $x_0$")
    ax.plot(x0, f0, "o", color=RED, ms=7, zorder=7)
    ax.annotate("$x_0=2$", (x0, f0), (x0 + 0.25, f0 - 0.9), color=RED, fontsize=11)
    ax.plot([x1, x1], [m1, f1], ":", color=RED, lw=1.2)
    ax.plot(x1, m1, "v", color=RED, ms=8, zorder=7)
    ax.plot(x1, f1, "o", color=RED, ms=7, zorder=7)
    ax.annotate(f"модель обещала ${m1:.2f}$", (x1, m1), (x1 + 0.3, -2.3), color=RED, fontsize=10.5)
    ax.annotate(f"получили ${f1:.2f}$\n$x_1={x1:.2f}$", (x1, f1), (x1 + 0.3, f1 + 0.5), color=RED, fontsize=10.5)
    ax.plot(x_half, f_half, "s", color=BLUE, ms=7, zorder=8)
    ax.annotate(f"$\\alpha=\\frac{{1}}{{2}}$: ${f_half:.2f}$", (x_half, f_half), (x_half - 0.1, f_half + 1.1),
                color=BLUE, fontsize=10.5, ha="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.plot(0, 0, "*", color=AQUA, ms=14, zorder=7)
    ax.set(xlim=(-5, 4), ylim=(-2.5, 5.5), xlabel="$x$", title="парабола-модель в $x_0=2$")
    ax.legend(loc="upper right", fontsize=10)

    ax = axes[1]                                       # f' = arctan x и касательные
    t = np.linspace(-5, 15, 800)
    ax.plot(t, np.arctan(t), color=INK, lw=2.2, label=r"$f'=\arctan x$")
    for y in (np.pi / 2, -np.pi / 2):
        ax.axhline(y, color=GRAY, ls=":", lw=1)
    ax.text(11.5, np.pi / 2 + 0.05, r"$+\pi/2$", color=GRAY, ha="right", va="bottom", fontsize=11)
    ax.text(14.8, -np.pi / 2 - 0.05, r"$-\pi/2$", color=GRAY, ha="right", va="top", fontsize=11)
    ax.axhline(0, color=GRAY, lw=0.8)
    for i, (xa, xb, ls, al) in enumerate(((x0, x1, "-", 1.0), (x1, x2, "--", 0.7))):
        g, h = np.arctan(xa), 1 / (1 + xa ** 2)
        tt = np.array([xb, xa]) if xa > xb else np.array([xa, xb])
        tt = np.linspace(tt[0] - 0.5, tt[1] + 0.5, 50)
        ax.plot(tt, g + h * (tt - xa), ls, color=RED, lw=1.6, alpha=al, label="касательная в $x_k$" if i == 0 else None)
        ax.plot(xa, g, "o", color=RED, ms=7, zorder=7, alpha=al)
        ax.plot([xb, xb], [0, np.arctan(xb)], ":", color=RED, lw=1, alpha=al)
        ax.plot(xb, 0, "o", color=RED, ms=6, zorder=7, alpha=al, mfc="white")
    ax.annotate("$x_0=2$", (x0, np.arctan(x0)), (x0 + 0.6, np.arctan(x0) - 0.35), color=RED, fontsize=11)
    ax.annotate(f"$x_1={x1:.2f}$", (x1, 0), (x1 - 1.2, -0.38), color=RED, fontsize=11, ha="left")
    ax.annotate(f"$x_2={x2:.2f}$", (x2, 0), (x2 - 0.3, -0.62), color=RED, fontsize=11, ha="right")
    ax.plot(0, 0, "*", color=AQUA, ms=14, zorder=7)
    for sgn in (1, -1):
        ax.plot([sgn * thr, sgn * thr], [-0.12, 0.12], color=GRAY, lw=2)
    ax.annotate(f"порог $\\pm{thr:.2f}$", (thr, -0.12), (thr + 1.3, -0.4), color=GRAY, fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.set(xlim=(-5, 15), ylim=(-1.85, 1.85), xlabel="$x$", title="$f'=\\arctan x$: касательная перелетает")
    ax.legend(loc="upper left", fontsize=10)
    save("00_atan_newton.png")


# ================================================================ 01 повторение
def fig_review(R):
    rv = R["rosen_review"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    rosen_contour(ax, xlim=(-2, 2), ylim=(-3.5, 3.2))
    kg, pg = rv["gd"]
    ax.plot(pg[:, 0], pg[:, 1], "-", color=GRAY, lw=1.5, label=f"спуск: {kg:,}".replace(",", " "))
    for key, col, lab in [("newton", RED, "Ньютон"), ("damped", BLUE, "Ньютон + Армихо"), ("bfgs", AQUA, "BFGS")]:
        k, p = rv[key]
        ax.plot(p[:, 0], p[:, 1], ".-", color=col, lw=1.6, ms=6, label=f"{lab}: {k}")
    ax.plot(*X0_ROSEN, "o", color=INK, ms=8, zorder=8)
    ax.set_title("Л4–Л5: одна формула, разные $B_k$ и $\\alpha_k$")
    ax.legend(loc="lower left", fontsize=10)

    ax = axs[1]
    k40, _, _ = gd_chain_count(40)
    names = ["спуск, шаг $1/L$ (Л4)", "BFGS, точный шаг (Л5)", "Ньютон (Л5)", "сегодня: CG"]
    vals = [k40, 20, 1, np.nan]
    cols = [ORANGE, AQUA, RED, GRAY]
    y = np.arange(len(names))[::-1]
    ax.barh(y[:3], vals[:3], color=cols[:3], height=0.6)
    ax.barh(y[3], 20, color="none", edgecolor=GRAY, ls="--", height=0.6)
    for yi, v in zip(y[:3], vals[:3]):
        ax.text(v * 1.3, yi, f"{v:,}".replace(",", " "), va="center", fontsize=11)
    ax.text(1.3, y[3], "?", va="center", fontsize=16, color=GRAY, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(0.7, 1e5)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set(xlabel="итераций до $\\|\\nabla E\\|\\leq10^{-6}$", title="биография цепи, $N=40$")
    ax.grid(axis="y", alpha=0)
    save("01_review.png")
    assert k40 == 10575, k40


# ================================================================ 02 Армихо и Вольфе
def line_at_x1():
    x1 = X0_ROSEN - np.linalg.solve(rosen_hess(X0_ROSEN), rosen_grad(X0_ROSEN))
    p1 = -np.linalg.solve(rosen_hess(x1), rosen_grad(x1))
    return x1, p1


def wolfe_intervals(x, p, c1=1e-4, c2=0.9, amax=1.0, n=200_001):
    al = np.linspace(0, amax, n)
    phi = np.array([rosen(x + a * p) for a in al]); dphi = np.array([rosen_grad(x + a * p) @ p for a in al])
    d0 = dphi[0]
    arm = phi <= phi[0] + c1 * al * d0
    wol = arm & (dphi >= c2 * d0)
    swol = arm & (np.abs(dphi) <= c2 * abs(d0))

    def first_interval(mask):
        idx = np.flatnonzero(mask[1:]) + 1
        start = idx[0]; stop = start
        while stop + 1 < len(mask) and mask[stop + 1]:
            stop += 1
        return al[start], al[stop]
    return al, phi, dphi, first_interval(arm), first_interval(wol), first_interval(swol)


def fig_armijo_wolfe():
    x1, p1 = line_at_x1()
    assert np.allclose(x1, [-1.1753, 1.3807], atol=1e-4) and abs(np.linalg.norm(p1) - 4.951) < 1e-3
    al, phi, dphi, arm, wol, swol = wolfe_intervals(x1, p1)
    assert abs(arm[1] - 0.176) < 0.002 and abs(wol[0] - 0.044) < 0.002 and abs(swol[1] - 0.138) < 0.002, (arm, wol, swol)
    assert abs(rosen(x1 + p1) - 1411.8) < 0.2 and abs(phi[0] - 4.732) < 1e-3, rosen(x1 + p1)
    a_min = al[np.argmin(phi)]; assert abs(a_min - 0.110) < 0.002 and abs(phi.min() - 4.061) < 2e-3, (a_min, phi.min())
    amax = 0.30
    sel = al <= amax
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axvspan(*arm, color=AQUA, alpha=0.10, lw=0)
    ax.axvspan(*wol, color=AQUA, alpha=0.18, lw=0)
    ax.plot(al[sel], phi[sel], color=INK, lw=2.2, label=r"$\varphi(\alpha)=f(x_1+\alpha p_1)$")
    ax.plot(al[sel], phi[0] + 1e-4 * al[sel] * dphi[0], color=ORANGE, ls="--", lw=1.6,
            label=r"прямая Армихо $\varphi(0)+c_1\alpha\varphi'(0)$")
    ax.plot(al[al <= 0.12], phi[0] + al[al <= 0.12] * dphi[0], color=GRAY, ls=":", lw=1.4, label="касательная в 0")
    for a, col, txt, dx, dy in [(0.02, YELLOW, "0.02: Армихо да,\nВольфе нет", 0.006, 0.55),
                                (0.125, AQUA, "0.125: оба условия", -0.03, 0.45),
                                (0.25, RED, "0.25: Армихо нет", -0.085, -0.75)]:
        y = rosen(x1 + a * p1)
        ax.plot(a, y, "o", color=col, ms=11, mec=INK, zorder=7)
        ax.annotate(txt, (a, y), xytext=(a + dx, y + dy), fontsize=11, color=INK)
    ax.text(arm[0] + 0.004, 9.0, f"Армихо: $\\alpha\\leq{arm[1]:.3f}$", fontsize=11, color="#0c7a54")
    ax.text(wol[0] + 0.004, 8.45, f"Вольфе: ${wol[0]:.3f}\\leq\\alpha\\leq{wol[1]:.3f}$", fontsize=11, color="#0c7a54")
    ax.annotate(r"$\alpha=1$: $\varphi=1412$", (0.30, 9.45), xytext=(0.262, 6.4), fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set(xlim=(0, amax), ylim=(3.6, 9.6), xlabel=r"длина шага $\alpha$", ylabel=r"$\varphi(\alpha)$",
           title="Розенброк в $x_1$ вдоль шага Ньютона ($\\| p_1\\|=4.95$)")
    ax.legend(loc="lower right", fontsize=10)
    save("02_armijo_wolfe.png")
    return arm, wol, swol


# ================================================================ 03 Вольфе и BFGS
def fig_bfgs_wolfe():
    rng = np.random.default_rng(2026)
    starts = rng.uniform(-6, 6, (400, 2))
    stat = {}
    cosines = {}
    for line in ["armijo", "wolfe"]:
        runs_bad, steps_bad, cs = 0, 0, []
        for x0 in starts:
            _, _, _, _, sy = bfgs(him, him_grad, x0, line=line, maxit=5000)
            neg = int(np.sum(sy[:, 0] <= 0)); runs_bad += neg > 0; steps_bad += neg; cs.append(sy[:, 1])
        stat[line] = (runs_bad, steps_bad); cosines[line] = np.concatenate(cs)
    assert stat["wolfe"] == (0, 0) and 30 <= stat["armijo"][0] <= 80, stat

    E, gE, H, b, v0, _ = make_chain(40)
    curves = {}
    for key, line, sc in [("armijo", "armijo", False), ("wolfe", "wolfe", False), ("scaled", "armijo", True)]:
        _, k, path, nf, _ = bfgs(E, gE, v0, line=line, scale_H0=sc, tol=1e-6, maxit=3000)
        curves[key] = ([np.linalg.norm(gE(v)) for v in path], k, nf)
    _, ke, pe, _, _ = bfgs(E, gE, v0, line="exact", Q=H)
    assert curves["armijo"][1] == 3000 and curves["wolfe"][1] == 3000, (curves["armijo"][1:], curves["wolfe"][1:])
    assert curves["scaled"][1:] == (27, 29) and ke == 20, (curves["scaled"][1:], ke)
    for N, k_expected in [(10, 10), (20, 17), (80, 47)]:
        EN, gEN, _, _, v0N, _ = make_chain(N)
        assert bfgs(EN, gEN, v0N, scale_H0=True)[1] == k_expected, N
    floor = np.sqrt(2 * np.linalg.eigvalsh(H)[-1] * np.finfo(float).eps * abs(E(np.linalg.solve(H, -b))))
    assert 1e-6 < floor < 3e-6, floor
    stall = min(min(curves["armijo"][0]), min(curves["wolfe"][0]))
    assert 1e-6 < stall < 2e-5, stall

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]
    rng2 = np.random.default_rng(0)
    for i, (line, lab) in enumerate([("armijo", "только Армихо"), ("wolfe", "Армихо + Вольфе")]):
        c = cosines[line]; jit = i + 0.28 * (rng2.random(len(c)) - 0.5)
        ax.scatter(jit[c > 0], c[c > 0], s=4, color=BLUE, alpha=0.25, lw=0)
        ax.scatter(jit[c <= 0], c[c <= 0], s=14, color=RED, alpha=0.9, lw=0)
        rb, sb = stat[line]
        ax.text(i, -1.12, f"{rb} запусков\n{sb} шагов с $s^\\top y\\leq0$", ha="center", va="top", fontsize=10,
                color=RED if rb else "#0c7a54")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["только Армихо", "Армихо + Вольфе"])
    ax.set(ylim=(-1.5, 1.05), ylabel=r"$\cos\angle(s_k,y_k)$", title="Химмельблау, BFGS из 400 стартов")
    ax.grid(axis="x", alpha=0)

    ax = axs[1]
    for key, col, lab in [("armijo", ORANGE, "$H_0=I$, Армихо"), ("wolfe", BLUE, "$H_0=I$, Вольфе")]:
        g = curves[key][0]
        ax.semilogy(np.arange(1, len(g) + 1), g, color=col, lw=1.4, label=f"{lab}: застрял")
    g = curves["scaled"][0]
    ax.semilogy(np.arange(1, len(g) + 1), g, "o-", color=AQUA, lw=2.2, ms=3,
                label=f"$H_0$ масштабирована, Армихо: {curves['scaled'][1]}")
    ge = [np.linalg.norm(gE(v)) for v in pe]
    ax.semilogy(np.arange(1, len(ge) + 1), ge, "--", color=VIOLET, lw=1.8, label=f"точный шаг: {ke}")
    ax.axhspan(1e-7, floor, color=GRAY, alpha=0.15, lw=0)
    ax.text(60, floor * 0.3, "пол округления значений $E$", fontsize=10, color=INK)
    ax.axhline(1e-6, color=GRAY, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set(xlabel="итерация $k$", ylabel=r"$\|\nabla E\|$", ylim=(1e-7, 60), title="цепь $N=40$: BFGS")
    ax.legend(loc="lower left", fontsize=9.5)
    save("03_bfgs_wolfe.png")
    return stat, floor


# ================================================================ 04 ловушка седла
TRAP = np.array([0.58, 2.15])


def fig_saddle_trap():
    xl, kl, pl = newton_ls(him, him_grad, him_hess, TRAP)
    lam = np.linalg.eigvalsh(him_hess(xl))
    assert kl == 5 and lam[0] < 0 < lam[1] and np.allclose(xl, [0.0867, 2.8843], atol=1e-3), (kl, xl, lam)
    xt, kt, pt, ht = trust_region(him, him_grad, him_hess, TRAP, step="exact", tol=1e-10)
    assert np.allclose(xt, [3, 2], atol=1e-8) and kt == 7, (xt, kt)
    xa, ka, _ = newton_ls(him, him_grad, him_hess, TRAP, shift="always")
    assert np.allclose(xa, [3, 2], atol=1e-6), xa

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    him_contour(ax, lim=4.5, levels=np.geomspace(0.5, 400, 14))
    ax.plot(pl[:, 0], pl[:, 1], "o-", color=ORANGE, ms=5, label=f"Ньютон + Армихо: седло за {kl}")
    ax.plot(pt[:, 0], pt[:, 1], "s-", color=BLUE, ms=5, label=f"доверительная область: минимум за {kt}")
    for (x, p, D, rho, acc) in ht[:3]:
        ax.add_patch(Circle(x, D, fill=False, ls="--", lw=1, color=BLUE, alpha=0.6))
    ax.plot(*TRAP, "o", color=INK, ms=9, zorder=8)
    ax.annotate("старт (0.58, 2.15)", TRAP, xytext=(-2.8, 0.6), fontsize=10, arrowprops=dict(arrowstyle="->", color=INK))
    ax.plot(*xl, "X", color=ORANGE, ms=13, mec=INK, zorder=9)
    ax.plot(3, 2, "*", color=RED, ms=18, mec=INK, zorder=9)
    ax.set(xlim=(-3, 4.5), ylim=(-1, 4.5), aspect="equal", xlabel="$x_1$", ylabel="$x_2$", title="Химмельблау: ловушка седла")
    ax.legend(loc="lower left", fontsize=9.5)
    ax = axs[1]
    ax.semilogy([np.linalg.norm(him_grad(x)) for x in pl], "o-", color=ORANGE, label="Ньютон + Армихо → седло")
    ax.semilogy([np.linalg.norm(him_grad(x)) for x in pt], "s-", color=BLUE, label="доверительная область → минимум")
    ax.set(xlabel="итерация $k$", ylabel=r"$\|\nabla f\|$", title="оба: $\\nabla f\\to0$ — теорема выполнена")
    ax.legend(loc="lower left", fontsize=10)
    save("04_saddle_trap.png")
    return kl, kt, xl


# ================================================================ 05 подзадача доверительной области
def model_grid(ax, x, g, B, lim, levels, n=300):
    P1, P2 = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    M = g[0] * P1 + g[1] * P2 + 0.5 * (B[0, 0] * P1 ** 2 + 2 * B[0, 1] * P1 * P2 + B[1, 1] * P2 ** 2)
    ax.contour(x[0] + P1, x[1] + P2, M, levels=levels, colors=GRAY, linewidths=0.7, alpha=0.8)


def fig_tr_subproblem():
    g, B = np.array([3.0, 4.0]), np.diag([1.0, 2.0])
    pN = -np.linalg.solve(B, g); pU = -(g @ g) / (g @ B @ g) * g
    m = lambda p: g @ p + 0.5 * p @ B @ p
    pc1 = cauchy_point(g, B, 1.0)
    assert np.allclose(pc1, [-0.6, -0.8]) and abs(m(pc1) + 4.18) < 1e-9
    assert np.allclose(pN, [-3, -2]) and abs(m(pN) + 8.5) < 1e-12 and abs(np.linalg.norm(pU) - 3.049) < 1e-3
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 4.6))
    ax = axs[0]
    o = np.zeros(2)
    model_grid(ax, o, g, B, 5.0, levels=np.sort(m(pN) + np.array([0.15, 0.7, 1.6, 3, 5, 8, 12])))
    Ds = np.linspace(0.02, np.linalg.norm(pN), 150)
    path = np.array([tr_exact(g, B, D) for D in Ds])
    ax.plot(path[:, 0], path[:, 1], color=BLUE, lw=2.6, label="точный шаг при разных $\\Delta$")
    ax.plot([0, pU[0], pN[0]], [0, pU[1], pN[1]], "--", color=ORANGE, lw=2, label="dogleg")
    for D in (1.0, 2.0, 3.0):
        ax.add_patch(Circle(o, D, fill=False, lw=1, color=INK, alpha=0.55))
        pc = cauchy_point(g, B, D)
        ax.plot(*pc, "D", color=YELLOW, mec=INK, ms=7, zorder=7)
        ax.text(D * 0.72 + 0.05, D * 0.72, f"$\\Delta={D:g}$", fontsize=10)
    ax.plot([], [], "D", color=YELLOW, mec=INK, ms=7, label="точки Коши")
    ax.plot(*pN, "o", color=RED, ms=9, mec=INK, zorder=8, label="точка Ньютона")
    ax.plot(0, 0, "o", color=INK, ms=7, zorder=8)
    arrow(ax, (0, 0), -0.9 * g / np.linalg.norm(g), GRAY, lw=1.5, ms=12)
    ax.text(-0.95, -0.2, "$-g$", fontsize=11, color=GRAY)
    ax.set(xlim=(-4.2, 3.4), ylim=(-4.2, 3.4), aspect="equal", xlabel="$p_1$", ylabel="$p_2$",
           title="модель-чаша: $g=(3,4)$, $B=\\mathrm{diag}(1,2)$")
    ax.legend(loc="lower right", fontsize=9)

    ax = axs[1]
    xs_ = np.array([0.4, 2.0])                          # у седла Химмельблау (0.09, 2.88): гессиан индефинитен
    g2, B2 = him_grad(xs_), him_hess(xs_)
    lam2 = np.linalg.eigvalsh(B2)
    assert lam2[0] < 0 < lam2[1], lam2
    model_grid(ax, xs_, g2, B2, 2.2, levels=np.linspace(-60, 60, 17))
    pN = -np.linalg.solve(B2, g2)
    for D, col in [(0.5, VIOLET), (1.0, BLUE), (1.6, AQUA)]:
        ax.add_patch(Circle(xs_, D, fill=False, lw=1, color=INK, alpha=0.6))
        p = tr_exact(g2, B2, D)
        assert abs(np.linalg.norm(p) - D) < 1e-8
        arrow(ax, xs_, xs_ + p, col, lw=2)
        ax.text(*(xs_ + p * 1.08), f"$\\Delta={D:g}$", fontsize=10, color=col)
    ax.plot(*(xs_ + pN), "o", color=RED, ms=9, mec=INK, zorder=8)
    ax.annotate("точка Ньютона =\nседло модели", xs_ + pN, xytext=(xs_[0] - 2.0, xs_[1] + 1.5), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.plot(*xs_, "o", color=INK, ms=7, zorder=8)
    ax.set(xlim=(xs_[0] - 2.2, xs_[0] + 2.2), ylim=(xs_[1] - 2.2, xs_[1] + 2.2), aspect="equal", xlabel="$x_1$",
           ylabel="$x_2$", title=f"Химмельблау в $(0.4,2)$: седло, $\\lambda_{{\\min}}={lam2[0]:.0f}$")
    save("05_tr_subproblem.png")
    return np.linalg.norm(pN), lam2


# ================================================================ 06 доверительная область на Розенброке
def fig_tr_rosenbrock(R):
    kt, pt, ht, rej, kdl, kc = R["tr_rosen"]
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]
    rosen_contour(ax, xlim=(-2, 2), ylim=(-1.2, 3))
    for (x, p, D, rho, acc) in ht:
        ax.add_patch(Circle(x, D, fill=False, lw=0.8, color=BLUE if acc else RED, alpha=0.35 if acc else 0.8,
                            ls="-" if acc else "--"))
        if not acc:
            ax.plot(*(x + p), "x", color=RED, ms=8, mew=2)
    ax.plot(pt[:, 0], pt[:, 1], "o-", color=BLUE, ms=4, lw=1.6, label=f"принятые шаги: {kt} итераций")
    ax.plot([], [], "x", color=RED, ms=8, mew=2, label=f"отвергнуто: {rej}")
    ax.plot(*X0_ROSEN, "o", color=INK, ms=8, zorder=8)
    ax.set(aspect="equal", title="точный шаг, $\\Delta_0=1$, $\\Delta_{\\max}=10$")
    ax.legend(loc="lower right", fontsize=9.5)
    ax = axs[1]
    Dk = np.array([h[2] for h in ht]); rk = np.array([h[3] for h in ht]); acc = np.array([h[4] for h in ht])
    it = np.arange(len(ht))
    ax.semilogy(it, Dk, "o-", color=BLUE, ms=4, label=r"$\Delta_k$")
    ax2 = ax.twinx()
    ax2.plot(it[acc], rk[acc], "s", color=AQUA, ms=5, label=r"$\rho_k$, шаг принят")
    ax2.plot(it[~acc], rk[~acc], "x", color=RED, ms=8, mew=2, label=r"$\rho_k$, шаг отвергнут")
    ax2.axhline(0.25, color=GRAY, ls=":", lw=1); ax2.axhline(0.75, color=GRAY, ls=":", lw=1)
    ax2.set_ylim(-1.2, 1.6); ax2.set_ylabel(r"$\rho_k$"); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax.set(xlabel="итерация (с отвергнутыми)", ylabel=r"$\Delta_k$", title=r"радиус $\Delta_k$ и отношение $\rho_k$")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9.5)
    save("06_tr_rosenbrock.png")


# ================================================================ 07 бассейны трёх методов
def basins_all(n=300, lim=6.0):
    t0 = time.perf_counter()
    xs = np.linspace(-lim, lim, n)
    X0, Y0 = np.meshgrid(xs, xs)
    out = {"xs": xs}
    for name, meth in [("pure", basins_pure), ("ls", basins_ls), ("tr", basins_tr)]:
        X, Y, its = meth(X0.copy(), Y0.copy())
        kind = classify(X, Y, its)
        out[name + "_kind"] = kind
        out[name + "_shares"] = [np.mean(kind == k) for k in (0, 1, 2, -1)]
        out[name + "_median"] = np.median(its[its >= 0])
    sp, sl, st = out["pure_shares"], out["ls_shares"], out["tr_shares"]
    assert abs(sp[0] - 0.644) < 0.01 and abs(sp[1] - 0.314) < 0.01 and abs(sp[2] - 0.043) < 0.005, sp
    assert abs(sl[0] - 0.792) < 0.01 and abs(sl[1] - 0.205) < 0.01 and sl[2] == 0, sl
    assert st[0] == 1.0, st
    print(f"      бассейны: {time.perf_counter() - t0:.1f} с; чистый {sp}, Армихо {sl}, область {st}")
    return out


def fig_basins_three(B):
    xs = B["xs"]
    cmap = ListedColormap([BLUE, ORANGE, RED])
    fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.5))
    titles = [("pure", "чистый Ньютон"), ("ls", "Ньютон + Армихо"), ("tr", "доверительная область")]
    for ax, (key, title) in zip(axs, titles):
        kind = B[key + "_kind"]
        ax.imshow(np.where(kind >= 0, kind, np.nan), origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]),
                  cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest", alpha=0.85)
        sh = B[key + "_shares"]
        ax.set_title(title, fontsize=12)
        ax.text(0.03, 0.03, f"мин. {sh[0]:.0%} · сёдла {sh[1]:.0%}\nмакс. {sh[2]:.0%}", transform=ax.transAxes,
                fontsize=9.5, va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set(aspect="equal", xticks=[-5, 0, 5], yticks=[-5, 0, 5]); ax.grid(False)
    axs[0].set_ylabel("$x_2$"); axs[1].set_xlabel("$x_1$")
    fig.suptitle("Химмельблау, 300×300 стартов: синий — минимум, оранжевый — седло, красный — максимум", fontsize=11, y=0.99)
    save("07_basins_three.png")


# ================================================================ 08 CG на эллипсе и на цепи
def fig_cg_ellipse():
    Q = np.diag([1.0, 50.0]); x0 = np.array([50.0, 1.0])
    xc, kc, _ = cg(lambda v: Q @ v, np.zeros(2), x0)
    assert kc == 2 and np.allclose(xc, 0, atol=1e-10)
    # спуск с точным шагом (Л4 §4.4)
    x = x0.copy(); path = [x.copy()]; k = 0
    while np.linalg.norm(Q @ x) > 1e-8:                     # допуск как в Л4 (567 и 1106)
        g = Q @ x; x = x - (g @ g) / (g @ Q @ g) * g; path.append(x.copy()); k += 1
    path = np.array(path)
    assert k == 567, k
    # CG-путь
    x = x0.copy(); r = Q @ x; d = -r; cgp = [x.copy()]
    for _ in range(2):
        a = (r @ r) / (d @ Q @ d); x = x + a * d; rn = r + a * Q @ d; d = -rn + (rn @ rn) / (r @ r) * d; r = rn; cgp.append(x.copy())
    cgp = np.array(cgp)

    E, gE, H, b, v0, _ = make_chain(40)
    _, k_cg, hist = cg(lambda v: H @ v, -b, v0)
    k_gd, _, gnorm = gd_chain_count(40)
    _, ke, pe, _, _ = bfgs(E, gE, v0, line="exact", Q=H)
    gl = []
    rs = minimize(E, v0, jac=gE, method="CG", options={"gtol": 1e-6, "maxiter": 100_000},
                  callback=lambda xk: gl.append(np.linalg.norm(gE(xk))))
    assert k_cg == 20 and ke == 20 and k_gd == 10575 and 240 <= rs.nit <= 300, (k_cg, ke, k_gd, rs.nit)

    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    X, Y = np.meshgrid(np.linspace(-5, 55, 400), np.linspace(-1.6, 1.6, 300))
    ax.contour(X, Y, 0.5 * (X ** 2 + 50 * Y ** 2), levels=np.geomspace(0.5, 1300, 12), colors=GRAY, linewidths=0.6)
    ax.plot(path[:80, 0], path[:80, 1], "-", color=ORANGE, lw=1.2, label=f"спуск, точный шаг: {k}")
    ax.plot(cgp[:, 0], cgp[:, 1], "o-", color=AQUA, lw=2.5, ms=8, label=f"CG: {kc}")
    ax.plot(0, 0, "*", color=RED, ms=16, zorder=8)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title=r"$\frac{1}{2}(x_1^2+50x_2^2)$ из $(50,1)$")
    ax.legend(loc="upper right", fontsize=10)
    ax = axs[1]
    ks = np.unique(np.geomspace(1, k_gd, 300).astype(int))
    ax.loglog(ks, [gnorm(k) for k in ks], color=ORANGE, label=f"спуск $1/L$: {k_gd:,}".replace(",", " "))
    ax.loglog(np.arange(1, len(gl) + 1), gl, color=VIOLET, lw=1.4, label=f"нелинейный CG SciPy: {rs.nit}")
    ge = [np.linalg.norm(gE(v)) for v in pe]
    ax.loglog(np.arange(1, len(ge) + 1), ge, color=BLUE, lw=3, alpha=0.5, label=f"BFGS, точный шаг: {ke}")
    ax.loglog(np.arange(1, len(hist) + 1), hist, "--", color=AQUA, lw=2, label=f"CG: {k_cg}")
    ax.axhline(1e-6, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="итерация $k$", ylabel=r"$\|\nabla E\|$", title="цепь $N=40$, 80 переменных")
    ax.legend(loc="lower left", fontsize=9.5)
    save("08_cg_ellipse.png")
    return k, rs.nit


# ================================================================ 09 CG на цепи: масштабирование
def fig_cg_chain():
    Ns = [10, 20, 40, 80, 160, 400]
    gd_counts, cg_counts, kappas, hists = [], [], [], {}
    for N in Ns:
        kg, kap, _ = gd_chain_count(N)
        _, _, H, b, v0, _ = make_chain(N)
        _, kc, hist = cg(lambda v: H @ v, -b, v0)
        gd_counts.append(kg); cg_counts.append(kc); kappas.append(kap); hists[N] = hist
        assert kc == N // 2, (N, kc)
    assert gd_counts[2] == 10575 and gd_counts[4] == 155943 and gd_counts[5] == 937503, gd_counts
    assert abs(kappas[2] - 681) < 1 and abs(kappas[5] - 65170) < 5, kappas
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.1))
    ax = axs[0]
    ax.loglog(Ns, gd_counts, "o-", color=ORANGE, label="спуск, шаг $1/L$ ($\\sim N^2$)")
    ax.loglog(Ns, cg_counts, "s-", color=AQUA, label="CG ($=N/2$)")
    ax.loglog(Ns, np.sqrt(kappas), ":", color=GRAY, label=r"$\sqrt{\kappa}$")
    ax.loglog(Ns, np.ones(len(Ns)), "^-", color=RED, label="Ньютон (1)")
    for N, kg, kc in zip(Ns, gd_counts, cg_counts):
        if N in (40, 400):
            ax.text(N * 1.08, kg * 0.55, f"{kg:,}".replace(",", " "), fontsize=10, color=ORANGE)
            ax.text(N * 1.08, kc * 0.55, f"{kc}", fontsize=10, color="#0c7a54")
    ax.set(xlabel="число грузов $N$", ylabel="итераций до $10^{-6}$", title="цепь: итерации от $N$")
    ax.legend(loc="upper left", fontsize=9.5)
    ax = axs[1]
    for N, col in [(40, BLUE), (160, VIOLET), (400, AQUA)]:
        h = hists[N]
        ax.semilogy(np.arange(len(h)), h, color=col, label=f"$N={N}$: {len(h) - 1} итераций")
    ax.axhline(1e-6, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="итерация CG", ylabel=r"$\| H v_k+b\|$", title="CG: обрыв на $N/2$")
    ax.legend(loc="lower left", fontsize=10)
    save("09_cg_chain.png")
    return gd_counts, cg_counts


# ================================================================ 10 Newton-CG в 1000 переменных
def fig_newton_cg():
    n = 1000
    x0 = np.full(n, 0.5)
    x, k, nmv, hist = newton_cg(rosen_nd, rosen_der, rosen_hess_prod, x0)
    assert (k, nmv) == (47, 225) and rosen_nd(x) < 1e-15, (k, nmv, rosen_nd(x))
    r_ncg = minimize(rosen_nd, x0, jac=rosen_der, hessp=rosen_hess_prod, method="Newton-CG", options={"xtol": 1e-8})
    gl = [np.linalg.norm(rosen_der(x0))]
    r_lb = minimize(rosen_nd, x0, jac=rosen_der, method="L-BFGS-B", options={"gtol": 1e-6, "ftol": 0.0, "maxiter": 10_000},
                    callback=lambda xk: gl.append(np.linalg.norm(rosen_der(xk))))
    assert (r_ncg.nit, r_ncg.nhev) == (54, 291) and 80 <= r_lb.nit <= 95, (r_ncg.nit, r_ncg.nhev, r_lb.nit)
    gn = np.array([h[0] for h in hist]); inner = np.array([h[1] for h in hist])
    fig, axs = plt.subplots(1, 2, figsize=(8.9, 4.0))
    ax = axs[0]
    ax.semilogy(np.arange(len(gn)), gn, "o-", color=BLUE, ms=4, label=f"Newton-CG (наш): {k}")
    ax.semilogy(np.arange(len(gl)), gl, "-", color=ORANGE, label=f"L-BFGS-B: {r_lb.nit}")
    ax.axhline(1e-6, color=GRAY, ls=":", lw=1)
    ax.set(xlabel="итерация", ylabel=r"$\|\nabla f\|$", title="Розенброк, $n=1000$, старт $0.5\\cdot\\mathbf{1}$")
    ax.legend(loc="lower left", fontsize=10)
    ax = axs[1]
    ax.bar(np.arange(len(inner) - 1), inner[:-1], color=AQUA, width=0.8)
    ax.set(xlabel="внешняя итерация", ylabel="итераций CG внутри",
           title=f"{nmv} произведений $\\nabla^2 f\\,v$, матрицы нет")
    save("10_newton_cg.png")
    return k, nmv, r_ncg.nit, r_ncg.nhev, r_lb.nit


# ================================================================ main
if __name__ == "__main__":
    t_start = time.perf_counter()
    print("рисую в", IMG)
    R = runs()
    B = basins_all()
    k_gd400, k_cg400 = fig_hooks(R, B)
    fig_atan_newton(R)
    fig_review(R)
    arm, wol, swol = fig_armijo_wolfe()
    stat, floor = fig_bfgs_wolfe()
    kl, kt, xl = fig_saddle_trap()
    pnorm, lam2 = fig_tr_subproblem()
    fig_tr_rosenbrock(R)
    fig_basins_three(B)
    k_ell, k_ncg_chain = fig_cg_ellipse()
    gd_counts, cg_counts = fig_cg_chain()
    ncg = fig_newton_cg()
    kt_, _, _, rej, kdl, kc = R["tr_rosen"]
    print(f"проверки: арктангенс {R['atan_pure'][-1]:.3g}, с Армихо {len(R['atan_damped']) - 1} итераций; "
          f"модель обещала {R['atan_promise'][1]:.3f}, получили {R['atan_promise'][2]:.3f}; порог {R['atan_thr']:.4f};")
    print(f"          Армихо α ≤ {arm[1]:.3f}, Вольфе {wol[0]:.3f}–{wol[1]:.3f}, сильный до {swol[1]:.3f};")
    print(f"          Химмельблау s^T y <= 0: Армихо {stat['armijo']}, Вольфе {stat['wolfe']}; пол округления {floor:.2e};")
    print(f"          ловушка: Армихо → седло {xl.round(3)} за {kl}, область → (3,2) за {kt};")
    print(f"          доверительная область на Розенброке: точный {kt_} (отвергнуто {rej}), dogleg {kdl}, Коши {kc};")
    print(f"          SciPy: {R['scipy_rosen']};")
    print(f"          CG: эллипс 2 против {k_ell}; цепь — CG {cg_counts}, спуск {gd_counts}; нелинейный CG SciPy {k_ncg_chain};")
    print(f"          Newton-CG n=1000: наш {ncg[0]} итераций / {ncg[1]} Hv, SciPy {ncg[2]} / {ncg[3]}, L-BFGS-B {ncg[4]}")
    print(f"готово за {time.perf_counter() - t_start:.1f} с")
