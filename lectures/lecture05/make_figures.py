"""Иллюстрации к конспекту лекции 5 -> папка img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture05/make_figures.py

Скрипт самодостаточен. Палитра и оформление — те же, что в лекциях 1-4.
Методы (gd, newton, newton_damped, bfgs, gauss_newton) определены здесь тем же
кодом, что в build_demo05.py; числа конспекта считаются в скрипте и
проверяются ассертами.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize

IMG = Path(__file__).resolve().parent / "img"
IMG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7")
GRAY, INK = "#8a8985", "#0b0b0b"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 150,
})


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(IMG / name, bbox_inches="tight")
    plt.close()
    print("  ", name)


def arrow(ax, p, q, color, lw=2, ms=14, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=6, **kw))


# ---------------------------------------------------------------- методы (те же, что в demo05 / l5helpers.py)
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


def rosen_contour(ax, xlim=(-2, 2), ylim=(-1, 3)):
    X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3.3, 14), colors=GRAY, linewidths=0.7, alpha=0.8)
    ax.plot(1, 1, "*", color=RED, ms=13, zorder=7)
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$")


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
    """Данные задачи 2(б) ДЗ 2: y = 2 sin(3t + 0.7) + шум, 60 точек, тот же генератор и порядок вызовов."""
    data_rng = np.random.default_rng(2026)
    N, omega = 60, 3.0
    t = np.sort(data_rng.uniform(0, 2 * np.pi, N))
    y = 2.0 * np.sin(3.0 * t + 0.7) + 0.2 * data_rng.standard_normal(N)
    starts = data_rng.uniform(-5, 5, (20, 3))
    return t, y, starts


def sine_problem(t, y):
    """resid, jac, f, grad, hess для модели x1 sin(x2 t + x3)."""
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
        S = np.zeros((3, 3))                            # сумма r_i * hess(r_i)
        S[0, 1] = S[1, 0] = r @ (t * c); S[0, 2] = S[2, 0] = r @ c
        S[1, 1] = r @ (-x[0] * t ** 2 * s); S[1, 2] = S[2, 1] = r @ (-x[0] * t * s); S[2, 2] = r @ (-x[0] * s)
        return J.T @ J + S
    return resid, jac, f, grad, hess


X0_SINE = np.array([1.0, 3.0, 0.0])


def himmelblau_newton_basins(n=300, lim=6.0, maxit=30):
    """Чистый Ньютон на функции Химмельблау из каждой точки сетки n x n на [-lim, lim]^2.
    Возвращает сетку, индекс стационарной точки (или -1), число итераций, список точек и их типы."""
    xs = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, xs); X = X.copy(); Y = Y.copy()
    its = np.full(X.shape, -1)
    for k in range(maxit):
        a, b = X ** 2 + Y - 11, X + Y ** 2 - 7
        gx, gy = 4 * X * a + 2 * b, 2 * a + 4 * Y * b
        hxx, hxy, hyy = 12 * X ** 2 + 4 * Y - 42, 4 * X + 4 * Y, 4 * X + 12 * Y ** 2 - 26
        det = hxx * hyy - hxy ** 2
        with np.errstate(all="ignore"):
            dx = (hyy * gx - hxy * gy) / det; dy = (hxx * gy - hxy * gx) / det
        done = (np.hypot(gx, gy) < 1e-8) & (its < 0)
        its[done] = k
        active = its < 0
        X = np.where(active, X - dx, X); Y = np.where(active, Y - dy, Y)
    pts = np.c_[X.ravel(), Y.ravel()]
    ok = its.ravel() >= 0
    uniq = np.unique(pts[ok].round(3), axis=0)
    idx = np.full(X.size, -1)
    for j, p in enumerate(uniq):
        idx[ok & np.all(np.isclose(pts, p, atol=1e-2), axis=1)] = j
    kinds = []
    for p in uniq:
        hxx, hxy, hyy = 12 * p[0] ** 2 + 4 * p[1] - 42, 4 * p[0] + 4 * p[1], 4 * p[0] + 12 * p[1] ** 2 - 26
        lam = np.linalg.eigvalsh([[hxx, hxy], [hxy, hyy]])
        kinds.append("min" if lam.min() > 0 else ("max" if lam.max() < 0 else "saddle"))
    return xs, idx.reshape(X.shape), its, uniq, kinds


# ================================================================ общие прогоны (числа конспекта)
def runs_rosen():
    xg, kg, pg = gd(rosen, rosen_grad, X0_ROSEN)
    xn, kn, pn = newton(rosen_grad, rosen_hess, X0_ROSEN)
    xd, kd, pd = newton_damped(rosen, rosen_grad, rosen_hess, X0_ROSEN)
    xb, kb, pb, nfb = bfgs(rosen, rosen_grad, X0_ROSEN)
    assert abs(kg - 13756) < 140, kg
    assert kn == 7 and np.allclose(xn, 1, atol=1e-8), (kn, xn)
    assert kd == 22 and np.allclose(xd, 1, atol=1e-8), (kd, xd)
    assert kb == 34 and nfb == 54 and np.allclose(xb, 1, atol=1e-6), (kb, nfb, xb)
    en = np.linalg.norm(pn - 1, axis=1)
    assert np.allclose(en[:7], [2.2, 2.208, 4.182, 0.4796, 0.05597, 9.62e-6, 1.85e-11], rtol=0.02), en
    assert np.allclose(pn[2], [0.763, -3.175], atol=2e-3), pn[2]
    return dict(gd=(kg, pg), newton=(kn, pn), damped=(kd, pd), bfgs=(kb, pb, nfb))


def runs_sine():
    t, y, starts = sine_data()
    resid, jac, f, grad, hess = sine_problem(t, y)
    xgn, kgn, pgn = gauss_newton(resid, jac, X0_SINE)
    xgd, kgd, pgd = gd(f, grad, X0_SINE)
    xnw, knw, pnw = newton(grad, hess, X0_SINE, tol=1e-6)
    assert kgn == 7 and np.allclose(xgn, [2.1014, 2.9967, 0.7003], atol=1e-3), (kgn, xgn)
    assert abs(kgd - 731) < 40 and np.allclose(xgd, xgn, atol=1e-3), (kgd, xgd)
    assert abs(xnw[0]) < 1e-6 and knw < 20, (knw, xnw)          # Ньютон пришёл в точку с нулевой амплитудой
    assert abs(f(xgn) - 1.1393) < 1e-3, f(xgn)
    lamJ = np.linalg.eigvalsh(jac(xgn).T @ jac(xgn)); lamH = np.linalg.eigvalsh(hess(xgn))
    assert np.allclose(lamJ, [17.57, 31.05, 1935.6], rtol=0.01) and np.allclose(lamH, [17.89, 31.12, 1927.3], rtol=0.01), (lamJ, lamH)
    return dict(t=t, y=y, starts=starts, resid=resid, jac=jac, f=f, grad=grad, hess=hess,
                gn=(kgn, pgn, xgn), gd=(kgd, pgd), newton=(knw, pnw, xnw))


# ================================================================ 00 три задачи-крючка
def fig_hooks(R, S):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw=dict(width_ratios=[1.1, 1.2, 1]))
    ax = axes[0]
    kg, pg = R["gd"]; kn, pn = R["newton"]
    rosen_contour(ax, xlim=(-2.2, 2.2), ylim=(-3.5, 3.2))
    ax.plot(pg[::50, 0], pg[::50, 1], "-", color=BLUE, lw=1.2, label=f"спуск, backtracking: {kg} итераций")
    ax.plot(pn[:, 0], pn[:, 1], "-o", color=ORANGE, ms=5, lw=1.6, label=f"Ньютон: {kn} итераций")
    ax.annotate("итерация 2", pn[2], (pn[2, 0] + 0.15, pn[2, 1] + 0.15), fontsize=8.5, color=ORANGE)
    ax.plot(*X0_ROSEN, "o", color=INK, ms=6, zorder=7)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Розенброк: 13 756 против 7 — и выброс")

    ax = axes[1]
    t, y = S["t"], S["y"]; kgn, pgn, xgn = S["gn"]; kgd, _ = S["gd"]; knw, _, xnw = S["newton"]
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(t, y, "o", ms=3.5, color=INK, alpha=0.7, label="данные ДЗ 2")
    ax.plot(tt, xgn[0] * np.sin(xgn[1] * tt + xgn[2]), color=AQUA, lw=2.2,
            label=f"Гаусс–Ньютон: {kgn} итераций (спуск: {kgd})")
    ax.plot(tt, xnw[0] * np.sin(xnw[1] * tt + xnw[2]), color=RED, lw=2.2, ls="--",
            label=f"Ньютон: {knw} итераций, $x_1=0$")
    ax.set(xlabel="$t$", ylabel="$y$", ylim=(-3.2, 3.6), title="Подгонка синуса: $x_1\\sin(x_2t+x_3)$")
    ax.legend(loc="upper right", fontsize=8.5)

    ax = axes[2]
    energy, grad_E, H, b, v0, unpack = make_chain(40)
    yy, zz = unpack(np.linalg.solve(H, -b))
    ax.plot(yy, zz, "-o", ms=3, color=BLUE, label="минимум энергии")
    ax.plot([-2, 2], [1, 1], "s", color=INK, ms=6)
    ax.set(xlabel="$y$", ylabel="$z$", title="Цепь: спуск 10 575, Ньютон — один solve")
    ax.legend(loc="upper center", fontsize=8.5)
    save("00_hooks.png")


# ================================================================ 01 квадратичная модель в двух точках пути
def fig_quadratic_model(R):
    kn, pn = R["newton"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, k, xlim, ylim, title in (
        (axes[0], 0, (-2.2, 0.6), (-0.4, 2.4), "старт $x_0=(-1.2,\\,1)$: модель ещё похожа на функцию"),
        (axes[1], 1, (-2.2, 2.2), (-3.6, 2.6), "$x_1=(-1.18,\\,1.38)$: минимум модели — за пределами долины"),
    ):
        xk = pn[k]; g = rosen_grad(xk); Hk = rosen_hess(xk); pstar = -np.linalg.solve(Hk, g)
        X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3.3, 14), colors=GRAY, linewidths=0.7, alpha=0.8)
        P1, P2 = X - xk[0], Y - xk[1]
        M = rosen(xk) + g[0] * P1 + g[1] * P2 + 0.5 * (Hk[0, 0] * P1 ** 2 + 2 * Hk[0, 1] * P1 * P2 + Hk[1, 1] * P2 ** 2)
        mmin = rosen(xk) + 0.5 * g @ pstar
        ax.contour(X, Y, M, levels=mmin + np.geomspace(0.5, 400, 9), colors=ORANGE, linewidths=1.0, alpha=0.9)
        ax.plot(*xk, "o", color=INK, ms=7, zorder=7)
        arrow(ax, xk, xk + pstar, ORANGE, lw=2.2)
        ax.plot(*(xk + pstar), "s", color=ORANGE, ms=7, zorder=7)
        ax.plot(1, 1, "*", color=RED, ms=13, zorder=7)
        ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$", title=title)
    axes[0].text(0.02, 0.03, "серые — линии уровня $f$, оранжевые — модели $m_k$;\nквадрат — минимум модели = следующая точка",
                 transform=axes[0].transAxes, fontsize=8.5, color=GRAY)
    axes[1].annotate(f"$x_2=({pn[2, 0]:.2f},\\,{pn[2, 1]:.2f})$ — минимум модели,\nно $f$ здесь больше, чем в $x_1$", pn[2], (pn[2, 0] + 0.25, pn[2, 1] + 0.35), fontsize=9, color=ORANGE)
    save("01_quadratic_model.png")


# ================================================================ 02 ln cosh: парабола едет по функции
def lncosh_newton(x0, n=4):
    xs = [x0]
    for _ in range(n):
        xs.append(xs[-1] - 0.5 * np.sinh(2 * xs[-1]))
    return np.array(xs)


def fig_newton_1d():
    f = lambda x: np.log(np.cosh(x)); df = np.tanh; d2f = lambda x: 1 / np.cosh(x) ** 2
    seq_a, seq_b = lncosh_newton(0.9), lncosh_newton(1.2, 3)
    assert abs(seq_a[4]) < 1e-8 and abs(seq_a[3] + 1.557e-3) < 1e-5 and abs(seq_b[1] + 1.533) < 2e-3 and abs(seq_b[2] - 3.82) < 0.01 and abs(seq_b[3]) > 100, (seq_a, seq_b)
    # порог: x1 = -x0  <=>  x0 - sinh(2x0)/2 = -x0
    from scipy.optimize import brentq
    thr = brentq(lambda x: 2 * x - 0.5 * np.sinh(2 * x), 0.5, 1.5)
    assert abs(thr - 1.0886) < 1e-3, thr
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    t = np.linspace(-2.6, 4.2, 600)
    for ax, seq, nsteps, title, color in ((axes[0], seq_a, 2, "старт $0.9$: четыре шага — и $10^{-8}$", AQUA),
                                          (axes[1], seq_b, 2, "старт $1.2$: вершина параболы перелетает всё дальше", RED)):
        ax.plot(t, f(t), color=INK, lw=2.2, label="$f(x)=\\ln\\cosh x$")
        for i in range(nsteps):
            xk = seq[i]; par = f(xk) + df(xk) * (t - xk) + 0.5 * d2f(xk) * (t - xk) ** 2
            ax.plot(t, par, color=color, lw=1.3, alpha=0.9 - 0.3 * i, ls="-" if i == 0 else "--",
                    label="парабола-модель в $x_k$" if i == 0 else None)
            ax.plot(xk, f(xk), "o", color=color, ms=7, zorder=7)
            ax.annotate(f"$x_{i}$", (xk, f(xk)), (xk + 0.08, f(xk) + 0.25), fontsize=9.5, color=color)
            ax.plot([seq[i + 1], seq[i + 1]], [0, f(seq[i + 1])], ":", color=color, lw=1)
        ax.plot(seq[nsteps], f(seq[nsteps]), "o", color=color, ms=7, zorder=7)
        ax.annotate(f"$x_{nsteps}={seq[nsteps]:.2f}$", (seq[nsteps], f(seq[nsteps])),
                    (seq[nsteps] + 0.15, f(seq[nsteps]) + (0.75 if nsteps == 2 and abs(seq[nsteps]) < 1 else 0.3)), fontsize=9.5, color=color)
        ax.plot(0, 0, "*", color=RED, ms=13, zorder=7)
        ax.set(xlim=(-2.6, 4.2), ylim=(-0.3, 3.3), xlabel="$x$", title=title)
        ax.legend(loc="upper left", fontsize=9)
    axes[1].text(0.03, 0.70, f"порог: $x_0\\approx{thr:.4f}$ —\nтам 2-цикл $x_1=-x_0$", transform=axes[1].transAxes, fontsize=9.5, color=RED)
    save("02_newton_1d.png")
    return thr


# ================================================================ 03 четыре метода: логарифм ошибки
def fig_rates_four(R):
    kg, pg = R["gd"]; kn, pn = R["newton"]; kd, pd = R["damped"]; kb, pb, nfb = R["bfgs"]
    err = {k: np.linalg.norm(p - 1, axis=1) + 1e-17 for k, p in (("gd", pg), ("newton", pn), ("damped", pd), ("bfgs", pb))}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    ax.plot(np.arange(41), np.log10(err["gd"][:41]), color=BLUE, label=f"спуск, backtracking ({kg} итераций)")
    ax.plot(np.arange(len(pd)), np.log10(err["damped"]), "-s", ms=3.5, color=VIOLET, label=f"Ньютон с backtracking ({kd})")
    ax.plot(np.arange(len(pb)), np.log10(err["bfgs"]), "-^", ms=3.5, color=AQUA, label=f"BFGS ({kb})")
    ax.plot(np.arange(len(pn)), np.log10(err["newton"]), "-o", ms=4.5, color=ORANGE, label=f"чистый Ньютон ({kn})")
    ax.set(xlabel="итерация $k$", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$", xlim=(0, 40), ylim=(-12.5, 1.2),
           title="Первые 40 итераций: прямая, загиб, обрыв")
    ax.legend(loc="lower left", fontsize=8.5)
    ax = axes[1]
    ax.plot(np.arange(1, len(pg) + 1), np.log10(err["gd"]), color=BLUE)
    ax.plot(np.arange(1, len(pd) + 1), np.log10(err["damped"]), "-s", ms=3, color=VIOLET)
    ax.plot(np.arange(1, len(pb) + 1), np.log10(err["bfgs"]), "-^", ms=3, color=AQUA)
    ax.plot(np.arange(1, len(pn) + 1), np.log10(err["newton"]), "-o", ms=4, color=ORANGE)
    ax.set_xscale("log")
    ax.set(xlabel="итерация $k$ (лог. шкала)", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$", ylim=(-12.5, 1.2),
           title="Все итерации: спуску нужно на три порядка больше")
    save("03_rates_four.png")


# ================================================================ 04 бассейны Ньютона на Химмельблау
def fig_basins():
    t0 = time.perf_counter()
    xs, idx, its, uniq, kinds = himmelblau_newton_basins()
    n_min = sum(k == "min" for k in kinds); n_sad = sum(k == "saddle" for k in kinds); n_max = sum(k == "max" for k in kinds)
    assert (n_min, n_sad, n_max) == (4, 4, 1), kinds
    ok = idx >= 0
    shares = {kind: np.mean([kinds[j] == kind for j in idx[ok]]) for kind in ("min", "saddle", "max")}
    assert ok.mean() > 0.999 and 0.6 < shares["min"] < 0.7 and 0.25 < shares["saddle"] < 0.36, (ok.mean(), shares)
    med = np.median(its[ok])
    # цвета: минимумы — оттенки синего/зелёного, сёдла — оттенки оранжевого, максимум — красный
    palette = []
    mins = ["#2a78d6", "#1baf7a", "#4a3aa7", "#5fb8e8"]; sads = ["#eb6834", "#eda100", "#f3a35f", "#c9a227"]
    im = isd = 0
    for kd in kinds:
        if kd == "min": palette.append(mins[im]); im += 1
        elif kd == "saddle": palette.append(sads[isd]); isd += 1
        else: palette.append(RED)
    cmap = ListedColormap(palette)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.imshow(np.where(idx >= 0, idx, np.nan), origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]), cmap=cmap,
              vmin=-0.5, vmax=len(uniq) - 0.5, interpolation="nearest", alpha=0.85)
    X, Y = np.meshgrid(xs, xs)
    F = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2
    ax.contour(X, Y, F, levels=np.geomspace(1, 400, 9), colors="white", linewidths=0.5, alpha=0.6)
    marker = {"min": ("*", 16, "white"), "saddle": ("s", 9, "white"), "max": ("X", 11, "white")}
    for p, kd in zip(uniq, kinds):
        mk, ms, mec = marker[kd]
        ax.plot(*p, mk, ms=ms, color=INK, mec=mec, mew=1.2, zorder=7)
    ax.plot([], [], "*", ms=12, color=INK, mec="white", label=f"минимумы — {shares['min']:.0%} стартов")
    ax.plot([], [], "s", ms=8, color=INK, mec="white", label=f"сёдла — {shares['saddle']:.0%}")
    ax.plot([], [], "X", ms=9, color=INK, mec="white", label=f"максимум — {shares['max']:.0%}")
    ax.legend(loc="upper left", fontsize=9, facecolor="white", frameon=True, framealpha=0.85)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal",
           title=f"Химмельблау: куда приходит чистый Ньютон из каждой точки (медиана — {med:.0f} итераций)")
    ax.grid(False)
    save("04_basins.png")
    print(f"      бассейны: {time.perf_counter() - t0:.1f} с, сошлось {ok.mean():.4%}, доли {shares}")
    return shares, med


# ================================================================ 05 где шаг Ньютона — не спуск
def fig_descent_or_not(R):
    kn, pn = R["newton"]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    xlim, ylim = (-2.2, 2.2), (-1.2, 4.2)
    rosen_contour(ax, xlim, ylim)
    xs = np.linspace(*xlim, 400)
    ax.fill_between(xs, xs ** 2 + 0.005, ylim[1], color=RED, alpha=0.08, zorder=0)
    ax.plot(xs, xs ** 2 + 0.005, color=RED, lw=1, ls="--")
    # где шаг Ньютона идёт ВВЕРХ (g^T p > 0): считаем на сетке
    X, Y = np.meshgrid(np.linspace(*xlim, 441), np.linspace(*ylim, 541))
    G1 = -2 * (1 - X) - 400 * X * (Y - X ** 2); G2 = 200 * (Y - X ** 2)
    Hxx = 2 - 400 * Y + 1200 * X ** 2; Hxy = -400 * X; Hyy = 200.0
    det = Hxx * Hyy - Hxy ** 2
    with np.errstate(all="ignore"):
        P1 = -(Hyy * G1 - Hxy * G2) / det; P2 = -(Hxx * G2 - Hxy * G1) / det
    uphill = (G1 * P1 + G2 * P2) > 0
    share_up, share_ind = uphill.mean(), (det < 0).mean()
    assert 0.003 < share_up < 0.012 and 0.4 < share_ind < 0.55 and np.all(det[uphill] < 0), (share_up, share_ind)
    ax.contourf(X, Y, uphill.astype(float), levels=[0.5, 1.5], colors=[RED], alpha=0.45, zorder=1)
    ax.text(-2.1, 3.75, "светлая зона: гессиан индефинитен ($x_2>x_1^2+0.005$) —\nмодель седловая, шаг идёт в её седло, обычно всё ещё вниз\nтёмная полоса над дном: шаг Ньютона идёт ВВЕРХ ($g^\\top p>0$)",
            color=RED, fontsize=9)
    pts = [np.array(p) for p in ((-1.7, 1.2), (-1.06, 1.18), (1.0, 2.6), (0.5, -0.6), (1.5, 1.2))]
    L = 0.75
    for x in pts:
        g = rosen_grad(x); H = rosen_hess(x); p = -np.linalg.solve(H, g)
        down = g @ p < 0
        arrow(ax, x, x - L * g / np.linalg.norm(g), BLUE, lw=1.8, ms=12)
        arrow(ax, x, x + L * p / np.linalg.norm(p), ORANGE if down else RED, lw=2.2, ms=12)
        ax.plot(*x, "o", color=INK, ms=6, zorder=7)
        ax.annotate(f"$g^\\top p{'<' if down else '>'}0$: {'спуск' if down else 'ПОДЪЁМ'}", x, (x[0] + 0.1, x[1] - 0.38), fontsize=8.5,
                    color=ORANGE if down else RED, fontweight="normal" if down else "bold")
    assert rosen_grad(pts[1]) @ (-np.linalg.solve(rosen_hess(pts[1]), rosen_grad(pts[1]))) > 0
    ax.plot([], [], color=BLUE, lw=2, label="антиградиент $-\\nabla f$ (нормирован)")
    ax.plot([], [], color=ORANGE, lw=2, label="шаг Ньютона $p=-H^{-1}g$ (нормирован): спуск")
    ax.plot([], [], color=RED, lw=2, label="шаг Ньютона: подъём")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("Розенброк: шаг Ньютона гарантированно спуск только там, где $\\nabla^2f\\succ0$")
    save("05_descent_or_not.png")
    return share_up, share_ind


# ================================================================ 06 секущая и BFGS учится эллипсу
def fig_secant_bfgs():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    F = lambda x: np.tanh(x); dF = lambda x: 1 / np.cosh(x) ** 2       # F = f' для f = ln cosh
    x0, x1 = 1.3, 0.8
    t = np.linspace(-1.1, 1.8, 400)
    ax.plot(t, F(t), color=INK, lw=2.2, label="$F(x)=f'(x)$ — ищем нуль")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.plot(t, F(x1) + dF(x1) * (t - x1), color=ORANGE, lw=1.6, label="касательная в $x_1$: Ньютон, нужна $f''$")
    slope = (F(x1) - F(x0)) / (x1 - x0)
    ax.plot(t, F(x1) + slope * (t - x1), color=AQUA, lw=1.6, ls="--", label="секущая через $x_0,x_1$: $f''\\approx y/s$")
    ax.plot([x0, x1], [F(x0), F(x1)], "o", color=INK, ms=7, zorder=7)
    xn_newton, xn_secant = x1 - F(x1) / dF(x1), x1 - F(x1) / slope
    ax.plot(xn_newton, 0, "s", color=ORANGE, ms=8, zorder=7); ax.plot(xn_secant, 0, "s", color=AQUA, ms=8, zorder=7)
    ax.annotate("$x_0$", (x0, F(x0)), (x0 + 0.05, F(x0) - 0.18), fontsize=10)
    ax.annotate("$x_1$", (x1, F(x1)), (x1 + 0.05, F(x1) - 0.18), fontsize=10)
    ax.text(0.04, 0.80, "$s=x_1-x_0$,  $y=F(x_1)-F(x_0)$,  $f''\\approx y/s$", transform=ax.transAxes, fontsize=9.5, color=AQUA)
    ax.set(xlim=(-1.1, 1.8), ylim=(-0.9, 1.1), xlabel="$x$", title="Секущее уравнение: кривизна из двух градиентов")
    ax.legend(loc="lower right", fontsize=8.5)

    ax = axes[1]
    Q = np.diag([1.0, 50.0])
    xq, kq, pq, _ = bfgs(lambda x: 0.5 * x @ Q @ x, lambda x: Q @ x, [50.0, 1.0], tol=1e-10, Q=Q)
    # восстанавливаем H_k по шагам, чтобы нарисовать эллипсы модели
    Hs = [np.eye(2)]
    for k in range(len(pq) - 1):
        s = pq[k + 1] - pq[k]; y = Q @ s; rho = 1 / (s @ y); I = np.eye(2)
        Hs.append((I - rho * np.outer(s, y)) @ Hs[-1] @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s))
    assert kq == 2 and np.allclose(Hs[2], np.linalg.inv(Q), atol=1e-10), (kq, Hs[2])
    th = np.linspace(0, 2 * np.pi, 300); circ = np.c_[np.cos(th), np.sin(th)]
    def ellipse(B):       # {p : p^T B p = 1}
        L = np.linalg.cholesky(np.linalg.inv(B)); return circ @ L.T
    e_true = ellipse(Q)
    for k, (Hk, col, lab) in enumerate(zip(Hs, (GRAY, VIOLET, ORANGE), ("$B_0=I$: круг", "$B_1$: кривизна вдоль первого шага", "$B_2=Q$: совпадение"))):
        e = ellipse(np.linalg.inv(Hk))
        ax.plot(e[:, 0], e[:, 1], color=col, lw=2.2 if k < 2 else 2.8, ls="-" if k < 2 else "--", label=lab)
    ax.plot(e_true[:, 0], e_true[:, 1], color=INK, lw=1.2, label="истинный гессиан $Q$")
    ax.set(xlabel="$p_1$", ylabel="$p_2$", aspect="equal", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3),
           title=f"$\\frac{{1}}{{2}}(x_1^2+50x_2^2)$: эллипсы $p^\\top B_kp=1$ — BFGS за {kq} шага учит $Q^{{-1}}$")
    ax.legend(loc="upper right", fontsize=8.5)
    save("06_secant_bfgs.png")
    return Hs


# ================================================================ 07 Гаусс–Ньютон на синусе
def fig_gn_sine(S):
    t, y = S["t"], S["y"]; kgn, pgn, xgn = S["gn"]; kgd, pgd = S["gd"]; knw, pnw, xnw = S["newton"]; f = S["f"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(t, y, "o", ms=3.5, color=INK, alpha=0.7, label="данные")
    ax.plot(tt, xgn[0] * np.sin(xgn[1] * tt + xgn[2]), color=AQUA, lw=2.4, label=f"Гаусс–Ньютон, {kgn} итераций: $f^*={f(xgn):.4f}$")
    ax.plot(tt, X0_SINE[0] * np.sin(X0_SINE[1] * tt + X0_SINE[2]), color=GRAY, lw=1.2, ls=":", label="старт $(1,\\,3,\\,0)$")
    ax.plot(tt, xnw[0] * np.sin(xnw[1] * tt + xnw[2]), color=RED, lw=2, ls="--", label=f"Ньютон, {knw} итераций: $x_1=0$, $f={f(xnw):.1f}$")
    ax.set(xlabel="$t$", ylabel="$y$", ylim=(-3.2, 4.0), title="Из одного старта — к разным стационарным точкам")
    ax.legend(loc="upper right", fontsize=8.5)
    ax = axes[1]
    egn = np.linalg.norm(pgn - xgn, axis=1) + 1e-17; egd = np.linalg.norm(pgd - xgn, axis=1) + 1e-17
    ax.plot(np.arange(1, len(egd) + 1), np.log10(egd), color=BLUE, label=f"спуск, backtracking: {kgd} итераций")
    ax.plot(np.arange(1, len(egn) + 1), np.log10(egn), "-o", ms=5, color=AQUA, label=f"Гаусс–Ньютон: {kgn}")
    ax.set_xscale("log")
    ax.set(xlabel="итерация $k$ (лог. шкала)", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$", ylim=(-7, 0.6),
           title="Невязки малы — Гаусс–Ньютон почти как Ньютон")
    ax.legend(loc="lower left", fontsize=8.5)
    save("07_gn_sine.png")


# ================================================================ 08 цепь: итерации от N
def fig_chain_methods():
    Ns = (10, 20, 40, 80, 160)
    rows = []
    t_total = time.perf_counter()
    for N in Ns:
        f, grad, H, b, v0, _ = make_chain(N)
        L = np.linalg.eigvalsh(H)[-1]
        t0 = time.perf_counter(); _, kg, _ = gd(f, grad, v0, alpha=1 / L, maxit=400_000); tg = time.perf_counter() - t0
        _, kb, _, _ = bfgs(f, grad, v0, Q=H)
        r = minimize(f, v0, jac=grad, method="L-BFGS-B", options={"gtol": 1e-6, "maxiter": 100_000, "maxcor": 5})
        rs = minimize(f, v0, jac=grad, method="BFGS", options={"gtol": 1e-6, "maxiter": 100_000})
        t0 = time.perf_counter(); np.linalg.solve(H, -b); tn = time.perf_counter() - t0
        rows.append((N, kg, kb, r.nit, tg, tn, rs.nit))
        print(f"      цепь N={N:3d}: спуск {kg} ({tg:.1f} с), BFGS точный шаг {kb}, L-BFGS m=5 {r.nit}, scipy BFGS (Вольфе) {rs.nit}, solve {tn * 1e3:.1f} мс")
    rows = np.array(rows, dtype=float)
    kg40 = rows[2, 1]; kb = rows[:, 2]
    assert abs(kg40 - 10575) < 110, kg40
    assert np.array_equal(kb, [5, 10, 20, 40, 80]), kb
    assert 60 <= rows[2, 3] <= 90, rows[2, 3]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(Ns, rows[:, 1], "o-", color=BLUE, label="градиентный спуск, шаг $1/L$  ($\\propto\\kappa\\sim N^2$)")
    ax.plot(Ns, rows[:, 3], "^-", color=AQUA, label="L-BFGS, память 5 (SciPy)")
    ax.plot(Ns, rows[:, 2], "s-", color=VIOLET, label="BFGS, точный шаг  ($=N/2$)")
    ax.plot(Ns, np.ones(len(Ns)), "*-", color=ORANGE, ms=10, label="Ньютон: один `solve`")
    for N, kg, kb_, kl, *_ in rows:
        ax.annotate(f"{int(kg):,}".replace(",", " "), (N, kg), (0, 6), textcoords="offset points", ha="center", fontsize=8, color=BLUE)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set(xlabel="число грузов $N$ (переменных — $2N$)", ylabel="итераций до $\\Vert\\nabla E\\Vert\\leq10^{-6}$",
           xticks=Ns, xticklabels=[str(n) for n in Ns], title="Цепь: чем больше кривизны знает метод, тем меньше итераций")
    ax.legend(loc="upper left", fontsize=8.5)
    save("08_chain_methods.png")
    print(f"      цепь всего: {time.perf_counter() - t_total:.1f} с")
    return rows


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("рисую в", IMG)
    R = runs_rosen()
    S = runs_sine()
    fig_hooks(R, S)
    fig_quadratic_model(R)
    thr = fig_newton_1d()
    fig_rates_four(R)
    shares, med = fig_basins()
    share_up, share_ind = fig_descent_or_not(R)
    Hs = fig_secant_bfgs()
    fig_gn_sine(S)
    rows = fig_chain_methods()
    kg, _ = R["gd"]; kn, _ = R["newton"]; kd, _ = R["damped"]; kb, _, nfb = R["bfgs"]
    print(f"проверки: Розенброк — спуск {kg}, Ньютон {kn}, демпфированный {kd}, BFGS {kb} ({nfb} вызовов f);")
    print(f"          синус — Гаусс–Ньютон {S['gn'][0]}, спуск {S['gd'][0]}, Ньютон {S['newton'][0]} (x1 = {S['newton'][2][0]:.1e});")
    print(f"          Розенброк: гессиан индефинитен на {share_ind:.0%} окна, шаг Ньютона вверх — на {share_up:.1%};")
    print(f"          порог ln cosh {thr:.4f}; Химмельблау: минимумы {shares['min']:.1%}, сёдла {shares['saddle']:.1%}, максимум {shares['max']:.1%}")
    print(f"готово за {time.perf_counter() - t_start:.1f} с")
