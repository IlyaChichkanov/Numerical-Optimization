"""GIF-анимации к конспекту лекции 5 -> папка ../img/ (коммитятся в репозиторий).

Запуск:  uv run python lectures/lecture05/scripts/make_animations.py

  13_parabola_rides.gif — «парабола едет по функции»: Ньютон на f(x) = ln cosh x; один параметр —
                          старт x_0 пробегает 0.3 -> 1.3; до порога 1.0886 метод сходится, после — расходится;
  14_bfgs_learns.gif    — «BFGS учится эллипсу»: BFGS на Розенброке из (-1.2, 1), параметр — номер итерации;
                          эллипс модели B_k против эллипса истинного гессиана в той же точке.
Палитра — та же, что в make_figures.py; размер — 900 px в ширину при dpi 100, шрифт 13,
чтобы при <img width="900"> подписи не уменьшались; размер файла проверяется (не больше 2 МБ).
"""

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import brentq

IMG = Path(__file__).resolve().parent.parent / "img"
IMG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7")
GRAY, INK = "#8a8985", "#0b0b0b"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 100,
})

FPS, DPI, MAX_BYTES = 12, 100, 2 * 1024 ** 2


def save_gif(anim, name):
    path = IMG / name
    t0 = time.perf_counter()
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(anim._fig)
    size = os.path.getsize(path)
    assert size <= MAX_BYTES, f"{name}: {size / 1024 ** 2:.2f} МБ > 2 МБ — уменьшите число кадров или dpi"
    print(f"   {name}: {size / 1024:.0f} КБ, кадров {anim._save_count}, {time.perf_counter() - t0:.1f} с")
    return path


# ================================================================ 09 парабола едет по ln cosh x
def lncosh_newton(x0, maxit=30, tol=1e-8):
    """Ньютон для min ln cosh x: x <- x - sinh(2x)/2. Возвращает (итераты, сошёлся ли, число итераций)."""
    xs = [float(x0)]
    for k in range(maxit):
        if abs(np.tanh(xs[-1])) <= tol:
            return np.array(xs), True, k
        xs.append(xs[-1] - 0.5 * np.sinh(2 * xs[-1]))
        if abs(xs[-1]) > 50:
            return np.array(xs), False, k + 1
    return np.array(xs), False, maxit


def anim_parabola_rides(n_frames=48, hold=8):
    f = lambda x: np.log(np.cosh(x)); df = np.tanh; d2f = lambda x: 1 / np.cosh(x) ** 2
    thr = brentq(lambda x: 2 * x - 0.5 * np.sinh(2 * x), 0.5, 1.5)
    assert abs(thr - 1.0886) < 1e-3, thr
    starts = np.concatenate([np.full(4, 0.3), np.linspace(0.3, 1.3, n_frames), np.full(hold, 1.3)])
    xlim, ylim = (-3.2, 4.6), (-0.45, 3.6)
    t = np.linspace(*xlim, 500)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.86)

    def draw(i):
        x0 = starts[i]
        xs, ok, k = lncosh_newton(x0)
        ax.cla()
        ax.plot(t, f(t), color=INK, lw=2.2)
        ax.plot(0, 0, "*", color=RED, ms=12, zorder=7)
        color = AQUA if ok else RED
        for j in range(min(3, len(xs) - 1)):            # три параболы-модели
            xk = xs[j]
            par = f(xk) + df(xk) * (t - xk) + 0.5 * d2f(xk) * (t - xk) ** 2
            ax.plot(t, par, color=color, lw=1.4, alpha=0.95 - 0.3 * j, ls="-" if j == 0 else "--")
        pts = xs[:5].copy(); pts[np.abs(pts) > xlim[1]] = np.nan
        ax.plot(pts, f(pts), "o-", color=color, ms=6, lw=1, zorder=6)          # точки за краем — NaN, не рисуются
        for j, xj in enumerate(pts[:4]):
            if np.isfinite(xj):
                ax.annotate(f"$x_{j}$", (xj, f(xj)), xytext=(4, 6), textcoords="offset points", color=color)
        ax.axvline(thr, color=GRAY, lw=0.8, ls=":"); ax.axvline(-thr, color=GRAY, lw=0.8, ls=":")
        ax.text(thr + 0.05, 3.3, f"порог ${thr:.4f}$", color=GRAY)
        status = f"сошёлся, итераций: {k}" if ok else "РАСХОДИТСЯ: вершина параболы всё дальше"
        ax.set(xlim=xlim, ylim=ylim, xlabel="$x$", ylabel="$f(x)=\\ln\\cosh x$")
        ax.set_title(f"старт $x_0={x0:.3f}$ — {status}", color=INK if ok else RED)
        ax.text(0.02, 0.04, "параболы — модели в $x_0$, $x_1$, $x_2$; следующая точка — вершина",
                transform=ax.transAxes, fontsize=11, color=GRAY)
        return []

    anim = FuncAnimation(fig, draw, frames=len(starts), blit=False)
    return anim


# ================================================================ 10 BFGS учится эллипсу (Розенброк)
def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def rosen_hess(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200.0]])


def bfgs_history(f, grad, x0, tol=1e-6, maxit=10_000, c=1e-4):
    """Тот же BFGS, что в make_figures.py / l5helpers.py, но с историей матриц H_k."""
    x = np.asarray(x0, float).copy(); n = len(x)
    H = np.eye(n); path = [x.copy()]; Hs = [H.copy()]; g = grad(x)
    for k in range(maxit):
        if np.linalg.norm(g) <= tol:
            return np.array(path), Hs, k
        p = -H @ g
        a, fx = 1.0, f(x)
        while f(x + a * p) > fx + c * a * (g @ p):
            a /= 2
        s = a * p; x_new = x + s; g_new = grad(x_new); y = g_new - g
        if s @ y > 1e-12:
            rho = 1.0 / (s @ y); I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, g = x_new, g_new; path.append(x.copy()); Hs.append(H.copy())
    return np.array(path), Hs, maxit


def ellipse(B, center, semi_major=0.45, n=200):
    """Кривая {center + p : p^T B p = c}, c подобрано так, чтобы большая полуось была semi_major."""
    lam, V = np.linalg.eigh(B)
    lam = np.maximum(lam, 1e-12)
    c = semi_major ** 2 * lam.min()
    th = np.linspace(0, 2 * np.pi, n)
    pts = (V * np.sqrt(c / lam)) @ np.vstack([np.cos(th), np.sin(th)])
    return center[0] + pts[0], center[1] + pts[1]


def anim_bfgs_learns(hold=8):
    path, Hs, k_total = bfgs_history(rosen, rosen_grad, [-1.2, 1.0])
    assert k_total == 34 and np.allclose(path[-1], 1, atol=1e-6), (k_total, path[-1])
    frames = np.concatenate([np.zeros(4, int), np.arange(len(path)), np.full(hold, len(path) - 1)])
    xlim, ylim = (-1.6, 1.6), (-0.6, 1.8)
    X, Y = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    eig_B = np.array([np.linalg.eigvalsh(np.linalg.inv(H)) for H in Hs])
    eig_H = np.array([np.linalg.eigvalsh(rosen_hess(x)) for x in path])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.15, top=0.85, wspace=0.32)

    def draw(i):
        k = frames[i]
        xk = path[k]; Bk = np.linalg.inv(Hs[k]); Hk = rosen_hess(xk)
        ax.cla()
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3.3, 14), colors=GRAY, linewidths=0.7, alpha=0.8, antialiased=False)
        ax.plot(1, 1, "*", color=RED, ms=12, zorder=7)
        ax.plot(path[:k + 1, 0], path[:k + 1, 1], "-o", color=AQUA, ms=3, lw=1.2, alpha=0.8)
        ex, ey = ellipse(Hk, xk); ax.plot(ex, ey, color=INK, lw=1.4, label="истинный гессиан $\\nabla^2f(x_k)$")
        ex, ey = ellipse(Bk, xk); ax.plot(ex, ey, color=ORANGE, lw=2.2, label="модель BFGS $B_k$")
        ax.plot(*xk, "o", color=ORANGE, ms=7, zorder=8)
        ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$")
        ax.set_title(f"итерация $k={k}$ из {k_total}:  $\\Vert x_k-x^*\\Vert={np.linalg.norm(xk - 1):.1e}$")
        ax.legend(loc="lower right")
        ax.text(0.02, 0.04, "сравнивайте форму и наклон эллипсов",
                transform=ax.transAxes, fontsize=11, color=GRAY)
        bx.cla()
        ks = np.arange(len(path))
        bx.plot(ks, eig_H[:, 0], color=INK, lw=1.2, ls="--"); bx.plot(ks, eig_H[:, 1], color=INK, lw=1.2, ls="--", label="$\\lambda(\\nabla^2f(x_k))$")
        bx.plot(ks[:k + 1], eig_B[:k + 1, 0], color=ORANGE, lw=2); bx.plot(ks[:k + 1], eig_B[:k + 1, 1], color=ORANGE, lw=2, label="$\\lambda(B_k)$")
        bx.plot([k, k], eig_B[k], "o", color=ORANGE, ms=6)
        bx.set_yscale("log")
        bx.set(xlim=(0, len(path) - 1), ylim=(3e-2, 3e4), xlabel="итерация $k$", ylabel="собственные числа",
               title="кривизна: модель и функция")
        bx.legend(loc="upper left")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    return anim


if __name__ == "__main__":
    xs, ok, k = lncosh_newton(1.08); assert ok, (xs, k)
    xs, ok, k = lncosh_newton(1.09); assert not ok, (xs, k)
    t0 = time.perf_counter()
    print("анимации ->", IMG)
    for anim, name in ((anim_parabola_rides(), "13_parabola_rides.gif"), (anim_bfgs_learns(), "14_bfgs_learns.gif")):
        save_gif(anim, name)
    print(f"готово за {time.perf_counter() - t0:.1f} с")
