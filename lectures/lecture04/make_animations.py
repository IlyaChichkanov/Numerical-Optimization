"""GIF-анимации к конспекту лекции 4 -> папка img/ (коммитятся в репозиторий).

Запуск:  uv run python lectures/lecture04/make_animations.py

Две анимации на одной и той же функции f(x) = 1/2 (x_1^2 + kappa x_2^2):
  10_valley_stretch.gif — «долина растягивается»: kappa растёт от 1 до 50, шаг 1/lambda_max;
  11_step_sweep.gif     — «ползунок шага»: kappa = 10, alpha пробегает от 0.2/lambda_max до 2.3/lambda_max.
Палитра и оформление — те же, что в make_figures.py; числа проверяются ассертами.
"""

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

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

FPS, DPI, MAX_BYTES = 12, 80, 2 * 1024 ** 2


def arrow(ax, p, q, color, lw=2, ms=14, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=6, **kw))


# ---------------------------------------------------------------- модель: f = 1/2 (x1^2 + kappa x2^2)
def grad(x, kappa):
    return np.array([x[0], kappa * x[1]])


def gd_const(x0, kappa, alpha, tol=1e-6, maxit=200_000):
    """Градиентный спуск с постоянным шагом: x <- x - alpha * grad f. Возвращает (путь, число итераций)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x, kappa)
        if np.linalg.norm(g) <= tol:
            return np.array(path), k
        x = x - alpha * g; path.append(x.copy())
    return np.array(path), maxit


def contours(ax, kappa, xlim, ylim, levels):
    X, Y = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 200))
    ax.contour(X, Y, 0.5 * (X ** 2 + kappa * Y ** 2), levels=levels, colors=GRAY, linewidths=0.7, alpha=0.85)
    ax.plot(0, 0, "*", color=RED, ms=12, zorder=7)


def save_gif(anim, name):
    path = IMG / name
    t0 = time.perf_counter()
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(anim._fig)
    size = os.path.getsize(path)
    assert size <= MAX_BYTES, f"{name}: {size / 1024 ** 2:.2f} МБ > 2 МБ — уменьшите число кадров или dpi"
    print(f"   {name}: {size / 1024:.0f} КБ, кадров {anim._save_count}, {time.perf_counter() - t0:.1f} с")
    return path


# ================================================================ 10 «долина растягивается»: kappa 1 -> 50
def anim_valley_stretch(n_frames=60, hold=8):
    kappas = np.concatenate([np.full(4, 1.0), np.geomspace(1, 50, n_frames), np.full(hold, 50.0)])
    xlim, ylim = (-5, 5), (-1.6, 1.6)
    levels = 0.98 * 1.6 ** np.arange(-3, 7)  # фиксированы: видно, как эллипсы сжимаются по x2
    x0, c = np.array([-4.0, 1.2]), 1.4       # c — длина стрелки по x1; концы стрелок лежат на уровне f = c^2/2 = 0.98
    n_show = 25

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.88)

    def draw(i):
        kappa = kappas[i]
        ax.cla()
        contours(ax, kappa, xlim, ylim, levels)
        # собственные направления гессиана diag(1, kappa): длины пропорциональны 1/sqrt(lambda)
        arrow(ax, (0, 0), (c, 0), AQUA, ms=12)
        arrow(ax, (0, 0), (0, c / np.sqrt(kappa)), VIOLET, ms=12)
        ax.annotate("$\\lambda=1$", (c, 0), xytext=(4, -4), textcoords="offset points", color=AQUA, va="top", fontsize=9)
        ax.annotate(f"$\\lambda=\\kappa={kappa:.3g}$", (0, c / np.sqrt(kappa)), xytext=(6, 0), textcoords="offset points",
                    color=VIOLET, va="center", fontsize=9)
        # градиентный спуск с шагом 1/lambda_max = 1/kappa
        path, k = gd_const(x0, kappa, 1 / kappa)
        p = path[:n_show + 1]
        ax.plot(p[:, 0], p[:, 1], "-o", color=BLUE, lw=1.4, ms=4, zorder=5)
        ax.plot(*x0, "o", color=BLUE, ms=7, zorder=6)
        ax.text(0.02, 0.04, f"шаг $\\alpha=1/\\lambda_{{\\max}}=1/\\kappa$, показаны первые {n_show} итераций",
                transform=ax.transAxes, fontsize=9, color=GRAY)
        ax.set(xlim=xlim, ylim=ylim, ylabel="$x_2$",
               xlabel="$x_1$   (оси в разных масштабах)",
               title=f"$\\kappa={kappa:.3g}$,  итераций до $\\Vert\\nabla f\\Vert\\leq10^{{-6}}$: {k}")
        return []

    anim = FuncAnimation(fig, draw, frames=len(kappas), blit=False)
    return anim


# ================================================================ 11 «ползунок шага»: kappa = 10, alpha 0.2..2.3 / lambda_max
def anim_step_sweep(n_frames=71, hold=8):
    kappa, lmax, lmin = 10.0, 10.0, 1.0
    alphas = np.linspace(0.2 / lmax, 2.3 / lmax, n_frames)   # шаг сетки 0.003: кадр 60 — ровно 2/lambda_max
    alphas = np.concatenate([alphas, np.full(hold, alphas[-1])])
    xlim, ylim = (-4.6, 4.6), (-3, 3)
    levels = np.geomspace(0.1, 40, 10)
    x0, n_show = np.array([4.0, 1.0]), 20
    a_best, a_edge = 2 / (lmax + lmin), 2 / lmax
    a_grid = np.linspace(0, 2.3 / lmax, 400)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.14, top=0.84, wspace=0.22)

    def draw(i):
        a = alphas[i]
        m_slow, m_fast = abs(1 - a * lmin), abs(1 - a * lmax)
        diverge = a > a_edge + 1e-12
        # --- слева: линии уровня и первые 20 итераций
        ax.cla()
        contours(ax, kappa, xlim, ylim, levels)
        path, _ = gd_const(x0, kappa, a, tol=0.0, maxit=n_show)
        p = path.copy()
        outside = (np.abs(p[:, 1]) > ylim[1]) | (np.abs(p[:, 0]) > xlim[1])
        p[outside] = np.nan                       # точки вне области не рисуем
        ax.plot(p[:, 0], p[:, 1], "-o", color=BLUE, lw=1.4, ms=4, zorder=5)
        ax.plot(*x0, "o", color=BLUE, ms=7, zorder=6)
        title = f"$\\alpha={a * lmax:.2f}\\cdot(1/\\lambda_{{\\max}})$" + (" — расходимость" if diverge else "")
        title += f"\nмножитель медленной компоненты {m_slow:.3f}, быстрой {m_fast:.2f}"
        ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$")
        ax.set_title(title, color=RED if diverge else INK)
        ax.text(0.02, 0.04, f"$\\kappa={kappa:g}$, старт $(4,1)$, первые {n_show} итераций",
                transform=ax.transAxes, fontsize=9, color=GRAY)
        # --- справа: множители компонент как функции шага
        bx.cla()
        bx.plot(a_grid, np.abs(1 - a_grid * lmin), color=BLUE)
        bx.plot(a_grid, np.abs(1 - a_grid * lmax), color=ORANGE)
        bx.axhline(1, color=RED, ls="--", lw=1.2)
        bx.plot([a_edge, a_edge], [0, 1.42], color=GRAY, ls=":", lw=1.2)
        bx.axvline(a, color=INK, lw=1, alpha=0.6)
        bx.plot(a, m_slow, "o", color=BLUE, ms=8, zorder=6)
        bx.plot(a, m_fast, "o", color=ORANGE, ms=8, zorder=6)
        bx.text(0.1, 0.62, "медленная: $|1-\\alpha\\cdot1|$", color=BLUE, ha="center", fontsize=9)
        bx.text(0.1, 0.38, "быстрая: $|1-\\alpha\\cdot10|$", color=ORANGE, ha="center", fontsize=9)
        bx.annotate("лучший шаг $2/(\\lambda_{\\max}+\\lambda_{\\min})$", (a_best, abs(1 - a_best)), xytext=(0.008, 1.5),
                    fontsize=9, va="center", arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
        bx.text(a_edge, 1.46, "граница\n$2/\\lambda_{\\max}$", ha="center", va="bottom", fontsize=9, color=INK)
        bx.set(xlim=(0, 0.245), ylim=(0, 1.72), xlabel="шаг $\\alpha$", ylabel="множитель компоненты за шаг",
               title="сходимость $\\Leftrightarrow$ обе ломаные ниже 1")
        return []

    anim = FuncAnimation(fig, draw, frames=len(alphas), blit=False)
    return anim


if __name__ == "__main__":
    # число из make_figures.py / demo04: kappa = 50, старт (50, 1), шаг 1/50 -> 1106 итераций до ||grad f|| <= 1e-8
    _, k = gd_const([50.0, 1.0], 50.0, 1 / 50, tol=1e-8)
    assert k == 1106, k
    # при alpha = 1/kappa координата x2 обнуляется за один шаг, x1 сжимается в (1 - 1/kappa)
    path, _ = gd_const([-4.0, 1.2], 50.0, 1 / 50)
    assert abs(path[1, 1]) < 1e-12 and np.isclose(path[1, 0], -4 * 0.98), path[1]

    t0 = time.perf_counter()
    print("анимации ->", IMG)
    for anim, name in ((anim_valley_stretch(), "10_valley_stretch.gif"), (anim_step_sweep(), "11_step_sweep.gif")):
        save_gif(anim, name)
    print(f"готово за {time.perf_counter() - t0:.1f} с")
