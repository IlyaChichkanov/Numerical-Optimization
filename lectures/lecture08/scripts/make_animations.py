"""GIF-анимации к конспекту лекции 8 -> папка ../img/ (коммитятся).

Запуск:  uv run python lectures/lecture08/scripts/make_animations.py

  10_licq_breaks.gif   точка скользит по клюву x2 = x1^(2/3) к острию: grad g укорачивается и ложится горизонтально,
                       оценка lambda растёт (§3.1); числа те же, что в make_figures.py (cusp_grad)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

IMG = Path(__file__).resolve().parent.parent / "img"
BLUE, ORANGE, AQUA, RED, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8985", "#0b0b0b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "lines.linewidth": 2, "legend.frameon": False})
FPS, DPI, HOLD = 12, 80, 8


def cusp_grad(x1):
    x2 = x1 ** (2 / 3)
    gf, gg = np.array([0.0, 1.0]), np.array([-2 * x1, 3 * x2 ** 2])
    return gg, gf @ gg / (gg @ gg), np.degrees(np.arccos(gf @ gg / np.linalg.norm(gg)))


def licq_breaks():
    xs = np.logspace(np.log10(0.6), np.log10(0.002), 64)
    frames = [xs[0]] * HOLD + list(xs) + [xs[-1]] * HOLD
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    t = np.linspace(-0.7, 0.7, 500)
    ax.plot(t, np.abs(t) ** (2 / 3), color=INK)
    ax.plot(0, 0, "o", color=RED, ms=7); ax.annotate("$x^\\ast$", (0, 0), (6, -14), textcoords="offset points", color=RED)
    ax.set_aspect("equal"); ax.set_xlim(-0.75, 0.95); ax.set_ylim(-0.12, 0.95); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    pt, = ax.plot([], [], "o", color=BLUE, ms=7, zorder=7)
    arrows = []

    def draw(x1):
        for a in arrows:
            a.remove()
        arrows.clear()
        gg, lam, ang = cusp_grad(x1)
        p = np.array([x1, x1 ** (2 / 3)])
        pt.set_data([p[0]], [p[1]])
        for q, col in ((p + np.array([0, 0.3]), RED), (p + 0.3 * gg, AQUA)):
            arrows.append(ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=col, lw=2, zorder=6)))
        ax.set_title(f"$x_1={x1:.3f}$:  $\\|\\nabla g\\|={np.linalg.norm(gg):.3f}$,  угол ${ang:.0f}^\\circ$,  $\\hat\\lambda={lam:.1f}$")
        return [pt] + arrows

    ax.annotate(r"$\nabla f$", (0.62, 0.86), color=RED); ax.annotate(r"$\nabla g$", (0.62, 0.78), color=AQUA)
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path = IMG / "10_licq_breaks.gif"
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    size = path.stat().st_size
    assert size <= 2_000_000, f"{path.name}: {size / 1e6:.2f} МБ > 2 МБ"
    print(f"   {path.name}: {len(frames)} кадров, {size / 1e6:.2f} МБ")


if __name__ == "__main__":
    licq_breaks()
