"""GIF-анимации к конспекту лекции 8 -> папка ../img/ (коммитятся).

Запуск:  uv run python lectures/lecture08/scripts/make_animations.py

  10_plane_touch.gif   плоскость x3 = c поднимается к верху сферы (§3.1, вид сбоку): окружность пересечения сжимается
                       в точку, угол между нормалями уходит в ноль, lambda_1 = 1 - sqrt5 / sqrt(1 - c^2) — в минус
                       бесконечность; числа те же, что в make_figures.py (sphere_plane)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch

IMG = Path(__file__).resolve().parent.parent / "img"
BLUE, AQUA, RED, VIOLET, GRAY, INK = "#2a78d6", "#1baf7a", "#e34948", "#4a3aa7", "#8a8985", "#0b0b0b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "lines.linewidth": 2, "legend.frameon": False})
FPS, DPI, HOLD = 12, 80, 8
S5 = np.sqrt(5.0)                                              # |(a1, a2)| для a = (2, 1, 1)


def plane_touch():
    t = np.linspace(0, 1, 64)
    cs = 1 - (1 - t) ** 2                                      # c от 0 до 1, у касания — медленнее
    frames = [cs[0]] * HOLD + list(cs) + [cs[-1]] * HOLD
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=INK, lw=2)          # сфера (вид сбоку)
    ax.plot(S5, 1.0, "o", color=INK, ms=6); ax.annotate("$a$", (S5, 1.0), (5, 5), textcoords="offset points")
    ax.set_aspect("equal"); ax.set_xlim(-1.45, 2.6); ax.set_ylim(-1.1, 1.7)
    ax.set_xlabel("вдоль $(a_1,a_2)$"); ax.set_ylabel("$x_3$")
    plane, = ax.plot([], [], color=GRAY, lw=2)
    chord, = ax.plot([], [], color=BLUE, lw=6, alpha=0.45, solid_capstyle="round")
    pt, = ax.plot([], [], "o", color=RED, ms=8, zorder=8)
    arrows = []
    ax.text(-1.4, 1.5, r"$\nabla g_1$", color=AQUA, fontsize=13); ax.text(-1.4, 1.25, r"$\nabla g_2$", color=VIOLET, fontsize=13)
    ax.text(-0.95, 1.5, r"$\nabla f$", color=RED, fontsize=13)

    def draw(c):
        for a in arrows:
            a.remove()
        arrows.clear()
        r = np.sqrt(max(1 - c ** 2, 0.0))                      # радиус окружности пересечения
        P = np.array([r, c])                                   # x* в координатах вида сбоку
        plane.set_data([-1.45, 2.6], [c, c]); chord.set_data([-r, r], [c, c]); pt.set_data([P[0]], [P[1]])
        dx = 0.0 if r > 1e-3 else 0.05                         # при касании нормали совпадают — разводим чуть-чуть
        g = np.array([r - S5, c - 1.0])                        # grad f = x* - a
        for p, q, col in ((P - [dx, 0], P - [dx, 0] + 0.55 * P / np.linalg.norm(P), AQUA),
                          (P + [dx, 0], P + [dx, 0] + np.array([0, 0.55]), VIOLET), (P, P + 0.35 * g, RED)):
            arrows.append(ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=col, lw=2.2, zorder=6)))
        if r > 1e-3:
            lam1 = 1 - S5 / r
            ax.set_title(f"$c={c:.3f}$:  угол между нормалями ${np.degrees(np.arccos(c)):.0f}^\\circ$,  $\\lambda_1={lam1:.1f}$")
        else:
            ax.set_title("$c=1$: нормали параллельны, подходящих $\\lambda$ нет")
        return [plane, chord, pt] + arrows

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    path = IMG / "10_plane_touch.gif"
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    size = path.stat().st_size
    assert size <= 2_000_000, f"{path.name}: {size / 1e6:.2f} МБ > 2 МБ"
    assert abs((1 - S5 / np.sqrt(1 - 0.9 ** 2)) + 4.1299) < 1e-3          # то же число, что в конспекте при c = 0.9
    print(f"   {path.name}: {len(frames)} кадров, {size / 1e6:.2f} МБ")


if __name__ == "__main__":
    plane_touch()
