"""GIF к конспекту лекции 9 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture09/scripts/make_animations.py

  06_ggn_height  §5: обобщённый Гаусс–Ньютон на дороге x2 = x1^2, дом a = (0, h) поднимается над вершиной:
                 h от 0.05 до 0.48. Ответ — вершина, lambda* = -h, множитель ошибки q = 2h;
                 при h -> 0.5 (радиус кривизны дороги в вершине) q -> 1 и метод почти останавливается.
Код метода — тот же, что в make_figures.py (newton_lagrange, road_ggn_kkt).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import newton_lagrange, road_ggn_kkt  # noqa: E402  один код метода на лекцию

IMG = Path(__file__).resolve().parent.parent / "img"
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b"
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False,
})

X0 = [1.0, 0.0]                     # старт, как в §2.1


def iters_word(n):
    """«32 итерации», «25 итераций», «1 итерация»."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} итерация"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} итерации"
    return f"{n} итераций"


def run(h, maxit=2000):
    """Итерации обобщённого Гаусса–Ньютона для дома (0, h)."""
    return newton_lagrange(road_ggn_kkt(np.array([0.0, h])), X0, [0.0], maxit=maxit)


def main():
    # проверки: множитель ошибки 2h и число итераций из таблицы §5
    for h in (0.2, 0.3, 0.4):
        hist = run(h)
        e = np.array([np.linalg.norm(t[0]) for t in hist])                         # ответ — вершина (0, 0)
        assert np.allclose(e[8:12] / e[7:11], 2 * h, atol=0.02), (h, e[8:12] / e[7:11])
    assert len(run(0.3)) - 1 == 43

    hold0, hold1 = 8, 10
    sweep = np.linspace(0.05, 0.48, 56)
    frames = [sweep[0]] * hold0 + list(sweep) + [sweep[-1]] * hold1
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))

    def draw(h):
        ax.clear(); bx.clear()
        hist = run(h)
        P = np.array([t[0] for t in hist])
        t = np.linspace(-0.8, 1.15, 200)
        ax.plot(t, t ** 2, color=INK, lw=2)
        ax.plot(P[:13, 0], P[:13, 1], "o-", color=ORANGE, ms=5, lw=1.4)
        ax.plot(0, h, "s", color=INK, ms=9); ax.annotate("дом", (0, h), (8, 4), textcoords="offset points")
        ax.plot(0, 0, "*", color=BLUE, ms=15, zorder=5)
        ax.set_aspect("equal"); ax.set_xlim(-0.8, 1.2); ax.set_ylim(-0.2, 1.1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title("12 шагов метода")
        e = np.array([np.linalg.norm(p) for p in P[:61]])
        bx.semilogy(np.arange(len(e)), np.maximum(e, 1e-16), "o-", color=ORANGE, ms=4, lw=1.4)
        bx.set_ylim(1e-11, 3); bx.set_xlim(-1, 61)
        bx.set_xlabel("итерация $k$"); bx.set_ylabel(r"$\|x^{(k)}-x^\ast\|$"); bx.set_title("Ошибка по итерациям")
        fig.suptitle(rf"дом на высоте $h={h:.2f}$: ошибка $\times{2 * h:.2f}$ за шаг, {iters_word(len(hist) - 1)}", fontsize=14)
        fig.tight_layout()

    anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames))
    path = IMG / "06_ggn_height.gif"
    anim.save(path, writer=PillowWriter(fps=12), dpi=80)
    size = path.stat().st_size / 1e6
    assert size <= 2.0, f"{path.name}: {size:.2f} МБ"
    print(f"   {path.name}: {len(frames)} кадров, {size:.2f} МБ")


if __name__ == "__main__":
    main()
