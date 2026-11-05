"""GIF к конспекту лекции 9 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture09/scripts/make_animations.py

  08_ggn_distance  §5: обобщённый Гаусс–Ньютон на окружности, точка a отодвигается от окружности:
                   |a| от 1.1 до 2.4; скорость сходимости = |a| - 1, при |a| >= 2 метод не сходится.
Код метода — тот же, что в make_figures.py (ggn_circle).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

IMG = Path(__file__).resolve().parent.parent / "img"
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8985", "#0b0b0b"
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False,
})

DIR = np.array([0.8, 0.6])          # направление на a; скорость зависит только от |a|
X0 = np.array([0.6, 0.8])           # старт на окружности, как на картинке 07


def ggn_circle(a, x0, iters):
    """Обобщённый Гаусс–Ньютон: в матрице ККТ B = I вместо H = (1 - lam) I. Точки x_0..x_iters."""
    x = np.array(x0, float); xs = [x.copy()]
    for _ in range(iters):
        J = x[None, :]
        K = np.block([[np.eye(2), -J.T], [J, np.zeros((1, 1))]])
        s = np.linalg.solve(K, -np.concatenate([x - a, [0.5 * (x @ x - 1)]]))
        x = x + s[:2]; xs.append(x.copy())
    return np.array(xs)


def angle_err(xs, a):
    return np.arctan2(xs[:, 1], xs[:, 0]) - np.arctan2(a[1], a[0])


def iterations(na, tol=1e-8, maxit=500):
    """Итераций до ошибки угла <= tol или None, если за maxit не сошлось."""
    e = np.abs(angle_err(ggn_circle(DIR * na, X0, maxit), DIR * na))
    hit = np.nonzero(e <= tol)[0]
    return int(hit[0]) if len(hit) else None


def main():
    # проверки: скорость = |a| - 1 и граница сходимости |a| = 2
    for na in (1.2, 1.5, 1.8):
        e = angle_err(ggn_circle(DIR * na, X0, 60), DIR * na)
        r = e[6:12] / e[5:11]
        assert np.allclose(r, -(na - 1), atol=1e-2), (na, r)
    assert (iterations(1.2), iterations(1.5), iterations(1.8)) == (11, 25, 75)
    assert iterations(2.0) is None and iterations(2.4) is None

    hold0, hold1 = 8, 10
    sweep = np.linspace(1.1, 2.4, 56)
    frames = [sweep[0]] * hold0 + list(sweep) + [sweep[-1]] * hold1
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))

    def draw(na):
        ax.clear(); bx.clear()
        a = DIR * na; xs = ggn_circle(a, X0, 12)
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(t), np.sin(t), color=INK, lw=2)
        ax.plot(xs[:, 0], xs[:, 1], "o-", color=ORANGE, ms=5, lw=1.4)
        ax.plot(*a, "o", color=INK, ms=7); ax.annotate("$a$", a, (6, 4), textcoords="offset points")
        ax.plot(*DIR, "*", color=BLUE, ms=15, zorder=5)
        ax.set_aspect("equal"); ax.set_xlim(-0.4, 2.2); ax.set_ylim(-0.6, 1.6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title("12 шагов метода")
        e = np.abs(angle_err(ggn_circle(a, X0, 40), a))
        bx.semilogy(np.arange(len(e)), np.maximum(e, 1e-16), "o-", color=ORANGE, ms=4, lw=1.4)
        bx.set_ylim(1e-12, 10); bx.set_xlim(-1, 41)
        bx.set_xlabel("итерация $k$"); bx.set_ylabel("ошибка угла, рад"); bx.set_title("Ошибка по итерациям")
        it = iterations(na)
        tail = f"{it} итераций" if it is not None else "не сходится"
        fig.suptitle(rf"$\|a\|={na:.2f}$: ошибка $\times{na - 1:.2f}$ за шаг, {tail}", fontsize=14)
        fig.tight_layout()

    anim = FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames))
    path = IMG / "08_ggn_distance.gif"
    anim.save(path, writer=PillowWriter(fps=12), dpi=80)
    size = path.stat().st_size / 1e6
    assert size <= 2.0, f"{path.name}: {size:.2f} МБ"
    print(f"   {path.name}: {len(frames)} кадров, {size:.2f} МБ")


if __name__ == "__main__":
    main()
