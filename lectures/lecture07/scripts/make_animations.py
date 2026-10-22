"""GIF-анимации к конспекту лекции 7 -> папка ../img/ (коммитятся в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture07/scripts/make_animations.py

Палитра, разности и задачи — из make_figures.py (тот же код, те же числа); 900 px в ширину при dpi 100,
fps 12, файл не больше 2 МБ (проверяется ассертом).

  09_ucurve   шаг h уменьшается от 1e-1 до 1e-15: слева секущая через 1 и 1 + h на графике sin x приближается
              к касательной, справа ошибка прямой разности рисует U-кривую; в заголовке h и ошибка
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import BLUE, GRAY, INK, ORANGE, RED, EPS, IMG, X_SIN, F_SIN, fd_forward  # noqa: E402

FPS, DPI, MAX_BYTES = 12, 100, 2 * 1024 ** 2


def save_gif(anim, name):
    path = IMG / name
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close()
    size = os.path.getsize(path)
    assert size <= MAX_BYTES, f"{name}: {size / 1024 ** 2:.2f} МБ > 2 МБ — уменьшите число кадров или dpi"
    print(f"   {name}: {size / 1024:.0f} КБ")


# ================================================================ 08 U-кривая
def gif_ucurve():
    x0 = X_SIN; ex = np.cos(1.0)
    hs = np.logspace(-1, -15, 71)                  # 71 шаг от 1e-1 вниз до 1e-15
    errs = np.array([abs(fd_forward(F_SIN, x0, 0, h) - ex) for h in hs])
    j_best = int(np.argmin(errs))
    assert 1e-9 < hs[j_best] < 1e-7, hs[j_best]
    g = np.sin                                     # f = sin x
    tt = np.linspace(0.7, 1.3, 300)
    hold = 6                                       # кадров задержки в начале и в конце
    frames = [0] * hold + list(range(len(hs))) + [len(hs) - 1] * hold

    fig, axs = plt.subplots(1, 2, figsize=(9, 4.1), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axs[0]
    ax.plot(tt, g(tt), color=INK, lw=2, label="$f(x)=\\sin x$")
    ax.plot(tt, g(x0[0]) + ex * (tt - x0[0]), "--", color=GRAY, lw=1.4, label="касательная")
    sec, = ax.plot([], [], color=ORANGE, lw=2, label="секущая")
    pts, = ax.plot([], [], "o", color=ORANGE, ms=7, zorder=7)
    ax.plot(x0[0], g(x0[0]), "o", color=INK, ms=6, zorder=7)
    ax.set(xlim=(tt[0], tt[-1]), ylim=(0.6, 1.0), xlabel="$x$", ylabel="$f$", title="Секущая и касательная в $x=1$")
    ax.legend(loc="upper right", fontsize=10)
    ax = axs[1]
    curve, = ax.loglog([], [], color=BLUE, lw=1.8)
    dot, = ax.plot([], [], "o", color=RED, ms=8, zorder=7)
    for h0, txt in ((np.sqrt(EPS), r"$\sqrt{\varepsilon}$"),):
        ax.axvline(h0, color=GRAY, ls=":", lw=1.2); ax.text(h0 * 1.5, 3e-10, txt, color=GRAY, fontsize=12)
    ax.set(xlim=(1e-16, 1), ylim=(1e-10, 1e0), xlabel="шаг $h$", ylabel="ошибка прямой разности", title="Ошибка по шагу")
    title = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))

    def update(k):
        h = hs[k]
        xa = x0[0] + h
        sec_slope = (g(xa) - g(x0[0])) / h
        sec.set_data(tt, g(x0[0]) + sec_slope * (tt - x0[0]))
        pts.set_data([x0[0], xa], [g(x0[0]), g(xa)])
        curve.set_data(hs[:k + 1], errs[:k + 1])
        dot.set_data([h], [errs[k]])
        state = "ошибка усечения падает" if k < j_best else ("минимум" if k == j_best else "ошибка округления растёт")
        title.set_text(f"$h = 10^{{{np.log10(h):.1f}}}$,  ошибка производной {errs[k]:.1e} — {state}")
        return sec, pts, curve, dot, title

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    save_gif(anim, "09_ucurve.gif")


if __name__ == "__main__":
    gif_ucurve()
