"""GIF-анимации к конспекту лекции 6 -> папка ../img/ (коммитятся в репозиторий).

Запуск:  uv run python lectures/lecture06/scripts/make_animations.py

Палитра, методы и задачи — из make_figures.py (тот же код, те же числа); размер — 900 px в ширину
при dpi 100, fps 12, файл не больше 2 МБ (проверяется ассертом). Последний кадр каждой анимации
совпадает с числами статичной картинки того же явления.

  11_backtracking   backtracking на phi(alpha) в точке x_1 Розенброка: 1, 1/2, 1/4 отвергнуты, 1/8 принят
  12_tr_breathes    доверительная область на Розенброке: круг Delta_k, пробный шаг, rho_k в заголовке
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (AQUA, BLUE, GRAY, INK, ORANGE, RED, X0_ROSEN, IMG, line_at_x1,  # noqa: E402
                          rosen, rosen_contour, rosen_grad, rosen_hess, trust_region)

FPS, DPI, MAX_BYTES = 12, 100, 2 * 1024 ** 2


def save_gif(anim, name):
    path = IMG / name
    anim.save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close()
    size = os.path.getsize(path)
    assert size <= MAX_BYTES, f"{name}: {size / 1024 ** 2:.2f} МБ > 2 МБ — уменьшите число кадров или dpi"
    print(f"   {name}: {size / 1024:.0f} КБ")


# ================================================================ 11 backtracking
def gif_backtracking():
    x1, p1 = line_at_x1()
    f0, d0 = rosen(x1), rosen_grad(x1) @ p1
    cands = [1.0, 0.5, 0.25, 0.125]
    phis = [rosen(x1 + a * p1) for a in cands]
    ok = [phi <= f0 + 1e-4 * a * d0 for a, phi in zip(cands, phis)]
    assert ok == [False, False, False, True] and abs(phis[0] - 1411.8) < 0.2, (ok, phis)

    al = np.linspace(0, 0.3, 600)
    phi = np.array([rosen(x1 + a * p1) for a in al])
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]
    ax.plot(al, phi, color=INK, lw=2.2)
    ax.plot(al, f0 + 1e-4 * al * d0, "--", color=ORANGE, lw=1.6)
    ax.set(xlim=(0, 0.3), ylim=(3.6, 9.6), xlabel=r"длина шага $\alpha$", ylabel=r"$\varphi(\alpha)$")
    ax.text(0.005, 4.9, "прямая Армихо", color=ORANGE, fontsize=11)
    ax2 = axs[1]
    rosen_contour(ax2, xlim=(-1.6, 1.2), ylim=(-3.4, 2.0))
    ray = np.array([x1 + a * p1 for a in np.linspace(0, 1, 50)])
    ax2.plot(ray[:, 0], ray[:, 1], ":", color=GRAY, lw=1.5)
    ax2.plot(*x1, "o", color=INK, ms=8, zorder=8)
    ax2.text(x1[0] - 0.35, x1[1] + 0.15, "$x_1$", fontsize=12)
    ax2.set_title("шаг Ньютона из $x_1$ и пробы")

    marks_l, marks_r, texts = [], [], []
    hold0, per, hold1 = 4, 12, 10
    frames = hold0 + per * len(cands) + hold1

    def update(i):
        j = min(max((i - hold0) // per, -1), len(cands) - 1)
        for a_ in marks_l + marks_r + texts:
            a_.remove()
        marks_l.clear(); marks_r.clear(); texts.clear()
        for m in range(j + 1):
            a, ph, good = cands[m], phis[m], ok[m]
            col = AQUA if good else RED
            if ph <= 9.6:
                marks_l.append(ax.plot(a, ph, "o", color=col, ms=12, mec=INK, zorder=7)[0])
            else:
                marks_l.append(ax.annotate("", (min(a, 0.297), 9.55), xytext=(min(a, 0.297) - 0.02, 8.6),
                                           arrowprops=dict(arrowstyle="->", color=RED, lw=2)))
            xm = x1 + a * p1
            marks_r.append(ax2.plot(*xm, "o" if good else "X", color=col, ms=11, mec=INK, zorder=9)[0])
        if j < 0:
            ax.set_title(r"$\varphi(0)=4.73$: пробуем полный шаг Ньютона $\alpha=1$")
        else:
            a, ph, good = cands[j], phis[j], ok[j]
            verdict = "принят" if good else "отказ, делим пополам"
            ax.set_title(f"$\\alpha={a:g}$: $\\varphi={ph:.4g}$ — {verdict}")
        return []

    ax.set_title(r"$\varphi(0)=4.73$: пробуем полный шаг Ньютона $\alpha=1$")
    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS)
    save_gif(anim, "11_backtracking.gif")


# ================================================================ 12 доверительная область дышит
def gif_tr_breathes():
    xt, kt, pt, ht = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step="exact")
    assert kt == 25 and sum(not h[4] for h in ht) == 3
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rosen_contour(ax, xlim=(-2.2, 2.2), ylim=(-1.2, 2.2))
    ax.set_aspect("equal")
    path_line, = ax.plot([], [], "o-", color=BLUE, ms=4, lw=1.6, zorder=6)
    circ = Circle((0, 0), 1, fill=False, lw=2, color=BLUE)
    ax.add_patch(circ)
    trial, = ax.plot([], [], "o", ms=10, mec=INK, zorder=9)
    step_line, = ax.plot([], [], "-", lw=2, zorder=8)
    hold0, per, hold1 = 4, 3, 10
    n = len(ht)
    frames = hold0 + per * n + hold1

    def update(i):
        k = min(max((i - hold0) // per, 0), n - 1)
        x, p, D, rho, acc = ht[k]
        done = np.array([h[0] for h in ht[:k + 1]] + ([x + p] if acc else []))
        path_line.set_data(done[:, 0], done[:, 1])
        circ.center = tuple(x); circ.set_radius(D)
        col = AQUA if acc else RED
        circ.set_edgecolor(col)
        trial.set_data([x[0] + p[0]], [x[1] + p[1]]); trial.set_color(col); trial.set_markeredgecolor(INK)
        step_line.set_data([x[0], x[0] + p[0]], [x[1], x[1] + p[1]]); step_line.set_color(col)
        if i >= hold0 + per * n:
            ax.set_title(f"готово: {kt} итераций, из них 3 пробных шага отвергнуты; $x^*=(1,1)$")
        else:
            verdict = "принят" if acc else "отвергнут"
            ax.set_title(f"итерация {k}: $\\Delta={D:.4g}$, $\\rho={rho:.2f}$ — шаг {verdict}")
        return path_line, circ, trial, step_line

    ax.set_title("итерация 0: $\\Delta=1$, $\\rho=1.00$ — шаг принят")       # место под заголовок до tight_layout
    plt.tight_layout()
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS)
    save_gif(anim, "12_tr_breathes.gif")


if __name__ == "__main__":
    print("рисую в", IMG)
    gif_backtracking()
    gif_tr_breathes()
