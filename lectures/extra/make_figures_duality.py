"""Иллюстрации к заготовке duality.md -> папка lectures/extra/img/.

Запуск:  uv run python lectures/extra/make_figures_duality.py

Скрипт самодостаточен. Палитра и оформление — те же, что в лекциях 1-2.
Числа во всех «настоящих» примерах (не помеченных как качественная схема)
проверены отдельно в demo_duality.ipynb — здесь только рисунки.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.optimize import linprog, minimize

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


# ---------------------------------------------------------------- 01 слабая двойственность на пальцах
def fig_weak_duality_line():
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.set(xlim=(0, 10), ylim=(-1, 1), xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=GRAY, lw=1.5)
    ax.annotate("", xy=(9.7, 0), xytext=(9.3, 0), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    duals = [1.2, 2.4, 3.5, 4.0]      # d(mu) для нескольких mu >= 0 -- растут к d*
    primals = [7.6, 6.8, 6.1]         # f(x) для нескольких допустимых x -- убывают к p*
    dstar, pstar = 4.0, 6.1

    for u in duals:
        ax.plot(u, 0, "o", color=AQUA, ms=9, zorder=5)
    for t in primals:
        ax.plot(t, 0, "o", color=BLUE, ms=9, zorder=5)
    ax.plot(dstar, 0, "o", color=AQUA, ms=12, mec=INK, mew=1.3, zorder=6)
    ax.plot(pstar, 0, "s", color=BLUE, ms=12, mec=INK, mew=1.3, zorder=6)

    ax.annotate("", xy=(pstar, 0.32), xytext=(dstar, 0.32),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text((dstar + pstar) / 2, 0.42, "зазор двойственности $p^\\ast - d^\\ast \\geq 0$",
            ha="center", color=RED, fontsize=9.5)

    ax.text(dstar, -0.32, "$d^\\ast = \\max_\\mu d(\\mu)$", ha="center", color=AQUA, fontsize=9.5)
    ax.text(pstar, -0.32, "$p^\\ast = \\min f(x)$", ha="center", color=BLUE, fontsize=9.5)
    ax.text(0.3, -0.75, "значения $d(\\mu)$ для допустимых $\\mu \\geq 0$", color=AQUA, fontsize=8.5)
    ax.text(5.6, -0.75, "значения $f(x)$ для допустимых $x$", color=BLUE, fontsize=8.5)
    ax.set(title="слабая двойственность: любой $d(\\mu)$ снизу, любой $f(x)$ сверху — между ними зазор")
    save("01_weak_duality_line.png")


# ---------------------------------------------------------------- 02 семейство функций Лагранжа (x-пространство)
def fig_lagrangian_family():
    x = np.linspace(0.5, 5.5, 400)
    f = (x - 2) ** 2
    mus = [0, 1, 2, 3]
    colors = [GRAY, ORANGE, RED, VIOLET]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axvspan(3, 5.6, color=BLUE, alpha=0.10, lw=0)
    ax.plot(x, f, color=BLUE, lw=2.6, label="$f(x) = (x-2)^2$", zorder=5)
    for mu, col in zip(mus, colors):
        L = f - mu * (x - 3)
        ax.plot(x, L, color=col, lw=1.6, ls="--",
                label=f"$L(x,{mu}) = f(x) - {mu}(x-3)$" if mu else "$L(x,0) = f(x)$")
        xs = 2 + mu / 2
        ax.plot(xs, (xs - 2) ** 2 - mu * (xs - 3), "o", color=col, ms=6, zorder=6)
    ax.axvline(3, color=INK, lw=1.2)
    ax.text(3.05, 5.6, "$x \\geq 3$", fontsize=10)
    ax.plot(3, 1, "o", color=INK, ms=10, zorder=7)
    ax.annotate("$x^\\ast = 3,\\ f^\\ast = 1$", xy=(3, 1), xytext=(3.5, 2.4),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), fontsize=10)
    ax.set(xlim=(0.5, 5.5), ylim=(-1.5, 6.5), xlabel="$x$", ylabel="$f,\\ L$",
           title="семейство $L(x,\\mu) = f(x) - \\mu\\,(x-3)$: точка минимума сдвигается с $\\mu$")
    ax.legend(loc="upper left", fontsize=8.5)
    save("02_lagrangian_family.png")


# ---------------------------------------------------------------- 03 геометрия: множество G и опорные прямые
def fig_geometric_duality():
    u = np.linspace(-2.6, 2.6, 400)
    t = (u + 1) ** 2                      # G = {(u, t) : u = x - 3, t = f(x)}, x = u + 3

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.axvspan(0, 2.8, color=BLUE, alpha=0.08, lw=0, label="допустимо: $u \\geq 0$")
    ax.plot(u, t, color=INK, lw=2.4, label=r"$G = \{(h(x),\,f(x)) : x \in \mathbb{R}\}$")

    for mu, col in zip((0, 1, 2, 3), (GRAY, ORANGE, RED, VIOLET)):
        # опорная прямая t = mu*u + d(mu), d(mu) = -mu^2/4 + mu
        d = -mu ** 2 / 4 + mu
        uu = np.linspace(-2.6, 2.6, 10)
        ax.plot(uu, mu * uu + d, color=col, lw=1.4, ls="--",
                label=f"$t = {mu} u + d({mu})$" if mu else "$t = d(0)$")
        ax.plot(0, d, "o", color=col, ms=7, zorder=6)

    ax.plot(0, 1, "*", color=INK, ms=18, zorder=7, mec="white", mew=0.6)
    ax.annotate("$(0,\\,p^\\ast{=}1) = (0,\\,d^\\ast)$\nкасание при $\\mu^\\ast = 2$", xy=(0, 1),
                xytext=(-2.5, 4.4), arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), fontsize=9.5)
    ax.set(xlim=(-2.6, 2.6), ylim=(-1.2, 6.5), xlabel="$u = h(x) = x - 3$", ylabel="$t = f(x)$",
           title="опорные прямые $t = \\mu u + d(\\mu)$: $d(\\mu)$ — их пересечение с осью $u=0$")
    ax.legend(loc="lower right", fontsize=8, ncol=1)
    save("03_geometric_duality.png")


# ---------------------------------------------------------------- 04 дуальная функция d(mu)
def fig_dual_function():
    mu = np.linspace(0, 4, 400)
    d = -mu ** 2 / 4 + mu
    pstar = 1.0

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(pstar, color=BLUE, lw=1.4, ls=":")
    ax.text(0.05, pstar + 0.12, "$p^\\ast = 1$", color=BLUE, fontsize=10)
    ax.plot(mu, d, color=AQUA, lw=2.4, label="$d(\\mu) = -\\mu^2/4 + \\mu$")
    ax.fill_between(mu, d, pstar, color=RED, alpha=0.08)
    ax.plot(2, 1, "o", color=INK, ms=10, zorder=6)
    ax.annotate("$\\mu^\\ast = 2,\\ d^\\ast = p^\\ast = 1$", xy=(2, 1), xytext=(2.15, 0.25),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), fontsize=10)
    ax.text(0.15, -1.05, "$d(\\mu)$ вогнута везде — даже когда исходная задача невыпукла", fontsize=9, color=INK)
    ax.set(xlim=(0, 4), ylim=(-1.3, 1.6), xlabel="$\\mu$", ylabel="$d(\\mu)$",
           title="двойственная функция: слабая двойственность как зазор, сильная — как касание")
    ax.legend(loc="lower center", fontsize=9)
    save("04_dual_function.png")


# ---------------------------------------------------------------- 05 качественная схема: выпуклый и невыпуклый G
def fig_nonconvex_gap():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))

    u = np.linspace(-2.7, 2.7, 500)
    t_convex = 0.55 * u ** 2 + 0.15
    ax = axes[0]
    ax.axvspan(0, 2.9, color=BLUE, alpha=0.08, lw=0)
    ax.plot(u, t_convex, color=INK, lw=2.4)
    ax.plot(0, 0.15, "*", color=INK, ms=16, zorder=6, mec="white", mew=0.6)
    uu = np.linspace(-2.7, 2.7, 6)
    ax.plot(uu, 0.15 * np.ones_like(uu), color=RED, lw=1.6, ls="--")
    ax.set(title="выпуклая задача: $G$ выпукло — опорная\nпрямая касается ровно в $p^\\ast$: $d^\\ast = p^\\ast$",
           xlim=(-2.7, 2.7), ylim=(-1.1, 3.6), xlabel="$u = h(x)$", ylabel="$t = f(x)$")

    # плавная невыпуклая кривая: две «ямы» разной глубины (левая — глобальный минимум по всем u,
    # правая, при u >= 0, — то, что видно допустимой области); участок между ними выгибается вверх,
    # так что кривая нигде не образует излома, но целиком невыпукла (вторая производная меняет знак)
    t_noncvx = (0.16 * (u + 1.3) ** 2 * (u - 1.1) ** 2 + 0.10 * u + 0.55)
    ax = axes[1]
    ax.axvspan(0, 2.9, color=BLUE, alpha=0.08, lw=0)
    ax.plot(u, t_noncvx, color=INK, lw=2.4)
    imin = np.argmin(np.where(u >= 0, t_noncvx, np.inf))
    ustar, tstar = u[imin], t_noncvx[imin]
    ax.plot(ustar, tstar, "*", color=INK, ms=16, zorder=6, mec="white", mew=0.6)
    # лучшая опорная прямая: касается кривой в её глобальном минимуме (u < 0, недопустимо),
    # но лежит не выше кривой нигде — это и есть d*, и он строго ниже p* = t(u*)
    iglob = np.argmin(t_noncvx)
    dstar = t_noncvx[iglob]
    ax.plot(uu, dstar * np.ones_like(uu), color=RED, lw=1.6, ls="--")
    ax.plot(u[iglob], dstar, "o", color=RED, ms=7, zorder=6)
    ax.annotate("", xy=(ustar, dstar), xytext=(ustar, tstar),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text(ustar + 0.12, (dstar + tstar) / 2 - 0.05, "зазор\n$p^\\ast{-}d^\\ast{>}0$", color=RED, fontsize=9)
    ax.set(title="невыпуклая задача: $G$ невыпукло — лучшая\nопорная прямая ниже $p^\\ast$: $d^\\ast < p^\\ast$",
           xlim=(-2.7, 2.7), ylim=(-1.1, 3.6), xlabel="$u = h(x)$", ylabel="$t = f(x)$")
    for a in axes:
        a.grid(False)
    save("05_nonconvex_gap.png")


# ---------------------------------------------------------------- 06 LP: планирование производства + теневые цены
def fig_lp_shadow_prices():
    P = np.array([[0, 0], [4, 0], [4, 3], [2, 6], [0, 6]])
    xstar, ystar = np.array([2.0, 6.0]), np.array([0.0, 1.5, 1.0])

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.add_patch(Polygon(P, color=BLUE, alpha=0.12, lw=0))
    ax.add_patch(Polygon(P, fill=False, color=BLUE, lw=1.5))
    xs = np.linspace(-0.4, 4.6, 10)
    style_active = dict(color=RED, lw=2.2)
    style_inactive = dict(color=GRAY, lw=1.4, ls=":")
    ax.plot([4, 4], [-0.4, 6.4], **style_inactive)                       # x1 <= 4, неактивно
    ax.plot(xs, np.full_like(xs, 6.0), **style_active)                   # 2x2 <= 12, активно
    ax.plot(xs, (18 - 3 * xs) / 2, **style_active)                       # 3x1+2x2 <= 18, активно
    for cc in np.linspace(20, 44, 7):
        ax.plot(xs, (cc - 3 * xs) / 5, color=GRAY, lw=0.7, alpha=0.6)
    ax.plot(*xstar, "o", color=INK, ms=11, zorder=6)
    ax.annotate(f"$x^\\ast=(2,6)$\n$3x_1+5x_2={36}$", xy=xstar, xytext=(2.7, 3.6),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), fontsize=9.5)
    ax.text(4.05, 3.0, f"$x_1 \\leq 4$\n$y_1^\\ast={ystar[0]:.1f}$ (не активно)", color=GRAY, fontsize=9)
    ax.text(-0.35, 6.55, f"$2x_2 \\leq 12$: $y_2^\\ast={ystar[1]:.1f}$", color=RED, fontsize=9)
    ax.text(2.6, 1.6, f"$3x_1{{+}}2x_2 \\leq 18$\n$y_3^\\ast={ystar[2]:.1f}$", color=RED, fontsize=9)
    ax.set(xlim=(-0.4, 5.6), ylim=(-0.4, 7.4), aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
           title="двойственные переменные LP = теневые цены активных ограничений")
    ax.grid(False)
    save("06_lp_shadow_prices.png")


# ---------------------------------------------------------------- 07 LP: диета, теневые цены веществ
def fig_diet_shadow_prices():
    labels = ["вещество 1\n($b_1=12$, активно)", "вещество 2\n($b_2=8$, активно)", "вещество 3\n($b_3=6$, слабо, $+0.4$)"]
    y = np.array([1.2, 0.4, 0.0])
    colors = [RED, RED, GRAY]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(labels, y, color=colors, width=0.55)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.1f}", ha="center", fontsize=10)
    ax.set(ylabel="теневая цена $y_i^\\ast$ (руб. за единицу вещества)", ylim=(0, 1.5),
           title="двойственные переменные диеты: цена, которую стоило бы\nзаплатить за дополнительную единицу вещества $i$")
    ax.grid(axis="x")
    save("07_diet_shadow_prices.png")


# ---------------------------------------------------------------- 08 QP с ящиком: множители на границе
def fig_qp_dual():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    c = np.array([-3.0, -1.0])
    f = lambda x: 0.5 * x @ Q @ x + c @ x
    df = lambda x: Q @ x + c
    xstar = minimize(f, x0=np.zeros(2), jac=df, method="L-BFGS-B", bounds=[(-1, 1), (-1, 1)]).x
    mu = np.array([0.75, 0.0, 0.0, 0.0])          # (1-x1, 1+x1, 1-x2, 1+x2) >= 0

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ax = axes[0]
    xs = np.linspace(-1.7, 2.1, 200)
    X1, X2 = np.meshgrid(xs, xs)
    F = 0.5 * (Q[0, 0] * X1 ** 2 + 2 * Q[0, 1] * X1 * X2 + Q[1, 1] * X2 ** 2) + c[0] * X1 + c[1] * X2
    ax.contour(X1, X2, F, levels=18, colors=[GRAY], linewidths=0.7)
    ax.add_patch(Rectangle((-1, -1), 2, 2, color=BLUE, alpha=0.12, lw=0))
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color=RED, lw=2.2))
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color=BLUE, lw=1.3, ls=":"))
    ax.plot(*xstar, "o", color=INK, ms=10, zorder=6)
    ax.annotate("$x^\\ast=(1,\\,0.5)$", xy=xstar, xytext=(1.1, 1.3),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), fontsize=9.5)
    ax.set(aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
           title="активная грань $x_1=1$ — красным")
    ax.grid(False)

    ax = axes[1]
    names = ["$\\mu_1$\n($1{-}x_1{\\geq}0$)", "$\\mu_2$\n($1{+}x_1{\\geq}0$)",
             "$\\mu_3$\n($1{-}x_2{\\geq}0$)", "$\\mu_4$\n($1{+}x_2{\\geq}0$)"]
    colors = [RED, GRAY, GRAY, GRAY]
    bars = ax.bar(names, mu, color=colors, width=0.55)
    for b, v in zip(bars, mu):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set(ylim=(0, 1.0), ylabel="$\\mu_i^\\ast$",
           title="комплементарная нежёсткость:\n$\\mu_i^\\ast > 0$ только у активного ограничения")
    ax.grid(axis="x")
    save("08_qp_dual.png")


# ---------------------------------------------------------------- 09 чувствительность: f*(b2) кусочно-линейна
def fig_sensitivity():
    A = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    c = np.array([3.0, 5.0])
    b2s = np.linspace(3, 22, 160)
    vals = []
    for b2 in b2s:
        r = linprog(-c, A_ub=A, b_ub=np.array([4.0, b2, 18.0]), bounds=[(0, None)] * 2, method="highs")
        vals.append(-r.fun)
    vals = np.array(vals)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(b2s, vals, color=BLUE, lw=2.4)
    for bp in (6, 18):
        ax.axvline(bp, color=GRAY, lw=0.9, ls=":")
    ax.plot(12, 36, "o", color=INK, ms=9, zorder=6)
    xs = np.linspace(9, 15, 5)
    ax.plot(xs, 36 + 1.5 * (xs - 12), color=RED, lw=1.6, ls="--")
    ax.annotate("наклон $= y_2^\\ast = 1.5$", xy=(14, 36 + 1.5 * 2), xytext=(15.4, 30),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2), fontsize=9.5)
    ax.text(4.3, vals[np.searchsorted(b2s, 4.3)] - 3.2, "наклон $2.5$\n(другое\nограничение\nактивно)", fontsize=8.5)
    ax.text(20.3, 41.5, "наклон $0$\n($b_2$ уже\nне мешает)", fontsize=8.5, ha="center")
    ax.set(xlabel="$b_2$ (мощность второго цеха)", ylabel="$f^\\ast(b_2)$ (максимальная прибыль)",
           title="теневая цена — это наклон $f^\\ast(b)$:\nпроизводная оптимума по ограничению", ylim=(17, 49))
    save("09_sensitivity.png")


# ---------------------------------------------------------------- 10 неделимый завод: конечный зазор двойственности
def fig_indivisible_gap():
    # завод: линия либо стоит (x=0, расходы 0), либо работает на полную (x=4, расходы 10);
    # контракт x >= 3 -> p* = 10; ставка mu: стоять и платить 3mu, или работать за 10 - mu
    mu = np.linspace(0, 5, 501)
    stand, work = 3 * mu, 10 - mu
    d = np.minimum(stand, work)
    i = np.argmax(d)
    assert abs(mu[i] - 2.5) < 1e-9 and abs(d[i] - 7.5) < 1e-9
    pstar = 10.0

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    ax.grid(False)
    ax.axvspan(-0.8, 3, color=RED, alpha=0.07, lw=0)
    ax.axvline(3, color=INK, lw=1.4)
    ax.text(3.08, 13.6, "контракт: $x \\geq 3$", fontsize=9.5)
    ax.plot([0, 4], [0, 10], "o", color=INK, ms=11, zorder=6)
    ax.plot(4, 10, "*", color=INK, ms=20, zorder=7, mec="white", mew=0.6)
    ax.text(0, 1.0, "линия стоит\n$x=0$, расходы $0$", ha="center", fontsize=9)
    ax.text(4, 7.4, "линия на полную\n$x=4$, расходы $10$", ha="center", fontsize=9)
    ax.annotate("единственная допустимая точка:\n$p^\\ast = 10$", xy=(4, 10), xytext=(0.2, 12.0),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1), fontsize=9.5)
    ax.text(1.2, 4.6, "«между» ничего нет:\nдопустимое множество —\nдве точки, задача невыпукла",
            fontsize=8.5, color=GRAY, ha="center")
    ax.set(xlim=(-0.8, 5.2), ylim=(-1.5, 14.5), xlabel="выпуск $x$", ylabel="расходы $f(x)$",
           title="неделимый завод: работать или нет")

    ax = axes[1]
    ax.plot(mu, stand, color=GRAY, lw=1.2, ls="--", label="стоять и платить штраф: $3\\mu$")
    ax.plot(mu, work, color=GRAY, lw=1.2, ls=":", label="работать и получать премию: $10-\\mu$")
    ax.plot(mu, d, color=AQUA, lw=2.6, label="$d(\\mu)=\\min(3\\mu,\\ 10-\\mu)$ — что платит завод")
    ax.axhline(pstar, color=BLUE, lw=1.4, ls=":")
    ax.text(0.1, pstar + 0.3, "$p^\\ast = 10$ (под запретом)", color=BLUE, fontsize=9.5)
    ax.fill_between(mu, d, pstar, color=RED, alpha=0.08)
    ax.plot(2.5, 7.5, "o", color=INK, ms=9, zorder=6)
    ax.annotate("", xy=(2.5, pstar), xytext=(2.5, 7.5), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(2.65, 8.6, "зазор $p^\\ast - d^\\ast = 2.5$", color=RED, fontsize=9.5)
    ax.text(2.5, 6.6, "$\\mu^\\ast = 2.5,\\ d^\\ast = 7.5$", ha="center", fontsize=9.5)
    ax.set(xlim=(0, 5), ylim=(-0.5, 12.5), xlabel="ставка штрафа $\\mu$", ylabel="расходы завода",
           title="никакая ставка не воспроизводит запрет:\nзавод предпочитает платить или перевыполнять")
    ax.legend(loc="lower right", fontsize=8)
    save("10_indivisible_gap.png")




if __name__ == "__main__":
    print("Сохраняю в", IMG)
    fig_weak_duality_line()
    fig_lagrangian_family()
    fig_geometric_duality()
    fig_dual_function()
    fig_nonconvex_gap()
    fig_lp_shadow_prices()
    fig_diet_shadow_prices()
    fig_qp_dual()
    fig_sensitivity()
    fig_indivisible_gap()
