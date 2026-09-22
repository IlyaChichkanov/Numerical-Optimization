"""Иллюстрации к конспекту лекции 3 -> папка img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture03/make_figures.py

Скрипт самодостаточен. Палитра и оформление — те же, что в лекциях 1-2.
Числа во всех «настоящих» примерах (не помеченных как качественная схема)
проверены отдельно в demo03.ipynb и exercises03.ipynb — здесь только рисунки.
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


# ---------------------------------------------------------------- 00 разминка: три задачки раздела 1
def fig_recap_problems():
    from scipy.optimize import minimize_scalar

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1.2, 1]))

    # --- (А) три «середины» одних данных: среднее, медиана (LP), середина размаха (LP)
    a = np.array([1.0, 2.0, 3.0, 4.0, 20.0])
    n = len(a)
    rows, rhs = [], []
    for i in range(n):                       # -s_i <= x - a_i <= s_i
        r1 = np.zeros(n + 1); r1[0], r1[1 + i] = 1, -1; rows.append(r1); rhs.append(a[i])
        r2 = np.zeros(n + 1); r2[0], r2[1 + i] = -1, -1; rows.append(r2); rhs.append(-a[i])
    lad = linprog(np.r_[0, np.ones(n)], A_ub=np.array(rows), b_ub=rhs,
                  bounds=[(None, None)] + [(0, None)] * n, method="highs")
    rows2 = [[1, -1] if k % 2 == 0 else [-1, -1] for k in range(2 * n)]
    rhs2 = [v for ai in a for v in (ai, -ai)]
    mm = linprog([0, 1], A_ub=np.array(rows2, float), b_ub=rhs2,
                 bounds=[(None, None), (0, None)], method="highs")
    mean, median, midrange = a.mean(), lad.x[0], mm.x[0]
    assert abs(median - np.median(a)) < 1e-7 and abs(midrange - (a.min() + a.max()) / 2) < 1e-7

    ax = axes[0]
    ax.set(xlim=(-0.5, 21.5), ylim=(-0.9, 1.5), yticks=[], xlabel="$x$")
    ax.grid(False)
    ax.axhline(0, color=GRAY, lw=1.4, zorder=1)
    ax.plot(a, np.zeros(n), "o", color=INK, ms=8, zorder=5)
    for ai in a:
        ax.text(ai, -0.28, f"{ai:g}", ha="center", fontsize=8.5, color=GRAY)
    for v, col, lab, y in ((median, AQUA, "медиана $3$\n$\\min\\sum|x-a_i|$ — LP", 0.55),
                           (mean, BLUE, "среднее $6$\n$\\min\\sum(x-a_i)^2$ — QP", 1.05),
                           (midrange, ORANGE, "середина размаха $10.5$\n$\\min\\max_i|x-a_i|$ — LP", 0.55)):
        ax.plot([v, v], [0, y - 0.12], color=col, lw=1.6, ls="--")
        ax.plot(v, 0, "v", color=col, ms=11, zorder=6)
        ax.text(v, y, lab, ha="center", fontsize=8.5, color=col)
    ax.set_title("(А) три «середины» одних данных", loc="left")

    # --- (Б) спасатель на пляже: минимум времени, закон Снеллиуса
    a_s, b_w, d, v1, v2 = 30.0, 20.0, 40.0, 5.0, 1.5
    T = lambda x: np.sqrt(a_s**2 + x**2) / v1 + np.sqrt(b_w**2 + (d - x)**2) / v2
    res = minimize_scalar(T, bounds=(0, d), method="bounded", options=dict(xatol=1e-10))
    xs = res.x
    s1 = xs / np.sqrt(a_s**2 + xs**2) / v1
    s2 = (d - xs) / np.sqrt(b_w**2 + (d - xs)**2) / v2
    assert abs(s1 - s2) < 1e-6, "закон Снеллиуса не выполнен"
    x_line = d * a_s / (a_s + b_w)         # по прямой: пересечение с берегом

    ax = axes[1]
    ax.grid(False)
    ax.axhspan(0, 36, color=YELLOW, alpha=0.18, zorder=0)
    ax.axhspan(-26, 0, color=BLUE, alpha=0.13, zorder=0)
    ax.axhline(0, color=GRAY, lw=1.2)
    ax.text(45, 32, "песок, $v_1 = 5$ м/с", fontsize=8.5, color=GRAY, ha="right")
    ax.text(1, -4.5, "вода, $v_2 = 1.5$ м/с", fontsize=8.5, color=GRAY)
    for x_in, col, lab, ls in ((xs, RED, f"оптимум $x^\\ast\\approx{xs:.1f}$ м: {res.fun:.2f} с", "-"),
                               (d, VIOLET, f"до траверза, затем поперёк: {T(d):.2f} с", "--"),
                               (x_line, GRAY, f"по прямой: {T(x_line):.2f} с", ":")):
        ax.plot([0, x_in, d], [a_s, 0, -b_w], color=col, lw=2.2 if col == RED else 1.5, ls=ls, label=lab)
    ax.plot(0, a_s, "s", color=INK, ms=8, zorder=6)
    ax.plot(d, -b_w, "*", color=INK, ms=13, zorder=6)
    ax.plot(xs, 0, "o", color=RED, ms=7, zorder=6)
    ax.text(-1.5, a_s + 1.5, "спасатель", fontsize=8.5, ha="left")
    ax.text(d - 1, -b_w - 3.5, "тонущий", fontsize=8.5, ha="right")
    ax.text(xs + 0.8, 4.5, "$\\frac{\\sin\\theta_1}{v_1}=\\frac{\\sin\\theta_2}{v_2}$", fontsize=10, color=RED)
    ax.set(xlim=(-3, 46), ylim=(-26, 36), xlabel="вдоль берега, м", yticks=[])
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(Б) спасатель на пляже: $\\min T(x)$", loc="left")

    # --- (В) центр Чебышёва пятиугольника — LP по (c, r)
    A = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 2.0], [3.0, 1.0], [1.0, -1.0]])
    b = np.array([0.0, 0.0, 8.0, 12.0, 3.0])
    norms = np.linalg.norm(A, axis=1)
    cheb = linprog([0, 0, -1], A_ub=np.c_[A, norms], b_ub=b,
                   bounds=[(None, None), (None, None), (0, None)], method="highs")
    c_, r_ = cheb.x[:2], cheb.x[2]
    mu = -cheb.ineqlin.marginals
    active = cheb.slack < 1e-9
    assert abs(r_ - (6 - 2 * np.sqrt(5))) < 1e-6 and active.sum() == 3
    assert np.all(mu[~active] < 1e-9) and np.allclose(mu @ A, 0, atol=1e-9) and abs(mu @ norms - 1) < 1e-9
    verts = np.array([[0, 0], [3, 0], [3.75, 0.75], [3.2, 2.4], [0, 4]])

    ax = axes[2]
    ax.grid(False)
    ax.add_patch(Polygon(verts, closed=True, facecolor=BLUE, alpha=0.10, edgecolor="none"))
    ax.add_patch(plt.Circle(c_, r_, facecolor=AQUA, alpha=0.25, edgecolor=AQUA, lw=2))
    ax.plot(*c_, "o", color=INK, ms=7, zorder=6)
    ax.plot([c_[0], c_[0] + r_], [c_[1], c_[1]], color=INK, lw=1)
    ax.text(c_[0] + r_ / 2, c_[1] + 0.12, f"$r={r_:.3f}$", fontsize=8.5, ha="center")
    # стороны: активные — жирно, с mu_i; неактивные — тонко серым, mu_i = 0
    edges = [(0, 4), (0, 1), (3, 4), (2, 3), (1, 2)]      # индексы вершин для ограничений 1..5
    label_pos = [(-0.35, 2.7), (2.0, -0.35), (1.5, 3.5), (3.75, 1.9), (3.6, 0.15)]
    for i, ((p, q), (lx, ly)) in enumerate(zip(edges, label_pos)):
        col, lw = (RED, 2.6) if active[i] else (GRAY, 1.3)
        ax.plot(verts[[p, q], 0], verts[[p, q], 1], color=col, lw=lw, zorder=3)
        ax.text(lx, ly, f"$\\mu_{i+1}={mu[i]:.3f}$" if active[i] else f"$\\mu_{i+1}=0$",
                fontsize=8.5, color=col, ha="center", rotation=90 if i == 0 else 0)
    ax.set(xlim=(-0.7, 4.4), ylim=(-0.7, 4.4), aspect="equal", xlabel="$x_1$", ylabel="$x_2$")
    ax.set_title("(В) самый большой круг — это LP", loc="left")

    save("00_recap_problems.png")


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


# ---------------------------------------------------------------- 11 градиенты: множитель как сила стенки
def fig_gradients():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), gridspec_kw=dict(width_ratios=[1, 1.05, 1]))
    arrow = lambda ax, p, v, col, lw=2.2: ax.annotate(
        "", xy=(p[0] + v[0], p[1] + v[1]), xytext=p,
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=16), zorder=8)

    # --- (а) одномерный завод: тяга -f'(3) и реакция стенки mu*h'(3)
    ax = axes[0]
    ax.grid(False)
    x = np.linspace(0.5, 5, 300)
    ax.axvspan(0.5, 3, color=RED, alpha=0.07, lw=0)
    ax.plot(x, (x - 2) ** 2, color=BLUE, lw=2.2)
    ax.axvline(3, color=INK, lw=1.6)
    ax.text(2.95, 6.9, "стенка $x=3$", ha="right", fontsize=9.5)
    ax.plot(3, 1, "o", color=INK, ms=10, zorder=9)
    arrow(ax, (3, 1), (-0.9, 0), RED)
    arrow(ax, (3, 1), (0.9, 0), AQUA)
    ax.text(1.75, 1.35, "тяга $-f'(3) = -2$\n(к вершине параболы)", ha="center", fontsize=8.5, color=RED)
    ax.text(3.75, 1.5, "реакция стенки\n$\\mu\\, h'(3) = +2$", ha="center", fontsize=8.5, color=AQUA)
    ax.text(1.0, 5.4, "равновесие:\n$f'(x^\\ast) = \\mu\\, h'(x^\\ast)$,\n$\\mu^\\ast = 2$", fontsize=9.5)
    ax.set(xlim=(0.5, 5), ylim=(-0.6, 7.6), xlabel="$x$", ylabel="$f(x) = (x-2)^2$",
           title="(а) стенка держит: множитель — сила реакции")

    # --- (б) ящик: антиградиент наружу через активную грань; на неоптимальной точке — скольжение
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    c = np.array([-3.0, -1.0])
    df = lambda x: Q @ x + c
    xstar = np.array([1.0, 0.5])
    grad_h = np.array([-1.0, 0.0])                       # h_1 = 1 - x_1
    mu = df(xstar)[0] / grad_h[0]
    assert np.allclose(df(xstar), mu * grad_h) and abs(mu - 0.75) < 1e-12
    xbad = np.array([1.0, -0.5])
    gb = -df(xbad)                                       # (1.25, 1)
    assert np.allclose(gb, [1.25, 1.0])

    ax = axes[1]
    ax.grid(False)
    xs = np.linspace(-1.6, 2.2, 200)
    X1, X2 = np.meshgrid(xs, xs)
    F = 0.5 * (Q[0, 0] * X1 ** 2 + 2 * Q[0, 1] * X1 * X2 + Q[1, 1] * X2 ** 2) + c[0] * X1 + c[1] * X2
    ax.contour(X1, X2, F, levels=16, colors=[GRAY], linewidths=0.6, alpha=0.8)
    ax.add_patch(Rectangle((-1, -1), 2, 2, color=BLUE, alpha=0.10, lw=0))
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color=BLUE, lw=1.2, ls=":"))
    ax.plot([1, 1], [-1, 1], color=RED, lw=2.4, zorder=5)
    s = 0.55
    ax.plot(*xstar, "o", color=INK, ms=9, zorder=9)
    arrow(ax, xstar, s * (-df(xstar)), RED)              # тяга наружу
    arrow(ax, xstar + np.array([0, 0.06]), s * mu * grad_h, AQUA)   # реакция внутрь (чуть выше, чтобы видеть обе)
    ax.text(1.06, 0.72, "$x^\\ast$: тяга $-\\nabla f$ наружу,\nреакция $\\mu_1\\nabla h_1$ внутрь —\nгасятся, $\\mu_1 = 0.75$",
            fontsize=8.3, color=INK)
    ax.plot(*xbad, "s", color=VIOLET, ms=8, zorder=9)
    arrow(ax, xbad, s * gb, VIOLET, lw=1.8)
    arrow(ax, xbad, s * np.array([0, gb[1]]), ORANGE, lw=2.0)
    ax.plot([1, 1 + s * gb[0]], [-0.5 + s * gb[1], -0.5 + s * gb[1]], color=VIOLET, lw=0.8, ls=":")
    ax.plot([1 + s * gb[0], 1 + s * gb[0]], [-0.5, -0.5 + s * gb[1]], color=VIOLET, lw=0.8, ls=":")
    ax.text(1.06, -0.98, "не оптимум: у $-\\nabla f$ есть\nсоставляющая вдоль стенки —\nскользим вверх", fontsize=8.3, color=VIOLET)
    for (lx, ly, txt) in ((-1.06, 0.0, "$\\mu_2 = 0$"), (0.0, 1.06, "$\\mu_3 = 0$"), (0.0, -1.06, "$\\mu_4 = 0$")):
        ax.text(lx, ly, txt, fontsize=8.5, color=GRAY, ha="center", va="center", rotation=90 if lx < 0 else 0)
    ax.set(xlim=(-1.6, 2.4), ylim=(-1.6, 1.6), aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
           title="(б) ящик: $\\nabla f(x^\\ast) = \\mu_1 \\nabla h_1(x^\\ast)$")

    # --- (в) вершина LP: градиент прибыли в конусе нормалей активных ограничений
    A = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    cc = np.array([3.0, 5.0])
    y23 = np.linalg.solve(A[[1, 2]].T, cc)
    assert np.allclose(y23, [1.5, 1.0])
    V = np.array([[0, 0], [4, 0], [4, 3], [2, 6], [0, 6]], float)
    v = np.array([2.0, 6.0])
    a2, a3 = A[1], A[2]
    k = 0.45                                             # масштаб векторов на картинке

    ax = axes[2]
    ax.grid(False)
    ax.add_patch(Polygon(V, closed=True, facecolor=BLUE, alpha=0.10, edgecolor=BLUE, lw=1.2, ls=":"))
    ax.plot(V[[2, 3], 0], V[[2, 3], 1], color=RED, lw=2.4, zorder=5)
    ax.plot(V[[3, 4], 0], V[[3, 4], 1], color=RED, lw=2.4, zorder=5)
    # конус внешних нормалей активных ограничений 2 и 3
    e2, e3 = a2 / np.linalg.norm(a2), a3 / np.linalg.norm(a3)
    R = 3.4
    ang = np.linspace(np.arctan2(e3[1], e3[0]), np.arctan2(e2[1], e2[0]), 40)
    cone = np.vstack([v, v + R * np.c_[np.cos(ang), np.sin(ang)]])
    ax.add_patch(Polygon(cone, closed=True, facecolor=AQUA, alpha=0.18, edgecolor="none"))
    arrow(ax, v, k * a2, AQUA, lw=1.8)
    arrow(ax, v, k * a3, AQUA, lw=1.8)
    ax.text(*(v + k * a2 + [-0.55, 0.05]), "$a_2$", color=AQUA, fontsize=10)
    ax.text(*(v + k * a3 + [0.1, -0.15]), "$a_3$", color=AQUA, fontsize=10)
    # разложение c = 1.5 a2 + 1 a3
    p2, p3, pc = v + k * y23[0] * a2, v + k * y23[1] * a3, v + k * cc
    arrow(ax, v, k * cc, RED, lw=2.4)
    ax.plot([p2[0], pc[0]], [p2[1], pc[1]], color=INK, lw=0.9, ls="--")
    ax.plot([p3[0], pc[0]], [p3[1], pc[1]], color=INK, lw=0.9, ls="--")
    ax.plot(*p2, "o", color=INK, ms=4); ax.plot(*p3, "o", color=INK, ms=4)
    ax.text(*(pc + [0.12, 0.05]), "$c = (3,5)$\nградиент прибыли", color=RED, fontsize=9.5)
    ax.text(*(p2 + [-1.55, 0.0]), "$1.5\\,a_2$", fontsize=9.5)
    ax.text(*(p3 + [0.12, -0.35]), "$1\\,a_3$", fontsize=9.5)
    ax.plot(*v, "o", color=INK, ms=9, zorder=9)
    ax.text(v[0] - 0.15, v[1] - 0.55, "$x^\\ast = (2,6)$", ha="right", fontsize=9.5)
    ax.text(4.08, 1.4, "$x_1 \\leq 4$: не касается,\n$y_1^\\ast = 0$", fontsize=8.5, color=GRAY)
    ax.text(0.2, 0.9, "$c = y_2^\\ast a_2 + y_3^\\ast a_3$,\n$y^\\ast = (0,\\ 1.5,\\ 1)$ — теневые цены", fontsize=9.5)
    ax.set(xlim=(-0.4, 6.8), ylim=(-0.5, 9.6), aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
           title="(в) вершина LP: $c$ в конусе нормалей активных стенок")

    save("11_gradients.png")


if __name__ == "__main__":
    print("Сохраняю в", IMG)
    fig_recap_problems()
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
    fig_gradients()
