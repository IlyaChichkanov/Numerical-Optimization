"""Иллюстрации к конспекту лекции 3 -> папка img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture03/make_figures.py

Скрипт самодостаточен. Палитра и оформление — те же, что в лекциях 1-2.
Числа во всех примерах считаются в самом скрипте и проверяются ассертами;
те же числа воспроизводит demo03.ipynb.
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
           title="множители: положителен только\nу активной грани, у остальных — ноль")
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


# ---------------------------------------------------------------- 12 где живёт минимум: внутри или на границе
def fig_where_is_min():
    Q = np.array([[2.0, 0.5], [0.5, 1.0]])
    Qinv = np.linalg.inv(Q)
    cases = [(np.array([-1.0, -0.5]), "(а) безусловный минимум внутри ящика:\nограничения не при чём, $\\nabla f(x^\\ast)=0$"),
             (np.array([-3.0, -1.0]), "(б) безусловный минимум снаружи:\nрешение на границе, грань $x_1=1$ активна")]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))
    xs = np.linspace(-1.6, 2.2, 240)
    X1, X2 = np.meshgrid(xs, xs)
    for ax, (c, title) in zip(axes, cases):
        f = lambda x: 0.5 * x @ Q @ x + c @ x
        x_free = -Qinv @ c                                   # безусловный минимум
        xstar = minimize(f, x0=np.zeros(2), jac=lambda x: Q @ x + c, method="L-BFGS-B",
                         bounds=[(-1, 1), (-1, 1)]).x
        F = 0.5 * (Q[0, 0] * X1 ** 2 + 2 * Q[0, 1] * X1 * X2 + Q[1, 1] * X2 ** 2) + c[0] * X1 + c[1] * X2
        ax.grid(False)
        ax.contour(X1, X2, F, levels=18, colors=[GRAY], linewidths=0.7, alpha=0.9)
        ax.add_patch(Rectangle((-1, -1), 2, 2, color=BLUE, alpha=0.10, lw=0))
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, color=BLUE, lw=1.3, ls=":"))
        inside = np.all(np.abs(x_free) <= 1)
        ax.plot(*x_free, "x", color=RED, ms=11, mew=2.4, zorder=6)
        ax.plot(*xstar, "o", color=INK, ms=10, zorder=7)
        if inside:
            assert np.allclose(x_free, [0.42857, 0.28571], atol=1e-4) and np.allclose(xstar, x_free, atol=1e-4)
            ax.annotate("$x^\\ast$ = безусловный минимум\n$(0.43,\\,0.29)$, все $\\mu_i = 0$", xy=xstar, xytext=(-1.45, 1.25),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1), fontsize=9.5)
        else:
            assert np.allclose(x_free, [1.42857, 0.28571], atol=1e-4) and np.allclose(xstar, [1.0, 0.5], atol=1e-4)
            ax.plot([1, 1], [-1, 1], color=RED, lw=2.6, zorder=5)
            ax.annotate("безусловный минимум $(1.43,\\,0.29)$\nза стенкой — недостижим", xy=x_free, xytext=(0.15, -1.45),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.1), fontsize=9.5, color=RED)
            ax.annotate("$x^\\ast = (1,\\,0.5)$ на активной грани", xy=xstar, xytext=(-1.45, 1.25),
                        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1), fontsize=9.5)
        ax.set(xlim=(-1.6, 2.2), ylim=(-1.6, 1.6), aspect="equal", xlabel="$x_1$", ylabel="$x_2$", title=title)
    save("12_where_is_min.png")


# ---------------------------------------------------------------- 13 равенство: касание линий уровня и кривой g = 0
def fig_equality_tangency():
    # прямоугольник в круге: min -4 x1 x2 при x1^2 + x2^2 = 1 (ДЗ 2); решение (1/sqrt2, 1/sqrt2)
    xs = np.array([1, 1]) / np.sqrt(2)
    grad_f = -4 * xs[::-1]                                   # (-4 x2, -4 x1)
    grad_g = 2 * xs                                          # (2 x1, 2 x2)
    lam = grad_f[0] / grad_g[0]
    assert np.allclose(grad_f, lam * grad_g) and abs(lam + 2) < 1e-12

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.grid(False)
    t = np.linspace(-1.5, 1.5, 300)
    X1, X2 = np.meshgrid(t, t)
    cs = ax.contour(X1, X2, -4 * X1 * X2, levels=[-3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 2], colors=[GRAY], linewidths=0.8)
    ax.clabel(cs, fmt="%.1f", fontsize=7)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=RED, lw=2.4, label="$g(x)=x_1^2+x_2^2-1=0$")
    ax.contour(X1, X2, -4 * X1 * X2, levels=[-2.0], colors=[BLUE], linewidths=2.2, linestyles="solid")
    ax.plot(*xs, "o", color=INK, ms=10, zorder=7)
    k = 0.18
    ax.annotate("", xy=xs + k * grad_f, xytext=xs, arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, mutation_scale=15), zorder=8)
    ax.annotate("", xy=xs + k * 0.5 * grad_g, xytext=xs, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2, mutation_scale=15), zorder=8)
    ax.text(-0.95, 0.22, "$\\nabla f = (-2\\sqrt{2},-2\\sqrt{2})$", color=BLUE, fontsize=9.5)
    ax.text(*(xs + k * 0.5 * grad_g + [0.05, 0.05]), "$\\nabla g = (\\sqrt{2},\\sqrt{2})$", color=RED, fontsize=9.5)
    ax.text(-1.45, 1.25, "линия уровня $f=-2$ (синяя)\nкасается окружности в $x^\\ast$:\n$\\nabla f = \\lambda\\,\\nabla g$, $\\lambda = -2$", fontsize=9.5)
    ax.text(0.78, 0.55, "$x^\\ast=(\\frac{1}{\\sqrt{2}},\\frac{1}{\\sqrt{2}})$", fontsize=9.5)
    for p_, lab in (((1, 0), "$(1,0)$: максимум; баланс сил дал бы\n$\\mu=-4<0$ для стенки $x_2\\geq0$"),):
        ax.plot(*p_, "s", color=GRAY, ms=7)
        ax.text(p_[0] - 0.02, p_[1] - 0.2, lab, fontsize=8.5, color=GRAY, ha="right", va="top")
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", xlabel="$x_1$", ylabel="$x_2$",
           title="равенство: в решении линия уровня $f$ касается кривой $g=0$")
    ax.legend(loc="lower left", fontsize=9)
    save("13_equality_tangency.png")


# ---------------------------------------------------------------- 14 цепь на полу: множители — силы реакции
def chain_with_floor():
    """Цепь из лекции 1 (§7.3): N грузов, пружины D, пол z >= 0.5 + 0.1 y. Возвращает решения и реакции пола."""
    from scipy.optimize import LinearConstraint
    N, D, g = 40, 70.0, 9.81
    m = 4.0 / N
    pl, pr = np.array([-2.0, 1.0]), np.array([2.0, 1.0])

    def unpack(v):
        return np.r_[pl[0], v[:N], pr[0]], np.r_[pl[1], v[N:], pr[1]]

    def energy(v):
        y, z = unpack(v)
        return 0.5 * D * np.sum(np.diff(y) ** 2 + np.diff(z) ** 2) + m * g * np.sum(z[1:-1])

    def grad(v):
        y, z = unpack(v)
        return np.r_[D * (2 * y[1:-1] - y[:-2] - y[2:]), D * (2 * z[1:-1] - z[:-2] - z[2:]) + m * g]

    v0 = np.r_[np.linspace(pl[0], pr[0], N + 2)[1:-1], np.linspace(pl[1], pr[1], N + 2)[1:-1]]
    free = minimize(energy, v0, jac=grad, method="BFGS", options={"gtol": 1e-8}).x
    A = np.hstack((-0.1 * np.eye(N), np.eye(N)))                       # h_i = z_i - 0.1 y_i - 0.5 >= 0
    con = minimize(energy, v0, jac=grad, method="SLSQP", constraints=[LinearConstraint(A, 0.5 * np.ones(N), np.inf)],
                   options={"ftol": 1e-12, "maxiter": 500}).x
    active = (A @ con - 0.5) < 1e-6
    G = grad(con)
    mu = np.zeros(N)
    mu[active] = G[N:][active]                                          # баланс по z: dE/dz_i = mu_i
    assert np.abs(G[:N][active] + 0.1 * mu[active]).max() < 1e-5       # баланс по y: dE/dy_i = -0.1 mu_i
    assert np.abs(G[~np.r_[active, active]]).max() < 1e-5               # у висящих грузов dE = 0
    assert active.sum() == 25 and abs(mu.sum() - 23.75) < 0.05
    return N, m, g, unpack, free, con, active, mu


def fig_chain_forces():
    N, m, g, unpack, free, con, active, mu = chain_with_floor()
    yf, zf = unpack(free)
    yc, zc = unpack(con)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    yy = np.linspace(-2.2, 2.2, 10)
    ax.fill_between(yy, -2.3, 0.5 + 0.1 * yy, color=GRAY, alpha=0.15, lw=0)
    ax.plot(yy, 0.5 + 0.1 * yy, color=INK, lw=1.4, ls="--", label="пол $z = 0.5 + 0.1y$")
    ax.plot(yf, zf, color=GRAY, lw=1.4, ls=":", label="без пола: цепь провисает до $z\\approx-1.9$")
    ax.plot(yc, zc, "-", color=BLUE, lw=1.8, zorder=4, label="с полом: 25 из 40 грузов лежат на полу")
    ax.plot(yc[1:-1][~active], zc[1:-1][~active], "o", color=BLUE, ms=4, zorder=5)
    ax.plot(yc[1:-1][active], zc[1:-1][active], "o", color=RED, ms=5, zorder=6)
    k = 0.45
    for i in np.where(active)[0][::2]:
        ax.annotate("", xy=(yc[i + 1] - 0.1 * k * mu[i], zc[i + 1] + k * mu[i]), xytext=(yc[i + 1], zc[i + 1]),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3, mutation_scale=10), zorder=7)
    ax.text(0, 1.25, "реакция пола $\\mu_i\\approx0.97$ Н на каждый лежащий груз\n($mg=0.98$ Н); у висящих грузов $\\mu_i = 0$",
            ha="center", fontsize=9.5, color=RED)
    ax.text(-2.1, -1.5, "всего вес $39$ Н: пол несёт $\\sum\\mu_i = 24$ Н,\nостальное — крепления", fontsize=9, color=GRAY)
    ax.plot([-2, 2], [1, 1], "s", color=INK, ms=7)
    ax.set(xlim=(-2.3, 2.3), ylim=(-2.2, 1.6), xlabel="$y$", ylabel="$z$",
           title="цепь на полу: множители ограничений $z_i \\geq 0.5 + 0.1 y_i$ — силы реакции пола")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(False)
    save("14_chain_forces.png")


# ---------------------------------------------------------------- 15 портфель Марковица: множители — цены
PF_R = np.array([0.04, 0.08, 0.11, 0.15])                              # ожидаемые доходности
PF_VOL = np.array([0.05, 0.12, 0.18, 0.30])                            # волатильности
PF_CORR = np.array([[1, 0.2, 0.1, 0.0], [0.2, 1, 0.5, 0.3], [0.1, 0.5, 1, 0.4], [0.0, 0.3, 0.4, 1.0]])
PF_SIGMA = np.outer(PF_VOL, PF_VOL) * PF_CORR


def portfolio(target):
    """min 1/2 x'Sx при r'x >= target, sum x = 1, x >= 0; множители — из баланса сил по активному набору."""
    S, r = PF_SIGMA, PF_R
    res = minimize(lambda x: 0.5 * x @ S @ x, x0=np.ones(4) / 4, jac=lambda x: S @ x, method="SLSQP", bounds=[(0, 1)] * 4,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones(4)},
                                {"type": "ineq", "fun": lambda x: r @ x - target, "jac": lambda x: r}],
                   options={"ftol": 1e-13, "maxiter": 500})
    x = res.x
    act0 = x < 1e-6
    M = np.c_[np.ones(4), r, np.eye(4)[:, act0]]                       # grad f = lam*1 + mu_r*r + sum mu_i e_i
    sol = np.linalg.lstsq(M, S @ x, rcond=None)[0]
    assert np.abs(M @ sol - S @ x).max() < 1e-6
    lam, mu_r, mu_b = sol[0], sol[1], np.zeros(4)
    mu_b[act0] = sol[2:]
    return x, res.fun, lam, mu_r, mu_b


def fig_portfolio():
    targets = np.linspace(0.045, 0.145, 41)
    risks = np.array([portfolio(tg)[1] for tg in targets])
    x, risk, lam, mu_r, mu_b = portfolio(0.12)
    assert np.allclose(x, [0, 0.1414, 0.5026, 0.356], atol=2e-3) and abs(mu_r - 0.473) < 5e-3 and mu_b[0] > 0
    slope_fd = (portfolio(0.1205)[1] - portfolio(0.1195)[1]) / 0.001
    assert abs(slope_fd - mu_r) < 5e-3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    ax.plot(100 * targets, risks, color=BLUE, lw=2.2)
    ax.plot(12, risk, "o", color=INK, ms=9, zorder=6)
    tt = np.linspace(0.105, 0.135, 5)
    ax.plot(100 * tt, risk + mu_r * (tt - 0.12), color=RED, lw=1.5, ls="--")
    ax.annotate(f"наклон $=\\mu_r={mu_r:.2f}$:\n$+1$ п.п. доходности стоит\n$+{mu_r/100:.4f}$ риска", xy=(12, risk), xytext=(5.2, 0.017),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1), fontsize=9, color=RED)
    ax.set(xlabel="требуемая доходность, % годовых", ylabel="минимальный риск $\\frac{1}{2} x^\\top\\Sigma x$",
           title="цена доходности: наклон границы = множитель")
    ax = axes[1]
    names = ["A\n4%", "B\n8%", "C\n11%", "D\n15%"]
    bars = ax.bar(names, x, color=[RED if a else BLUE for a in mu_b > 0], width=0.55)
    for bb, v in zip(bars, x):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9.5)
    ax.text(0, 0.12, f"доля $0$: запрет шортов\nактивен, $\\mu_A={mu_b[0]:.4f}$\n(«хотелось бы\nзашортить»)", ha="center", fontsize=8.5, color=RED)
    ax.set(ylim=(0, 0.62), ylabel="доля в портфеле $x_i^\\ast$", title="портфель при цели 12%: кто не куплен")
    ax.grid(axis="x")
    save("15_portfolio.png")


if __name__ == "__main__":
    print("Сохраняю в", IMG)
    fig_recap_problems()
    fig_lp_shadow_prices()
    fig_qp_dual()
    fig_sensitivity()
    fig_gradients()
    fig_where_is_min()
    fig_equality_tangency()
    fig_chain_forces()
    fig_portfolio()
