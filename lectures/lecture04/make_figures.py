"""Иллюстрации к конспекту лекции 4 -> папка img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture04/make_figures.py

Скрипт самодостаточен. Палитра и оформление — те же, что в лекциях 1-3.
Числа во всех примерах считаются в самом скрипте и проверяются ассертами;
те же числа воспроизводит demo04.ipynb.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

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


def arrow(ax, p, q, color, lw=2, ms=14, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=6, **kw))


# ---------------------------------------------------------------- методы (те же, что в demo04)
def gd(f, grad, x0, alpha=None, tol=1e-6, maxit=100_000, c=1e-4):
    """Градиентный спуск: alpha=None — backtracking (Армихо с константой c), иначе постоянный шаг."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        if alpha is None:
            a, fx = 1.0, f(x)
            while f(x - a * g) > fx - c * a * (g @ g):
                a /= 2
        else:
            a = alpha
        x = x - a * g; path.append(x.copy())
    return x, maxit, np.array(path)


def newton(grad, hess, x0, tol=1e-10, maxit=50):
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        x = x - np.linalg.solve(hess(x), g); path.append(x.copy())
    return x, maxit, np.array(path)


# ---------------------------------------------------------------- Розенброк
def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def rosen_hess(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200.0]])


def rosen_contour(ax, xlim=(-2, 2), ylim=(-1, 3)):
    X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3.3, 14), colors=GRAY, linewidths=0.7, alpha=0.8)
    ax.plot(1, 1, "*", color=RED, ms=13, zorder=7)
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$")


# ---------------------------------------------------------------- цепь (лекция 1, §7.3)
def chain_matrices(N=40, D=70.0, g=9.81):
    m = 4.0 / N
    K = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    H = np.kron(np.eye(2), D * K)
    p_left, p_right = np.array([-2.0, 1.0]), np.array([2.0, 1.0])
    # градиент в нуле: вклад закреплённых концов и сила тяжести
    b = np.zeros(2 * N)
    b[0] -= D * p_left[0]; b[N - 1] -= D * p_right[0]
    b[N] -= D * p_left[1]; b[2 * N - 1] -= D * p_right[1]
    b[N:] += m * g
    return H, b, p_left, p_right


def chain_solution(N=40):
    H, b, pl, pr = chain_matrices(N)
    v = np.linalg.solve(H, -b)
    y = np.concatenate(([pl[0]], v[:N], [pr[0]])); z = np.concatenate(([pl[1]], v[N:], [pr[1]]))
    return y, z


# ---------------------------------------------------------------- логистическая регрессия (синтетика)
def logistic_data(seed=4, n=200):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(1.5 * z1 - 1.0 * z2 + 0.3)))
    y = (rng.uniform(size=n) < p).astype(float)
    age, exp_ = 40 + 10 * z1, 5 + z2
    return age, exp_, y


# ================================================================ 00 три задачи-крючка
def fig_hooks():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    rosen_contour(ax)
    ax.plot(-1.2, 1, "o", color=BLUE, ms=7, zorder=7)
    ax.annotate("старт $(-1.2,\\,1)$", (-1.2, 1), (-1.9, 1.9), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("минимум $(1,1)$", (1, 1), (0.2, 2.5), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.set_title("Розенброк: изогнутая долина")

    ax = axes[1]
    y, z = chain_solution(40)
    ax.plot(y, z, "-o", ms=3, color=BLUE)
    ax.plot([-2, 2], [1, 1], "s", color=INK, ms=6)
    ax.set(xlabel="$y$", ylabel="$z$", title="Цепь: 40 грузов, 80 переменных, минимум энергии")

    ax = axes[2]
    age, exp_, lab = logistic_data()
    ax.plot(age[lab == 1], exp_[lab == 1], "o", ms=4, color=BLUE, label="$y=1$")
    ax.plot(age[lab == 0], exp_[lab == 0], "s", ms=4, color=ORANGE, label="$y=0$")
    ax.set(xlabel="возраст, лет", ylabel="стаж, лет", title="Логистическая регрессия: найти разделяющую прямую")
    ax.legend(loc="upper left", fontsize=8.5)
    save("00_hooks.png")


# ================================================================ 01 минимум, максимум, седло
def fig_stationary():
    cases = [
        (np.diag([1.0, 2.0]), "минимум: $\\nabla^2 f \\succ 0$", "$f=\\frac{1}{2}(x_1^2+2x_2^2)$"),
        (-np.diag([1.0, 2.0]), "максимум: $\\nabla^2 f \\prec 0$", "$f=-\\frac{1}{2}(x_1^2+2x_2^2)$"),
        (np.diag([1.0, -1.0]), "седло: $\\nabla^2 f$ индефинитен", "$f=\\frac{1}{2}(x_1^2-x_2^2)$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    lim = 2.2
    X, Y = np.meshgrid(np.linspace(-lim, lim, 300), np.linspace(-lim, lim, 300))
    for ax, (Q, title, formula) in zip(axes, cases):
        Z = 0.5 * (Q[0, 0] * X ** 2 + Q[1, 1] * Y ** 2)
        ax.contour(X, Y, Z, levels=np.linspace(-4, 4, 17), colors=GRAY, linewidths=0.7)
        xs = np.linspace(-lim + 0.3, lim - 0.3, 7)
        U, V = np.meshgrid(xs, xs)
        GX, GY = -(Q[0, 0] * U), -(Q[1, 1] * V)
        ax.quiver(U, V, GX, GY, color=BLUE, width=0.005, scale=28, alpha=0.9)
        ax.plot(0, 0, "o", color=RED, ms=8, zorder=7)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="$x_1$", ylabel="$x_2$", title=title, aspect="equal")
        ax.text(0.03, 0.97, formula, transform=ax.transAxes, va="top", fontsize=9.5)
    fig.suptitle("Во всех трёх точках $\\nabla f=0$; стрелки — антиградиент $-\\nabla f$ (куда пойдёт спуск)", y=1.02, fontsize=10.5)
    save("01_stationary.png")


# ================================================================ 02 второй порядок: зазор и четыре квадратичных ландшафта
def fig_second_order():
    fig = plt.figure(figsize=(14, 3.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[0])
    t = np.linspace(-1.3, 1.3, 300)
    ax.plot(t, t ** 4, color=BLUE, label="$x^4$ — минимум")
    ax.plot(t, t ** 3, color=ORANGE, label="$x^3$ — не экстремум")
    ax.plot(t, -t ** 4, color=RED, label="$-x^4$ — максимум")
    ax.plot(0, 0, "o", color=INK, ms=6, zorder=7)
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.6, 1.6), xlabel="$x$", title="У всех $f'(0)=f''(0)=0$: условия молчат")
    ax.legend(loc="upper left", fontsize=8.5)

    lim = 2.0
    X, Y = np.meshgrid(np.linspace(-lim, lim, 300), np.linspace(-lim, lim, 300))
    cases = [
        (0.5 * (X ** 2 + 2 * Y ** 2), "$Q\\succ0$:\nчаша, один минимум", "min"),
        (0.5 * (X ** 2 - Y ** 2), "$Q$ индефинитна:\nседло, минимума нет", "saddle"),
        (0.5 * X ** 2, "$Q\\succeq0$, $c\\in\\mathrm{range}\\,Q$:\nпрямая минимумов", "line"),
        (0.5 * X ** 2 + Y, "$Q\\succeq0$, $c\\notin\\mathrm{range}\\,Q$:\nминимума нет", "none"),
    ]
    for k, (Z, title, kind) in enumerate(cases):
        ax = fig.add_subplot(gs[k + 1])
        ax.contour(X, Y, Z, levels=np.linspace(-3, 3, 19), colors=GRAY, linewidths=0.7)
        if kind == "min":
            ax.plot(0, 0, "*", color=RED, ms=12, zorder=7)
        elif kind == "saddle":
            ax.plot(0, 0, "o", color=ORANGE, ms=7, zorder=7)
        elif kind == "line":
            ax.plot([0, 0], [-lim, lim], color=RED, lw=2.5, zorder=7)
        else:
            arrow(ax, (0, 1.0), (0, -1.6), RED, lw=2.2)
            ax.text(0.12, -0.3, "$f\\to-\\infty$", color=RED, fontsize=9)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[], title=title, aspect="equal")
        ax.title.set_fontsize(9.2)
    save("02_second_order.png")


# ================================================================ 03 направление спуска и выбор шага
def fig_descent_direction():
    Q = np.diag([1.0, 10.0])
    f = lambda x: 0.5 * x @ Q @ x
    x = np.array([3.0, 1.0]); g = Q @ x
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.05, 1]))

    ax = axes[0]
    X, Y = np.meshgrid(np.linspace(-1, 5, 300), np.linspace(-1.5, 2.6, 300))
    Z = 0.5 * (X ** 2 + 10 * Y ** 2)
    ax.contour(X, Y, Z, levels=[0.3, 1, 2.5, 5, 9.5, 15, 25, 40, 60], colors=GRAY, linewidths=0.7)
    # полуплоскость направлений спуска: p^T g < 0
    t = np.array([-g[1], g[0]]) / np.linalg.norm(g)
    n = -g / np.linalg.norm(g)
    L = 4
    poly = Polygon([x + L * t, x - L * t, x - L * t + L * n, x + L * t + L * n], closed=True,
                   color=AQUA, alpha=0.12, zorder=1)
    ax.add_patch(poly)
    ax.plot([x[0] - L * t[0], x[0] + L * t[0]], [x[1] - L * t[1], x[1] + L * t[1]], "--", color=AQUA, lw=1.2)
    s = 1.3
    arrow(ax, x, x + s * n, BLUE, lw=2.5)
    ax.text(*(x + s * n + np.array([0.05, 0.08])), "$-\\nabla f$", color=BLUE, fontsize=10)
    p1 = np.array([-1.0, 0.15]); p1 /= np.linalg.norm(p1)
    arrow(ax, x, x + s * p1, VIOLET, lw=1.8)
    ax.text(*(x + s * p1 + np.array([-0.25, 0.12])), "тоже спуск", color=VIOLET, fontsize=9)
    p2 = np.array([1.0, 0.6]); p2 /= np.linalg.norm(p2)
    arrow(ax, x, x + s * p2, RED, lw=1.8)
    ax.text(*(x + s * p2 + np.array([0.05, 0.02])), "подъём", color=RED, fontsize=9)
    ax.plot(*x, "o", color=INK, ms=7, zorder=7)
    ax.text(x[0] + 0.1, x[1] - 0.25, "$x$", fontsize=10)
    ax.text(0.2, 2.2, "направления спуска: $p^\\top\\nabla f<0$\n(полуплоскость, зелёная)", fontsize=9, color=AQUA)
    ax.set(xlim=(-1, 5), ylim=(-1.5, 2.6), xlabel="$x_1$", ylabel="$x_2$", aspect="equal",
           title="Градиент $\\perp$ линии уровня; $-\\nabla f$ — самый крутой спуск")

    ax = axes[1]
    a = np.linspace(0, 0.32, 400)
    phi = np.array([f(x - ai * g) for ai in a])
    a_exact = (g @ g) / (g @ Q @ g)
    a_const = 1 / 10.0
    a_bt, fx = 1.0, f(x)
    while f(x - a_bt * g) > fx - 1e-4 * a_bt * (g @ g):
        a_bt /= 2
    assert abs(a_bt - 0.125) < 1e-12 and abs(a_exact - 109 / 1009) < 1e-12
    ax.plot(a, phi, color=INK, lw=2.2)
    ax.axhline(fx, color=GRAY, lw=0.8, ls="--")
    ax.text(0.005, fx + 0.5, "$\\varphi(0)=f(x)$ — где стоим", color=GRAY, fontsize=9)
    ax.plot(a_exact, f(x - a_exact * g), "*", color=RED, ms=14, zorder=7)
    ax.annotate(f"точный шаг $\\alpha^*={a_exact:.3f}$:\nниже некуда", (a_exact, f(x - a_exact * g)), (0.13, 1.2),
                fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.annotate("слишком малый шаг:\nпочти не сдвинулись", (0.012, phi[np.searchsorted(a, 0.012)]), (0.015, 13.5),
                fontsize=9.5, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("слишком большой шаг:\nперелетели дно, стало хуже", (0.24, phi[np.searchsorted(a, 0.24)]), (0.115, 16.5),
                fontsize=9.5, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    ax.set(xlim=(0, 0.32), ylim=(0, 20), xlabel="длина шага $\\alpha$", ylabel="$\\varphi(\\alpha)=f(x-\\alpha\\nabla f)$",
           title="Функция одного шага: идём из $x=(3,1)$ против градиента")
    save("03_descent_direction.png")
    return a_exact, a_const, a_bt


# ================================================================ 04 зигзаг: κ = 2 и κ = 50
def gd_exact_quadratic(Q, x0, tol=1e-8, maxit=100_000):
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = Q @ x
        if np.linalg.norm(g) <= tol:
            return np.array(path), k
        a = (g @ g) / (g @ Q @ g)
        x = x - a * g; path.append(x.copy())
    return np.array(path), maxit


def fig_zigzag():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), gridspec_kw=dict(width_ratios=[1, 1.35]))
    results = {}
    for ax, kappa, nshow in zip(axes, (2, 50), (12, 20)):
        Q = np.diag([1.0, float(kappa)])
        path, k = gd_exact_quadratic(Q, [kappa, 1.0])
        fvals = np.array([0.5 * p @ Q @ p for p in path])
        ratio = fvals[1:6] / fvals[:5]
        theory = ((kappa - 1) / (kappa + 1)) ** 2
        assert np.allclose(ratio, theory, atol=1e-6), (kappa, ratio, theory)
        _, k_const, _ = gd(lambda x: 0.5 * x @ Q @ x, lambda x: Q @ x, [kappa, 1.0], alpha=1 / kappa, tol=1e-8)
        results[kappa] = (k, k_const, theory)
        xl = kappa * 1.15
        X, Y = np.meshgrid(np.linspace(-xl * 0.15, xl, 400), np.linspace(-1.4, 1.4, 300))
        Z = 0.5 * (X ** 2 + kappa * Y ** 2)
        ax.contour(X, Y, Z, levels=np.geomspace(0.02, 0.5 * (kappa ** 2 + kappa), 14), colors=GRAY, linewidths=0.7)
        ax.plot(path[:nshow + 1, 0], path[:nshow + 1, 1], "-o", color=BLUE, ms=3.5, lw=1.4)
        ax.plot(0, 0, "*", color=RED, ms=12, zorder=7)
        ax.set(xlabel="$x_1$", ylabel="$x_2$",
               title=f"$\\kappa={kappa}$: точный шаг, {k} итераций до $\\Vert\\nabla f\\Vert\\leq10^{{-8}}$")
        ax.text(0.03, 0.05, f"$f$ сжимается за шаг в $\\left(\\frac{{\\kappa-1}}{{\\kappa+1}}\\right)^2={theory:.3f}$",
                transform=ax.transAxes, fontsize=9)
        if kappa == 50:
            ax.text(0.03, 0.93, "оси в разных масштабах: эллипсы вытянуты по $x_1$ в $\\sqrt{50}\\approx7$ раз",
                    transform=ax.transAxes, fontsize=8.5, color=GRAY)
    save("04_zigzag.png")
    return results


# ================================================================ 05 скорости сходимости
def fig_rates(zig):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    K = 12
    ks = np.arange(K + 1)
    lin = 0.5 * 0.5 ** ks
    sup = [0.5]
    for _ in range(K): sup.append(sup[-1] ** 1.5)
    quad = [0.5]
    for _ in range(K): quad.append(quad[-1] ** 2)
    for seq, col, lab in ((lin, BLUE, "линейная: $e_{k+1}=0.5\\,e_k$"),
                          (sup, ORANGE, "сверхлинейная: $e_{k+1}=e_k^{1.5}$"),
                          (quad, RED, "квадратичная: $e_{k+1}=e_k^2$")):
        seq = np.array(seq, float); mask = seq > 1e-16
        ax.plot(ks[mask], np.log10(seq[mask]), "-o", color=col, ms=4, label=lab)
    ax.axhline(-16, color=GRAY, lw=0.8, ls="--"); ax.text(8.3, -15.5, "точность float64", color=GRAY, fontsize=8.5)
    ax.set(xlabel="итерация $k$", ylabel="$\\log_{10}\\Vert e_k\\Vert$", ylim=(-17, 0.5),
           title="Три скорости: сколько цифр прибавляет шаг")
    ax.legend(loc="lower left", fontsize=8.5)

    ax = axes[1]
    for kappa, col in ((2, AQUA), (50, BLUE)):
        Q = np.diag([1.0, float(kappa)])
        _, k, path = gd(lambda x: 0.5 * x @ Q @ x, lambda x: Q @ x, [kappa, 1.0], alpha=1 / kappa, tol=1e-8)
        err = np.linalg.norm(path, axis=1)
        late = err[-1] / err[-2]
        assert abs(late - (1 - 1 / kappa)) < 1e-3, (kappa, late)
        ax.plot(np.arange(len(err)), np.log10(err), color=col, label=f"GD, $\\alpha=1/\\lambda_{{\\max}}$, $\\kappa={kappa}$: {k} итераций")
    ax.set(xlabel="итерация $k$", ylabel="$\\log_{10}\\Vert x_k\\Vert$", xlim=(0, 1150),
           title="Измерено на $\\frac{1}{2}(x_1^2+\\kappa x_2^2)$: прямая с наклоном $\\log_{10}(1-1/\\kappa)$")
    ax.legend(loc="upper right", fontsize=8.5)
    save("05_rates.png")


# ================================================================ 06 масштабирование переменных
def fig_scaling():
    kappa = 25
    Q = np.diag([1.0, float(kappa)])
    f = lambda x: 0.5 * x @ Q @ x
    x0 = np.array([5.0, 1.0])
    _, k1, path1 = gd(f, lambda x: Q @ x, x0, alpha=1 / kappa, tol=1e-6)
    # после замены u2 = sqrt(kappa) x2 задача — 1/2 (u1^2 + u2^2), шаг 1 попадает в минимум
    _, k2, path2 = gd(lambda u: 0.5 * u @ u, lambda u: u, [x0[0], np.sqrt(kappa) * x0[1]], alpha=1.0, tol=1e-6)
    assert k2 == 1 and k1 > 200, (k1, k2)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    X, Y = np.meshgrid(np.linspace(-1, 5.8, 400), np.linspace(-1.6, 1.6, 300))
    ax.contour(X, Y, 0.5 * (X ** 2 + kappa * Y ** 2), levels=np.geomspace(0.05, 30, 12), colors=GRAY, linewidths=0.7)
    ax.plot(path1[:40, 0], path1[:40, 1], "-o", color=BLUE, ms=3, lw=1.3)
    ax.plot(0, 0, "*", color=RED, ms=12, zorder=7)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal", title=f"$\\frac{{1}}{{2}}(x_1^2+25x_2^2)$: $\\kappa=25$, GD — {k1} итераций")
    ax = axes[1]
    X, Y = np.meshgrid(np.linspace(-1, 5.8, 400), np.linspace(-1.6, 5.8, 400))
    ax.contour(X, Y, 0.5 * (X ** 2 + Y ** 2), levels=np.geomspace(0.05, 30, 12), colors=GRAY, linewidths=0.7)
    ax.plot(path2[:, 0], path2[:, 1], "-o", color=BLUE, ms=5, lw=1.6)
    ax.plot(0, 0, "*", color=RED, ms=12, zorder=7)
    ax.set(xlabel="$u_1=x_1$", ylabel="$u_2=5x_2$", aspect="equal", title=f"после замены $u_2=5x_2$: $\\kappa=1$, GD — {k2} итерация")
    save("06_scaling.png")
    return k1


# ================================================================ 07 тизер: Ньютон против GD на Розенброке
def fig_newton_teaser():
    x0 = [-1.2, 1.0]
    xg, kg, pg = gd(rosen, rosen_grad, x0)                       # backtracking
    xn, kn, pn = newton(rosen_grad, rosen_hess, x0)
    lam = np.linalg.eigvalsh(rosen_hess([1, 1]))
    kappa = lam[1] / lam[0]
    assert abs(kappa - 2508) < 2, kappa
    assert abs(kg - 13756) < 140, kg
    assert kn == 7 and np.allclose(xn, 1, atol=1e-8), (kn, xn)
    _, kc, _ = gd(rosen, rosen_grad, x0, alpha=1e-3)
    assert abs(kc - 32076) < 330, kc
    # шаг 0.002 (чуть больше 2/λ_max в минимуме): метод не сходится, застревает в 2-цикле поперёк долины
    _, kd, pd = gd(rosen, rosen_grad, x0, alpha=2e-3, maxit=20_000)
    fd = np.array([rosen(p) for p in pd[-4:]])
    d = np.diff(fd)
    assert kd == 20_000 and np.isfinite(pd).all() and np.abs(pd).max() < 2 and (d[0] * d[1] < 0), (kd, fd)
    _, k19, _ = gd(rosen, rosen_grad, x0, alpha=1.9e-3)
    assert abs(k19 - 16851) < 170, k19

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    rosen_contour(ax, xlim=(-2.2, 2.2), ylim=(-1.2, 4.6))
    ax.plot(pg[::50, 0], pg[::50, 1], "-", color=BLUE, lw=1.2, label=f"градиентный спуск: {kg} итераций")
    ax.plot(pn[:, 0], pn[:, 1], "-o", color=ORANGE, ms=5, lw=1.6, label=f"Ньютон: {kn} итераций")
    for i, p in enumerate(pn[:4]):
        if -1.2 < p[1] < 4.6:
            ax.annotate(str(i), p, (p[0] + 0.08, p[1] + 0.1), fontsize=8.5, color=ORANGE)
    ax.text(pn[2, 0] + 0.1, -1.0, f"итерация 2: $({pn[2, 0]:.2f},\\,{pn[2, 1]:.2f})$, за краем", fontsize=8.5, color=ORANGE)
    ax.plot(*x0, "o", color=INK, ms=6, zorder=7)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Розенброк из $(-1.2,\\,1)$: пути двух методов")

    ax = axes[1]
    eg = np.linalg.norm(pg - 1, axis=1); en = np.linalg.norm(pn - 1, axis=1)
    ax.plot(np.arange(1, len(eg) + 1), np.log10(eg + 1e-17), color=BLUE, label="градиентный спуск")
    ax.plot(np.arange(1, len(en) + 1), np.log10(en + 1e-17), "-o", color=ORANGE, ms=5, label="Ньютон")
    ax.set_xscale("log")
    ax.set(xlabel="итерация $k$ (лог. шкала)", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$",
           title="Ошибка: линейное сползание против квадратичного обрыва")
    ax.legend(loc="lower left", fontsize=9)
    save("07_newton_teaser.png")
    return kappa, kg, kn, kc


# ================================================================ 08 лемма о спуске — одна панель
def fig_descent_lemma():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    L = 10.0
    f = lambda t: 0.5 * L * t ** 2 * 0.55 + 0.3 * np.sin(3 * t)          # гладкая, кривизна <= L
    df = lambda t: L * 0.55 * t + 0.9 * np.cos(3 * t)
    assert L * 0.55 + 0.3 * 9 <= L + 1e-9
    xk = 1.2
    t = np.linspace(-0.6, 1.7, 400)
    g = df(xk)
    par = f(xk) + g * (t - xk) + 0.5 * L * (t - xk) ** 2
    ax.plot(t, f(t), color=INK, lw=2.4, label="функция $f$")
    ax.plot(t, par, color=ORANGE, lw=1.8, label="парабола кривизны $L$: $f(x_k)+f'(x_k)\\,p+\\frac{L}{2}p^2$")
    ax.plot(t, f(xk) + g * (t - xk), color=GRAY, lw=1, ls=":", label="касательная")
    ax.axhline(f(xk), color=GRAY, lw=0.8, ls="--")
    x1 = xk - g / L; x2 = xk - 2 * g / L
    y1 = f(xk) + g * (x1 - xk) + 0.5 * L * (x1 - xk) ** 2
    ax.plot(xk, f(xk), "o", color=INK, ms=8, zorder=7)
    ax.plot(x1, y1, "*", color=RED, ms=15, zorder=7)
    ax.plot(x2, f(xk), "s", color=ORANGE, ms=8, zorder=7)
    ax.annotate("$x_k$: здесь стоим", (xk, f(xk)), (xk + 0.05, f(xk) + 1.0), fontsize=10)
    ax.annotate("шаг $1/L$: минимум параболы,\nфункция здесь ещё ниже", (x1, y1), (x1 - 0.45, y1 - 1.9), fontsize=10, color=RED,
                ha="center", arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.annotate("шаг $2/L$: парабола вернулась\nна уровень $f(x_k)$ — дальше гарантии нет", (x2, f(xk)), (x2 + 0.05, f(xk) + 2.4),
                fontsize=10, color=ORANGE, arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))
    ax.set(xlim=(-0.6, 1.7), ylim=(-1.6, f(xk) + 3.6), xlabel="$x$", yticks=[],
           title="Лемма о спуске: кривизна $f$ не больше $L$, значит $f$ лежит под параболой")
    ax.legend(loc="lower right", fontsize=9)
    save("08_descent_lemma.png")


# ================================================================ 09 множители сжатия компонент как функции шага
def fig_step_multipliers():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    kappa = 50.0; lmin, lmax = 1.0, kappa
    a = np.linspace(0, 2.4 / kappa, 400)
    ax.plot(a, np.abs(1 - a * lmin), color=BLUE, lw=2.2, label="медленная компонента (вдоль $\\lambda_{\\min}=1$): $|1-\\alpha\\lambda_{\\min}|$")
    ax.plot(a, np.abs(1 - a * lmax), color=ORANGE, lw=2.2, label="быстрая компонента (вдоль $\\lambda_{\\max}=50$): $|1-\\alpha\\lambda_{\\max}|$")
    ax.axhline(1, color=RED, lw=1, ls="--")
    ax.text(0.001, 1.03, "множитель 1: компонента не убывает", color=RED, fontsize=9)
    a_best = 2 / (lmin + lmax); q_best = (kappa - 1) / (kappa + 1)
    assert abs(np.abs(1 - a_best * lmin) - q_best) < 1e-12 and abs(np.abs(1 - a_best * lmax) - q_best) < 1e-12
    ax.plot(a_best, q_best, "*", color=RED, ms=15, zorder=7)
    ax.plot(1 / lmax, 1 - 1 / kappa, "o", color=BLUE, ms=8, zorder=7)
    ax.plot(2 / lmax, 1, "s", color=RED, ms=8, zorder=7)
    ax.annotate(f"шаг $1/\\lambda_{{\\max}}$: быстрая компонента исчезает за шаг,\nмедленная сжимается лишь в $1-1/\\kappa={1 - 1 / kappa:.2f}$",
                (1 / lmax, 1 - 1 / kappa), (0.0255, 0.12), fontsize=9.5, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate(f"лучший шаг $\\frac{{2}}{{\\lambda_{{\\max}}+\\lambda_{{\\min}}}}$: обе компоненты\nсжимаются одинаково, в $\\frac{{\\kappa-1}}{{\\kappa+1}}={q_best:.3f}$",
                (a_best, q_best), (0.0012, 0.16), fontsize=9, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.annotate("шаг $2/\\lambda_{\\max}$: быстрая компонента\nперестаёт убывать — граница", (2 / lmax, 1), (0.022, 1.28),
                fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
    ax.set(xlim=(0, 2.4 / kappa), ylim=(0, 1.7), xlabel="длина шага $\\alpha$", ylabel="во сколько раз сжалась компонента за итерацию",
           title="$\\frac{1}{2}(x_1^2+50x_2^2)$: каждая компонента ошибки живёт своей жизнью")
    ax.legend(loc="upper left", fontsize=9)
    save("09_step_multipliers.png")


# ================================================================ 12: квадратичные ландшафты в 3D
def fig_quadric_3d():
    """Три поверхности 1/2 x^T Q x: чаша, седло, жёлоб — таблица раздела 3 в 3D."""
    cases = (
        (np.diag([1.0, 3.0]), "чаша: $\\lambda=1,\\;3$ — минимум"),
        (np.diag([1.0, -2.0]), "седло: $\\lambda=1,\\;-2$ — не минимум"),
        (np.diag([1.0, 0.0]), "жёлоб: $\\lambda=1,\\;0$ — прямая минимумов"),
    )
    lim = 1.6
    X, Y = np.meshgrid(np.linspace(-lim, lim, 120), np.linspace(-lim, lim, 120))
    fig = plt.figure(figsize=(12, 3.9))
    for i, (Q, title) in enumerate(cases, 1):
        Z = 0.5 * (Q[0, 0] * X**2 + Q[1, 1] * Y**2)
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="Blues_r", alpha=0.72, rcount=60, ccount=60, linewidth=0)
        ax.contour(X, Y, Z, levels=7, colors=GRAY, linewidths=0.5, offset=float(Z.min()))
        if i == 1:
            ax.computed_zorder = False
            ax.scatter([0], [0], [0.05], color=RED, s=70, depthshade=False, zorder=10, edgecolors="white", linewidths=0.8)
        if i == 3:
            ax.computed_zorder = False
            t = np.linspace(-lim, lim, 50)
            ax.plot(np.zeros_like(t), t, np.full_like(t, 0.03), color=RED, lw=2.5, zorder=10)
        ax.view_init(elev=27, azim=-60)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("$x_1$", labelpad=-4); ax.set_ylabel("$x_2$", labelpad=-4)
        ax.set_zticks([]); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(pad=-2, labelsize=7)
        ax.grid(False)
    plt.savefig(IMG / "12_quadric_3d.png", bbox_inches="tight")
    plt.close()
    print("  ", "12_quadric_3d.png")


# ================================================================ проверки чисел для конспекта
def check_chain():
    for N, expect in ((10, 48.4), (20, 178.1), (40, 680.6), (80, 2658.4)):
        H, b, _, _ = chain_matrices(N)
        lam = np.linalg.eigvalsh(H)
        kappa = lam[-1] / lam[0]
        formula = (2 * (N + 1) / np.pi) ** 2
        assert abs(kappa - expect) < 0.5 and abs(kappa - formula) / formula < 0.02, (N, kappa, formula)
    H, b, _, _ = chain_matrices(40)
    L = np.linalg.eigvalsh(H)[-1]
    assert abs(L - 279.6) < 0.5, L
    return L


if __name__ == "__main__":
    print("рисую в", IMG)
    fig_hooks()
    fig_stationary()
    fig_second_order()
    a_exact, a_const, a_bt = fig_descent_direction()
    zig = fig_zigzag()
    fig_rates(zig)
    k_scaling = fig_scaling()
    kappa, kg, kn, kc = fig_newton_teaser()
    fig_descent_lemma()
    fig_step_multipliers()
    fig_quadric_3d()
    L = check_chain()
    print(f"проверки: κ Розенброка = {kappa:.1f}; GD backtracking {kg}, постоянный шаг 0.001: {kc}, Ньютон {kn};")
    print(f"          зигзаг: {zig}; масштабирование: {k_scaling} итераций до замены; цепь N=40: L = {L:.1f}")
