"""Иллюстрации к конспекту лекции 8 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture08/scripts/make_figures.py

Скрипт самодостаточен: окружность, клюв, параболы, цепь на стержнях и портфель определены здесь тем же
кодом, что в build_demo08.py / l8helpers.py; числа конспекта считаются в скрипте и проверяются ассертами.
Цепь на стержнях и её множители — через CasADi + IPOPT.

Картинки в порядке появления в конспекте:
  01_review               касание (Л3): линия уровня f = 1/2|x - a|^2 касается окружности; рядом точка без касания
  02_circle_two_points    две точки Лагранжа на окружности: lambda = 1 - sqrt5 (минимум) и 1 + sqrt5 (максимум)
  03_tangent_space        разложение grad f на нормальную часть lambda grad g и касательную часть; в оптимуме касательной нет
  04_cusp_licq            клюв x2 = x1^(2/3): grad g -> 0, угол между grad f и grad g -> 90 градусов, оценка lambda -> inf
  05_parabola_curvature   f = x2 на параболах x2 = x1^2 и x2 = -x1^2: одно условие первого порядка, минимум и максимум
  06_curve_cut            разрез f вдоль кривой: phi(t) = c t^2 для c = 1, 0, -1
  07_chain_rods           цепь на стержнях: форма, натяжения по стержням, постоянная горизонтальная составляющая
  08_portfolio_price      f*(rho) портфеля с наклоном lambda_rho; энергия цепи от общей длины с наклоном по множителям
  09_ipopt_residuals      inf_pr и inf_du IPOPT на цепи на стержнях по итерациям
"""

from pathlib import Path

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy.linalg import null_space
from scipy.optimize import fsolve

IMG = Path(__file__).resolve().parent.parent / "img"
IMG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7")
GRAY, INK = "#8a8985", "#0b0b0b"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "legend.frameon": False,
    "figure.dpi": 100, "savefig.dpi": 100,
})

MAX_WIDTH = 920        # px: картинка показывается с width="900", уменьшать её нельзя


def save(name: str) -> None:
    plt.tight_layout()
    path = IMG / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    w, h = Image.open(path).size
    assert 600 <= w <= MAX_WIDTH, f"{name}: ширина {w} px — нужна 600–{MAX_WIDTH}, иначе подписи нечитаемы"
    print(f"   {name}: {w}x{h}")


def arrow(ax, p, q, color, lw=2, ms=14, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=6, **kw))


# ---------------------------------------------------------------- задачи (те же, что в demo08 / l8helpers.py)
A_CIRCLE = np.array([2.0, 1.0])            # введение и §2.1: ближайшая к a точка единичной окружности


def circle_system(z, a=A_CIRCLE):
    """Система условий первого порядка для min 1/2|x - a|^2 при 1/2(|x|^2 - 1) = 0: (x - a) - lam x = 0, g = 0."""
    x, lam = z[:2], z[2]
    return np.r_[(x - a) - lam * x, 0.5 * (x @ x - 1)]


def circle_points(a=A_CIRCLE):
    """Оба решения системы: (x, lam) ближней и дальней точки."""
    na = np.linalg.norm(a)
    return (a / na, 1 - na), (-a / na, 1 + na)


def cusp_grad(x1):
    """Клюв g = x2^3 - x1^2 = 0, f = x2: в точке кривой x2 = x1^(2/3) возвращает grad g, оценку lambda, угол к grad f (град)."""
    x2 = x1 ** (2 / 3)
    gf, gg = np.array([0.0, 1.0]), np.array([-2 * x1, 3 * x2 ** 2])
    lam = gf @ gg / (gg @ gg)                                  # наименьшие квадраты для grad f = lam grad g
    ang = np.degrees(np.arccos(gf @ gg / np.linalg.norm(gg)))
    return gg, lam, ang


def rod_chain(N=40, L_total=5.0, g_acc=9.81):
    """Цепь на стержнях: N грузов между (-2, 1) и (2, 1), N + 1 стержней длины L_total / (N + 1), f = w sum z.
    Возвращает словарь с символьными v, f, g (CasADi), весом w, длиной стержня ell и функцией старта."""
    m = 4.0 / N; w = m * g_acc; ell = L_total / (N + 1)
    v = ca.SX.sym("v", 2 * N)
    y = ca.vertcat(-2.0, v[:N], 2.0); z = ca.vertcat(1.0, v[N:], 1.0)
    f = w * ca.sum1(z[1:-1])
    g = ca.vertcat(*[(y[j + 1] - y[j]) ** 2 + (z[j + 1] - z[j]) ** 2 - ell ** 2 for j in range(N + 1)])
    t = np.linspace(-2, 2, N + 2)[1:-1]
    return dict(v=v, f=f, g=g, w=w, ell=ell, N=N, v0=np.r_[t, np.zeros(N)])


def solve_rod_chain(ch, v0=None, tol=1e-10):
    """IPOPT для цепи на стержнях. Возвращает (v*, lam* в нашем знаке L = f - lam^T g, stats)."""
    solver = ca.nlpsol("s", "ipopt", {"x": ch["v"], "f": ch["f"], "g": ch["g"]},
                       {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.tol": tol})
    sol = solver(x0=ch["v0"] if v0 is None else v0, lbg=0, ubg=0)
    return np.array(sol["x"]).ravel(), -np.array(sol["lam_g"]).ravel(), solver.stats()   # IPOPT: L = f + lam_g^T g


def chain_xy(ch, v):
    N = ch["N"]
    return np.r_[-2.0, v[:N], 2.0], np.r_[1.0, v[N:], 1.0]


def chain_checks(ch, v, lam):
    """Три проверки §5.1: невязка стационарности, ранг J, собственные числа редуцированного гессиана; натяжения."""
    J = ca.Function("J", [ch["v"]], [ca.jacobian(ch["g"], ch["v"])])(v).full()
    gf = ca.Function("gf", [ch["v"]], [ca.gradient(ch["f"], ch["v"])])(v).full().ravel()
    lsym = ca.SX.sym("l", ch["N"] + 1)
    H = ca.Function("H", [ch["v"], lsym], [ca.hessian(ch["f"] - ca.dot(lsym, ch["g"]), ch["v"])[0]])(v, lam).full()
    Z = null_space(J)
    y, z = chain_xy(ch, v)
    F = 2 * np.abs(lam) * ch["ell"]                            # натяжение стержня
    Fh = 2 * lam * np.diff(y)                                  # горизонтальная составляющая
    return dict(res=np.linalg.norm(gf - J.T @ lam), rank=np.linalg.matrix_rank(J), eig=np.linalg.eigvalsh(Z.T @ H @ Z),
                Z=Z, F=F, Fh=Fh, Fv=2 * lam * np.diff(z))


PF_R = np.array([0.04, 0.08, 0.11, 0.15])                              # портфель Л3: доходности
PF_VOL = np.array([0.05, 0.12, 0.18, 0.30])
PF_CORR = np.array([[1, 0.2, 0.1, 0.0], [0.2, 1, 0.5, 0.3], [0.1, 0.5, 1, 0.4], [0.0, 0.3, 0.4, 1.0]])
PF_SIGMA = np.outer(PF_VOL, PF_VOL) * PF_CORR


def portfolio_eq(rho):
    """min 1/2 x'Sx при 1'x = 1, r'x = rho — только равенства: система ККТ линейна. Возвращает (x, f*, lam)."""
    Aeq = np.c_[np.ones(4), PF_R].T
    K = np.block([[PF_SIGMA, -Aeq.T], [Aeq, np.zeros((2, 2))]])
    sol = np.linalg.solve(K, np.r_[np.zeros(4), 1.0, rho])
    return sol[:4], 0.5 * sol[:4] @ PF_SIGMA @ sol[:4], sol[4:]


# ---------------------------------------------------------------- числа конспекта
def numbers():
    (xmin, lmin), (xmax, lmax) = circle_points()
    assert abs(lmin + 1.2361) < 1e-4 and abs(lmax - 3.2361) < 1e-4 and np.allclose(xmin, [0.8944, 0.4472], atol=1e-4)
    for z0 in ([1, 0.5, 0], [-1, -0.5, 0]):
        z = fsolve(circle_system, z0, xtol=1e-14)
        assert min(abs(z[2] - lmin), abs(z[2] - lmax)) < 1e-8
    for x1, lam_exp, ang_exp in ((0.1, 2.3, 55), (0.01, 14.6, 72), (0.001, 73, 81)):
        gg, lam, ang = cusp_grad(x1)
        assert abs(lam - lam_exp) < 0.15 * lam_exp and abs(ang - ang_exp) < 1, (x1, lam, ang)

    ch = rod_chain()
    v, lam, st = solve_rod_chain(ch)
    chk = chain_checks(ch, v, lam)
    y, z = chain_xy(ch, v)
    f_star = ch["w"] * z[1:-1].sum()
    print(f"   цепь: {st['iter_count']} итераций, f* = {f_star:.4f}, z_min = {z.min():.4f}, lam концы/середина = "
          f"{lam[0]:.2f}/{lam[20]:.2f}, F = {chk['F'][0]:.2f}/{chk['F'][20]:.2f} Н, F_h = {chk['Fh'].mean():.3f}, "
          f"rank J = {chk['rank']}, eig red = {chk['eig'].min():.2f}..{chk['eig'].max():.1f}")
    assert st["iter_count"] == 13 and st["return_status"] == "Solve_Succeeded", st["iter_count"]
    assert abs(f_star - 5.7564) < 1e-3 and abs(z.min() + 0.327) < 1e-3
    assert abs(lam[0] + 97.88) < 0.05 and abs(lam[20] + 55.76) < 0.05
    assert abs(chk["F"][0] - 23.87) < 0.02 and abs(chk["F"][20] - 13.60) < 0.02
    assert np.ptp(chk["Fh"]) < 1e-6 and abs(abs(chk["Fh"].mean()) - 13.60) < 0.02       # горизонтальная одинакова
    assert abs(abs(chk["Fv"][0]) - 19.62) < 0.02 and abs(ch["w"] * ch["N"] - 39.24) < 1e-6   # половина веса
    assert chk["res"] < 1e-9 and chk["rank"] == 41 and chk["Z"].shape == (80, 39)
    assert chk["eig"].min() > 2.1 and 650 < chk["eig"].max() < 700, (chk["eig"].min(), chk["eig"].max())
    # чувствительность по длине стержней: +1 %
    ch2 = rod_chain(L_total=5.05)
    v2, _, _ = solve_rod_chain(ch2, v0=v)
    f2 = ch2["w"] * chain_xy(ch2, v2)[1][1:-1].sum()
    f_pred = f_star + lam.sum() * (ch2["ell"] ** 2 - ch["ell"] ** 2)
    assert abs(f2 - 4.886) < 1e-3 and abs(f_pred - 4.873) < 1e-3, (f2, f_pred)

    x, fs, lam_pf = portfolio_eq(0.10)
    assert np.allclose(x, [0.0946, 0.3822, 0.3209, 0.2023], atol=2e-4) and abs(fs - 0.008204) < 1e-6
    assert abs(lam_pf[1] - 0.2571) < 1e-4 and abs(lam_pf[0] + 0.0093) < 1e-4, lam_pf
    d = 1e-5
    assert abs((portfolio_eq(0.10 + d)[1] - portfolio_eq(0.10 - d)[1]) / (2 * d) - lam_pf[1]) < 1e-6
    return ch, v, lam, st, chk


# ---------------------------------------------------------------- рисовалки
def level_circles(ax, a, levels, color=BLUE, lw=1.2, alpha=0.8):
    th = np.linspace(0, 2 * np.pi, 400)
    for f in levels:
        r = np.sqrt(2 * f)
        ax.plot(a[0] + r * np.cos(th), a[1] + r * np.sin(th), color=color, lw=lw, alpha=alpha)


def unit_circle(ax, color=INK, lw=2.2):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=color, lw=lw, label="$g(x)=0$")


def fig_review():
    a = A_CIRCLE
    (xs, lam), _ = circle_points()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, x, title in ((axes[0], xs, "В минимуме: касание"), (axes[1], np.array([0.5, np.sqrt(3) / 2]), "Не минимум: касания нет")):
        unit_circle(ax)
        fx = 0.5 * np.sum((x - a) ** 2)
        level_circles(ax, a, [fx] + [v for v in (0.3, 1.5, 3.0) if abs(v - fx) > 0.2], color=BLUE)
        ax.plot(*a, "o", color=INK, ms=6); ax.annotate("$a$", a, (6, 4), textcoords="offset points")
        gf, gg = x - a, x
        s = 0.45
        arrow(ax, x, x + s * gf, RED); arrow(ax, x, x + s * gg, AQUA)
        ax.annotate(r"$\nabla f$", x + s * gf, (4, -14), textcoords="offset points", color=RED)
        ax.annotate(r"$\nabla g$", x + s * gg, (4, 4), textcoords="offset points", color=AQUA)
        if ax is axes[1]:
            t = np.array([-x[1], x[0]]); tp = (gf @ t) * t
            arrow(ax, x, x + s * tp, ORANGE, lw=2)
            ax.annotate("касательная\nчасть", x + s * tp, (-70, 6), textcoords="offset points", color=ORANGE, fontsize=10)
        ax.plot(*x, "o", color=ORANGE if ax is axes[1] else RED, ms=7, zorder=7)
        ax.set_aspect("equal"); ax.set_xlim(-1.4, 2.6); ax.set_ylim(-1.4, 2.0)
        ax.set_title(title); ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].annotate(f"$f={0.5 * np.sum((xs - a) ** 2):.3f}$, $\\lambda={lam:.3f}$", (-1.3, 1.7), fontsize=11)
    axes[0].legend(loc="lower left")
    save("01_review.png")


def fig_circle_two_points():
    a = A_CIRCLE
    (x1, l1), (x2, l2) = circle_points()
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    unit_circle(ax)
    level_circles(ax, a, [0.5 * np.sum((x1 - a) ** 2), 0.5 * np.sum((x2 - a) ** 2)], color=BLUE, lw=1.8)
    level_circles(ax, a, [0.2, 2.0, 3.5], color=BLUE, lw=0.8, alpha=0.5)
    ax.plot(*a, "o", color=INK, ms=6); ax.annotate("$a=(2,1)$", a, (6, 4), textcoords="offset points")
    for x, lam, col, name in ((x1, l1, RED, "минимум"), (x2, l2, VIOLET, "максимум")):
        gf, gg = x - a, x
        s = 0.35
        arrow(ax, x, x + s * gf, RED); arrow(ax, x, x + s * gg, AQUA)
        ax.plot(*x, "o", color=col, ms=8, zorder=7)
        ax.annotate(f"$x=({x[0]:.3f},{x[1]:.3f})$\n$\\lambda={lam:.3f}$, $f={0.5 * np.sum((x - a) ** 2):.3f}$\n{name} (§4)",
                    (1.45, -0.55) if lam < 0 else (-2.1, -1.75), fontsize=11, color=col)
    ax.annotate(r"$\nabla f$", x1 + 0.35 * (x1 - a), (-22, 6), textcoords="offset points", color=RED)
    ax.annotate(r"$\nabla g$", x1 + 0.35 * x1, (4, 4), textcoords="offset points", color=AQUA)
    ax.set_aspect("equal"); ax.set_xlim(-2.2, 3.6); ax.set_ylim(-1.9, 2.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Две точки касания, два $\\lambda$")
    save("02_circle_two_points.png")


def fig_tangent_space():
    a = A_CIRCLE
    (xs, lam), _ = circle_points()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, x, title in ((axes[0], np.array([np.cos(1.1), np.sin(1.1)]), "Не оптимум: есть касательная часть"),
                         (axes[1], xs, r"Оптимум: $\nabla f=\lambda\nabla g$")):
        unit_circle(ax)
        gf, gg = x - a, x
        t = np.array([-x[1], x[0]])
        lam_ls = gf @ gg / (gg @ gg); normal = lam_ls * gg; tang = gf - normal
        s = 0.5
        ax.plot([x[0] - 1.2 * t[0], x[0] + 1.2 * t[0]], [x[1] - 1.2 * t[1], x[1] + 1.2 * t[1]], color=GRAY, lw=1.2, ls="--")
        ax.annotate("$T=\\ker J$", x + 1.1 * t, (4, 4), textcoords="offset points", color=GRAY, fontsize=11)
        arrow(ax, x, x + s * gf, RED, lw=2.4)
        arrow(ax, x, x + s * normal, AQUA)
        if np.linalg.norm(tang) > 1e-6:
            arrow(ax, x, x + s * tang, ORANGE)
            ax.plot([x[0] + s * normal[0], x[0] + s * gf[0]], [x[1] + s * normal[1], x[1] + s * gf[1]], color=GRAY, lw=0.8, ls=":")
            ax.plot([x[0] + s * tang[0], x[0] + s * gf[0]], [x[1] + s * tang[1], x[1] + s * gf[1]], color=GRAY, lw=0.8, ls=":")
            ax.annotate("касательная часть", x + s * tang, (-50, 8), textcoords="offset points", color=ORANGE, fontsize=10)
        ax.annotate(r"$\nabla f$", x + s * gf, (-22, -2) if ax is axes[0] else (6, 6), textcoords="offset points", color=RED)
        ax.annotate(r"$\lambda\nabla g$", x + s * normal, (4, -14) if ax is axes[0] else (-34, -14), textcoords="offset points", color=AQUA)
        ax.plot(*x, "o", color=INK, ms=6, zorder=7)
        ax.plot(*a, "o", color=INK, ms=5); ax.annotate("$a$", a, (6, 4), textcoords="offset points")
        ax.set_aspect("equal"); ax.set_xlim(-1.5, 2.6); ax.set_ylim(-1.3, 2.1); ax.set_title(title); ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    save("03_tangent_space.png")


def fig_cusp_licq():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    x1 = np.linspace(-0.7, 0.7, 600)
    ax.plot(x1, np.abs(x1) ** (2 / 3), color=INK)
    ax.plot(0, 0, "o", color=RED, ms=8, zorder=7); ax.annotate("$x^\\ast=(0,0)$", (0, 0), (-70, -16), textcoords="offset points", color=RED)
    for xx, col, off in ((0.5, BLUE, (-6, 6)), (0.1, AQUA, (8, 4)), (0.01, ORANGE, (8, -10))):
        p = np.array([xx, xx ** (2 / 3)]); gg = cusp_grad(xx)[0]
        arrow(ax, p, p + 0.3 * gg, col)
        ax.annotate(f"$\\nabla g$, $x_1={xx}$", p + 0.3 * gg, off, textcoords="offset points", color=col, fontsize=9)
    arrow(ax, (-0.55, 0.45), (-0.55, 0.75), RED); ax.annotate(r"$\nabla f=(0,1)$", (-0.55, 0.75), (-28, 5), textcoords="offset points", color=RED, fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(-0.75, 1.0); ax.set_ylim(-0.15, 0.95)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Клюв $x_2^3-x_1^2=0$: $\\nabla g\\to0$")
    xs = np.logspace(-4, -0.3, 80)
    lam = np.array([cusp_grad(x)[1] for x in xs]); ang = np.array([cusp_grad(x)[2] for x in xs])
    axes[1].semilogx(xs, ang, color=VIOLET); axes[1].axhline(90, color=GRAY, ls="--", lw=1)
    axes[1].set_xlabel("$x_1$"); axes[1].set_ylabel("градусы"); axes[1].set_title(r"Угол между $\nabla f$ и $\nabla g$")
    axes[2].loglog(xs, lam, color=ORANGE, label=r"$\hat\lambda$"); axes[2].loglog(xs, 0.75 * xs ** (-2 / 3), color=GRAY, ls="--", lw=1, label="$0.75\\,x_1^{-2/3}$")
    axes[2].set_xlabel("$x_1$"); axes[2].set_title(r"Оценка $\lambda$ растёт"); axes[2].legend(fontsize=10)
    for xx in (0.1, 0.01, 0.001):
        axes[2].plot(xx, cusp_grad(xx)[1], "o", color=ORANGE, ms=5)
    save("04_cusp_licq.png")


def fig_parabola_curvature():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=True)
    x1 = np.linspace(-1.1, 1.1, 300)
    for ax, c, name, col in ((axes[0], 1.0, "минимум", RED), (axes[1], -1.0, "максимум", VIOLET)):
        for lev in np.arange(-1.0, 1.01, 0.25):                         # линии уровня f = x2 — горизонтали
            ax.axhline(lev, color=BLUE, lw=0.8, alpha=0.45)
        ax.plot(x1, c * x1 ** 2, color=INK, label="$g=x_2-x_1^2=0$" if c > 0 else "$g=x_2+x_1^2=0$")
        arrow(ax, (0, 0), (0, 0.5), RED, lw=2.6); arrow(ax, (0, 0), (0, 0.5), AQUA, lw=1.2)
        ax.annotate(r"$\nabla f=\nabla g=(0,1)$, $\lambda=1$", (0, 0.5), (8, 2), textcoords="offset points", fontsize=11)
        ax.plot(0, 0, "o", color=col, ms=8, zorder=7); ax.annotate(name, (0, 0), (8, -16), textcoords="offset points", color=col, fontsize=12)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xlabel("$x_1$"); ax.legend(loc="lower left" if c > 0 else "upper left", fontsize=10)
        ax.set_title(f"$f=x_2$ на параболе, $c={c:+.0f}$".replace("+1", "1"))
    axes[0].set_ylabel("$x_2$")
    save("05_parabola_curvature.png")


def fig_curve_cut():
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    t = np.linspace(-1, 1, 200)
    for c, col, name in ((1, RED, "$c=1$: минимум"), (0, GRAY, "$c=0$: прямая"), (-1, VIOLET, "$c=-1$: максимум")):
        ax.plot(t, c * t ** 2, color=col, label=name)
    ax.plot(0, 0, "o", color=INK, ms=6, zorder=7)
    ax.set_xlabel("$t$ — параметр вдоль кривой $x(t)=(t,\\,ct^2)$"); ax.set_ylabel(r"$\varphi(t)=f(x(t))$")
    ax.set_title(r"Разрез $f$ вдоль кривой: $\varphi''(0)=p^\top\nabla^2_{xx}L\,p=2c$"); ax.legend()
    save("06_curve_cut.png")


def fig_chain_rods(ch, v, lam, chk):
    y, z = chain_xy(ch, v)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    ax.plot(y, z, "-", color=INK, lw=1.5); ax.plot(y[1:-1], z[1:-1], "o", color=BLUE, ms=3.5)
    ax.plot([-2, 2], [1, 1], "s", color=INK, ms=7)
    for j, lab in ((0, "конец"), (20, "середина")):                      # сила стержня j на груз j + 1 (и на груз j — в обратную сторону)
        p = np.array([y[j + 1], z[j + 1]]); d = np.array([y[j] - y[j + 1], z[j] - z[j + 1]]); d /= np.linalg.norm(d)
        q = p + 0.025 * chk["F"][j] * d
        arrow(ax, p, q, ORANGE, lw=2.4)
        ax.annotate(f"$F_{{{j}}}={chk['F'][j]:.1f}$ Н", q, (4, 2) if j == 0 else (12, -18), textcoords="offset points", color=ORANGE, fontsize=11)
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.3); ax.set_ylim(-0.7, 1.55)
    ax.set_xlabel("$y$"); ax.set_ylabel("$z$"); ax.set_title(f"Цепь на стержнях: $f^\\ast={ch['w'] * z[1:-1].sum():.3f}$ Дж")
    ax = axes[1]
    j = np.arange(ch["N"] + 1)
    ax.plot(j, chk["F"], "o-", color=ORANGE, ms=3, label="натяжение $F_j=2|\\lambda_j|\\ell$")
    ax.plot(j, np.abs(chk["Fh"]), "-", color=AQUA, label=f"горизонтальная: ${abs(chk['Fh'].mean()):.1f}$ Н везде")
    ax.set_xlabel("номер стержня $j$"); ax.set_ylabel("Н"); ax.set_ylim(0, 27); ax.set_title("$\\lambda_j$ — это сила")
    ax.legend(loc="lower center", fontsize=10)
    save("07_chain_rods.png")


def fig_portfolio_price(ch, v, lam):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax = axes[0]
    rhos = np.linspace(0.06, 0.14, 81)
    fs = np.array([portfolio_eq(r)[1] for r in rhos])
    x0, f0, l0 = portfolio_eq(0.10)
    ax.plot(100 * rhos, fs, color=BLUE, label="$f^\\ast(\\rho)$ — минимальный риск")
    ax.plot(100 * rhos, f0 + l0[1] * (rhos - 0.10), color=ORANGE, ls="--", label=f"касательная, наклон $\\lambda_\\rho={l0[1]:.3f}$")
    ax.plot(10, f0, "o", color=RED, ms=7, zorder=7)
    ax.set_xlabel(r"цель доходности $\rho$, %"); ax.set_ylabel(r"$\frac{1}{2}x^\top\Sigma x$"); ax.set_title("Портфель: $\\lambda_\\rho$ — цена доходности")
    ax.set_ylim(0, 0.025); ax.legend(fontsize=10, loc="upper left")
    ax = axes[1]
    Ls = np.linspace(4.9, 5.1, 9)
    fL = []
    for L in Ls:
        chL = rod_chain(L_total=L); vL, _, _ = solve_rod_chain(chL, v0=v)
        fL.append(chL["w"] * chain_xy(chL, vL)[1][1:-1].sum())
    fL = np.array(fL)
    f_star = ch["w"] * chain_xy(ch, v)[1][1:-1].sum()
    slope = lam.sum() * 2 * ch["ell"] / (ch["N"] + 1)                      # d f*/dL = sum lam_j d(ell^2)/dL
    ax.plot(Ls, fL, "o-", color=BLUE, ms=4, label="$f^\\ast(L)$ — пересчёт IPOPT")
    ax.plot(Ls, f_star + slope * (Ls - 5.0), color=ORANGE, ls="--", label=f"наклон по множителям: ${slope:.1f}$ Дж/м")
    ax.plot(5.0, f_star, "o", color=RED, ms=7, zorder=7)
    ax.set_xlabel("общая длина цепи $L$, м"); ax.set_ylabel("$f^\\ast$, Дж"); ax.set_title("Цепь: $\\sum\\lambda_j$ — цена длины")
    ax.legend(fontsize=10, loc="upper right")
    assert abs(fL[4] - f_star) < 1e-9 and -18 < slope < -17, slope   # (f(5.05) - f(5)) / 0.05 = -17.4
    save("08_portfolio_price.png")


def fig_ipopt_residuals(st):
    it = st["iterations"]
    pr, du = np.array(it["inf_pr"]), np.array(it["inf_du"])
    k = np.arange(len(pr))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.semilogy(k, np.maximum(pr, 1e-17), "o-", color=BLUE, ms=4, label=r"inf_pr $=\|g(x_k)\|_\infty$ — допустимость")
    ax.semilogy(k, np.maximum(du, 1e-17), "s-", color=ORANGE, ms=4, label=r"inf_du $=\|\nabla_xL(x_k,\lambda_k)\|_\infty$ — стационарность")
    ax.axhline(1e-8, color=GRAY, ls="--", lw=1); ax.annotate("tol $=10^{-8}$", (0.3, 2e-8), color=GRAY, fontsize=10)
    ax.set_xlabel("итерация IPOPT"); ax.set_ylabel("невязка"); ax.set_title(f"Две невязки системы: {len(pr) - 1} итераций")
    ax.set_ylim(1e-17, 1e4); ax.legend(fontsize=10, loc="lower left")
    save("09_ipopt_residuals.png")


if __name__ == "__main__":
    print("числа конспекта:")
    ch, v, lam, st, chk = numbers()
    print("картинки:")
    fig_review()
    fig_circle_two_points()
    fig_tangent_space()
    fig_cusp_licq()
    fig_parabola_curvature()
    fig_curve_cut()
    fig_chain_rods(ch, v, lam, chk)
    fig_portfolio_price(ch, v, lam)
    fig_ipopt_residuals(st)
