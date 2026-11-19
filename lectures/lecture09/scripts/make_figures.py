"""Иллюстрации к конспекту лекции 9 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture09/scripts/make_figures.py

Скрипт самодостаточен: метод Ньютона–Лагранжа, обобщённый Гаусс–Ньютон, дом у дороги и бросок мяча определены
здесь тем же кодом, что в build_demo09.py / l9helpers.py; числа конспекта считаются в скрипте и проверяются
ассертами. Бросок и сверка с IPOPT — через CasADi.

Картинки в порядке появления в конспекте:
  00_road           введение, задача 1: дорога x2 = x1^2, дом a = (2, 1) и старт (1, 0) под дорогой
  00_throw          введение, задача 2: броски под разными углами в одну точку D = 30 м
  01_nl_road        §2.3: Ньютон–Лагранж на дороге из (1, 0) — первый шаг до касательной, норма невязки
  02_nl_throw       §2.3: бросок — полёт мяча по итерациям; две половины невязки и лог IPOPT
  03_road_vertex    §3.4, спрятанный ответ: дом над вершиной, из вершины метод не уходит — это самая дальняя точка
  04_step_split     §4: шаг из (1, 0) = часть к линеаризованному ограничению + часть вдоль него
  05_ggn_road       §5: обобщённый Гаусс–Ньютон на дороге — медленные шаги к вершине и ошибка при трёх домах
"""

from pathlib import Path

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image

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


def save(name: str, min_width: int = 600) -> None:
    plt.tight_layout()
    path = IMG / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    w, h = Image.open(path).size
    assert min_width <= w <= MAX_WIDTH, f"{name}: ширина {w} px — нужна {min_width}–{MAX_WIDTH}, иначе подписи нечитаемы"
    print(f"   {name}: {w}x{h}")


def arrow(ax, p, q, color, lw=2, ms=14, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=6, **kw))


# ---------------------------------------------------------------- методы (те же, что в demo09 / l9helpers.py)
def newton_lagrange(kkt, x0, lam0, tol=1e-10, maxit=50):
    """Метод Ньютона–Лагранжа для min f при g(x) = 0, знак L = f - lam^T g.
    kkt(x, lam) -> (grad f, J, H, g), где H — гессиан L по x.
    Шаг: [[H, -J^T], [J, 0]] (p, dlam) = -(grad_x L, g).
    Возвращает список (x_k, lam_k, |F_k|) по итерациям, включая старт."""
    x, lam = np.array(x0, float), np.array(lam0, float)
    n, m = len(x), len(lam)
    hist = []
    for k in range(maxit + 1):
        gf, J, H, g = kkt(x, lam)
        F = np.concatenate([gf - J.T @ lam, g])                                     # невязка системы
        hist.append((x.copy(), lam.copy(), np.linalg.norm(F)))
        if hist[-1][2] <= tol or k == maxit:
            break
        K = np.block([[H, -J.T], [J, np.zeros((m, m))]])                           # матрица ККТ
        s = np.linalg.solve(K, -F)
        x, lam = x + s[:n], lam + s[n:]
    return hist


def casadi_kkt(x, f, g):
    """Производные для newton_lagrange от CasADi (алгоритмическое дифференцирование, лекция 7).
    x, f, g — символьные выражения CasADi. Возвращает kkt(x, lam) -> (grad f, J, H, g) в numpy, H — гессиан L = f - lam^T g."""
    lam = ca.SX.sym("lam", g.shape[0])
    fun = ca.Function("kkt", [x, lam], [ca.gradient(f, x), ca.jacobian(g, x), ca.hessian(f - ca.dot(lam, g), x)[0], g])

    def kkt(xv, lv):
        gf, J, H, gv = fun(xv, lv)
        return np.array(gf.full()).ravel(), np.array(J.full()), np.array(H.full()), np.array(gv.full()).ravel()
    return kkt


def lam_ls(kkt, x, m):
    """Стартовые множители по МНК: J^T lam ≈ grad f в точке x."""
    gf, J, _, _ = kkt(x, np.zeros(m))
    return np.linalg.lstsq(J.T, gf, rcond=None)[0]


# ---------------------------------------------------------------- задача 1: дом у дороги
A_ROAD = np.array([2.0, 1.0])          # дом


def road_kkt(a):
    """f = 1/2 |x - a|^2, дорога g = x2 - x1^2: grad f = x - a, J = (-2 x1, 1), H = diag(1 + 2 lam, 1)."""
    def kkt(x, lam):
        return x - a, np.array([[-2 * x[0], 1.0]]), np.array([[1 + 2 * lam[0], 0.0], [0.0, 1.0]]), np.array([x[1] - x[0] ** 2])
    return kkt


def road_ggn_kkt(a):
    """Обобщённый Гаусс–Ньютон на дороге: невязка r = x - a, J_r = I, в матрице I вместо H."""
    kkt = road_kkt(a)

    def kkt_gn(x, lam):
        gf, J, _, g = kkt(x, lam)
        return gf, J, np.eye(2), g
    return kkt_gn


# ---------------------------------------------------------------- задача 2: бросок мяча
D_THROW = 30.0


def throw_problem(D=D_THROW):
    """x = (v1, v2, T), f = 1/2 (v1^2 + v2^2), g1 = v1 T - D, g2 = v2 T - 4.905 T^2.
    Производные — от CasADi (лекция 7). Возвращает словарь: x, f, g (символьные), kkt(x, lam) в numpy."""
    x = ca.SX.sym("x", 3)
    v1, v2, T = x[0], x[1], x[2]
    f = 0.5 * (v1 ** 2 + v2 ** 2)
    g = ca.vertcat(v1 * T - D, v2 * T - 4.905 * T ** 2)
    return dict(x=x, f=f, g=g, kkt=casadi_kkt(x, f, g))


def solve_throw_ipopt(pr, x0, tol=1e-10):
    """IPOPT для броска. Возвращает (x*, lam* в нашем знаке, stats); lam_g IPOPT = -lam."""
    solver = ca.nlpsol("s", "ipopt", {"x": pr["x"], "f": pr["f"], "g": pr["g"]},
                       {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.tol": tol})
    sol = solver(x0=x0, lbg=0, ubg=0)
    return np.array(sol["x"]).ravel(), -np.array(sol["lam_g"]).ravel(), solver.stats()


def throw_path(v1, v2, T, n=200):
    """Полёт без воздуха до момента T: s1 = v1 t, s2 = v2 t - 4.905 t^2."""
    t = np.linspace(0, T, n)
    return v1 * t, v2 * t - 4.905 * t ** 2


# ---------------------------------------------------------------- числа конспекта
def numbers():
    out = {}
    kkt = road_kkt(A_ROAD)
    h = newton_lagrange(kkt, [1.0, 0.0], [0.0])
    xs = np.array([t[0] for t in h]); ls = np.array([t[1][0] for t in h]); Fn = np.array([t[2] for t in h])
    xst, lst = xs[-1], ls[-1]
    # введение: подстановка даёт кубическое уравнение 2 x1^3 - x1 - 2 = 0
    assert np.isclose(2 * xst[0] ** 3 - xst[0] - 2, 0)

    # §2.1: шаг руками из (1, 0) при lam = 0: F = (-1, -1, -1)
    gf0, J0, H0, g0 = kkt(xs[0], [0.0])
    assert np.allclose(np.concatenate([gf0, g0]), [-1, -1, -1])
    assert np.allclose(xs[1], [1.2, 1.4]) and np.isclose(ls[1], 0.4)
    assert np.allclose(Fn[:5], [1.732, 0.165, 3.02e-3, 1.13e-6, 1.57e-13], rtol=6e-3), Fn
    # §2.3: x(1) лежит на касательной x2 = 2 x1 - 1, нарушение x2 - x1^2 падает квадратично, ответ на дороге
    assert np.isclose(xs[1, 1], 2 * xs[1, 0] - 1)
    gviol = xs[:, 1] - xs[:, 0] ** 2
    assert np.allclose(gviol[:4], [-1, -0.04, -1.15e-3, -5.84e-7], rtol=6e-3), gviol
    assert abs(gviol[4]) < 1e-12
    assert len(h) - 1 == 4
    assert np.allclose(xst, [1.1654, 1.3581], atol=1e-4) and np.isclose(lst, 0.3581, atol=1e-4)
    assert np.isclose(np.linalg.norm(A_ROAD - xst), 0.908, atol=1e-3)
    out["road"] = (xs, ls, Fn)

    # §3.1: на дороге H = diag(1 + 2 lam, 1), Z ~ (1, 2 x1); в старте H = I и Z^T H Z = 1
    assert np.allclose(H0, np.eye(2)) and np.allclose(J0, [[-2.0, 1.0]])
    # §3.2: det K = 1 + 2 lam + 4 x1^2 = Z~^T H Z~, Z~ = (1, 2 x1)
    for x_, l_ in ((xs[0], 0.0), (xst, lst), (np.array([0.3, -2.0]), -0.7)):
        _, J, H, _ = kkt(x_, [l_])
        K = np.block([[H, -J.T], [J, np.zeros((1, 1))]])
        Zt = np.array([1.0, 2 * x_[0]])
        assert np.isclose(np.linalg.det(K), 1 + 2 * l_ + 4 * x_[0] ** 2) and np.isclose(Zt @ H @ Zt, np.linalg.det(K))
    assert np.isclose(1 + 2 * lst + 4 * xst[0] ** 2, 7.15, atol=5e-3)
    Z0 = np.array([1.0, 2.0]) / np.sqrt(5)                                          # шаги доказательства в (1, 0), lam = 0
    assert np.allclose(J0 @ Z0, 0) and np.isclose(Z0 @ H0 @ Z0, 1)

    # §4: шаг по частям из (1, 0)
    pN = -J0.T.ravel() * g0[0] / (J0 @ J0.T)[0, 0]
    u = -(Z0 @ (gf0 + H0 @ pN)) / (Z0 @ H0 @ Z0)
    lam_plus = (J0 @ (H0 @ (xs[1] - xs[0]) + gf0))[0] / (J0 @ J0.T)[0, 0]
    lam_range = (J0 @ np.linalg.solve(H0, gf0) - g0)[0] / (J0 @ np.linalg.solve(H0, J0.T))[0, 0]
    assert np.allclose(pN, [-0.4, 0.2]) and np.allclose(Z0 * u, [0.6, 1.2]) and np.allclose(pN + Z0 * u, xs[1] - xs[0])
    assert np.isclose(u, 3 / np.sqrt(5)) and np.allclose(gf0 + H0 @ pN, [-1.4, -0.8])
    assert np.isclose(lam_plus, 0.4) and np.isclose(lam_range, 0.4) and np.isclose((J0 @ J0.T)[0, 0], 5)
    assert np.isclose(xs[0][1] + pN[1], 2 * (xs[0][0] + pN[0]) - 1)                 # p_N кончается на прямой x2 = 2 x1 - 1
    out["split"] = (pN, Z0 * u)

    # §2.3: бросок — Ньютон–Лагранж с lam0 = 0 и по МНК, против IPOPT
    pr = throw_problem(); tk = pr["kkt"]; x0 = np.array([10.0, 10.0, 2.0])
    h0 = newton_lagrange(tk, x0, [0.0, 0.0])
    assert len(h0) - 1 == 5 and np.allclose([t[2] for t in h0][:4], [17.3, 8.4, 0.725, 2.37e-3], rtol=6e-3)
    l0 = lam_ls(tk, x0, 2)
    assert np.allclose(l0, [4.903, 5.093], atol=1e-3)
    ht = newton_lagrange(tk, x0, l0)
    Ft = np.array([t[2] for t in ht])
    assert len(ht) - 1 == 4 and np.allclose(Ft[:4], [10.0, 1.47, 0.0256, 6.9e-6], rtol=6e-3), Ft
    v1, v2, T = ht[-1][0]
    assert np.isclose(v1, v2) and np.isclose(np.hypot(v1, v2), 17.155, atol=1e-3) and np.isclose(T, 2.473, atol=1e-3)
    assert np.allclose(ht[-1][1], [4.905, 4.905])
    xi, li, st = solve_throw_ipopt(pr, x0)
    assert st["iter_count"] == 4 and np.allclose(xi, ht[-1][0], atol=1e-8) and np.allclose(li, ht[-1][1], atol=1e-6)
    it = st["iterations"]
    for k in range(4):                                                              # лог IPOPT = половины нашей невязки
        x, lam, _ = ht[k]; gf, J, _, gv = tk(x, lam)
        assert np.isclose(np.abs(gv).max(), it["inf_pr"][k], rtol=1e-8) and np.isclose(np.abs(gf - J.T @ lam).max(), it["inf_du"][k], rtol=1e-8)
    assert np.allclose(it["alpha_pr"][1:], 1.0) and np.allclose(it["regularization_size"], 0.0)
    out["throw"] = (ht, (np.array(it["inf_pr"]), np.array(it["inf_du"])))

    # §3.3: старт T = 0 — строки J совпадают, шаг не определён
    _, JT0, _, gT0 = tk(np.array([10.0, 10.0, 0.0]), np.zeros(2))
    assert np.allclose(JT0, [[0, 0, 10], [0, 0, 10]]) and np.allclose(gT0, [-30, 0]) and np.linalg.matrix_rank(JT0) == 1
    try:
        newton_lagrange(tk, [10.0, 10.0, 0.0], [0.0, 0.0])
        raise AssertionError("при T = 0 шаг должен быть не определён")
    except np.linalg.LinAlgError:
        pass
    # упражнение 9.3: в ответе броска det H = 9.81 lam2 - lam1^2 - lam2^2 = 0
    assert abs(np.linalg.det(tk(ht[-1][0], ht[-1][1])[2])) < 1e-8

    # §3.4: дом над вершиной — из вершины метод не уходит
    av = np.array([0.0, 2.0])
    hv = newton_lagrange(road_kkt(av), [0.0, 0.0], [0.0])
    assert len(hv) - 1 == 1 and np.allclose(hv[-1][0], 0) and np.isclose(hv[-1][1][0], -2)
    e1 = np.array([1.0, 0.0])                                                       # в вершине J = (0, 1), Z = (1, 0)
    assert np.isclose(e1 @ road_kkt(av)(hv[-1][0], hv[-1][1])[2] @ e1, -3)            # Z^T H Z = -3
    hc = newton_lagrange(road_kkt(av), [1.0, 0.0], [0.0])
    assert np.allclose(hc[-1][0], [np.sqrt(1.5), 1.5]) and np.isclose(hc[-1][1][0], -0.5)
    assert np.isclose(np.linalg.norm(av - hc[-1][0]), np.sqrt(1.75)) and 2 > np.sqrt(1.75)   # вершина дальше: 2 против 1.32
    out["vertex"] = (av, hc[-1][0])

    # §5: обобщённый Гаусс–Ньютон, множитель ошибки q = 1 - Z^T H Z / Z^T Z = -2 lam* / (1 + 4 x1*^2)
    ggn = {}
    for a, its_g, its_n, q_ref in (((1.5, 2.0), 6, 4, -0.0059), ((2.0, 1.0), 11, 4, -0.1113), ((0.0, 0.3), 43, 6, 0.6)):
        a = np.array(a)
        hn = newton_lagrange(road_kkt(a), [1.0, 0.0], [0.0])
        hg = newton_lagrange(road_ggn_kkt(a), [1.0, 0.0], [0.0], maxit=500)
        xa, la = hn[-1][0], hn[-1][1][0]
        q = -2 * la / (1 + 4 * xa[0] ** 2)
        R = (1 + 4 * xa[0] ** 2) ** 1.5 / 2                                         # радиус кривизны дороги в x*
        assert len(hn) - 1 == its_n and len(hg) - 1 == its_g, (a, len(hn) - 1, len(hg) - 1)
        assert np.isclose(q, q_ref, atol=5e-4) and np.isclose(abs(q), np.linalg.norm(a - xa) / R)
        e = np.array([np.linalg.norm(t[0] - xa) for t in hg])
        if abs(q) > 0.05:
            assert np.allclose(e[6:9] / e[5:8], abs(q), atol=0.02), (a, e[6:9] / e[5:8])
        ggn[tuple(a)] = (hg, hn, q)
    assert np.allclose(ggn[(2.0, 1.0)][0][1][0], [1.2, 1.4])                       # первый шаг = шаг §2.1 (при lam = 0 H = I)
    ggn_throw = lambda x, lam: (*tk(x, lam)[:2], np.diag([1.0, 1.0, 0.0]), tk(x, lam)[3])   # J_r^T J_r = diag(1, 1, 0)
    hgt = newton_lagrange(ggn_throw, x0, l0, maxit=60)
    assert len(hgt) - 1 == 60 and hgt[-1][2] > 1                                    # на броске обобщённый Гаусс–Ньютон не сходится
    out["ggn"] = ggn

    print("дорога: |F| =", np.array2string(Fn, precision=3), " итераций", len(h) - 1, " x* =", xst.round(4), " lam* =", round(lst, 4))
    print("бросок: |F| =", np.array2string(Ft, precision=3), " итераций", len(ht) - 1, " IPOPT", st["iter_count"], " lam0 =", l0.round(3))
    print("Гаусс–Ньютон (итераций ОГН, НЛ, q):", {k: (len(v[0]) - 1, len(v[1]) - 1, round(v[2], 4)) for k, v in ggn.items()})
    return out, pr


# ---------------------------------------------------------------- рисовалки
def num(e):
    """Число для подписи: 0.48, 0.056 или 9.6·10^-6."""
    if e >= 1e-3:
        return f"{e:.3g}" if e >= 1 else f"{e:.2g}"
    ex = int(np.floor(np.log10(e))); m = e / 10 ** ex
    return f"${m:.1f}\\cdot10^{{{ex}}}$"


def road(ax, x1=(-1.5, 1.75), color=INK, lw=2.0):
    t = np.linspace(*x1, 300)
    ax.plot(t, t ** 2, color=color, lw=lw)


def house(ax, a, text=None, off=(8, 6)):
    ax.plot(*a, "s", color=INK, ms=10, zorder=7)
    if text:
        ax.annotate(text, a, off, textcoords="offset points", fontsize=12)


def fig_road():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    road(ax, (-1.6, 1.75))
    ax.annotate(r"дорога $x_2=x_1^2$", (-1.6, 2.56), (6, 0), textcoords="offset points", fontsize=12)
    house(ax, A_ROAD, "дом $a=(2,\\ 1)$", (-40, 12))
    ax.plot(1, 0, "o", color=ORANGE, ms=9, zorder=7)
    ax.annotate("старт $x^{(0)}=(1,\\ 0)$", (1, 0), (10, -16), textcoords="offset points", fontsize=12, color=ORANGE)
    ax.set_aspect("equal"); ax.set_xlim(-1.8, 2.8); ax.set_ylim(-0.6, 3.1)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Дом у дороги")
    save("00_road.png", min_width=560)


def fig_throw():
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    cols = {30: AQUA, 45: BLUE, 60: VIOLET, 75: ORANGE}
    for th, c in cols.items():
        v = np.sqrt(9.81 * D_THROW / np.sin(2 * np.radians(th)))
        v1, v2 = v * np.cos(np.radians(th)), v * np.sin(np.radians(th))
        X, Y = throw_path(v1, v2, v2 / 4.905)
        assert np.isclose(X[-1], D_THROW) and abs(Y[-1]) < 1e-9
        ax.plot(X, Y, color=c, lw=2.2)
        i = int(0.42 * len(X)) if th < 70 else int(0.3 * len(X))
        ax.annotate(f"{th}°", (X[i], Y[i]), (4, 6) if th < 70 else (-30, 0), textcoords="offset points", color=c, fontsize=12, fontweight="bold")
    ax.plot(0, 0, "o", color=INK, ms=8, zorder=5)
    ax.plot(D_THROW, 0, "v", color=RED, ms=12, zorder=5)
    ax.annotate("старт", (0, 0), (6, -16), textcoords="offset points", fontsize=11)
    ax.annotate(f"цель: $D={D_THROW:.0f}$ м", (D_THROW, 0), (-8, -18), textcoords="offset points", fontsize=11, ha="right", color=RED)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlim(-1.5, 33); ax.set_ylim(-3.5, 29)
    ax.set_xlabel("расстояние, м"); ax.set_ylabel("высота, м")
    ax.set_title("Четыре броска в одну точку")
    save("00_throw.png")


def fig_nl_road(xs, ls, Fn):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.4), gridspec_kw=dict(width_ratios=[1, 1.05]))
    road(ax, (0.3, 1.6))
    t = np.linspace(0.62, 1.55, 2)
    ax.plot(t, 2 * t - 1, "--", color=GRAY, lw=1.4)                                 # g(x0) + J p = 0: x2 = 2 x1 - 1
    ax.plot(1, 1, "o", color=GRAY, ms=5, zorder=6)                                   # точка касания (1, 1)
    ax.annotate("касательная\nв $(1,1)$", (1.5, 2.0), (6, -4), textcoords="offset points", fontsize=11, color=GRAY)
    house(ax, A_ROAD, "$a$", (6, 6))
    for k in range(2):
        arrow(ax, xs[k], xs[k + 1], ORANGE, lw=1.8, ms=12)
    ax.plot(xs[:3, 0], xs[:3, 1], "o", color=ORANGE, ms=7, zorder=7)
    for k, off in ((0, (8, -4)), (1, (-34, 6))):
        ax.annotate(f"$x^{{({k})}}$", xs[k], off, textcoords="offset points", fontsize=12, color=ORANGE)
    ax.plot(*xs[-1], "*", color=BLUE, ms=15, zorder=8)
    ax.annotate(r"$x^\ast$", xs[-1], (10, -12), textcoords="offset points", fontsize=12, color=BLUE)
    ax.set_aspect("equal"); ax.set_xlim(0.3, 2.3); ax.set_ylim(-0.25, 2.2)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Первый шаг — до касательной")
    k = np.arange(len(Fn))
    bx.semilogy(k, Fn, "o-", color=ORANGE, ms=8)
    for i in range(len(Fn)):
        bx.annotate(num(Fn[i]), (i, Fn[i]), (8, 3), textcoords="offset points", fontsize=11)
    bx.set_xlabel("итерация $k$"); bx.set_ylabel(r"$\|F\|$"); bx.set_xticks(k)
    bx.set_xlim(-0.3, 4.9); bx.set_title("Невязка системы")
    save("01_nl_road.png")


def fig_nl_throw(pr, ht, log):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for k, c in zip(range(3), (GRAY, YELLOW, AQUA)):
        v1, v2, T = ht[k][0]
        X, Y = throw_path(v1, v2, T)
        ax.plot(X, Y, "--", color=c, lw=1.8, label=f"$k={k}$")
        ax.plot(X[-1], Y[-1], "o", color=c, ms=7)
    v1, v2, T = ht[-1][0]
    X, Y = throw_path(v1, v2, T)
    ax.plot(X, Y, color=BLUE, lw=2.4, label=f"$k={len(ht) - 1}$"); ax.plot(X[-1], Y[-1], "o", color=BLUE, ms=8)
    ax.plot(D_THROW, 0, "v", color=RED, ms=11, zorder=6)
    ax.axhline(0, color=GRAY, lw=1)
    ax.legend(loc="upper left", ncol=2, fontsize=11, handlelength=1.6)
    ax.set_xlim(-1, 33); ax.set_ylim(-2, 12)
    ax.set_xlabel("расстояние, м"); ax.set_ylabel("высота, м"); ax.set_title("Полёт до момента $T$")
    tk = pr["kkt"]; pr_, du = [], []
    for x, lam, _ in ht:
        gf, J, _, gv = tk(x, lam)
        pr_.append(np.abs(gv).max()); du.append(np.abs(gf - J.T @ lam).max())
    k = np.arange(len(ht)); inf_pr, inf_du = log
    bx.semilogy(k, pr_, "o-", color=ORANGE, ms=7, label=r"$\max|g_j|$")
    bx.semilogy(k, du, "o-", color=BLUE, ms=7, label=r"$\max|\nabla_xL|$")
    bx.semilogy(k, inf_pr, "x", color=INK, ms=11, mew=2, label="лог IPOPT")
    bx.semilogy(k, inf_du, "x", color=INK, ms=11, mew=2)
    bx.legend(loc="lower left", fontsize=11)
    bx.set_xlabel("итерация $k$"); bx.set_ylabel("невязка"); bx.set_xticks(k)
    bx.set_xlim(-0.3, 4.3); bx.set_ylim(3e-15, 60); bx.set_title("Наш метод и IPOPT")
    save("02_nl_throw.png")


def fig_road_vertex(av, xc):
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    road(ax, (-1.75, 1.75))
    house(ax, av, "дом $(0,\\ 2)$", (10, 4))
    r = np.linalg.norm(av - xc); t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(av[0] + r * np.cos(t), av[1] + r * np.sin(t), ":", color=BLUE, lw=1.4)
    for s in (1, -1):
        ax.plot(s * xc[0], xc[1], "*", color=BLUE, ms=15, zorder=8)
    ax.annotate("ближайшие точки,\n$\\lambda=-0.5$", (xc[0], xc[1]), (12, -34), textcoords="offset points", fontsize=11, color=BLUE)
    ax.plot(0, 0, "o", color=ORANGE, ms=11, zorder=8)
    ax.annotate("старт и ответ метода:\nвершина, $\\lambda=-2$", (0, 0), (14, -30), textcoords="offset points", fontsize=11, color=ORANGE)
    ax.plot([0, 0], [0, 2], "--", color=ORANGE, lw=1.2)
    ax.set_aspect("equal"); ax.set_xlim(-2.2, 2.7); ax.set_ylim(-0.8, 3.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Из вершины — никуда")
    save("03_road_vertex.png", min_width=560)


def fig_step_split(pN, pZ):
    x0 = np.array([1.0, 0.0]); x1 = x0 + pN + pZ
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    road(ax, (0.2, 1.6))
    t = np.array([0.35, 1.45])
    ax.plot(t, 2 * t - 1, "--", color=GRAY, lw=1.4)                                 # g(x0) + J p = 0: x2 = 2 x1 - 1
    ax.annotate("$x_2=2x_1-1$:\nограничение,\nлинеаризованное в $x^{(0)}$", (1.7, 1.15), fontsize=11, color=GRAY, ha="left")
    house(ax, A_ROAD, "$a$", (6, 6))
    arrow(ax, x0, x0 + pN, ORANGE, lw=2.6, ms=16)
    arrow(ax, x0 + pN, x1, BLUE, lw=2.6, ms=16)
    arrow(ax, x0, x1, INK, lw=1.4, ms=12, linestyle=":")
    ax.plot(*x0, "o", color=INK, ms=8, zorder=8); ax.plot(*x1, "o", color=INK, ms=8, zorder=8)
    ax.annotate("$x^{(0)}=(1,\\ 0)$", x0, (10, -6), textcoords="offset points", fontsize=12)
    ax.annotate("$x^{(1)}=(1.2,\\ 1.4)$", x1, (-120, 4), textcoords="offset points", fontsize=12)
    ax.plot([], [], color=ORANGE, lw=2.6, label="$p_N=(-0.4,\\ 0.2)$: к прямой")
    ax.plot([], [], color=BLUE, lw=2.6, label="$Zu=(0.6,\\ 1.2)$: вдоль прямой")
    ax.plot([], [], ":", color=INK, lw=1.4, label="$p=(0.2,\\ 1.4)$: весь шаг")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_aspect("equal"); ax.set_xlim(-0.3, 3.5); ax.set_ylim(-0.3, 2.6)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Шаг из двух частей")
    save("04_step_split.png", min_width=560)


def fig_ggn_road(ggn):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.3), gridspec_kw=dict(width_ratios=[1, 1.1]))
    hg, hn, q = ggn[(0.0, 0.3)]
    road(ax, (-0.75, 1.15))
    house(ax, (0.0, 0.3))
    ax.annotate("дом $(0,\\ 0.3)$", (0.0, 0.3), (0.55, 0.55), fontsize=12,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=1))
    P = np.array([t[0] for t in hg][:14]); N = np.array([t[0] for t in hn])
    ax.plot(P[:, 0], P[:, 1], "o-", color=ORANGE, ms=5, lw=1.3, label=f"Гаусс–Ньютон: {len(hg) - 1} итерации")
    ax.plot(N[:, 0], N[:, 1], "s--", color=BLUE, ms=5, lw=1.1, label=f"Ньютон–Лагранж: {len(hn) - 1}")
    ax.legend(loc="upper left", fontsize=10.5)
    ax.set_aspect("equal"); ax.set_xlim(-0.8, 1.3); ax.set_ylim(-0.25, 1.35)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Медленный подход к вершине")
    cols = {(1.5, 2.0): AQUA, (2.0, 1.0): BLUE, (0.0, 0.3): ORANGE}
    for a, (hg_, hn_, q_) in ggn.items():
        xa = hn_[-1][0]
        e = np.array([np.linalg.norm(t[0] - xa) for t in hg_])
        e = e[e > 1e-13]
        bx.semilogy(np.arange(len(e)), e, "o-", color=cols[a], ms=3.5, lw=1.4,
                    label=f"дом $({a[0]:g},\\ {a[1]:g})$: $q={q_:.2g}$")
    bx.legend(loc="upper right", fontsize=10.5)
    bx.set_xlabel("итерация $k$"); bx.set_ylabel(r"$\|x^{(k)}-x^\ast\|$"); bx.set_ylim(1e-12, 5); bx.set_xlim(-1, 46)
    bx.set_title("Ошибка Гаусса–Ньютона")
    save("05_ggn_road.png")


def main():
    out, pr = numbers()
    fig_road()
    fig_throw()
    fig_nl_road(*out["road"])
    ht, log = out["throw"]
    fig_nl_throw(pr, ht, log)
    fig_road_vertex(*out["vertex"])
    fig_step_split(*out["split"])
    fig_ggn_road(out["ggn"])


if __name__ == "__main__":
    main()
