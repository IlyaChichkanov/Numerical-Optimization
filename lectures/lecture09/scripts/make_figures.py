"""Иллюстрации к конспекту лекции 9 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture09/scripts/make_figures.py

Скрипт самодостаточен: метод Ньютона–Лагранжа, обобщённый Гаусс–Ньютон, окружность, цепь на стержнях и бросок
определены здесь тем же кодом, что в build_demo09.py / l9helpers.py; числа конспекта считаются в скрипте и
проверяются ассертами. Цепь на стержнях и сверка с IPOPT — через CasADi.

Картинки в порядке появления в конспекте:
  00_circle_start   введение, задача 1: окружность, точка a = (2, 1) и старт (1, 1) — не на окружности
  00_chain_start    введение, задача 2: цепь на стержнях в стартовом положении — прямая z = 0, длины стержней не те
  01_newton_sqrt2   §1: Ньютон лекции 5 для x^2 = 2 из 1.5 — касательная и ошибки 0.086, 0.0025, 2.1e-6
  02_nl_circle      §2.1: Ньютон–Лагранж на окружности из (1, 1) — точки вне окружности и норма невязки
  03_chain_singular §3.2: цепь в старте при lambda = 0 — два касательных направления, в которых шаг не определён
  04_nl_chain       §3.3: Ньютон–Лагранж на цепи — формы по итерациям; две половины невязки и лог IPOPT
  05_chain_arch     §3.3, спрятанный ответ: из арки метод приходит в арку — это максимум
  06_step_split     §4: шаг из (1, 1) = часть к ограничению + часть вдоль касательной
  07_ggn_circle     §5: обобщённый Гаусс–Ньютон на окружности — шаг-проекция и ошибка угла при трёх a
  09_throw          семинар: броски под разными углами в одну точку D = 30 м
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


# ---------------------------------------------------------------- окружность (лекция 8, задача 1)
A_CIRCLE = np.array([2.0, 1.0])


def circle_kkt(a):
    """f = 1/2 |x - a|^2, g = 1/2 (|x|^2 - 1): grad f = x - a, J = x^T, H = (1 - lam) I."""
    def kkt(x, lam):
        return x - a, x[None, :], (1 - lam[0]) * np.eye(2), np.array([0.5 * (x @ x - 1)])
    return kkt


def ggn_circle(a, x0, iters):
    """Обобщённый Гаусс–Ньютон на окружности: R = x - a, J_R = I, в матрице ККТ B = I вместо H.
    Шаг — проекция a на прямую g(x_k) + J p = 0. Возвращает точки x_0..x_iters."""
    x = np.array(x0, float); xs = [x.copy()]
    for _ in range(iters):
        J = x[None, :]
        K = np.block([[np.eye(2), -J.T], [J, np.zeros((1, 1))]])
        s = np.linalg.solve(K, -np.concatenate([x - a, [0.5 * (x @ x - 1)]]))      # lam не входит: правая часть без -J^T lam
        x = x + s[:2]; xs.append(x.copy())
    return np.array(xs)


def angle_err(xs, a):
    """Угол между точками и лучом к a, рад (знак — по какую сторону)."""
    return np.arctan2(xs[:, 1], xs[:, 0]) - np.arctan2(a[1], a[0])


# ---------------------------------------------------------------- цепь на стержнях (лекция 8, задача 2)
def rod_chain(N=3, L_total=5.0, m=1.0, g_acc=9.81):
    """N грузов массы m между (-2, 1) и (2, 1), N + 1 стержней длины L_total / (N + 1), f = w sum z.
    Переменные v = (y_1..y_N, z_1..z_N). Возвращает словарь с символьными v, f, g, функцией kkt, весом w, длиной ell."""
    w = m * g_acc; ell = L_total / (N + 1)
    v = ca.SX.sym("v", 2 * N)
    y = ca.vertcat(-2.0, v[:N], 2.0); z = ca.vertcat(1.0, v[N:], 1.0)              # опоры по краям
    f = w * ca.sum1(z[1:-1])                                                        # энергия: вес на высоту
    g = ca.vertcat(*[(y[j + 1] - y[j]) ** 2 + (z[j + 1] - z[j]) ** 2 - ell ** 2 for j in range(N + 1)])
    t = np.linspace(-2, 2, N + 2)[1:-1]
    return dict(v=v, f=f, g=g, kkt=casadi_kkt(v, f, g), w=w, ell=ell, N=N, v0=np.concatenate([t, np.zeros(N)]))


def lam_ls(kkt, x, m):
    """Стартовые множители по МНК: grad f ≈ J^T lam."""
    gf, J, _, _ = kkt(x, np.zeros(m))
    return np.linalg.lstsq(J.T, gf, rcond=None)[0]


def solve_rod_chain(ch, v0=None, tol=1e-10):
    """IPOPT для цепи. Возвращает (v*, lam* в нашем знаке, stats)."""
    solver = ca.nlpsol("s", "ipopt", {"x": ch["v"], "f": ch["f"], "g": ch["g"]},
                       {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.tol": tol})
    sol = solver(x0=ch["v0"] if v0 is None else v0, lbg=0, ubg=0)
    return np.array(sol["x"]).ravel(), -np.array(sol["lam_g"]).ravel(), solver.stats()


def chain_xy(ch, v):
    N = ch["N"]
    return np.concatenate([[-2.0], v[:N], [2.0]]), np.concatenate([[1.0], v[N:], [1.0]])


# ---------------------------------------------------------------- Ньютон для уравнения (лекция 5, §2.1)
def newton_sqrt2(x0=1.5, iters=3):
    """Ньютон для F(x) = x^2 - 2: x <- x - F(x) / F'(x). Возвращает точки x_0..x_iters."""
    xs = [x0]
    for _ in range(iters):
        x = xs[-1]; xs.append(x - (x * x - 2) / (2 * x))
    return np.array(xs)


# ---------------------------------------------------------------- бросок без сопротивления
D_THROW = 30.0


def throw_path(theta_deg, D=D_THROW, n=200):
    """Бросок без сопротивления под углом theta в точку D: скорость v^2 = 9.81 D / sin(2 theta)."""
    th = np.radians(theta_deg)
    v = np.sqrt(9.81 * D / np.sin(2 * th))
    T = 2 * v * np.sin(th) / 9.81                                                   # время полёта
    t = np.linspace(0, T, n)
    return v * np.cos(th) * t, v * np.sin(th) * t - 4.905 * t ** 2


# ---------------------------------------------------------------- числа конспекта
def numbers():
    out = {}
    # §1: Ньютон для x^2 = 2 из 1.5 (лекция 5, §2.1 и упражнение 11.0(б))
    xs2 = newton_sqrt2()
    errs = np.abs(xs2 - np.sqrt(2))
    assert np.allclose(xs2[1], 17 / 12) and np.allclose(errs[:3], [0.0858, 0.00245, 2.12e-6], rtol=3e-3), errs
    out["sqrt2"] = xs2

    # §2.1: окружность из (1, 1), lam0 = 0
    kkt = circle_kkt(A_CIRCLE)
    h = newton_lagrange(kkt, [1.0, 1.0], [0.0])
    xs = np.array([t[0] for t in h]); ls = np.array([t[1][0] for t in h]); Fn = np.array([t[2] for t in h])
    assert np.allclose(xs[1], [1.25, 0.25]) and np.isclose(ls[1], -0.75)            # шаг руками
    assert np.allclose(xs[2], [0.9437, 0.5316], atol=1e-4) and np.isclose(ls[2], -1.0288, atol=1e-4)
    assert np.allclose(Fn[:6], [1.118, 0.670, 0.145, 1.83e-2, 1.64e-4, 9.47e-9], rtol=6e-3), Fn
    assert len(h) - 1 == 6                                                          # итераций до |F| <= 1e-10
    assert np.allclose(xs[-1], A_CIRCLE / np.sqrt(5)) and np.isclose(ls[-1], 1 - np.sqrt(5))
    out["circle"] = (xs, ls, Fn)

    # §3.1: редуцированный гессиан в ответе окружности
    xst, lst = xs[-1], ls[-1]
    Z = np.array([-xst[1], xst[0]])
    assert np.isclose(Z @ ((1 - lst) * np.eye(2)) @ Z, 2.236, atol=1e-3)

    # §4: разложение шага из (1, 1)
    x0, g0, J0, gf0 = np.array([1.0, 1.0]), 0.5, np.array([[1.0, 1.0]]), np.array([-1.0, 0.0])
    pN = -J0.T.ravel() * g0 / (J0 @ J0.T)[0, 0]                                     # к ограничению
    Zc = np.array([1.0, -1.0]) / np.sqrt(2)
    u = -(Zc @ (gf0 + pN)) / (Zc @ Zc)                                              # H = I при lam0 = 0
    lam_range = (J0 @ gf0 - g0)[0] / (J0 @ J0.T)[0, 0]                              # J H^-1 J^T lam+ = J H^-1 grad f - g
    assert np.allclose(pN, [-0.25, -0.25]) and np.isclose(u, 1 / np.sqrt(2)) and np.allclose(Zc * u, [0.5, -0.5])
    assert np.isclose(lam_range, -0.75) and np.allclose(pN + Zc * u, xs[1] - xs[0])
    out["split"] = (pN, Zc * u)

    # §3.3: цепь — Ньютон–Лагранж против IPOPT
    ch = rod_chain(); ck = ch["kkt"]
    l0 = lam_ls(ck, ch["v0"], 4)
    assert np.allclose(l0, -4.905)
    hc = newton_lagrange(ck, ch["v0"], l0)
    Fc = np.array([t[2] for t in hc])
    assert len(hc) - 1 == 5, len(hc)
    assert np.allclose(Fc[:4], [9.86, 2.19, 0.549, 0.019], rtol=3e-3), Fc
    vi, li, st = solve_rod_chain(ch)
    assert st["iter_count"] == 5
    assert np.allclose(hc[-1][0], vi, atol=1e-8) and np.allclose(hc[-1][1], li, atol=1e-6)
    assert np.allclose(li, [-7.87, -5.58, -5.58, -7.87], atol=6e-3)
    it = st["iterations"]                                                           # лог IPOPT = наши невязки
    for k in range(5):
        x, lam, _ = hc[k]; gf, J, _, gv = ck(x, lam)
        assert np.isclose(np.abs(gv).max(), it["inf_pr"][k], rtol=1e-8) and np.isclose(np.abs(gf - J.T @ lam).max(), it["inf_du"][k], rtol=1e-8)
    assert np.allclose(it["alpha_pr"][1:], 1.0)
    out["chain"] = (hc, vi, li)

    # §3.3: старт в арке -> арка, lam > 0
    v_arch = np.array([-1.0, 0.0, 1.0, 2.0, 2.5, 2.0])
    ha = newton_lagrange(ck, v_arch, lam_ls(ck, v_arch, 4))
    va, la = ha[-1][0], ha[-1][1]
    assert len(ha) - 1 == 4 and np.all(la > 0)
    assert np.isclose(ch["w"] * va[3:].sum(), 61.25, atol=5e-3)
    assert np.allclose(va[3:], 2 - vi[3:]) and np.allclose(la, -li)                  # зеркало висящей цепи
    out["arch"] = (ha, v_arch)

    # §3.2: старт цепи при lam = 0 — H = 0, матрица ККТ вырождена
    gf, J, H, _ = ck(ch["v0"], np.zeros(4))
    K0 = np.block([[H, -J.T], [J, np.zeros((4, 4))]])
    assert np.allclose(H, 0) and np.linalg.matrix_rank(K0) == 8 and null_space(J).shape[1] == 2
    perm = [0, 3, 1, 4, 2, 5]                                                       # v = (y1, y2, y3, z1, z2, z3) -> (y1, z1, y2, z2, y3, z3)
    assert np.allclose(J[:, perm], [[2, -2, 0, 0, 0, 0], [-2, 0, 2, 0, 0, 0], [0, 0, -2, 0, 2, 0], [0, 0, 0, 0, -2, -2]])
    d1 = np.array([1, 1, 1, 0, 1, -1.0]); d2 = np.array([0, 0, 0, 1, 0, 0.0])        # в порядке конспекта (y1, z1, y2, z2, y3, z3)
    inv = np.argsort(perm)
    for d in (d1, d2):
        dv = d[inv]                                                                 # обратно в порядок v
        assert np.allclose(J @ dv, 0) and np.allclose(K0 @ np.concatenate([dv, np.zeros(4)]), 0)
    assert np.allclose(gf[perm], [0, 9.81, 0, 9.81, 0, 9.81])
    _, J_sol, H_sol, _ = ck(vi, li)
    K_sol = np.block([[H_sol, -J_sol.T], [J_sol, np.zeros((4, 4))]])
    assert np.linalg.matrix_rank(K_sol) == 10
    out["singular"] = (J, d1[inv], d2[inv])

    # §3.3: с lam0 по МНК H = 4.905 * sum hess g_j положительно определена
    H_l0 = ck(ch["v0"], l0)[2]
    ev = np.linalg.eigvalsh(H_l0)
    assert np.isclose(ev[0], 5.747, atol=1e-3) and np.isclose(ev[-1], 33.493, atol=1e-3)
    out["ipopt_log"] = (np.array(it["inf_pr"]), np.array(it["inf_du"]))

    # §5: обобщённый Гаусс–Ньютон на окружности, скорость = |a| - 1
    for na, rate in ((1.2, 0.2), (1.5, 0.5)):
        a = np.array([0.8, 0.6]) * na
        e = angle_err(ggn_circle(a, [0.0, 1.0], 30), a)
        assert np.allclose(e[8:20] / e[7:19], -rate, atol=2e-3), (na, e[8:20] / e[7:19])
    e12 = angle_err(ggn_circle(np.array([0.96, 0.72]), [0.6, 0.8], 60), np.array([0.96, 0.72]))
    assert int(np.argmax(np.abs(e12) <= 1e-8)) == 11                                  # таблица §5: |a| = 1.2 — 11 итераций
    assert np.allclose(ggn_circle(A_CIRCLE, [0.6, 0.8], 1)[1], [1.4, 0.2])            # §5: первый шаг из (0.6, 0.8) при a = (2, 1)
    e15 = angle_err(ggn_circle(np.array([1.2, 0.9]), [0.6, 0.8], 60), np.array([1.2, 0.9]))
    it15 = int(np.argmax(np.abs(e15) <= 1e-8))
    assert it15 == 25                                                                 # тот же старт, что в GIF 08
    e2 = angle_err(ggn_circle(A_CIRCLE, [0.6, 0.8], 40), A_CIRCLE)
    assert np.allclose(np.abs(e2[20:]), 0.2525, atol=2e-3)                            # цикл из двух точек, ±0.25 рад
    out["ggn"] = (e15, it15, e2)
    out["ggn_errs"] = {na: angle_err(ggn_circle(a_, [0.6, 0.8], 24), a_)
                       for na, a_ in ((1.2, np.array([0.96, 0.72])), (1.5, np.array([1.2, 0.9])), (2.236, A_CIRCLE))}

    # введение: бросок
    for th in (30, 45, 60, 75):
        X, Y = throw_path(th)
        assert np.isclose(X[-1], D_THROW) and abs(Y[-1]) < 1e-9
    assert np.isclose(np.sqrt(9.81 * D_THROW), 17.155, atol=1e-3)                      # 45°: наименьшая скорость

    print("x^2 = 2, ошибки Ньютона:", np.array2string(errs, precision=3))
    print("окружность: |F| =", np.array2string(Fn, precision=3), " итераций", len(h) - 1)
    print("цепь: |F| =", np.array2string(Fc, precision=3), " итераций", len(hc) - 1, " IPOPT", st["iter_count"])
    print("цепь: lam =", np.round(li, 3), " арка: lam =", np.round(la, 3), " f =", round(ch["w"] * va[3:].sum(), 3))
    print("ОГН, |a| = 1.5: итераций до 1e-8:", it15, " a = (2, 1): цикл", np.round(e2[-2:], 4))
    return out, ch


# ---------------------------------------------------------------- рисовалки
def unit_circle(ax, color=INK, lw=2.0, **kw):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=color, lw=lw, **kw)


def draw_chain(ax, ch, v, color=BLUE, lw=2.2, ms=12, alpha=1.0, ls="-", zorder=3, label=None):
    y, z = chain_xy(ch, v)
    ax.plot(y, z, ls, color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)
    ax.plot(y[1:-1], z[1:-1], "o", color=color, ms=ms, alpha=alpha, zorder=zorder + 1)
    ax.plot([y[0], y[-1]], [z[0], z[-1]], "s", color=INK, ms=10, zorder=zorder + 2)


def fig_throw():
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    cols = {30: AQUA, 45: BLUE, 60: VIOLET, 75: ORANGE}
    for th, c in cols.items():
        X, Y = throw_path(th)
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
    save("09_throw.png")


def fig_circle_start():
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    unit_circle(ax)
    ax.annotate(r"окружность $g(x)=0$", (np.cos(3.9), np.sin(3.9)), (-6, -10), textcoords="offset points", ha="right", va="top", fontsize=11)
    a = A_CIRCLE; x0 = np.array([1.0, 1.0])
    ax.plot(*a, "o", color=INK, ms=8, zorder=6)
    ax.annotate("$a=(2,\\ 1)$", a, (0, 12), textcoords="offset points", ha="center", fontsize=12)
    ax.plot(*x0, "o", color=ORANGE, ms=9, zorder=6)
    ax.annotate("старт $x^{(0)}=(1,\\ 1)$", x0, (10, -20), textcoords="offset points", fontsize=12, color=ORANGE)
    assert np.isclose(0.5 * (x0 @ x0 - 1), 0.5)                                       # старт не на окружности: g = 0.5
    ax.plot(0, 0, "+", color=GRAY, ms=10)
    ax.set_aspect("equal"); ax.set_xlim(-2.1, 2.6); ax.set_ylim(-1.45, 1.55)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Ближайшая к $a$ точка окружности")
    save("00_circle_start.png", min_width=560)


def fig_chain_start(ch):
    v0 = ch["v0"]; y, z = chain_xy(ch, v0)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    draw_chain(ax, ch, v0, color=GRAY, ls="--")
    ax.plot(y[1:-1], z[1:-1], "o", color=BLUE, ms=13, zorder=7)
    for i in (1, 2, 3):
        ax.annotate(f"груз {i}\n$(y_{i},\\ z_{i})$", (y[i], z[i]), (0, -14), textcoords="offset points", ha="center", va="top",
                    color=BLUE, fontsize=12)
    for i, dx in ((0, -8), (4, 8)):
        ax.annotate(f"опора {i}", (y[i], z[i]), (dx, 10), textcoords="offset points", ha="right" if i == 0 else "left", fontsize=11)
    L = np.hypot(np.diff(y), np.diff(z))
    assert np.allclose(L, [np.sqrt(2), 1, 1, np.sqrt(2)])
    for j in range(4):
        mid = ((y[j] + y[j + 1]) / 2, (z[j] + z[j + 1]) / 2)
        off = {0: (-34, -6), 1: (0, 12), 2: (0, 12), 3: (34, -6)}[j]
        ax.annotate(f"{L[j]:.2f} м", mid, off, textcoords="offset points", ha="center", color=ORANGE, fontsize=11)
    ax.text(0, 1.3, r"нужно: все стержни по $\ell=1.25$ м", ha="center", fontsize=11, color=INK)
    ax.set_aspect("equal"); ax.set_xlim(-2.7, 2.7); ax.set_ylim(-0.9, 1.5)
    ax.set_xlabel("$y$, м"); ax.set_ylabel("$z$, м"); ax.set_title("Старт: грузы на прямой $z=0$")
    save("00_chain_start.png")


def num(e):
    """Число для подписи: 0.48, 0.056 или 9.6·10^-6."""
    if e >= 1e-3:
        return f"{e:.3g}" if e >= 1 else f"{e:.2g}"
    ex = int(np.floor(np.log10(e))); m = e / 10 ** ex
    return f"${m:.1f}\\cdot10^{{{ex}}}$"


def fig_newton_sqrt2(xs):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    t = np.linspace(1.36, 1.54, 200)
    ax.plot(t, t ** 2 - 2, color=INK, lw=2, label="$F(x)=x^2-2$")
    ax.axhline(0, color=GRAY, lw=1)
    for k, c in ((0, ORANGE), (1, BLUE)):
        xk = xs[k]; Fk = xk ** 2 - 2
        tt = np.linspace(xs[k + 1] - 0.01, xk + 0.01, 2)
        ax.plot(tt, Fk + 2 * xk * (tt - xk), "--", color=c, lw=1.6)                  # касательная — линейная модель F
        ax.plot([xk, xk], [0, Fk], ":", color=c, lw=1.2)
        ax.plot(xk, Fk, "o", color=c, ms=7, zorder=6)
        ax.plot(xs[k + 1], 0, "o", color=c, ms=7, mfc="white", zorder=6)
    ax.annotate("$x^{(0)}=1.5$", (xs[0], 0), (0, -18), textcoords="offset points", ha="center", color=ORANGE, fontsize=11)
    ax.annotate("$x^{(1)}=1.4167$", (xs[1], 0), (-6, 10), textcoords="offset points", ha="right", color=BLUE, fontsize=11)
    ax.annotate("пунктир — касательная\nв $x^{(0)}$: линейная\nмодель $F$", (1.365, 0.17), fontsize=11, color=ORANGE)
    ax.set_xlim(1.36, 1.54); ax.set_ylim(-0.15, 0.32)
    ax.set_xlabel("$x$"); ax.set_ylabel("$F(x)$"); ax.set_title("Шаг: касательная равна нулю")
    errs = np.abs(xs - np.sqrt(2)); k = np.arange(len(errs))
    bx.semilogy(k, errs, "o-", color=BLUE, ms=8)
    for i, e in enumerate(errs):
        bx.annotate(num(e), (i, e), (8, 4), textcoords="offset points", fontsize=11)
    bx.set_xlabel("итерация $k$"); bx.set_ylabel(r"$|x^{(k)}-\sqrt{2}|$")
    bx.set_xlim(-0.3, 3.9); bx.set_ylim(3e-13, 0.6); bx.set_xticks(k)
    bx.set_title("Ошибка возводится в квадрат")
    save("01_newton_sqrt2.png")


def fig_nl_circle(xs, ls, Fn):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.3), gridspec_kw=dict(width_ratios=[1, 1.05]))
    unit_circle(ax)
    ax.plot(*A_CIRCLE, "o", color=INK, ms=7); ax.annotate("$a$", A_CIRCLE, (6, 4), textcoords="offset points", fontsize=12)
    for k in range(3):
        arrow(ax, xs[k], xs[k + 1], ORANGE, lw=1.8, ms=12)
    ax.plot(xs[:3, 0], xs[:3, 1], "o", color=ORANGE, ms=7, zorder=7)
    offs = {0: (6, 4), 1: (6, -4), 2: (6, 6)}
    for k in range(3):
        ax.annotate(f"$x^{{({k})}}$", xs[k], offs[k], textcoords="offset points", fontsize=12, color=ORANGE)
    ax.plot(*xs[-1], "*", color=BLUE, ms=15, zorder=8)
    ax.annotate(r"$x^\ast$", xs[-1], (-30, -6), textcoords="offset points", fontsize=12, color=BLUE)
    ax.set_aspect("equal"); ax.set_xlim(0.35, 2.25); ax.set_ylim(0.0, 1.2)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Точки идут вне окружности")
    k = np.arange(len(Fn))
    bx.semilogy(k, Fn, "o-", color=ORANGE, ms=8)
    for i in range(len(Fn)):
        if Fn[i] > 1e-12:
            bx.annotate(num(Fn[i]), (i, Fn[i]), (8, 3), textcoords="offset points", fontsize=11)
    bx.set_xlabel("итерация $k$"); bx.set_ylabel(r"$\|F\|$")
    bx.set_xlim(-0.3, 6.8); bx.set_title("Невязка системы")
    save("02_nl_circle.png")


def fig_nl_chain(ch, hc, log):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for k, c in zip(range(3), (GRAY, YELLOW, AQUA)):
        draw_chain(ax, ch, hc[k][0], color=c, lw=1.6, ms=8, ls="--", label=f"$k={k}$")
    draw_chain(ax, ch, hc[-1][0], color=BLUE, lw=2.4, ms=11, label=f"$k={len(hc) - 1}$")
    ax.legend(loc="upper center", ncol=4, fontsize=11, handlelength=1.4, columnspacing=1.0)
    ax.set_aspect("equal"); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.8, 1.6)
    ax.set_xlabel("$y$, м"); ax.set_ylabel("$z$, м"); ax.set_title("Форма цепи по итерациям")
    ck = ch["kkt"]; pr, du = [], []
    for x, lam, _ in hc:
        gf, J, _, gv = ck(x, lam)
        pr.append(np.abs(gv).max()); du.append(np.abs(gf - J.T @ lam).max())
    k = np.arange(len(hc)); inf_pr, inf_du = log
    bx.semilogy(k, du, "o-", color=BLUE, ms=7, label=r"$\max|\nabla_xL|$")
    bx.semilogy(k, pr, "o-", color=ORANGE, ms=7, label=r"$\max|g_j|$")
    bx.semilogy(k, inf_du, "x", color=INK, ms=11, mew=2, label="лог IPOPT")
    bx.semilogy(k, inf_pr, "x", color=INK, ms=11, mew=2)
    bx.legend(loc="lower left", fontsize=11)
    bx.set_xlabel("итерация $k$"); bx.set_ylabel("невязка"); bx.set_xlim(-0.3, 5.3); bx.set_ylim(1e-13, 60)
    bx.set_title("Наш метод и IPOPT")
    save("04_nl_chain.png")


def fig_chain_arch(ch, ha, v_arch, vi):
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    draw_chain(ax, ch, v_arch, color=GRAY, lw=1.6, ms=8, ls="--", label="старт")
    draw_chain(ax, ch, ha[-1][0], color=ORANGE, lw=2.4, ms=11, label=r"ответ метода: $\lambda_j>0$")
    draw_chain(ax, ch, vi, color=BLUE, lw=1.6, ms=8, alpha=0.35, label=r"висящая цепь: $\lambda_j<0$")
    ax.legend(loc="lower center", ncol=3, fontsize=11, handlelength=1.6)
    ax.set_aspect("equal"); ax.set_xlim(-3.7, 3.7); ax.set_ylim(-1.15, 2.75)
    ax.set_xlabel("$y$, м"); ax.set_ylabel("$z$, м"); ax.set_title("Из арки — в арку")
    save("05_chain_arch.png")


def fig_chain_singular(ch, J0, d1, d2):
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    v0 = ch["v0"]; y, z = chain_xy(ch, v0)
    draw_chain(ax, ch, v0, color=GRAY, lw=1.8, ms=10)
    for d, c, sc in ((d1, ORANGE, 0.45), (d2, VIOLET, 0.55)):
        assert np.allclose(J0 @ d, 0)                                                 # оба — касательные: J d = 0
        for i in range(3):
            p = np.array([y[i + 1], z[i + 1]]); q = p + sc * np.array([d[i], d[3 + i]])
            if np.linalg.norm(q - p) > 1e-9:
                arrow(ax, p, q, c, lw=2.2, ms=13)
    for i in (1, 2, 3):
        ax.annotate(f"груз {i}", (y[i], z[i]), (0, -16), textcoords="offset points", ha="center", va="top", fontsize=11)
    ax.plot([], [], color=ORANGE, lw=2.2, label="$d_1$: все грузы вправо, груз 1 вверх, груз 3 вниз")
    ax.plot([], [], color=VIOLET, lw=2.2, label="$d_2$: груз 2 вверх")
    ax.legend(loc="upper center", fontsize=11, bbox_to_anchor=(0.5, 1.03))
    ax.set_aspect("equal"); ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.85, 1.75)
    ax.set_xlabel("$y$, м"); ax.set_ylabel("$z$, м")
    ax.set_title(r"Старт при $\lambda=0$: $K\,(d,\ 0)=0$")
    save("03_chain_singular.png")


def fig_step_split(pN, pZ):
    x0 = np.array([1.0, 1.0]); x1 = x0 + pN + pZ
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    unit_circle(ax)
    t = np.linspace(-0.6, 0.9, 2)
    ax.plot(0.75 + t, 0.75 - t, "--", color=GRAY, lw=1.4)                            # прямая g(x0) + J p = 0: x1 + x2 = 1.5
    ax.annotate("$x_1+x_2=1.5$:\nограничение,\nлинеаризованное в $x^{(0)}$", (1.45, 0.5), fontsize=11, color=GRAY, ha="left")
    arrow(ax, x0, x0 + pN, ORANGE, lw=2.6, ms=16)
    arrow(ax, x0 + pN, x1, BLUE, lw=2.6, ms=16)
    arrow(ax, x0, x1, INK, lw=1.4, ms=12, linestyle=":")
    ax.plot(*x0, "o", color=INK, ms=8, zorder=8); ax.plot(*x1, "o", color=INK, ms=8, zorder=8)
    ax.annotate("$x^{(0)}=(1,\\ 1)$", x0, (8, 4), textcoords="offset points", fontsize=12)
    ax.annotate("$x^{(1)}=(1.25,\\ 0.25)$", x1, (8, -4), textcoords="offset points", fontsize=12)
    ax.plot([], [], color=ORANGE, lw=2.6, label="$p_N=(-0.25,\\ -0.25)$: к ограничению")
    ax.plot([], [], color=BLUE, lw=2.6, label="$Zu=(0.5,\\ -0.5)$: вдоль касательной")
    ax.plot([], [], ":", color=INK, lw=1.4, label="$p=(0.25,\\ -0.75)$: весь шаг")
    ax.legend(loc="upper left", fontsize=11, bbox_to_anchor=(0.0, 1.02))
    ax.set_aspect("equal"); ax.set_xlim(-0.1, 2.15); ax.set_ylim(-0.15, 1.75)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Шаг из двух частей")
    save("06_step_split.png", min_width=560)


def fig_ggn_circle(errs):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.2), gridspec_kw=dict(width_ratios=[1, 1.1]))
    a = A_CIRCLE; x0 = np.array([0.6, 0.8]); x1 = ggn_circle(a, x0, 1)[1]
    unit_circle(ax)
    tdir = np.array([-x0[1], x0[0]]); s_ = np.linspace(-0.75, 1.25, 2)
    ax.plot(x0[0] + s_ * tdir[0], x0[1] + s_ * tdir[1], "--", color=GRAY, lw=1.4)      # касательная в x0
    ax.plot([a[0], x1[0]], [a[1], x1[1]], ":", color=INK, lw=1.4)                       # перпендикуляр из a
    arrow(ax, x0, x1, ORANGE, lw=2.4, ms=15)
    ax.plot(*a, "o", color=INK, ms=7); ax.annotate("$a$", a, (6, 4), textcoords="offset points", fontsize=12)
    ax.plot(*x0, "o", color=ORANGE, ms=8, zorder=8); ax.plot(*x1, "o", color=ORANGE, ms=8, zorder=8)
    ax.annotate("$x^{(0)}$", x0, (-34, -2), textcoords="offset points", fontsize=12, color=ORANGE)
    ax.annotate("$x^{(1)}$", x1, (6, -18), textcoords="offset points", fontsize=12, color=ORANGE)
    xst = a / np.linalg.norm(a)
    ax.plot(*xst, "*", color=BLUE, ms=15, zorder=9)
    ax.annotate(r"$x^\ast$", xst, (-26, -6), textcoords="offset points", fontsize=12, color=BLUE)
    assert np.arctan2(x1[1], x1[0]) < np.arctan2(a[1], a[0]) < np.arctan2(x0[1], x0[0])  # x1 по другую сторону от x*
    ax.set_aspect("equal"); ax.set_xlim(-0.2, 2.25); ax.set_ylim(-0.2, 1.45)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Шаг: проекция $a$ на касательную")
    cols = {1.2: AQUA, 1.5: BLUE, 2.236: ORANGE}
    for na, e in errs.items():
        k = np.arange(len(e))
        bx.semilogy(k, np.abs(e), "o-", color=cols[na], ms=5, lw=1.6)
    bx.annotate(r"$\|a\|=1.2$: $\times0.2$", (10, abs(errs[1.2][10])), (10, 0), textcoords="offset points", color=AQUA, fontsize=11)
    bx.annotate(r"$\|a\|=1.5$: $\times0.5$", (14, abs(errs[1.5][14])), (6, 6), textcoords="offset points", color=BLUE, fontsize=11)
    bx.annotate(r"$a=(2,\ 1)$: качается", (12, abs(errs[2.236][12])), (0, 10), textcoords="offset points", color=ORANGE, fontsize=11)
    bx.set_xlabel("итерация $k$"); bx.set_ylabel("ошибка угла, рад"); bx.set_ylim(1e-12, 8)
    bx.set_title("Ошибка по итерациям")
    save("07_ggn_circle.png")


def main():
    out, ch = numbers()
    fig_circle_start()
    fig_chain_start(ch)
    fig_newton_sqrt2(out["sqrt2"])
    fig_nl_circle(*out["circle"])
    fig_chain_singular(ch, *out["singular"])
    hc, vi, li = out["chain"]
    fig_nl_chain(ch, hc, out["ipopt_log"])
    ha, v_arch = out["arch"]
    fig_chain_arch(ch, ha, v_arch, vi)
    fig_step_split(*out["split"])
    fig_ggn_circle(out["ggn_errs"])
    fig_throw()


if __name__ == "__main__":
    main()
