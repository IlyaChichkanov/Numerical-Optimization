"""Иллюстрации к конспекту лекции 8 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture08/scripts/make_figures.py

Скрипт самодостаточен: окружность, окружность в пространстве (и плоскость, касающаяся сферы), параболы, цепь на стержнях и портфель
определены здесь тем же кодом, что в build_demo08.py / l8helpers.py; числа конспекта считаются в скрипте и
проверяются ассертами. Цепь на стержнях и её множители — через CasADi + IPOPT.

Картинки в порядке появления в конспекте:
  00_circle_hook       введение, задача 1: два решения условия касания — ближайшая и самая далёкая точки окружности
  00_chain_hook        введение, задача 2: цепь из трёх грузов на четырёх стержнях — обозначения и проекции стержня 0
  01_review            повторение: задача Л3 §6 — квадрат в круге, линия уровня касается окружности, lambda = -2
  02_circle_tangent    §2.1: касательная часть grad f — в точке (0.5, 0.866) она есть, в минимуме её нет
  03_sphere_plane      §2.2: окружность = сфера ∩ плоскость, grad f = lambda1 grad g1 + lambda2 grad g2
  04_sphere_touch      §3.1: плоскость x3 = c касается сферы — вид сбоку при c = 0.6 и c = 1, нормали становятся параллельными
  05_parabola_curvature  §4.1: f = x2 на параболах x2 = x1^2 и x2 = -x1^2 — минимум и максимум при одном lambda
  06_circle_cut        §4.3: f вдоль окружности, кривизна +sqrt5 и -sqrt5 — редуцированный гессиан
  07_chain_forces      §5.1: три силы на груз 1 и их треугольник — равновесие
  08_chain_rods        §5.1: натяжения стержней, их горизонтальные и вертикальные составляющие
  09_portfolio_price   §5.2: минимальный риск f*(rho) и касательная с наклоном lambda_rho
  10_ipopt_residuals   §6: inf_pr и inf_du IPOPT на цепи на стержнях по итерациям
"""

from pathlib import Path

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
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


# ---------------------------------------------------------------- задачи (те же, что в demo08 / l8helpers.py)
A_CIRCLE = np.array([2.0, 1.0])            # введение и §2.1: ближайшая к a точка единичной окружности


def circle_points(a=A_CIRCLE):
    """Оба решения системы: (x, lam) ближней и дальней точки."""
    na = np.linalg.norm(a)
    return (a / na, 1 - na), (-a / na, 1 + na)


A_SPACE = np.array([2.0, 1.0, 1.0])       # §2.2: та же окружность в пространстве, точка a поднята над плоскостью


def sphere_plane(a=A_SPACE, c=0.0):
    """Сфера g1 = 1/2(|x|^2 - 1) = 0 и плоскость g2 = x3 - c = 0; ближайшая к a точка их пересечения.
    При c < 1 пересечение — окружность радиуса sqrt(1 - c^2) (§2.2: c = 0), при c = 1 — одна точка (0, 0, 1) (§3.1).
    Возвращает (x*, grad f, J, lam, базис T = null_space(J)); при c = 1 lam — лучшее приближение, точного нет."""
    r = np.sqrt(max(1.0 - c ** 2, 0.0))                       # радиус окружности пересечения
    s = np.hypot(a[0], a[1])                                   # длина горизонтальной части a
    xs = np.array([r * a[0] / s, r * a[1] / s, c])             # ближайшая точка: над направлением (a1, a2), на высоте c
    gf = xs - a                                                # grad f для f = 1/2 |x - a|^2
    J = np.vstack([xs, [0.0, 0.0, 1.0]])                       # строки: grad g1 = x, grad g2 = e3
    lam = np.linalg.lstsq(J.T, gf, rcond=None)[0]              # grad f = J^T lam методом наименьших квадратов
    return xs, gf, J, lam, null_space(J)


def solve_eq(x, f, g, x0):
    """IPOPT для min f при g(x) = 0 (CasADi). Возвращает (x*, lam* в нашем знаке L = f - lam^T g, итерации, rank J)."""
    solver = ca.nlpsol("s", "ipopt", {"x": x, "f": f, "g": g}, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
    sol = solver(x0=x0, lbg=0, ubg=0)                          # lbg = ubg = 0: ограничения — равенства
    xs = np.array(sol["x"]).ravel()
    J = ca.Function("J", [x], [ca.jacobian(g, x)])(xs).full()  # якобиан ограничений в ответе
    return xs, -np.array(sol["lam_g"]).ravel(), solver.stats()["iter_count"], np.linalg.matrix_rank(J, tol=1e-6)


def rod_chain(N=3, L_total=5.0, m=1.0, g_acc=9.81):
    """Цепь на стержнях: N грузов массы m между (-2, 1) и (2, 1), N + 1 стержней длины L_total / (N + 1), f = w sum z.
    Возвращает словарь с символьными v, f, g (CasADi), весом груза w, длиной стержня ell и стартом v0."""
    w = m * g_acc; ell = L_total / (N + 1)
    v = ca.SX.sym("v", 2 * N)
    y = ca.vertcat(-2.0, v[:N], 2.0); z = ca.vertcat(1.0, v[N:], 1.0)
    f = w * ca.sum1(z[1:-1])
    g = ca.vertcat(*[(y[j + 1] - y[j]) ** 2 + (z[j + 1] - z[j]) ** 2 - ell ** 2 for j in range(N + 1)])
    t = np.linspace(-2, 2, N + 2)[1:-1]
    v0 = np.concatenate([t, np.zeros(N)])                     # старт: грузы равномерно по y на прямой z = 0
    return dict(v=v, f=f, g=g, w=w, ell=ell, N=N, v0=v0)


def solve_rod_chain(ch, v0=None, tol=1e-10):
    """IPOPT для цепи на стержнях. Возвращает (v*, lam* в нашем знаке L = f - lam^T g, stats)."""
    solver = ca.nlpsol("s", "ipopt", {"x": ch["v"], "f": ch["f"], "g": ch["g"]},
                       {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.tol": tol})
    sol = solver(x0=ch["v0"] if v0 is None else v0, lbg=0, ubg=0)
    return np.array(sol["x"]).ravel(), -np.array(sol["lam_g"]).ravel(), solver.stats()   # IPOPT: L = f + lam_g^T g


def chain_xy(ch, v):
    """Координаты y и z всех точек цепи вместе с опорами (-2, 1) и (2, 1) по краям."""
    N = ch["N"]
    y = np.concatenate([[-2.0], v[:N], [2.0]])                # v[:N] — y-координаты грузов
    z = np.concatenate([[1.0], v[N:], [1.0]])                 # v[N:] — z-координаты грузов
    return y, z


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
    rhs = np.concatenate([np.zeros(4), [1.0, rho]])           # правая часть: 0 — стационарность, 1 и rho — ограничения
    sol = np.linalg.solve(K, rhs)
    return sol[:4], 0.5 * sol[:4] @ PF_SIGMA @ sol[:4], sol[4:]


# ---------------------------------------------------------------- числа конспекта
def circle_system(z, a=A_CIRCLE):
    """Три уравнения задачи 1: две компоненты grad f = lambda grad g и ограничение (в демо (a) — та же функция)."""
    x1, x2, lam = z
    return [x1 - a[0] - lam * x1, x2 - a[1] - lam * x2, x1 ** 2 + x2 ** 2 - 1]


def numbers():
    (xmin, lmin), (xmax, lmax) = circle_points()
    assert abs(lmin + 1.2361) < 1e-4 and abs(lmax - 3.2361) < 1e-4 and np.allclose(xmin, [0.8944, 0.4472], atol=1e-4)
    for z0 in ([1, 0.5, 0], [-1, -0.5, 0]):
        z = fsolve(circle_system, z0, xtol=1e-14)
        assert min(abs(z[2] - lmin), abs(z[2] - lmax)) < 1e-8
    # §2.1: касательная часть grad f в точке (0.5, 0.866) и в минимуме
    P = np.array([0.5, np.sqrt(3) / 2])
    assert abs((P - A_CIRCLE) @ np.array([-P[1], P[0]]) - 1.2321) < 1e-4
    assert abs((xmin - A_CIRCLE) @ np.array([-xmin[1], xmin[0]])) < 1e-12
    # §2.2: два ограничения — два множителя
    xs, gf, J, lam2, T = sphere_plane()
    assert np.allclose(gf, [-1.1056, -0.5528, -1.0], atol=1e-4) and np.allclose(lam2, [-1.2361, -1.0], atol=1e-4)
    assert np.linalg.matrix_rank(J) == 2 and T.shape == (3, 1) and np.linalg.norm(gf - J.T @ lam2) < 1e-12
    # повторение: задача Л3 §6, f = -4 x1 x2 на окружности x1^2 + x2^2 = 1
    x3 = np.array([1, 1]) / np.sqrt(2)
    assert np.allclose(np.array([-4 * x3[1], -4 * x3[0]]), -2 * 2 * x3)          # grad f = lambda grad g, lambda = -2
    # §3.1: плоскость x3 = c поднимается к касанию; lambda_1 = 1 - sqrt5 / sqrt(1 - c^2)
    for c, lam1 in ((0.6, -1.7951), (0.9, -4.1299), (0.99, -14.8511), (0.999, -49.0125)):
        xs_c, gf_c, J_c, lam_c, T_c = sphere_plane(c=c)
        assert abs(lam_c[0] - lam1) < 1e-3 and abs(lam_c[0] - (1 - np.sqrt(5) / np.sqrt(1 - c ** 2))) < 1e-9
        assert np.linalg.norm(gf_c - J_c.T @ lam_c) < 1e-9 and T_c.shape == (3, 1)
    xs1, gf1, J1, lam_1, T1 = sphere_plane(c=1.0)
    assert np.allclose(xs1, [0, 0, 1]) and np.allclose(gf1, [-2, -1, 0]) and np.linalg.matrix_rank(J1) == 1
    assert T1.shape == (3, 2) and abs(np.linalg.norm(gf1 - J1.T @ lam_1) - np.sqrt(5)) < 1e-12   # точного lambda нет
    x3 = ca.SX.sym("x", 3); f3 = 0.5 * ca.sumsqr(x3 - ca.DM(A_SPACE))
    for c, it_exp in ((0.0, 5), (0.9, 5), (0.99, 8), (0.999, 10), (1.0, 47)):
        xs_i, lam_i, it, rk = solve_eq(x3, f3, ca.vertcat(0.5 * (ca.sumsqr(x3) - 1), x3[2] - c), [0.5, 0.5, c])
        print(f"   плоскость c = {c}: {it} итераций IPOPT, rank J = {rk}, lam = {lam_i.round(3)}")
        assert it == it_exp and rk == (1 if c == 1.0 else 2), (c, it, rk)
        if c < 1.0:
            assert abs(lam_i[0] - (1 - np.sqrt(5) / np.sqrt(1 - c ** 2))) < 1e-4
        else:                                                  # точка найдена, а множители — порядка 10^8
            assert np.allclose(xs_i, [0, 0, 1], atol=1e-6) and 1e8 < np.abs(lam_i).min() < 1e9
    # §4.3: кривизна f вдоль окружности = редуцированный гессиан 1 - lambda
    th0, h = np.arctan2(1, 2), 1e-4
    fc = lambda t: 0.5 * np.sum((np.array([np.cos(t), np.sin(t)]) - A_CIRCLE) ** 2)
    for t, lam in ((th0, lmin), (th0 + np.pi, lmax)):
        assert abs((fc(t + h) - 2 * fc(t) + fc(t - h)) / h ** 2 - (1 - lam)) < 1e-5
    # §3.3: окружность задачи 1 тремя записями — точка одна, множители разные
    y = ca.SX.sym("y", 2); fy = 0.5 * ca.sumsqr(y - ca.DM(A_CIRCLE)); gy = 0.5 * (ca.sumsqr(y) - 1)
    for g, x0, it_exp, rk_exp, lam_lo, lam_hi in ((gy, [1.0, 0.5], 3, 1, [-1.2362], [-1.2360]),
                                                  (ca.vertcat(gy, gy), [1.0, 0.5], 3, 1, [-2.0001, 0.7638], [-1.9999, 0.7640]),
                                                  ((ca.sumsqr(y) - 1) ** 2, [1.0, 0.5], 33, 0, [-6.8e9], [-6.6e9]),
                                                  ((ca.sumsqr(y) - 1) ** 2, [0.5, 0.2], 33, 0, [3.9e9], [4.0e9])):
        ys, lam_y, it, rk = solve_eq(y, fy, g, x0)
        print(f"   окружность, старт {x0}: {it} итераций, rank J = {rk}, lam = {lam_y}")
        assert np.allclose(ys, xmin, atol=1e-6) and it == it_exp and rk == rk_exp, (it, rk, lam_y)
        assert np.all(np.array(lam_lo) < lam_y) and np.all(lam_y < np.array(lam_hi)), lam_y
    # введение и §5.1: цепь на стержнях — 3 груза по 1 кг, 4 стержня по 1.25 м
    ch = rod_chain()
    v, lam, st = solve_rod_chain(ch)
    chk = chain_checks(ch, v, lam)
    y, z = chain_xy(ch, v)
    f_star = ch["w"] * z[1:-1].sum()
    print(f"   цепь: {st['iter_count']} итераций, f* = {f_star:.4f}, z_min = {z.min():.4f}, lam = {lam.round(4)}, "
          f"F = {chk['F'].round(3)} Н, F_h = {chk['Fh'].round(3)}, F_v = {chk['Fv'].round(3)}, "
          f"rank J = {chk['rank']}, eig red = {chk['eig'].round(3)}, res = {chk['res']:.1e}")
    assert abs(ch["ell"] - 1.25) < 1e-12 and abs(ch["w"] - 9.81) < 1e-12
    assert st["iter_count"] == 5 and st["return_status"] == "Solve_Succeeded", st["iter_count"]
    assert abs(f_star + 2.3936) < 1e-3 and abs(z.min() + 0.3743) < 1e-3
    assert np.allclose(lam, [-7.8703, -5.5808, -5.5808, -7.8703], atol=2e-3)
    assert np.allclose(chk["F"], [19.676, 13.952, 13.952, 19.676], atol=2e-3)           # натяжения 19.7 и 14.0 Н
    assert np.ptp(chk["Fh"]) < 1e-6 and abs(abs(chk["Fh"].mean()) - 13.061) < 2e-3      # горизонтальная одинакова
    assert np.allclose(chk["Fv"], [14.715, 4.905, -4.905, -14.715], atol=2e-3)          # 1.5 w и 0.5 w
    assert abs(chk["Fv"][0] - chk["Fv"][-1] - ch["w"] * ch["N"]) < 1e-6                # опоры держат вес цепи 3w
    assert chk["res"] < 1e-9 and chk["rank"] == 4 and chk["Z"].shape == (6, 2)
    assert np.allclose(chk["eig"], [18.99, 35.21], atol=0.01), chk["eig"]
    # введение: равновесие груза 1 — вес и силы стержней 0 и 1 в сумме дают ноль
    p0, p1, p2 = (np.array([y[i], z[i]]) for i in range(3))
    rod0, rod1, weight = 2 * lam[0] * (p1 - p0), 2 * lam[1] * (p1 - p2), np.array([0.0, -ch["w"]])
    assert np.allclose(rod0, [-13.061, 14.715], atol=2e-3) and np.allclose(rod1, [13.061, -4.905], atol=2e-3)
    assert np.linalg.norm(weight + rod0 + rod1) < 1e-9
    # цена длины стержней (упражнение 8.4, демо (e)): +1 % и +5 %
    for k, f_exp, pred_exp in ((1.01, -3.2260, -3.2385), (1.05, -6.4084, -6.7022)):
        ch2 = rod_chain(L_total=5.0 * k)
        v2, _, _ = solve_rod_chain(ch2, v0=v)
        f2 = ch2["w"] * chain_xy(ch2, v2)[1][1:-1].sum()
        f_pred = f_star + lam.sum() * (ch2["ell"] ** 2 - ch["ell"] ** 2)
        assert abs(f2 - f_exp) < 1e-3 and abs(f_pred - pred_exp) < 1e-3, (k, f2, f_pred)
    # §5.2: портфель
    x, fs, lam_pf = portfolio_eq(0.10)
    assert np.allclose(x, [0.0946, 0.3822, 0.3209, 0.2023], atol=2e-4) and abs(fs - 0.008204) < 1e-6
    assert abs(lam_pf[1] - 0.2571) < 1e-4 and abs(lam_pf[0] + 0.0093) < 1e-4, lam_pf
    d = 1e-5
    assert abs((portfolio_eq(0.10 + d)[1] - portfolio_eq(0.10 - d)[1]) / (2 * d) - lam_pf[1]) < 1e-6
    # §6: лог IPOPT — в конце число верных знаков обеих невязок примерно удваивается за итерацию
    it = st["iterations"]; pr, du = np.array(it["inf_pr"]), np.array(it["inf_du"])
    print(f"   лог: inf_pr = {np.array2string(pr, precision=1)}, inf_du = {np.array2string(du, precision=1)}")
    lpr, ldu = np.log10(pr[3:]), np.log10(du[3:])
    assert np.all(lpr[1:] / lpr[:-1] > 1.7) and np.all(ldu[1:] / ldu[:-1] > 1.7), (pr, du)
    assert 1e-4 < pr[3] < 1e-3 and 1e-7 < pr[4] < 1e-6 and 1e-13 < pr[5] < 1e-12 and du[-1] < 1e-11, (pr, du)
    return ch, v, lam, st, chk


# ---------------------------------------------------------------- рисовалки
def level_circles(ax, a, levels, color=BLUE, lw=1.2, alpha=0.8):
    """Линии уровня f = 1/2 |x - a|^2 — окружности вокруг a радиуса sqrt(2 f)."""
    th = np.linspace(0, 2 * np.pi, 400)
    for f in levels:
        r = np.sqrt(2 * f)
        ax.plot(a[0] + r * np.cos(th), a[1] + r * np.sin(th), color=color, lw=lw, alpha=alpha)


def unit_circle(ax, color=INK, lw=2.2, **kw):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=color, lw=lw, **kw)


def fig_circle_hook():
    a = A_CIRCLE
    (x1, l1), (x2, l2) = circle_points()
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    unit_circle(ax, label="окружность $g(x)=0$")
    level_circles(ax, a, [0.5 * np.sum((x1 - a) ** 2), 0.5 * np.sum((x2 - a) ** 2)], color=BLUE, lw=1.4, alpha=0.7)
    ax.plot([], [], color=BLUE, lw=1.4, alpha=0.7, label="линии уровня $f$")
    ax.plot([x2[0], a[0]], [x2[1], a[1]], ":", color=GRAY, lw=1.2)          # обе точки — на прямой через 0 и a
    ax.plot(*a, "o", color=INK, ms=7, zorder=7); ax.annotate("$a=(2,1)$", a, (8, 6), textcoords="offset points", fontsize=12)
    for x, lam, col, name, pos in ((x1, l1, RED, "ближайшая", (1.15, -1.35)), (x2, l2, VIOLET, "самая далёкая", (-2.35, -1.95))):
        ax.plot(*x, "o", color=col, ms=9, zorder=8)
        ax.annotate(f"{name}\n$x=({x[0]:.3f},\\ {x[1]:.3f})$\n$\\lambda={lam:.3f}$,  $f={0.5 * np.sum((x - a) ** 2):.3f}$",
                    xy=x, xytext=pos, fontsize=11, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, shrinkA=0, shrinkB=6))
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 3.7); ax.set_ylim(-2.2, 2.4)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Два решения условия касания")
    ax.legend(loc="upper left")
    save("00_circle_hook.png")


def fig_chain_hook(ch, v, chk):
    y, z = chain_xy(ch, v)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(y, z, "-", color=INK, lw=2.2, zorder=3)                                   # стержни
    ax.plot([y[0], y[-1]], [z[0], z[-1]], "s", color=INK, ms=10, zorder=6)            # опоры
    ax.plot(y[1:-1], z[1:-1], "o", color=BLUE, ms=13, zorder=7)                       # грузы
    for i in (1, 2, 3):
        ax.annotate(f"груз {i}", (y[i], z[i]), (0, -26), textcoords="offset points", ha="center", color=BLUE, fontsize=12)
    for i, dx in ((0, -8), (4, 8)):
        ax.annotate(f"опора {i}", (y[i], z[i]), (dx, 10), textcoords="offset points", ha="right" if i == 0 else "left",
                    color=INK, fontsize=11)
    for j in range(4):                                                                 # номер стержня — снаружи цепи
        mid = np.array([(y[j] + y[j + 1]) / 2, (z[j] + z[j + 1]) / 2])
        d = np.array([y[j + 1] - y[j], z[j + 1] - z[j]])
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        nrm = nrm if nrm[1] < 0 else -nrm
        off = 46 if j in (0, 3) else 26                                                # крутые стержни — подпись дальше
        ax.annotate(f"стержень {j}", mid, tuple(off * nrm), textcoords="offset points", ha="center", va="center",
                    color=GRAY, fontsize=11)
    corner = (y[1], z[0])                                                              # проекции стержня 0
    ax.plot([y[0], corner[0], corner[0]], [z[0], corner[1], z[1]], "--", color=ORANGE, lw=1.6, zorder=2)
    ax.annotate(r"$\Delta y_0$", ((y[0] + y[1]) / 2, z[0]), (0, 7), textcoords="offset points", ha="center", color=ORANGE, fontsize=13)
    ax.annotate(r"$\Delta z_0$", (corner[0], (z[0] + z[1]) / 2), (7, 0), textcoords="offset points", va="center", color=ORANGE, fontsize=13)
    ax.text(0, 1.32, r"стержни по $\ell=1.25$ м, грузы по 1 кг", ha="center", fontsize=11, color=GRAY)
    ax.set_aspect("equal"); ax.set_xlim(-2.7, 2.7); ax.set_ylim(-0.85, 1.5)
    ax.set_xlabel("$y$, м"); ax.set_ylabel("$z$, м"); ax.set_title("Цепь: три груза, четыре стержня")
    save("00_chain_hook.png")


def fig_chain_forces(ch, v, lam):
    y, z = chain_xy(ch, v)
    p0, p1, p2 = (np.array([y[i], z[i]]) for i in range(3))
    weight = np.array([0.0, -ch["w"]])                                                # вес = -grad f по грузу 1
    rod0, rod1 = 2 * lam[0] * (p1 - p0), 2 * lam[1] * (p1 - p2)                        # силы стержней 0 и 1 на груз 1
    forces = ((weight, BLUE, "вес"), (rod0, ORANGE, "стержень 0"), (rod1, VIOLET, "стержень 1"))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 4.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    # слева: груз 1, его стержни и три силы
    ax.plot(y[:4], z[:4], "-", color=GRAY, lw=1.6, zorder=2)
    ax.plot(*p0, "s", color=INK, ms=9, zorder=6); ax.plot(y[1:4], z[1:4], "o", color=BLUE, ms=11, zorder=7)
    ax.annotate("опора 0", p0, (6, 8), textcoords="offset points", fontsize=10, color=INK)
    ax.annotate("груз 1", p1, (-52, -4), textcoords="offset points", fontsize=11, color=BLUE)
    ax.annotate("груз 2", p2, (0, -22), textcoords="offset points", ha="center", fontsize=11, color=BLUE)
    s_ = 0.035                                                                         # метров на ньютон
    offs = {"вес": (6, -6), "стержень 0": (10, -2), "стержень 1": (4, 10)}                   # подписи — в пустом месте
    for F, col, name in forces:
        arrow(ax, p1, p1 + s_ * F, col, lw=2.6, ms=16)
        ax.annotate(f"{name}\n{np.linalg.norm(F):.1f} Н", p1 + s_ * F, offs[name], textcoords="offset points",
                    color=col, fontsize=11, fontweight="bold", ha="left")
    ax.set_aspect("equal"); ax.set_xlim(-2.3, 0.45); ax.set_ylim(-0.75, 1.25)
    ax.set_xlabel("$y$, м"); ax.set_ylabel("$z$, м"); ax.set_title("Три силы на груз 1")
    # справа: те же силы одна за другой — треугольник замыкается
    q = np.zeros(2)
    for F, col, name in (forces[0], forces[2], forces[1]):
        arrow(bx, q, q + F, col, lw=2.6, ms=16)
        mid = q + F / 2
        off = {"вес": (-10, 0), "стержень 1": (-20, -30), "стержень 0": (10, 6)}[name]        # снаружи треугольника
        bx.annotate(f"{name}\n{np.linalg.norm(F):.1f} Н", mid, off, textcoords="offset points", color=col, fontsize=11,
                    fontweight="bold", ha={"вес": "right", "стержень 1": "center", "стержень 0": "left"}[name], va="center")
        q = q + F
    assert np.linalg.norm(q) < 1e-9                                                    # треугольник замкнулся
    bx.plot(0, 0, "o", color=INK, ms=6, zorder=7)
    bx.annotate("начало = конец:\nсумма $=0$", (0, 0), (10, 2), textcoords="offset points", fontsize=11, color=INK, va="bottom")
    bx.set_aspect("equal"); bx.set_xlim(-9, 19); bx.set_ylim(-19.5, 5.5)
    bx.set_xlabel("Н"); bx.set_ylabel("Н"); bx.set_title("Сложим их по очереди")
    save("07_chain_forces.png")


def fig_review():
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    s = 1 / np.sqrt(2)
    ax.add_patch(Rectangle((-s, -s), 2 * s, 2 * s, facecolor=GRAY, alpha=0.15, edgecolor=GRAY, lw=1.2))
    ax.text(-0.2, -0.45, "вписанный\nквадрат", ha="center", va="center", color=GRAY, fontsize=11)
    unit_circle(ax, label="$g(x)=x_1^2+x_2^2-1=0$")
    t = np.linspace(0.3, 1.7, 300)
    ax.plot(t, 0.5 / t, color=BLUE, lw=2, label="линия уровня $f=-2$")
    x = np.array([s, s])
    arrow(ax, x, x + 0.22 * np.array([-4 * s, -4 * s]), RED); arrow(ax, x, x + 0.22 * np.array([2 * s, 2 * s]), AQUA)
    ax.annotate(r"$\nabla f$", x + 0.22 * np.array([-4 * s, -4 * s]), (-34, 4), textcoords="offset points", color=RED, fontsize=13)
    ax.annotate(r"$\nabla g$", x + 0.22 * np.array([2 * s, 2 * s]), (4, 2), textcoords="offset points", color=AQUA, fontsize=13)
    ax.plot(*x, "o", color=INK, ms=7, zorder=7)
    ax.annotate("$x^\\ast=(0.707,\\ 0.707)$\n$\\nabla f=\\lambda\\nabla g$, $\\lambda=-2$", xy=x, xytext=(-1.2, 1.3), fontsize=12,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=0, shrinkB=6))
    ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.9); ax.set_ylim(-1.3, 1.75)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("Лекция 3: в решении — касание")
    ax.legend(loc="lower left", fontsize=10)
    save("01_review.png")


def fig_circle_tangent():
    a = A_CIRCLE
    (xs, lam), _ = circle_points()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=True)
    s = 0.5
    # слева: не минимум
    ax = axes[0]; P = np.array([0.5, np.sqrt(3) / 2]); t = np.array([-P[1], P[0]]); gf = P - a
    tang = (gf @ t) * t; norm = (gf @ P) * P
    unit_circle(ax)
    ax.plot(*np.c_[P - 1.1 * t, P + 1.1 * t], "--", color=GRAY, lw=1)
    arrow(ax, P, P + s * gf, RED, lw=2.4)
    arrow(ax, P, P + s * tang, ORANGE)
    ax.plot(*np.c_[P + s * tang, P + s * gf], ":", color=GRAY, lw=1)
    ax.plot(*np.c_[P + s * norm, P + s * gf], ":", color=GRAY, lw=1)
    ax.annotate(r"$\nabla f$", P + s * gf, (-30, -4), textcoords="offset points", color=RED, fontsize=13)
    ax.annotate("касательная\nчасть", P + s * tang, (-78, 4), textcoords="offset points", color=ORANGE, fontsize=11)
    th = np.linspace(np.pi / 3 - 0.06, 0.62, 40)                          # спуск по окружности — против касательной части
    ax.plot(np.cos(th) * 1.12, np.sin(th) * 1.12, color=BLUE, lw=2.2)
    arrow(ax, (1.12 * np.cos(th[-2]), 1.12 * np.sin(th[-2])), (1.12 * np.cos(th[-1]), 1.12 * np.sin(th[-1])), BLUE, ms=16)
    ax.annotate("движение сюда\nуменьшает $f$", (1.12 * np.cos(0.75), 1.12 * np.sin(0.75)), (0.95, 1.38), color=BLUE, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8, shrinkA=2, shrinkB=4))      # подпись в свободном углу
    ax.plot(*P, "o", color=ORANGE, ms=8, zorder=7)
    ax.set_title("Не минимум: есть касательная часть")
    # справа: минимум
    ax = axes[1]; t = np.array([-xs[1], xs[0]]); gf = xs - a
    unit_circle(ax)
    ax.plot(*np.c_[xs - 1.1 * t, xs + 1.1 * t], "--", color=GRAY, lw=1)
    arrow(ax, xs, xs + s * gf, RED, lw=2.4); arrow(ax, xs, xs + s * xs, AQUA)
    ax.annotate(r"$\nabla f$", xs + s * gf, (-28, -14), textcoords="offset points", color=RED, fontsize=13)
    ax.annotate(r"$\nabla g$", xs + s * xs, (4, 2), textcoords="offset points", color=AQUA, fontsize=13)
    ax.annotate(f"$\\nabla f=\\lambda\\nabla g$\n$\\lambda={lam:.3f}$", xs, (-120, -70), textcoords="offset points", fontsize=12)
    ax.plot(*xs, "o", color=RED, ms=8, zorder=7)
    ax.set_title("Минимум: касательной части нет")
    for ax in axes:
        ax.plot(*a, "o", color=INK, ms=6); ax.annotate("$a$", a, (6, 4), textcoords="offset points", fontsize=12)
        ax.set_aspect("equal"); ax.set_xlim(-1.45, 2.4); ax.set_ylim(-1.25, 1.75); ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    save("02_circle_tangent.png")


def fig_sphere_plane():
    xs, gf, J, lam, T = sphere_plane()
    a = A_SPACE
    fig = plt.figure(figsize=(8.6, 6.0))
    ax = fig.add_subplot(projection="3d")
    Y1, Y2 = np.meshgrid(np.linspace(-1.4, 2.4, 2), np.linspace(-1.4, 1.8, 2))
    ax.plot_surface(Y1, Y2, 0 * Y1, color=GRAY, alpha=0.12, linewidth=0)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), 0 * th, color=INK, lw=2.2, label="окружность: $g_1=0$, $g_2=0$")
    ax.plot(*a, "o", color=INK, ms=6); ax.plot([a[0], a[0]], [a[1], a[1]], [0, a[2]], ":", color=GRAY, lw=1.2)
    ax.text(a[0] - 1.25, a[1], a[2] + 0.05, "$a=(2,1,1)$", fontsize=12)
    n1, n2 = lam[0] * J[0], lam[1] * J[1]
    for vec, col, name, off in ((n1, AQUA, r"$\lambda_1\nabla g_1$", (-0.2, 0.0, 0.22)), (n2, VIOLET, r"$\lambda_2\nabla g_2$", (0.08, 0.05, -0.05)),
                                (gf, RED, r"$\nabla f$", (-0.35, -0.2, -0.12))):
        ax.quiver(*xs, *vec, color=col, lw=2.4, arrow_length_ratio=0.12)
        ax.text(*(xs + vec + np.array(off)), name, color=col, fontsize=13)
    for p, q in ((xs + n1, xs + gf), (xs + n2, xs + gf)):              # параллелограмм: grad f = сумма двух нормалей
        ax.plot(*np.c_[p, q], ":", color=GRAY, lw=1.2)
    tt = T[:, 0]
    ax.plot(*np.c_[xs - 0.9 * tt, xs + 0.9 * tt], "--", color=GRAY, lw=1.4)
    ax.text(*(xs - 0.95 * tt + np.array([0.05, 0.0, 0.08])), "$T$", color=GRAY, fontsize=13)
    ax.plot(*xs, "o", color=RED, ms=7)
    ax.set_xlim(-1.4, 2.4); ax.set_ylim(-1.4, 1.8); ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3$")
    ax.set_box_aspect((3.8, 3.2, 2.4)); ax.view_init(elev=22, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0)); axis.set_major_locator(plt.MaxNLocator(3))
    ax.set_title("Два ограничения — две нормали"); ax.legend(loc="upper left", fontsize=10)
    save("03_sphere_plane.png")


def side_view(ax, c):
    """Вид сбоку на §3.1: разрез через ось x3 и точку a. Сфера — окружность, плоскость x3 = c — прямая.
    Горизонтальная ось идёт вдоль (a1, a2), поэтому a = (sqrt5, 1), а x* = (r, c)."""
    xs, gf, J, lam, T = sphere_plane(c=c)
    r = np.hypot(xs[0], xs[1])
    P, A = np.array([r, c]), np.array([np.sqrt(5), 1.0])     # x* и a в координатах вида сбоку
    unit_circle(ax, lw=2)
    ax.plot([-1.45, 2.55], [c, c], color=GRAY, lw=2)          # плоскость x3 = c
    if r > 0:
        ax.plot([-r, r], [c, c], color=BLUE, lw=6, alpha=0.45, solid_capstyle="round")   # окружность пересечения с ребра
    ax.plot(*A, "o", color=INK, ms=6); ax.annotate("$a$", A, (5, 5), textcoords="offset points", fontsize=12)
    n1 = P / np.linalg.norm(P)                                  # grad g1 = x*, единичный
    dx = 0.0 if r > 0 else 0.05                                 # при касании нормали совпадают — разводим чуть-чуть
    arrow(ax, P - [dx, 0], P - [dx, 0] + 0.55 * n1, AQUA, lw=2.4)
    arrow(ax, P + [dx, 0], P + [dx, 0] + np.array([0, 0.55]), VIOLET, lw=2.4)
    g = np.array([r - np.sqrt(5), c - 1.0])                     # grad f = x* - a в тех же координатах
    arrow(ax, P, P + 0.35 * g, RED, lw=2.4)
    ax.plot(*P, "o", color=RED, ms=8, zorder=8)
    return P, n1, g


def fig_sphere_touch():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharey=True)
    ax = axes[0]
    P, n1, g = side_view(ax, 0.6)
    ax.annotate(r"$\nabla g_1$", P + 0.55 * n1, (4, 2), textcoords="offset points", color=AQUA, fontsize=13)
    ax.annotate(r"$\nabla g_2$", P + np.array([0, 0.55]), (-30, 4), textcoords="offset points", color=VIOLET, fontsize=13)
    ax.annotate(r"$\nabla f$", P + 0.35 * g, (-2, -22), textcoords="offset points", color=RED, fontsize=13)
    ax.text(-0.85, 0.44, "окружность\nпересечения", color=BLUE, fontsize=10, va="top")          # под хордой, внутри сферы
    ax.set_title("$c=0.6$: нормали не параллельны")
    ax = axes[1]
    P, n1, g = side_view(ax, 1.0)
    ax.annotate(r"$\nabla g_1$", P + 0.55 * n1 - [0.05, 0], (-34, -4), textcoords="offset points", color=AQUA, fontsize=13)
    ax.annotate(r"$\nabla g_2$", P + [0.05, 0.55], (5, -4), textcoords="offset points", color=VIOLET, fontsize=13)
    ax.annotate(r"$\nabla f$", P + 0.35 * g, (-4, -22), textcoords="offset points", color=RED, fontsize=13)
    ax.text(1.06, 0.93, "$Jp=0$: вся плоскость\n$p_3=0$", color=GRAY, fontsize=11, va="top")      # справа под прямой
    ax.text(0, 0.78, "допустимая\nточка одна", color=RED, fontsize=11, va="top", ha="center")    # внутри сферы, под точкой
    ax.set_title("$c=1$: нормали параллельны")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlim(-1.45, 2.7); ax.set_ylim(-1.15, 1.75)
        ax.set_xlabel("вдоль $(a_1,a_2)$")
    axes[0].set_ylabel("$x_3$")
    save("04_sphere_touch.png")


def fig_parabola_curvature():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharey=True)
    x1 = np.linspace(-1.1, 1.1, 300)
    for ax, c, name, col in ((axes[0], 1.0, "минимум", RED), (axes[1], -1.0, "максимум", VIOLET)):
        for lev in np.arange(-1.0, 1.01, 0.25):                         # линии уровня f = x2 — горизонтали
            ax.axhline(lev, color=BLUE, lw=0.8, alpha=0.45)
        ax.plot(x1, c * x1 ** 2, color=INK, label="$x_2=x_1^2$" if c > 0 else "$x_2=-x_1^2$")
        arrow(ax, (0, 0), (0, 0.5), RED, lw=2.6)
        ax.text(-0.5, 0.74, r"$\nabla f=\nabla g=(0,1)$, $\lambda=1$", fontsize=11)
        ax.plot(0, 0, "o", color=col, ms=8, zorder=7); ax.annotate(name, (0, 0), (8, -18), textcoords="offset points", color=col, fontsize=13)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xlabel("$x_1$"); ax.legend(loc="lower left" if c > 0 else "upper left", fontsize=11)
        ax.set_title("$c=1$: парабола вверх" if c > 0 else "$c=-1$: парабола вниз")
    axes[0].set_ylabel("$x_2$")
    save("05_parabola_curvature.png")


def fig_circle_cut():
    (xmin, lmin), (xmax, lmax) = circle_points()
    th0 = np.arctan2(xmin[1], xmin[0]); th1 = th0 + np.pi - 2 * np.pi          # дальняя точка, в отрезке [-pi, pi]
    t = np.linspace(-np.pi, np.pi, 400)
    fc = 3 - np.sqrt(5) * np.cos(t - th0)                                      # f(cos t, sin t) = 3 - sqrt5 cos(t - th0)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t, fc, color=BLUE, label="$f$ в точке окружности с углом $\\theta$")
    for tc, lam, col, name, pos in ((th0, lmin, RED, "минимум", (th0 + 0.2, 0.2)), (th1, lmax, VIOLET, "максимум", (-3.05, 5.75))):
        k = 1 - lam                                                            # редуцированный гессиан
        fv = 3 - np.sqrt(5) * np.cos(tc - th0)
        tt = np.linspace(max(tc - 1.4, -np.pi), min(tc + 1.4, np.pi), 60)
        ax.plot(tt, fv + 0.5 * k * (tt - tc) ** 2, "--", color=col, lw=2)
        ax.plot(tc, fv, "o", color=col, ms=8, zorder=7)
        ax.text(*pos, f"{name}: кривизна ${k:+.3f}$", color=col, fontsize=12)
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(0, 6.4)
    ax.set_xlabel(r"угол $\theta$ точки на окружности, рад"); ax.set_ylabel("$f$"); ax.set_title("$f$ вдоль окружности")
    ax.legend(loc="upper right", fontsize=10)
    save("06_circle_cut.png")


def fig_chain_rods(chk):
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    j = np.arange(len(chk["F"])); wd = 0.26
    bars = ((-wd, chk["F"], ORANGE, r"натяжение $2|\lambda_j|\ell$"),
            (0.0, np.abs(chk["Fh"]), AQUA, "горизонтальная"),
            (wd, np.abs(chk["Fv"]), VIOLET, "вертикальная"))
    for dx, val, col, lab in bars:
        b = ax.bar(j + dx, val, wd, color=col, label=lab)
        ax.bar_label(b, fmt="%.1f", fontsize=10, padding=2)
    ax.set_xticks(j, [f"стержень {k}" for k in j]); ax.set_ylabel("сила, Н"); ax.set_ylim(0, 31)
    ax.set_title("Горизонтальная составляющая везде одна"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, fontsize=10)
    save("08_chain_rods.png")


def fig_portfolio_price():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    rhos = np.linspace(0.06, 0.14, 81)
    fs = np.array([portfolio_eq(r)[1] for r in rhos])
    x0, f0, l0 = portfolio_eq(0.10)
    ax.plot(100 * rhos, fs, color=BLUE, label="$f^\\ast(\\rho)$ — минимальный риск")
    ax.plot(100 * rhos, f0 + l0[1] * (rhos - 0.10), color=ORANGE, ls="--", label=f"касательная, наклон $\\lambda_\\rho={l0[1]:.3f}$")
    ax.plot(10, f0, "o", color=RED, ms=8, zorder=7)
    ax.set_xlabel(r"цель доходности $\rho$, %"); ax.set_ylabel(r"риск $\frac{1}{2}x^\top\Sigma x$")
    ax.set_title("Цена доходности — наклон $f^\\ast$"); ax.set_ylim(0, 0.025); ax.legend(fontsize=11, loc="upper left")
    save("09_portfolio_price.png")


def fig_ipopt_residuals(st):
    it = st["iterations"]
    pr, du = np.array(it["inf_pr"]), np.array(it["inf_du"])
    k = np.arange(len(pr))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.semilogy(k, np.maximum(pr, 1e-17), "o-", color=BLUE, ms=4, label=r"inf_pr $=\|g(x_k)\|_\infty$ — ограничения")
    ax.semilogy(k, np.maximum(du, 1e-17), "s-", color=ORANGE, ms=4, label=r"inf_du $=\|\nabla_xL(x_k,\lambda_k)\|_\infty$ — стационарность")
    ax.axhline(1e-8, color=GRAY, ls="--", lw=1); ax.annotate("tol $=10^{-8}$", (0.3, 2e-8), color=GRAY, fontsize=10)
    ax.set_xlabel("итерация IPOPT"); ax.set_ylabel("невязка"); ax.set_title(f"Две невязки IPOPT: {len(pr) - 1} итераций")
    ax.set_ylim(1e-17, 1e4); ax.legend(fontsize=10, loc="lower left")
    save("10_ipopt_residuals.png")


if __name__ == "__main__":
    print("числа конспекта:")
    ch, v, lam, st, chk = numbers()
    print("картинки:")
    fig_circle_hook()
    fig_chain_hook(ch, v, chk)
    fig_review()
    fig_circle_tangent()
    fig_sphere_plane()
    fig_sphere_touch()
    fig_parabola_curvature()
    fig_circle_cut()
    fig_chain_forces(ch, v, lam)
    fig_chain_rods(chk)
    fig_portfolio_price()
    fig_ipopt_residuals(st)
