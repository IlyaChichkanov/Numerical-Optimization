"""Вспомогательный код демонстраций лекции 8.

Окружность: A_CIRCLE = (2, 1), circle_points() -> ((x, lam) ближней, дальней точки); саму систему демо (a) пишет явно.
Окружность в пространстве: A_SPACE, sphere_plane(a, c) -> (x*, grad f, J, lam, базис T) — сфера и плоскость x3 = c (§2.2, §3.1).
IPOPT для равенств: solve_eq(x, f, g, x0) -> (x*, lam* в нашем знаке, итерации, rank J).
Цепь на стержнях: rod_chain(N, L_total, m) -> словарь (v, f, g — CasADi; w, ell, N, v0); solve_rod_chain(ch, v0) -> (v*, lam*, stats)
        в нашем знаке L = f - lam^T g; chain_xy(ch, v) -> (y, z) с опорами; chain_checks(ch, v, lam) -> невязка, rank J,
        собственные числа редуцированного гессиана, Z, натяжения F и составляющие Fh, Fv.
Портфель: PF_R, PF_SIGMA, portfolio_eq(rho) -> (x, f*, lam) — два равенства, система ККТ линейна.
Рисовалки: level_circles(ax, a, levels), unit_circle(ax). Нужен CasADi (--group casadi).

Файл генерируется скриптом scripts/build_lecture08.py из scripts/build_demo08.py — не правьте руками.
"""

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


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


def level_circles(ax, a, levels, color=BLUE, lw=1.2, alpha=0.8):
    """Линии уровня f = 1/2 |x - a|^2 — окружности вокруг a радиуса sqrt(2 f)."""
    th = np.linspace(0, 2 * np.pi, 400)
    for f in levels:
        r = np.sqrt(2 * f)
        ax.plot(a[0] + r * np.cos(th), a[1] + r * np.sin(th), color=color, lw=lw, alpha=alpha)


def unit_circle(ax, color=INK, lw=2.2):
    """Единичная окружность — ограничение задачи 1."""
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=color, lw=lw)
