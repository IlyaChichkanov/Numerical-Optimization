"""Вспомогательный код демонстраций лекции 9.

Метод Ньютона–Лагранжа: newton_lagrange(kkt, x0, lam0) -> список (x_k, lam_k, |F_k|); kkt(x, lam) -> (grad f, J, H, g),
        знак L = f - lam^T g, шаг [[H, -J^T], [J, 0]] (p, dlam) = -(grad_x L, g).
Дом у дороги: A_ROAD = (2, 1), road_kkt(a) -> kkt для дороги x2 = x1^2; road_ggn_kkt(a) -> kkt обобщённого Гаусса–Ньютона.
Бросок: throw_problem(D) -> словарь (x, f, g — CasADi; kkt); solve_throw_ipopt(pr, x0) -> (x*, lam* в нашем знаке, stats IPOPT);
        throw_path(v1, v2, T) -> полёт до момента T; lam_ls(kkt, x, m) -> множители по МНК.
Рисовалка: draw_road(ax). Нужен CasADi (--group casadi).

Файл генерируется скриптом scripts/build_lecture09.py из scripts/build_demo09.py — не правьте руками.
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


def draw_road(ax, x1=(-1.5, 1.75)):
    """Дорога x2 = x1^2 — ограничение задачи 1."""
    t = np.linspace(*x1, 300)
    ax.plot(t, t ** 2, color=INK, lw=2)
