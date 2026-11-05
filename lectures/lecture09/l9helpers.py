"""Вспомогательный код демонстраций лекции 9.

Метод Ньютона–Лагранжа: newton_lagrange(kkt, x0, lam0) -> список (x_k, lam_k, |F_k|); kkt(x, lam) -> (grad f, J, H, g),
        знак L = f - lam^T g, шаг [[H, -J^T], [J, 0]] (p, dlam) = -(grad_x L, g).
Окружность: A_CIRCLE = (2, 1), circle_kkt(a) -> kkt; ggn_circle(a, x0, iters) -> точки обобщённого Гаусса–Ньютона;
        angle_err(xs, a) -> угол до луча к a.
Цепь на стержнях: rod_chain(N, L_total, m) -> словарь (v, f, g — CasADi; kkt; w, ell, N, v0); lam_ls(kkt, x, m) -> множители по МНК;
        solve_rod_chain(ch, v0) -> (v*, lam* в нашем знаке, stats IPOPT); chain_xy(ch, v) -> (y, z) с опорами.
Рисовалки: unit_circle(ax), draw_chain(ax, ch, v). Нужен CasADi (--group casadi).

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


def unit_circle(ax, color=INK, lw=2.0):
    """Единичная окружность — ограничение задачи §2.1."""
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=color, lw=lw)


def draw_chain(ax, ch, v, color=BLUE, ls="-", label=None):
    """Цепь: стержни линией, грузы кружками, опоры квадратами."""
    y, z = chain_xy(ch, v)
    ax.plot(y, z, ls, color=color, lw=2, label=label)
    ax.plot(y[1:-1], z[1:-1], "o", color=color, ms=9)
    ax.plot([y[0], y[-1]], [z[0], z[-1]], "s", color=INK, ms=9)
