"""Вспомогательный код демонстраций лекции 7.

Разности: fd_forward(f, x, i, h), fd_central(f, x, i, h), ucurve(f, x, i, exact) -> (ошибки прямой, центральной) по HS.
Пример: tape_example() — f = x1 x2 + sin(x1 x2) в три операции: значения, прямой и обратный режим;
        OPS, PROGRAM, forward_mode(program, x, d), reverse_mode(program, x) — мини-АД по списку операций.
Цепь: make_elastic_chain(N) -> (energy, grad, v0) на numpy; chain_casadi(N) -> (v, E) — граф CasADi;
        chain_costs(N) -> (операций E, grad, hess, H d, ненулевых); chain_pattern(N) -> (портрет, равновесие, Function H).
Задачи: rosen, rosen_grad, X0_ROSEN; ucurve_plot(ax, curves, labels, colors). Нужен CasADi (--group casadi).

Файл генерируется скриптом scripts/build_lecture07.py из scripts/build_demo07.py — не правьте руками.
"""

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import time

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})
EPS = np.finfo(float).eps                                # машинная точность 2.2e-16
HS = np.logspace(-15, -1, 141)                           # сетка шагов h для U-кривых


# ---------------------------------------------------------------- разности и задачи (те же, что в demo07 / l7helpers.py)
def fd_forward(f, x, i, h):
    """Прямая разность по i-й координате: (f(x + h e_i) - f(x)) / h."""
    e = np.zeros(len(x)); e[i] = 1.0
    return (f(x + h * e) - f(x)) / h


def fd_central(f, x, i, h):
    """Центральная разность по i-й координате: (f(x + h e_i) - f(x - h e_i)) / (2h)."""
    e = np.zeros(len(x)); e[i] = 1.0
    return (f(x + h * e) - f(x - h * e)) / (2 * h)


def ucurve(f, x, i, exact, hs=HS):
    """Ошибки прямой и центральной разности по сетке шагов hs."""
    ef = np.array([abs(fd_forward(f, x, i, h) - exact) for h in hs])
    ec = np.array([abs(fd_central(f, x, i, h) - exact) for h in hs])
    return ef, ec


def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


X0_ROSEN = np.array([-1.2, 1.0])

X_SIN = np.array([1.0])                     # пример раздела 2: f = sin x в точке 1, f' = cos 1
F_SIN = lambda x: np.sin(x[0])


def tape_example(x1=2.0, x2=0.5, d=(1.0, 0.0)):
    """f = x1 x2 + sin(x1 x2) как список из трёх операций: значения, прямой режим по d, обратный режим.
    Возвращает словарь с v1, v2, f, dot (v1, v2, f), bar (v1, v2, x1, x2)."""
    v1 = x1 * x2; v2 = np.sin(v1); f = v1 + v2
    dv1 = d[0] * x2 + x1 * d[1]; dv2 = np.cos(v1) * dv1; df = dv1 + dv2            # вперёд
    bf = 1.0; bv1 = bf * 1.0; bv2 = bf * 1.0                                      # назад: f = v1 + v2
    bv1 += bv2 * np.cos(v1)                                                       # v2 = sin v1
    bx1 = bv1 * x2; bx2 = bv1 * x1                                                # v1 = x1 x2
    return dict(x1=x1, x2=x2, v1=v1, v2=v2, f=f, dv1=dv1, dv2=dv2, df=df, bv1=bv1, bv2=bv2, bx1=bx1, bx2=bx2)


def make_elastic_chain(N=40, L_total=5.0, D=70.0, g=9.81):
    """Упругая цепь (ДЗ 4): N грузиков между (-2, 1) и (2, 1), N + 1 пружин с длиной покоя L_total / (N + 1).
    Возвращает (energy, grad, v0) на numpy — градиент выписан руками."""
    m = 4.0 / N; L = L_total / (N + 1)

    def energy(v):
        y, z = np.r_[-2.0, v[:N], 2.0], np.r_[1.0, v[N:], 1.0]
        dd = np.hypot(np.diff(y), np.diff(z))
        return 0.5 * D * np.sum((dd - L) ** 2) + m * g * np.sum(z[1:-1])

    def grad(v):
        y, z = np.r_[-2.0, v[:N], 2.0], np.r_[1.0, v[N:], 1.0]
        dy, dz = np.diff(y), np.diff(z); c = D * (1 - L / np.hypot(dy, dz))
        return np.r_[c[:-1] * dy[:-1] - c[1:] * dy[1:], c[:-1] * dz[:-1] - c[1:] * dz[1:] + m * g]

    return energy, grad, np.r_[np.linspace(-2, 2, N + 2)[1:-1], np.ones(N)]


def chain_casadi(N=40, L_total=5.0, D=70.0, g=9.81):
    """Та же энергия как граф CasADi: цикл по пружинам. Возвращает (v, E) — символьная точка и выражение."""
    m = 4.0 / N; L = L_total / (N + 1)
    v = ca.SX.sym("v", 2 * N)
    y = ca.vertcat(-2.0, v[:N], 2.0); z = ca.vertcat(1.0, v[N:], 1.0)
    E = 0
    for j in range(N + 1):                                   # пружина j между грузиками j и j + 1
        dlen = ca.sqrt((y[j + 1] - y[j]) ** 2 + (z[j + 1] - z[j]) ** 2)
        E += 0.5 * D * (dlen - L) ** 2
    E += m * g * ca.sum1(z[1:-1])
    return v, E


def chain_pattern(N=40):
    """Портрет разреженности гессиана энергии (структурный, из графа) и равновесие цепи из IPOPT."""
    v, E = chain_casadi(N)
    Hs = ca.hessian(E, v)[0]
    pattern = np.array(ca.DM(Hs.sparsity(), 1.0))                     # 1 там, где элемент может быть ненулевым
    solver = ca.nlpsol("s", "ipopt", {"x": v, "f": E}, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
    v_star = np.array(solver(x0=make_elastic_chain(N)[2])["x"]).ravel()
    return pattern, v_star, ca.Function("H", [v], [Hs])


def chain_costs(N):
    """Число операций энергии, градиента, гессиана, произведения гессиана на вектор и ненулевых в гессиане."""
    v, E = chain_casadi(N)
    dvec = ca.SX.sym("d", 2 * N)
    nf = ca.Function("E", [v], [E]).n_instructions()
    ng = ca.Function("G", [v], [ca.gradient(E, v)]).n_instructions()
    Hs = ca.hessian(E, v)[0]
    nh = ca.Function("H", [v], [Hs]).n_instructions()
    nhd = ca.Function("Hd", [v, dvec], [ca.jtimes(ca.gradient(E, v), v, dvec)]).n_instructions()
    return nf, ng, nh, nhd, Hs.sparsity().nnz()


# ---------------------------------------------------------------- мини-АД: программа как список операций (разделы 3-4)
OPS = {                                                  # операция: (как считать, локальные производные по аргументам)
    "add": (lambda a, b: a + b, lambda a, b: (1.0, 1.0)),
    "mul": (lambda a, b: a * b, lambda a, b: (b, a)),
    "sin": (lambda a: np.sin(a), lambda a: (np.cos(a),)),
    "exp": (lambda a: np.exp(a), lambda a: (np.exp(a),)),
    "sq": (lambda a: a * a, lambda a: (2 * a,)),
}
PROGRAM = [("v1", "mul", ("x1", "x2")), ("v2", "sin", ("v1",)), ("f", "add", ("v1", "v2"))]   # f = x1 x2 + sin(x1 x2)


def forward_mode(program, x, d):
    """Прямой режим: вместе с каждым v_i считаем dot v_i = sum_j (d phi_i / d v_j) dot v_j. x, d — словари по входам."""
    val, dot = dict(x), dict(d)
    for name, op, args in program:
        fn, dfn = OPS[op]; a = [val[k] for k in args]
        val[name] = fn(*a)
        dot[name] = sum(p * dot[k] for p, k in zip(dfn(*a), args))
    return val, dot


def reverse_mode(program, x):
    """Обратный режим: прямой проход запоминает все v_i, обратный разносит bar v_i = d f / d v_i по аргументам."""
    val = dict(x)
    for name, op, args in program:                       # прямой проход
        val[name] = OPS[op][0](*[val[k] for k in args])
    bar = {k: 0.0 for k in val}; bar[program[-1][0]] = 1.0
    for name, op, args in reversed(program):             # обратный проход, операции в обратном порядке
        a = [val[k] for k in args]
        for p, k in zip(OPS[op][1](*a), args):
            bar[k] += bar[name] * p
    return val, bar


def ucurve_plot(ax, curves, labels, colors, hs=HS):
    """U-кривые в двойных логарифмических осях; пунктир — sqrt(eps) и eps^(1/3)."""
    for e, lab, col in zip(curves, labels, colors):
        ax.loglog(hs, np.maximum(e, 1e-17), color=col, lw=1.6, label=lab)
    for h0, txt in ((np.sqrt(EPS), r"$\sqrt{\varepsilon}$"), (EPS ** (1 / 3), r"$\varepsilon^{1/3}$")):
        ax.axvline(h0, color=GRAY, ls=":", lw=1); ax.text(h0 * 1.4, 0.04, txt, color=GRAY, transform=ax.get_xaxis_transform())
    ax.set(xlabel="шаг $h$", ylabel="ошибка производной"); ax.legend(fontsize=8)
