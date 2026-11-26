"""Иллюстрации к конспекту лекции 10 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture10/scripts/make_figures.py

Скрипт самодостаточен: задача «такси», портфель с запретом коротких продаж и путь при mu*h = tau определены здесь
тем же кодом, что в build_demo10.py / l10helpers.py; числа конспекта считаются в скрипте и проверяются ассертами.
Сверка портфеля с IPOPT — через CasADi.

Картинки в порядке появления в конспекте:
  00_taxi              введение, задача 1: четыре поездки и прямая без ограничений — плата за посадку b = -0.60
  00_portfolio         введение, задача 2: портфель без запрета при доходности 12 % — доля актива 1 отрицательна
  01_wall_sign         §1: знак множителя у стенки — min (x - 2)^2 при x >= 3 (mu = 2) и при x >= 1 (у стенки mu = -2)
  02_taxi_guesses      §2.1: плоскость (a, b), линии уровня f и четыре догадки; «обрезка» — не ответ
  04_complementarity   §4: условие mu*h = 0 — «уголок»; mu*h = tau — гипербола; путь точки в задаче «такси»
"""

from itertools import combinations
from pathlib import Path

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy.optimize import brentq

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

MAX_WIDTH = 920        # px: картинка показывается с width до 900


def save(name: str, min_width: int = 500) -> None:
    plt.tight_layout()
    path = IMG / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    w, h = Image.open(path).size
    assert min_width <= w <= MAX_WIDTH, f"{name}: ширина {w} px — нужна {min_width}–{MAX_WIDTH}"
    print(f"   {name}: {w}x{h}")


def arrow(ax, p, q, color, lw=2.2, ms=16):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=6))


# ---------------------------------------------------------------- методы (те же, что в demo10 / l10helpers.py)
# задача «такси»: цена поездки y ~ a t + b, a >= 0, b >= 0 (t — км, y — сотни рублей)
TAXI_T = np.array([1.0, 2.0, 3.0, 4.0])
TAXI_Y = np.array([0.6, 1.7, 3.0, 4.1])
TAXI_A = np.column_stack([TAXI_T, np.ones(4)])          # матрица МНК: строка (t_i, 1)


def taxi_f(x):
    """Половина суммы квадратов отклонений модели a t + b от данных; x = (a, b)."""
    return 0.5 * np.sum((TAXI_A @ x - TAXI_Y) ** 2)


def taxi_grad(x):
    """grad f = A^T A x - A^T y = (30a + 10b - 29.4, 10a + 4b - 9.4)."""
    return TAXI_A.T @ (TAXI_A @ x - TAXI_Y)


def taxi_guess(active):
    """Догадка: ограничения из active (0 — a >= 0, 1 — b >= 0) — равенства, остальные выброшены.
    Возвращает (x, mu): x решает систему, mu = grad f(x) (градиенты ограничений — e_1 и e_2)."""
    G, c = TAXI_A.T @ TAXI_A, TAXI_A.T @ TAXI_Y
    free = [i for i in range(2) if i not in active]
    x = np.zeros(2)
    if free:
        x[free] = np.linalg.solve(G[np.ix_(free, free)], c[free])
    return x, taxi_grad(x)


def taxi_tau(tau):
    """Точка, где вместо mu_i h_i = 0 стоит mu_i h_i = tau: grad f = (mu_a, mu_b), mu_a a = mu_b b = tau, a, b > 0.
    Из второй строки b(a) — положительный корень 4b^2 + (10a - 9.4) b - tau = 0; a — корень первой строки."""
    def b_of(a):
        q = 10 * a - 9.4
        return (-q + np.sqrt(q * q + 16 * tau)) / 8
    a = brentq(lambda a: 30 * a + 10 * b_of(a) - 29.4 - tau / a, 1e-9, 5)
    x = np.array([a, b_of(a)])
    return x, tau / x


# ближайшая точка круга — упражнение 10.1
def park_solution(a):
    """min 1/2 |x - a|^2 при h(x) = 1 - |x|^2 >= 0. Возвращает (x*, mu*).
    Внутри (|a| <= 1): x* = a, mu = 0. Снаружи: x* = a/|a|, mu = (|a| - 1)/2 — из x - a = -2 mu x."""
    a = np.asarray(a, float)
    na = np.linalg.norm(a)
    if na <= 1:
        return a.copy(), 0.0
    return a / na, (na - 1) / 2


# портфель: четыре актива, учебные данные
PF_R = np.array([0.04, 0.08, 0.11, 0.15])
PF_VOL = np.array([0.05, 0.12, 0.18, 0.30])
PF_CORR = np.array([[1, 0.2, 0.1, 0.0], [0.2, 1, 0.5, 0.3], [0.1, 0.5, 1, 0.4], [0.0, 0.3, 0.4, 1.0]])
PF_SIGMA = np.outer(PF_VOL, PF_VOL) * PF_CORR


def portfolio_active(rho, active):
    """Портфель с равенствами 1^T x = 1, r^T x = rho и границами x_i = 0 для i из active (активное множество).
    Система ККТ: Sigma x - lam1 1 - lam_rho r - sum mu_i e_i = 0 и ограничения. Возвращает (x, lam1, lam_rho, mu)."""
    active = list(active)
    k = len(active)
    one = np.ones(4)
    E = np.eye(4)[active].reshape(k, 4)
    K = np.block([[PF_SIGMA, -one[:, None], -PF_R[:, None], -E.T],
                  [one[None, :], np.zeros((1, 2 + k))],
                  [PF_R[None, :], np.zeros((1, 2 + k))],
                  [E, np.zeros((k, 2 + k))]])
    rhs = np.concatenate([np.zeros(4), [1.0, rho], np.zeros(k)])
    z = np.linalg.solve(K, rhs)
    return z[:4], z[4], z[5], z[6:]


def portfolio_bounded(rho):
    """Портфель с x >= 0 перебором активных множеств: годится то, где x >= 0 и mu >= 0 (задача выпуклая, ответ один)."""
    found = []
    for k in range(3):                       # 2 равенства + не больше 2 границ: иначе ограничений больше, чем переменных
        for A in combinations(range(4), k):
            x, l1, lr, mu = portfolio_active(rho, A)
            if x.min() >= -1e-12 and (mu.min() if k else 0) >= -1e-12:
                found.append((A, x, l1, lr, mu))
    assert len(found) == 1, (rho, [f[0] for f in found])
    return found[0]


# ---------------------------------------------------------------- числа конспекта
print("числа лекции 10:")

# ---- задача «такси»
G_taxi, c_taxi = TAXI_A.T @ TAXI_A, TAXI_A.T @ TAXI_Y
assert np.allclose(G_taxi, [[30, 10], [10, 4]]) and np.allclose(c_taxi, [29.4, 9.4])      # суммы во введении
guess = {A: taxi_guess(A) for A in ((), (0, 1), (0,), (1,))}
x_free, _ = guess[()]
assert np.allclose(x_free, [1.18, -0.60])                                                  # без ограничений: b < 0
assert np.allclose(guess[(0, 1)][1], [-29.4, -9.4])                                        # оба как равенства: mu < 0
assert np.allclose(guess[(0,)][0], [0, 2.35]) and np.allclose(guess[(0,)][1], [-5.9, 0])    # только a = 0: mu_1 < 0
x_taxi, mu_taxi = guess[(1,)]
assert np.allclose(x_taxi, [0.98, 0]) and np.allclose(mu_taxi, [0, 0.4])                  # только b = 0: ответ
x_clip = np.array([x_free[0], 0.0])
assert np.allclose(taxi_grad(x_clip), [6, 2.4])                                            # «обрезка»: стационарность нарушена
assert abs(taxi_f(x_clip) - 0.724) < 5e-4 and abs(taxi_f(x_taxi) - 0.124) < 5e-4 and abs(taxi_f(x_free) - 0.004) < 5e-4
print(f"   такси: без ограничений {x_free.round(2)}, ответ {x_taxi.round(2)}, mu = {mu_taxi.round(2)}; "
      f"f: обрезка {taxi_f(x_clip):.3f}, ответ {taxi_f(x_taxi):.3f}")

tau_rows = {}
for tau in (1.0, 0.1, 0.01, 0.001):
    x_t, mu_t = taxi_tau(tau)
    assert x_t.min() > 0 and np.allclose(taxi_grad(x_t), mu_t) and np.allclose(mu_t * x_t, tau)
    tau_rows[tau] = (x_t, mu_t, np.linalg.norm(x_t - x_taxi))
assert np.allclose(tau_rows[0.1][0], [0.924, 0.180], atol=5e-4) and abs(tau_rows[0.1][2] - 0.19) < 5e-3
assert abs(tau_rows[0.01][2] - 0.025) < 5e-4 and abs(tau_rows[0.001][2] - 0.0026) < 5e-5
print("   путь mu*h = tau:", ", ".join(f"tau={t}: x=({r[0][0]:.3f}, {r[0][1]:.3f}), до ответа {r[2]:.4f}" for t, r in tau_rows.items()))

# ---- ближайшая точка круга (упражнение 10.1)
assert np.allclose(park_solution((2, 1))[0], [0.894, 0.447], atol=5e-4) and abs(park_solution((2, 1))[1] - 0.618) < 5e-4
assert park_solution((0.5, 0.3))[1] == 0
x_c, mu_c = park_solution((0, 3))
assert np.allclose(x_c, [0, 1]) and mu_c == 1
x_c, mu_c = park_solution((0.6, 0.8))
assert np.allclose(x_c, [0.6, 0.8]) and mu_c == 0 and abs(1 - x_c @ x_c) < 1e-12              # на границе: h = 0 и mu = 0

# ---- портфель
x0_12, l1_0, lr_0, _ = portfolio_active(0.12, [])
assert np.allclose(x0_12, [-0.2015, 0.4981, 0.4324, 0.2710], atol=5e-5) and abs(lr_0 - 0.3511) < 5e-5
x1_12, l1_1, lr_1, mu_1 = portfolio_active(0.12, [0])
assert np.allclose(x1_12, [0, 0.1414, 0.5026, 0.3560], atol=5e-5)
assert abs(l1_1 + 0.0265) < 5e-5 and abs(lr_1 - 0.4731) < 5e-5 and abs(mu_1[0] - 0.0082) < 5e-5
rho_zero = brentq(lambda r: portfolio_active(r, [])[0][0], 0.09, 0.12)
assert abs(rho_zero - 0.1064) < 5e-5
A12, xb12, *_ = portfolio_bounded(0.12)
assert A12 == (0,) and np.allclose(xb12, x1_12)
x10 = portfolio_active(0.10, [])[0]
assert np.allclose(x10, [0.0946, 0.3822, 0.3209, 0.2023], atol=5e-5)                 # без запрета при rho = 10 %
risk0, risk1 = 0.5 * x0_12 @ PF_SIGMA @ x0_12, 0.5 * x1_12 @ PF_SIGMA @ x1_12
assert np.allclose(PF_SIGMA, [[0.0025, 0.0012, 0.0009, 0], [0.0012, 0.0144, 0.0108, 0.0108],
                             [0.0009, 0.0108, 0.0324, 0.0216], [0, 0.0108, 0.0216, 0.09]])   # матрица во введении
assert abs(np.sqrt(2 * risk0) - 0.169) < 5e-4 and abs(np.sqrt(2 * risk1) - 0.174) < 5e-4      # СКО портфеля 16.9 % и 17.4 %
assert abs(-0.2 * 4 + 0.5 * 8 + 0.43 * 11 + 0.27 * 15 - 12) < 0.05                           # проверка «≈ 12 %» во введении
assert abs(risk0 - 0.01429) < 5e-6 and abs(risk1 - 0.01512) < 5e-6
rho_two = brentq(lambda r: portfolio_active(r, [0])[0][1], 0.121, 0.135)            # когда обнуляется и x2
assert abs(rho_two - 0.1269) < 5e-5
for rho in (0.13, 0.14):
    A, xb, *_ = portfolio_bounded(rho)
    assert A == (0, 1), (rho, A)
print(f"   портфель 12 %: без запрета x = {x0_12.round(4)}, lam_rho = {lr_0:.4f}; "
      f"с x1 = 0: x = {x1_12.round(4)}, lam_rho = {lr_1:.4f}, mu1 = {mu_1[0]:.4f}; x1 = 0 при {rho_zero:.4%}, x2 = 0 при {rho_two:.4%}")

# ---- IPOPT: те же числа и знаки
xs = ca.SX.sym("x", 4)
nlp = {"x": xs, "f": 0.5 * ca.mtimes([xs.T, ca.DM(PF_SIGMA), xs]),
       "g": ca.vertcat(ca.sum1(xs) - 1, ca.dot(ca.DM(PF_R), xs) - 0.12)}
solver = ca.nlpsol("pf", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.tol": 1e-10})
res = solver(x0=np.ones(4) / 4, lbx=0, ubx=ca.inf, lbg=0, ubg=0)
xi, lam_g, lam_x = (np.array(res[k]).ravel() for k in ("x", "lam_g", "lam_x"))
assert np.allclose(xi, x1_12, atol=1e-6) and solver.stats()["iter_count"] == 9
assert np.allclose(lam_g, [-l1_1, -lr_1], atol=1e-6) and np.allclose(lam_x, [-mu_1[0], 0, 0, 0], atol=1e-6)
print(f"   IPOPT: 9 итераций, lam_g = {lam_g.round(4)}, lam_x = {lam_x.round(4)} — минус наши множители")

# ---- семинар у доски
# С1: min (x + 1)^2 при x >= c: c = 0 -> x = 0, mu = 2(x + 1) = 2; c = -2 -> x = -1, mu = 0
for c, (xs_, mus) in ((0, (0, 2)), (-2, (-1, 0))):
    xx = max(-1, c)
    assert xx == xs_ and 2 * (xx + 1) == mus
# С2: проекция a = (-1, 2) на x >= 0: x* = (0, 2), x - a = (1, 0) = mu1 e1 + mu2 e2 -> mu = (1, 0)
a2 = np.array([-1.0, 2.0])
assert np.allclose(np.maximum(a2, 0) - a2, [1, 0])
# С3: треугольник x1 >= 0, x2 >= 0, 1 - x1 - x2 >= 0, дом a = (0.2, -1)
a3 = np.array([0.2, -1.0])
g_v = np.array([1.0, 0.0]) - a3                                       # вершина (1, 0): активны x2 >= 0 и x1 + x2 <= 1
mu_v = np.linalg.solve(np.array([[0.0, -1.0], [1.0, -1.0]]), g_v)      # grad f = mu2 (0, 1) + mu3 (-1, -1)
assert np.allclose(mu_v, [0.2, -0.8])                                  # mu3 < 0: ограничение 3 из догадки убрать
x3 = np.array([0.2, 0.0])
assert np.allclose(x3 - a3, [0, 1]) and x3.sum() <= 1                  # mu2 = 1, остальные неактивны
print("   семинар: С1 mu = 0 и 2; С2 x* = (0, 2), mu = (1, 0); С3 в вершине (1, 0) mu = (0.2, -0.8), ответ (0.2, 0), mu2 = 1")

print("картинки:")

# ---------------------------------------------------------------- 00_taxi
fig, ax = plt.subplots(figsize=(6.4, 4.2))
tt = np.linspace(0, 4.4, 50)
ax.axhline(0, color=INK, lw=0.8)
ax.plot(tt, x_free[0] * tt + x_free[1], color=ORANGE, lw=2, label="без ограничений: $b=-0.60$")
ax.plot(TAXI_T, TAXI_Y, "o", color=BLUE, ms=8, zorder=5, label="четыре поездки")
ax.plot(0, x_free[1], "o", color=ORANGE, ms=7, zorder=5)
ax.text(0.12, x_free[1] - 0.05, "плата за посадку $-60$ руб.", color=ORANGE, fontsize=11, va="center")
ax.set(xlim=(-0.1, 4.4), ylim=(-0.9, 4.6), xlabel="длина поездки $t$, км", ylabel="цена $y$, сотни руб.",
       title="Цена поездки: $y\\approx at+b$")
ax.legend(loc="upper left")
save("00_taxi.png")

# ---------------------------------------------------------------- 00_portfolio
fig, ax = plt.subplots(figsize=(6.4, 3.8))
labels = [f"актив {i + 1}\n{int(round(r * 100))} %" for i, r in enumerate(PF_R)]
colors = [RED if v < 0 else BLUE for v in x0_12]
bars = ax.bar(labels, x0_12, color=colors, width=0.6)
for b, v in zip(bars, x0_12):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}", ha="center", fontsize=11)
ax.axhline(0, color=INK, lw=0.8)
ax.set(ylabel="доля капитала", ylim=(-0.32, 0.62), title="Портфель без запрета, доходность 12 %")
save("00_portfolio.png")

# ---------------------------------------------------------------- 01_wall_sign
# одна стенка: f(x) = (x - 2)^2, ограничение h(x) = x - c >= 0; у стенки x = c условие f'(c) = mu * h'(c), h' = 1
fp = lambda x: 2 * (x - 2)
assert fp(3) == 2 and fp(1) == -2                     # ограничение x >= 3: mu = 2; ограничение x >= 1, точка у стенки: mu = -2
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
xx = np.linspace(0, 4.4, 200)
for ax, c, title in ((axes[0], 3, "Ограничение $x\\geq 3$: $\\mu=2$"), (axes[1], 1, "Ограничение $x\\geq 1$: у стенки $\\mu=-2$")):
    ax.axvspan(c, 4.4, color=AQUA, alpha=0.15)                       # допустимая область h >= 0
    ax.axvline(c, color=AQUA, lw=3)                                 # стенка h = 0
    ax.plot(xx, (xx - 2) ** 2, color=GRAY, lw=2)
    ax.plot(c, (c - 2) ** 2, "o", color=INK, ms=7, zorder=7)
    ax.text(c + 0.08, 3.6, "допустимо", color=AQUA, fontsize=11)
    ax.set(xlim=(0, 4.4), ylim=(-0.9, 4.2), xlabel="$x$", title=title)
y0 = -0.45
arrow(axes[0], (3, y0), (2.2, y0), ORANGE)                          # -f'(3) = -2: f уменьшается влево, за стенку
arrow(axes[0], (3, y0 + 0.5), (3.8, y0 + 0.5), BLUE)                # mu h' = 2: стенка толкает внутрь
axes[0].text(2.15, y0 - 0.05, "$-f'=-2$", color=ORANGE, ha="right", va="center", fontsize=12)
axes[0].text(3.85, y0 + 0.5, "$\\mu h'=2$", color=BLUE, va="center", fontsize=12)
axes[0].set_ylabel("$f(x)=(x-2)^2$")
arrow(axes[1], (1, y0), (1.8, y0), ORANGE)                          # -f'(1) = 2: f уменьшается вправо, внутрь
axes[1].add_patch(FancyArrowPatch((1, y0 + 0.5), (0.2, y0 + 0.5), arrowstyle="-|>", mutation_scale=16,
                                  color=BLUE, lw=2.2, ls="--", zorder=6))   # mu h' = -2: стенке пришлось бы тянуть
axes[1].text(1.85, y0, "$-f'=2$", color=ORANGE, va="center", fontsize=12)
axes[1].text(0.15, y0 + 0.85, "$\\mu h'=-2$", color=BLUE, fontsize=12)
axes[1].plot(2, 0, "*", color=INK, ms=13, zorder=7)
axes[1].text(2.1, 0.25, "минимум $x^\\ast=2$, $\\mu=0$", fontsize=11)
save("01_wall_sign.png", min_width=800)

# ---------------------------------------------------------------- 02_taxi_guesses
fig, ax = plt.subplots(figsize=(7.6, 5.0))
aa, bb = np.meshgrid(np.linspace(-0.3, 1.6, 300), np.linspace(-1.0, 2.6, 300))
ff = 0.5 * sum((aa * t_ + bb - y_) ** 2 for t_, y_ in zip(TAXI_T, TAXI_Y))
ax.contour(aa, bb, ff, levels=[0.124, 0.724, 2.0, 3.485, 8.0, 14.53], colors=GRAY, linewidths=0.9)
ax.fill_between([0, 1.6], 0, 2.6, color=AQUA, alpha=0.15)                       # допустимо: a >= 0, b >= 0
ax.axhline(0, color=AQUA, lw=2.5)
ax.axvline(0, color=AQUA, lw=2.5)
marks = [(x_free, "x", RED, "без ограничений\n$(1.18,\\ -0.60)$", (0.05, -0.28)),
         (guess[(0, 1)][0], "s", VIOLET, "оба как равенства\n$(0,\\ 0)$, $\\mu<0$", (0.05, -0.42)),
         (guess[(0,)][0], "s", VIOLET, "только $a=0$\n$(0,\\ 2.35)$, $\\mu_1<0$", (0.06, -0.42)),
         (x_clip, "D", ORANGE, "«обрезка»\n$(1.18,\\ 0)$, $f=0.724$", (0.06, 0.12)),
         (x_taxi, "*", INK, "ответ $(0.98,\\ 0)$\n$f=0.124$", (-0.62, 0.22))]
for pt, m, col, label, (dx, dy) in marks:
    ax.plot(*pt, m, color=col, ms=12 if m == "*" else 9, zorder=6)
    ax.text(pt[0] + dx, pt[1] + dy, label, fontsize=10, color=col if col != INK else INK)
ax.text(1.0, 2.25, "допустимо: $a\\geq 0$, $b\\geq 0$", color=AQUA, fontsize=11)
ax.set(xlim=(-0.3, 1.6), ylim=(-1.0, 2.6), xlabel="цена километра $a$", ylabel="плата за посадку $b$",
       title="Задача «такси»: четыре догадки")
save("02_taxi_guesses.png", min_width=600)

# ---------------------------------------------------------------- 04_complementarity
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
ax = axes[0]
ax.plot([0, 1.1], [0, 0], color=INK, lw=3.5)
ax.plot([0, 0], [0, 1.1], color=INK, lw=3.5, label="$\\mu h=0$, $\\mu\\geq 0$, $h\\geq 0$")
for tau, col in ((0.1, BLUE), (0.01, VIOLET)):
    hh = np.linspace(tau / 1.1, 1.1, 300)
    ax.plot(hh, tau / hh, color=col, lw=2, label=f"$\\mu h={tau}$")
ax.plot(x_taxi[0], 0, "s", color=ORANGE, ms=9, zorder=6)                       # a >= 0: h_1 = 0.98, mu_1 = 0
ax.plot(0, mu_taxi[1], "o", color=ORANGE, ms=9, zorder=6)                      # b >= 0: h_2 = 0, mu_2 = 0.4
ax.text(x_taxi[0], 0.06, "$a\\geq 0$", ha="center", fontsize=11)
ax.text(0.05, mu_taxi[1] + 0.04, "$b\\geq 0$", fontsize=11)
ax.set(xlim=(-0.05, 1.1), ylim=(-0.05, 1.1), xlabel="$h$", ylabel="$\\mu$", title="Уголок и гиперболы")
ax.legend(loc="upper right", fontsize=10)
ax = axes[1]
ax.fill_between([0, 1.3], 0, 1.0, color=AQUA, alpha=0.15)
ax.axhline(0, color=AQUA, lw=2.5)
ax.axvline(0, color=AQUA, lw=2.5)
pts = np.array([tau_rows[tau][0] for tau in (1.0, 0.1, 0.01)])
ax.plot(pts[:, 0], pts[:, 1], "o-", color=BLUE, ms=6, lw=1.2)
for tau, p, (dx, dy, ha) in zip((1.0, 0.1, 0.01), pts, ((-0.05, 0.03, "right"), (-0.05, 0.03, "right"), (-0.06, 0.06, "right"))):
    ax.text(p[0] + dx, p[1] + dy, f"$\\tau={tau}$", fontsize=11, ha=ha)
ax.plot(*x_taxi, "*", color=INK, ms=12, zorder=6)
ax.text(x_taxi[0] + 0.04, x_taxi[1] + 0.05, "ответ", fontsize=11)
ax.set(xlim=(-0.05, 1.3), ylim=(-0.08, 1.0), xlabel="$a$", ylabel="$b$", title="Задача «такси»: точки при $\\mu_ih_i=\\tau$")
save("04_complementarity.png", min_width=800)

print("готово")
