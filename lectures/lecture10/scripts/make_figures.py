"""Иллюстрации к конспекту лекции 10 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture10/scripts/make_figures.py

Скрипт самодостаточен: задача «парк», портфель с запретом коротких продаж и путь при mu*h = tau определены здесь
тем же кодом, что в build_demo10.py / l10helpers.py; числа конспекта считаются в скрипте и проверяются ассертами.
Сверка портфеля с IPOPT — через CasADi.

Картинки в порядке появления в конспекте:
  00_park              введение, задача 1: круглый парк и два дома — внутри и снаружи
  00_portfolio         введение, задача 2: портфель лекции 8 при доходности 12 % — доля актива 1 отрицательна
  01_park_cases        §2.1: два случая — дом внутри (mu = 0) и дом снаружи (h = 0, -grad f и grad h на одной прямой)
  02_portfolio_weights §3.2: доли портфеля от доходности — без запрета (пунктир) и с запретом x >= 0
  03_complementarity   §4: условие mu*h = 0 — «уголок»; mu*h = tau — гипербола; путь точки в задаче «парк»
"""

from itertools import combinations
from pathlib import Path

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
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
# задача «парк»
HOUSE_OUT = np.array([2.0, 1.0])       # дом снаружи парка
HOUSE_IN = np.array([0.5, 0.3])        # дом внутри парка


def park_solution(a):
    """min 1/2 |x - a|^2 при h(x) = 1 - |x|^2 >= 0. Возвращает (x*, mu*).
    Внутри (|a| <= 1): x* = a, mu = 0. Снаружи: x* = a/|a|, mu = (|a| - 1)/2 — из x - a = -2 mu x."""
    a = np.asarray(a, float)
    na = np.linalg.norm(a)
    if na <= 1:
        return a.copy(), 0.0
    return a / na, (na - 1) / 2


def park_tau(a, tau):
    """Точка, где вместо mu*h = 0 стоит mu*h = tau: x = a/(1 + 2 mu), mu*(1 - |x|^2) = tau.
    Ветвь mu > 0, h > 0: при mu > (|a| - 1)/2 точка x внутри парка, поэтому корень ищется правее этого числа."""
    a = np.asarray(a, float)
    na = np.linalg.norm(a)
    phi = lambda mu: mu * (1 - na**2 / (1 + 2 * mu) ** 2) - tau
    mu = brentq(phi, max((na - 1) / 2, 0) + 1e-12, 1e3)
    return a / (1 + 2 * mu), mu


# портфель (данные лекций 3 и 8)
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

# ---- задача «парк»
x_out, mu_out = park_solution(HOUSE_OUT)
x_in, mu_in = park_solution(HOUSE_IN)
assert np.allclose(x_out, [0.894, 0.447], atol=5e-4) and abs(mu_out - 0.618) < 5e-4
assert np.allclose(x_in, HOUSE_IN) and mu_in == 0 and abs(1 - x_in @ x_in - 0.66) < 1e-12
assert np.allclose(x_out - HOUSE_OUT + 2 * mu_out * x_out, 0)                # стационарность снаружи
x_far, mu_far = -x_out, -(np.linalg.norm(HOUSE_OUT) + 1) / 2                   # второе решение системы на границе
assert np.allclose(x_far - HOUSE_OUT + 2 * mu_far * x_far, 0) and abs(mu_far + 1.618) < 5e-4   # самая дальняя точка, mu < 0
print(f"   парк: снаружи x* = {x_out.round(3)}, mu = {mu_out:.3f}; внутри x* = a, mu = 0, h = {1 - x_in @ x_in:.2f}")

tau_rows = {}
for tau in (1.0, 0.1, 0.01, 0.001):
    x_t, mu_t = park_tau(HOUSE_OUT, tau)
    tau_rows[tau] = (x_t, mu_t, 1 - x_t @ x_t, np.linalg.norm(x_t - x_out))
x_t, mu_t, h_t, d_t = tau_rows[0.01]
assert np.allclose(x_t, [0.887, 0.444], atol=5e-4) and abs(mu_t - 0.627) < 5e-4 and abs(h_t - 0.016) < 5e-4
assert abs(d_t - 0.008) < 5e-4 and abs(tau_rows[0.1][3] - 0.0735) < 1e-4 and abs(tau_rows[0.001][3] - 0.0008) < 5e-5
print("   путь mu*h = tau:", ", ".join(f"tau={t}: x=({r[0][0]:.3f}, {r[0][1]:.3f}), до x* {r[3]:.4f}" for t, r in tau_rows.items()))

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
assert np.allclose(x10, [0.0946, 0.3822, 0.3209, 0.2023], atol=5e-5)                 # лекция 8, семинар
risk0, risk1 = 0.5 * x0_12 @ PF_SIGMA @ x0_12, 0.5 * x1_12 @ PF_SIGMA @ x1_12
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

# ---------------------------------------------------------------- 00_park
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.add_patch(Circle((0, 0), 1, facecolor=AQUA, alpha=0.18, edgecolor=AQUA, lw=2))
ax.text(-0.55, -0.55, "парк\n$h(x)=1-\\Vert x\\Vert^2\\geq 0$", color=INK, fontsize=11, ha="center")
for a, label, dy in ((HOUSE_OUT, "дом снаружи\n$a=(2,\\ 1)$", 0.2), (HOUSE_IN, "дом внутри\n$a=(0.5,\\ 0.3)$", 0.18)):
    ax.plot(*a, "s", color=ORANGE, ms=11, zorder=5)
    ax.text(a[0], a[1] + dy, label, ha="center", va="bottom", fontsize=11)
ax.set(xlim=(-1.4, 2.8), ylim=(-1.25, 1.9), xlabel="$x_1$", ylabel="$x_2$", title="Ближайшая к дому точка парка")
ax.set_aspect("equal")
save("00_park.png")

# ---------------------------------------------------------------- 00_portfolio
fig, ax = plt.subplots(figsize=(6.4, 3.8))
labels = [f"актив {i + 1}\n{int(round(r * 100))} %" for i, r in enumerate(PF_R)]
colors = [RED if v < 0 else BLUE for v in x0_12]
bars = ax.bar(labels, x0_12, color=colors, width=0.6)
for b, v in zip(bars, x0_12):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}", ha="center", fontsize=11)
ax.axhline(0, color=INK, lw=0.8)
ax.set(ylabel="доля капитала", ylim=(-0.32, 0.62), title="Портфель лекции 8 при доходности 12 %")
save("00_portfolio.png")

# ---------------------------------------------------------------- 01_park_cases
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
t = np.linspace(0, 2 * np.pi, 200)
for ax, a, title in ((axes[0], HOUSE_IN, "Дом внутри: $\\mu=0$"), (axes[1], HOUSE_OUT, "Дом снаружи: $h=0$")):
    xs_, mu_ = park_solution(a)
    ax.add_patch(Circle((0, 0), 1, facecolor=AQUA, alpha=0.18, edgecolor=AQUA, lw=2))
    for rr in (0.4, 0.8, 1.236, 1.7):                                  # линии уровня f — окружности вокруг дома
        ax.plot(a[0] + rr * np.cos(t), a[1] + rr * np.sin(t), color=GRAY, lw=0.8, alpha=0.7)
    ax.plot(*a, "s", color=ORANGE, ms=10, zorder=5)
    ax.plot(*xs_, "o", color=INK, ms=7, zorder=7)
    ax.set(xlim=(-1.3, 2.6), ylim=(-1.3, 1.8), xlabel="$x_1$", title=title)
    ax.set_aspect("equal")
axes[0].set_ylabel("$x_2$")
axes[0].text(HOUSE_IN[0], -0.95, "$x^\\ast=a$, $h=0.66$", ha="center", fontsize=12)
xs_ = x_out
arrow(axes[1], xs_, xs_ + 0.55 * (HOUSE_OUT - xs_) / np.linalg.norm(HOUSE_OUT - xs_), ORANGE)       # -grad f: к дому
arrow(axes[1], xs_, xs_ - 0.55 * xs_ / np.linalg.norm(xs_), BLUE)                                  # grad h = -2x: внутрь
axes[1].text(1.05, 0.05, "$-\\nabla f$", color=ORANGE, fontsize=13)
axes[1].text(0.12, 0.42, "$\\nabla h$", color=BLUE, fontsize=13)
axes[1].text(0.9, -0.85, "$\\nabla f=0.618\\,\\nabla h$", fontsize=12, ha="center")
save("01_park_cases.png", min_width=800)

# ---------------------------------------------------------------- 02_portfolio_weights
rhos = np.linspace(0.06, 0.145, 120)
w_free = np.array([portfolio_active(r, [])[0] for r in rhos])
w_box = np.array([portfolio_bounded(r)[1] for r in rhos])
fig, ax = plt.subplots(figsize=(9.0, 4.4))
for i, col in enumerate((BLUE, ORANGE, AQUA, VIOLET)):
    ax.plot(rhos * 100, w_free[:, i], "--", color=col, lw=1.4, alpha=0.8)
    ax.plot(rhos * 100, w_box[:, i], "-", color=col, lw=2.4, label=f"актив {i + 1}")
ax.axhline(0, color=INK, lw=0.8)
ax.axvline(rho_zero * 100, color=GRAY, lw=1, ls=":")
ax.axvline(12, color=GRAY, lw=1, ls=":")
ax.text(rho_zero * 100 - 0.08, 0.86, "10.64 %", ha="right", fontsize=11, color=GRAY)
ax.text(12.08, 0.86, "12 %", ha="left", fontsize=11, color=GRAY)
ax.plot([], [], "--", color=GRAY, lw=1.4, label="без запрета")
ax.plot([], [], "-", color=GRAY, lw=2.4, label="с $x\\geq 0$")
ax.set(xlabel="требуемая доходность $\\rho$, %", ylabel="доля капитала", ylim=(-0.75, 1.0),
       title="Доли портфеля от доходности")
ax.legend(loc="lower left", ncol=3)
save("02_portfolio_weights.png", min_width=800)

# ---------------------------------------------------------------- 03_complementarity
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
ax = axes[0]
ax.plot([0, 1.1], [0, 0], color=INK, lw=3.5)
ax.plot([0, 0], [0, 1.1], color=INK, lw=3.5, label="$\\mu h=0$, $\\mu\\geq 0$, $h\\geq 0$")
for tau, col in ((0.1, BLUE), (0.01, VIOLET)):
    hh = np.linspace(tau / 1.1, 1.1, 300)
    ax.plot(hh, tau / hh, color=col, lw=2, label=f"$\\mu h={tau}$")
ax.plot(1 - x_in @ x_in, 0, "s", color=ORANGE, ms=9, zorder=6)
ax.plot(0, mu_out, "o", color=ORANGE, ms=9, zorder=6)
ax.text(1 - x_in @ x_in, 0.06, "дом внутри", ha="center", fontsize=10)
ax.text(0.05, mu_out + 0.03, "дом снаружи", fontsize=10)
ax.set(xlim=(-0.05, 1.1), ylim=(-0.05, 1.1), xlabel="$h$", ylabel="$\\mu$", title="Уголок и гиперболы")
ax.legend(loc="upper right", fontsize=10)
ax = axes[1]
ax.add_patch(Circle((0, 0), 1, facecolor=AQUA, alpha=0.18, edgecolor=AQUA, lw=2))
ax.plot(*HOUSE_OUT, "s", color=ORANGE, ms=10)
pts = np.array([tau_rows[tau][0] for tau in (1.0, 0.1, 0.01)])
ax.plot(pts[:, 0], pts[:, 1], "o-", color=BLUE, ms=6, lw=1.2)
for tau, p, (dx, dy, ha) in zip((1.0, 0.1, 0.01), pts, ((-0.05, 0.06, "right"), (0.04, -0.2, "left"), (-0.06, 0.1, "right"))):
    ax.text(p[0] + dx, p[1] + dy, f"$\\tau={tau}$", fontsize=11, ha=ha)
ax.plot(*x_out, "o", color=INK, ms=6, zorder=6)
ax.text(x_out[0] + 0.08, x_out[1] + 0.02, "$x^\\ast$", fontsize=13)
ax.set(xlim=(-0.4, 2.3), ylim=(-0.5, 1.4), xlabel="$x_1$", ylabel="$x_2$", title="Задача «парк»: точки при $\\mu h=\\tau$")
ax.set_aspect("equal")
save("03_complementarity.png", min_width=800)

print("готово")
