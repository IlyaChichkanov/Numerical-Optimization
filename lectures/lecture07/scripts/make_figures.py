"""Иллюстрации к конспекту лекции 7 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run --group casadi python lectures/lecture07/scripts/make_figures.py

Скрипт самодостаточен: разности, пример из трёх операций и упругая цепь определены здесь тем же кодом,
что в build_demo07.py / l7helpers.py; числа конспекта считаются в скрипте и проверяются ассертами.
Картинки 07, 08 считают через CasADi (граф энергии цепи, число операций, разреженность).

Картинки в порядке появления в конспекте:
  01_review           BFGS на Розенброке с точным и разностным градиентом: пути и норма точного градиента
  02_secants          sin x около x = 1, h = 0.5: касательная, прямая и центральная секущие с наклонами
  03_fd_ucurve        U-кривые прямой и центральной разности для sin x в x = 1 и две оценки ошибок пунктиром
  04_fd_noise         те же U-кривые для f с относительным шумом 1e-10 (ответ на вопрос залу, §2.8)
  05_graph_forward    граф вычислений f = x1 x2 + sin(x1 x2): значения и производные по направлению d = (1, 0)
  06_graph_reverse    тот же граф назад: сопряжённые величины и градиент за один проход
  07_cost             цена градиента и гессиана цепи в единицах цены энергии от n; время при n = 2000
  08_sparsity         гессиан цепи 80x80: портрет ненулевых из CasADi и гессиан как разность разностного градиента
"""

import time
from pathlib import Path

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image
from scipy.optimize import minimize

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
EPS = np.finfo(float).eps
HS = np.logspace(-15, -1, 141)


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


# ---------------------------------------------------------------- числа конспекта
def numbers():
    ex = np.cos(1.0)
    ef, ec = ucurve(F_SIN, X_SIN, 0, ex)
    jf, jc = int(np.argmin(ef)), int(np.argmin(ec))
    assert 1e-9 < HS[jf] < 1e-7 and 1e-7 < HS[jc] < 1e-5, (HS[jf], HS[jc])
    assert 1e-9 < ef[jf] < 5e-9 and 1e-13 < ec[jc] < 2e-12, (ef[jf], ec[jc])
    table = {h: fd_forward(F_SIN, X_SIN, 0, h) for h in (1e-1, 1e-4, 1e-8, 1e-12)}   # таблица §2.2
    assert abs(table[1e-1] - 0.49736) < 1e-5 and abs(table[1e-8] - ex) < 5e-9 and 1e-5 < abs(table[1e-12] - ex) < 1e-4, table
    costs = {N: chain_costs(N) for N in (40, 200, 1000)}
    for nf, ng, nh, nhd, nnz in costs.values():
        assert 1.5 < ng / nf < 1.7 and 5 < nh / nf < 6 and 3 < nhd / nf < 3.6
    assert costs[40][4] == 472 and costs[40][0] == 537 and costs[40][1] == 860
    return dict(ex=ex, ef=ef, ec=ec, jf=jf, jc=jc, costs=costs)


def ucurve_axes(ax, hs, curves, labels, colors, vlines=True):
    for e, lab, col in zip(curves, labels, colors):
        ax.loglog(hs, np.maximum(e, 1e-17), color=col, lw=1.8, label=lab)
    if vlines:
        for h0, txt in ((np.sqrt(EPS), r"$\sqrt{\varepsilon}$"), (EPS ** (1 / 3), r"$\varepsilon^{1/3}$")):
            ax.axvline(h0, color=GRAY, ls=":", lw=1.2)
            ax.text(h0 * 1.4, 0.03, txt, fontsize=12, color=GRAY, transform=ax.get_xaxis_transform())
    ax.set(xlabel="шаг $h$", ylabel="ошибка производной")


# ================================================================ 01 review: BFGS с разностями
def fig_review():
    runs = {}
    for name, jac in (("точный градиент", rosen_grad), ("разностный градиент (jac=None)", None)):
        r = minimize(rosen, X0_ROSEN, jac=jac, method="BFGS", options={"gtol": 1e-6, "return_all": True})
        runs[name] = (r, np.array(r.allvecs), [np.linalg.norm(rosen_grad(x)) for x in r.allvecs])
    r1, r2 = runs["точный градиент"][0], runs["разностный градиент (jac=None)"][0]
    assert r1.nit == 33 and r2.nit == 33 and r1.nfev == 40 and r2.nfev == 120, (r1.nit, r2.nit, r1.nfev, r2.nfev)
    assert np.linalg.norm(rosen_grad(r2.x)) > 1e-6 > 1e-7 > np.linalg.norm(rosen_grad(r1.x))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axs[0]
    X, Y = np.meshgrid(np.linspace(-1.6, 1.4, 300), np.linspace(-0.6, 1.6, 300))
    ax.contour(X, Y, (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2, levels=np.geomspace(0.3, 500, 12), colors=GRAY, linewidths=0.6, alpha=0.7)
    for (name, (r, P, _)), col, mk in zip(runs.items(), (BLUE, ORANGE), ("o", "x")):
        ax.plot(P[:, 0], P[:, 1], "-" + mk, color=col, ms=4 if mk == "o" else 6, lw=1.3, label=f"{name}: {r.nit} итераций, {r.nfev} вызовов $f$")
    ax.plot(1, 1, "*", color=RED, ms=14, mec="white", zorder=8)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Пути совпадают"); ax.grid(False)
    ax.legend(loc="upper left", fontsize=9)
    ax = axs[1]
    for (name, (r, P, gn)), col in zip(runs.items(), (BLUE, ORANGE)):
        ax.semilogy(np.arange(len(gn)), gn, "-o", color=col, ms=3.5, lw=1.5, label=name)
    ax.axhline(1e-6, color=GRAY, ls="--", lw=1.2); ax.text(1, 1.6e-6, "gtol = $10^{-6}$", color=GRAY, fontsize=10)
    ax.set(xlabel="итерация", ylabel=r"$\|\nabla f(x_k)\|$, точный", title="Точность ответа разная")
    ax.legend(loc="upper right", fontsize=9)
    save("01_review.png")


# ================================================================ 02 секущие
def fig_secants():
    x, h = 1.0, 0.5
    ex = np.cos(x)
    d_plus = (np.sin(x + h) - np.sin(x)) / h
    d_zero = (np.sin(x + h) - np.sin(x - h)) / (2 * h)
    assert abs(d_plus - 0.312) < 1e-3 and abs(d_zero - 0.518) < 1e-3 and abs(ex - 0.540) < 1e-3, (d_plus, d_zero, ex)
    t = np.linspace(0.25, 1.85, 400)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(t, np.sin(t), color=INK, lw=2.2, label="$f(x)=\\sin x$")
    ax.plot(t, np.sin(x) + ex * (t - x), "--", color=GRAY, lw=1.8, label=f"касательная: $f'(1)=\\cos 1={ex:.3f}$")
    ax.plot(t, np.sin(x) + d_plus * (t - x), color=BLUE, lw=2, label=f"прямая секущая через $1$ и $1.5$: $D_+={d_plus:.3f}$")
    ax.plot(t, np.sin(x) + d_zero * (t - x), color=ORANGE, lw=2, label=f"центральная секущая через $0.5$ и $1.5$: $D_0={d_zero:.3f}$")
    for px, col, lab, dy in ((x - h, ORANGE, "$x-h$", -0.09), (x, INK, "$x$", -0.09), (x + h, BLUE, "$x+h$", -0.09)):
        ax.plot(px, np.sin(px), "o", color=col, ms=9, mec="white", mew=1.5, zorder=7)
        ax.annotate(lab, (px, np.sin(px)), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=11, color=col)
    ax.set(xlim=(t[0], t[-1]), ylim=(0.1, 1.12), xlabel="$x$", ylabel="$f$", title="Секущие вместо касательной: $x=1$, $h=0.5$")
    ax.legend(loc="upper left", fontsize=10)
    save("02_secants.png")


# ================================================================ 03 U-кривая
def fig_fd_ucurve(num):
    ef, ec, jf, jc, ex = num["ef"], num["ec"], num["jf"], num["jc"], num["ex"]
    f0, f2 = np.sin(1.0), np.sin(1.0)                       # |f| и |f''| для sin x в точке 1
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.loglog(HS, HS / 2 * f2, "--", color=BLUE, lw=1.3, alpha=0.8, label=r"оценка усечения $\frac{h}{2}|f''|$")
    ax.loglog(HS, 2 * EPS * f0 / HS, "--", color=INK, lw=1.3, alpha=0.8, label=r"оценка округления $2\varepsilon|f|/h$")
    ucurve_axes(ax, HS, [ef, ec], ["прямая разность: измерено", "центральная разность: измерено"], [BLUE, ORANGE])
    for j, e, col, txt, xy in ((jf, ef, BLUE, "$h=2.5\\cdot10^{-9}$, ошибка $2\\cdot10^{-9}$", (-200, 10)),
                               (jc, ec, ORANGE, "$h=3\\cdot10^{-6}$, ошибка $6\\cdot10^{-13}$", (18, -4))):
        ax.plot(HS[j], e[j], "o", color=col, ms=8, mec="white", zorder=7)
        ax.annotate(txt, (HS[j], e[j]), textcoords="offset points", xytext=xy, fontsize=10, color=col)
    ax.text(1e-3, 3e-5, "наклон 2", color=ORANGE, fontsize=10.5, rotation=34)
    ax.set(title=r"U-кривая: ошибка разности для $f=\sin x$ в $x=1$ по шагу $h$", ylim=(1e-14, 1e2))
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    save("03_fd_ucurve.png")


# ================================================================ 04 шумная функция
def fig_fd_noise(num):
    rng = np.random.default_rng(0)
    noisy = lambda x: np.sin(x[0]) * (1 + 1e-10 * rng.standard_normal())
    efn, ecn = ucurve(noisy, X_SIN, 0, num["ex"])
    assert 1e-6 < HS[np.argmin(efn)] < 1e-4 and 1e-8 < efn.min() < 1e-6, (HS[np.argmin(efn)], efn.min())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ucurve_axes(ax, HS, [num["ef"], num["ec"]], ["прямая, точная $f$", "центральная, точная $f$"], [BLUE, ORANGE], vlines=False)
    for e, lab, col in ((efn, "прямая, $f$ с шумом $10^{-10}$", BLUE), (ecn, "центральная, $f$ с шумом $10^{-10}$", ORANGE)):
        ax.loglog(HS, np.maximum(e, 1e-17), color=col, lw=1.8, alpha=0.4, label=lab)
        j = int(np.argmin(e)); ax.plot(HS[j], e[j], "o", color=col, ms=8, mec="white", alpha=0.7, zorder=7)
    ax.annotate(f"$h={HS[np.argmin(efn)]:.1e}$, ошибка ${efn.min():.1e}$".replace("e-0", "\\cdot10^{-").replace("$, ", "}$, ").replace("e-1", "\\cdot10^{-1") + "}$" if False else
                "$h=1.6\\cdot10^{-5}$, ошибка $1.4\\cdot10^{-7}$", (HS[np.argmin(efn)], efn.min()), textcoords="offset points", xytext=(14, 18), fontsize=10, color=BLUE)
    ax.annotate("$h=8\\cdot10^{-4}$, ошибка $1.6\\cdot10^{-9}$", (HS[np.argmin(ecn)], ecn.min()), textcoords="offset points", xytext=(12, -18), fontsize=10, color=ORANGE)
    ax.set(title="Шумная функция: левая ветка поднялась, дно уехало вправо", ylim=(1e-14, 1e2))
    ax.legend(loc="upper center", ncol=2, fontsize=10)
    save("04_fd_noise.png")


# ================================================================ 05–06 граф вычислений
NODES = {"x1": (0.0, 1.1), "x2": (0.0, -0.1), "v1": (1.8, 0.5), "v2": (3.6, 1.4), "f": (5.4, 0.5)}
EDGES = [("x1", "v1"), ("x2", "v1"), ("v1", "v2"), ("v1", "f"), ("v2", "f")]
EDGE_OFFSET = {("x1", "v1"): (0.0, 0.22), ("x2", "v1"): (0.0, -0.24), ("v1", "v2"): (-0.3, 0.22), ("v1", "f"): (0.0, -0.2), ("v2", "f"): (0.3, 0.22)}
LABEL_ABOVE = {"v2"}            # у v2 вторая строка пишется над узлом, чтобы не лечь на стрелку v1 -> f


def draw_graph(ax, top, bottom, edge_labels, reverse=False, bottom_color=BLUE):
    """Граф f = x1 x2 + sin(x1 x2): в узле сверху — имя и значение (top), снизу — вторая строка (bottom)."""
    R = 0.42
    for a, b in EDGES:
        p, q = np.array(NODES[a]), np.array(NODES[b])
        u = (q - p) / np.linalg.norm(q - p)
        p2, q2 = p + R * u * 1.08, q - R * u * 1.08
        if reverse:
            p2, q2 = q2, p2
        arrow(ax, p2, q2, GRAY if not reverse else VIOLET, lw=1.6, ms=16)
        mid = 0.5 * (p + q) + np.array(EDGE_OFFSET[(a, b)])
        ax.text(*mid, edge_labels[(a, b)], fontsize=10.5, ha="center", va="center", color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=8)
    for name, (x, y) in NODES.items():
        ax.add_patch(Circle((x, y), R, fc="white", ec=INK, lw=1.4, zorder=5))
        ax.text(x, y + 0.02, top[name], fontsize=10.5, ha="center", va="center", zorder=7)
        if name in LABEL_ABOVE:
            ax.text(x, y + R + 0.12, bottom[name], fontsize=11, ha="center", va="bottom", color=bottom_color, zorder=7)
        else:
            ax.text(x, y - R - 0.12, bottom[name], fontsize=11, ha="center", va="top", color=bottom_color, zorder=7)
    ax.set(xlim=(-0.7, 6.1), ylim=(-1.25, 2.3), aspect="equal"); ax.axis("off")


def fig_graphs():
    t = tape_example()
    assert abs(t["bx1"] - 0.7702) < 5e-5 and abs(t["bx2"] - 3.0806) < 5e-5 and abs(t["df"] - t["bx1"]) < 1e-12
    top = {"x1": "$x_1=2$", "x2": "$x_2=0.5$", "v1": "$v_1=1$", "v2": f"$v_2={t['v2']:.4f}$", "f": f"$f={t['f']:.4f}$"}
    labels = {("x1", "v1"): "$x_2=0.5$", ("x2", "v1"): "$x_1=2$", ("v1", "v2"): r"$\cos v_1=0.5403$", ("v1", "f"): "$1$", ("v2", "f"): "$1$"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottom = {"x1": r"$\dot x_1=1$", "x2": r"$\dot x_2=0$", "v1": r"$\dot v_1=0.5$", "v2": f"$\\dot v_2={t['dv2']:.4f}$", "f": f"$\\dot f={t['df']:.4f}$"}
    draw_graph(ax, top, bottom, labels)
    ax.set_title(r"Прямой режим: $v_i$ и $\dot v_i$ текут слева направо, $d=(1,0)$")
    ax.text(2.7, -1.2, r"$\dot v_i=\sum_j \frac{\partial\varphi_i}{\partial v_j}\dot v_j$:  $\dot v_1=0.5\cdot1+2\cdot0$,  $\dot v_2=0.5403\cdot0.5$,  $\dot f=\dot v_1+\dot v_2=\partial f/\partial x_1$",
            fontsize=10.5, ha="center", color=BLUE)
    save("05_graph_forward.png")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bottom = {"x1": f"$\\bar x_1={t['bx1']:.4f}$", "x2": f"$\\bar x_2={t['bx2']:.4f}$", "v1": f"$\\bar v_1=1+0.5403$", "v2": r"$\bar v_2=1$", "f": r"$\bar f=1$"}
    draw_graph(ax, top, bottom, labels, reverse=True, bottom_color=VIOLET)
    ax.set_title(r"Обратный режим: $\bar v_i=\partial f/\partial v_i$ текут от $f$ к входам — один проход")
    ax.text(2.7, -1.2, r"$\bar v_j\ {+}{=}\ \bar v_i\,\partial\varphi_i/\partial v_j$:  $\bar v_1=1+1\cdot0.5403=1.5403$,  $\nabla f=(\bar v_1 x_2,\ \bar v_1 x_1)=(0.7702,\ 3.0806)$",
            fontsize=10.5, ha="center", color=VIOLET)
    save("06_graph_reverse.png")


# ================================================================ 07 цена
def fig_cost(num):
    ns = np.array([80, 400, 2000])
    c = num["costs"]
    grad_ad = np.array([c[N][1] / c[N][0] for N in (40, 200, 1000)])
    hess_ad = np.array([c[N][2] / c[N][0] for N in (40, 200, 1000)])
    hd_ad = np.array([c[N][3] / c[N][0] for N in (40, 200, 1000)])
    # время при n = 2000
    energy, grad, v0 = make_elastic_chain(1000)
    v, E = chain_casadi(1000); G = ca.Function("G", [v], [ca.gradient(E, v)])
    t0 = time.perf_counter(); [G(v0) for _ in range(50)]; t_ad = (time.perf_counter() - t0) / 50
    t0 = time.perf_counter(); [fd_forward(energy, v0, i, 1e-8) for i in range(2000)]; t_fd = time.perf_counter() - t0
    assert t_fd > 100 * t_ad, (t_fd, t_ad)      # в конспекте: «в сотни раз»
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]
    nn = np.geomspace(50, 4000, 50)
    ax.loglog(nn, nn + 1, color=ORANGE, lw=1.8, label="градиент разностями: $n+1$ вычислений $f$")
    ax.loglog(ns, ns + 1, "o", color=ORANGE, ms=7)
    for y, col, lab, mk in ((hess_ad, RED, "гессиан, CasADi", "s"), (hd_ad, AQUA, r"$\nabla^2 f\,d$, CasADi", "D"), (grad_ad, BLUE, "градиент, обратный режим", "o")):
        ax.loglog(ns, y, "-" + mk, color=col, ms=7, lw=1.8, label=f"{lab}: {y.mean():.1f}")
    for n_, g_ in zip(ns, ns + 1):
        ax.annotate(str(g_), (n_, g_), textcoords="offset points", xytext=(8, -4), fontsize=10, color=ORANGE)
    ax.set(xlabel="переменных $n$", ylabel="цена, в операциях $f$", title="Упругая цепь: цена производных от $n$", ylim=(1, 1e5), xticks=[80, 400, 2000], xticklabels=["80", "400", "2000"])
    ax.legend(loc="upper left", fontsize=9.5)
    ax = axs[1]
    bars = ax.bar(["разности", "CasADi"], [t_fd * 1e3, t_ad * 1e3], color=[ORANGE, BLUE], width=0.55)
    ax.set_yscale("log"); ax.set(ylabel="мс на один градиент", title="$n=2000$: время градиента", ylim=(0.05, 2000))
    for b, val in zip(bars, (t_fd * 1e3, t_ad * 1e3)):
        ax.text(b.get_x() + b.get_width() / 2, val * 1.3, f"{val:.2f} мс" if val < 1 else f"{val:.0f} мс", ha="center", fontsize=11)
    ax.grid(False, axis="x")
    save("07_cost.png")


# ================================================================ 08 разреженность
def fig_sparsity():
    energy, grad, _ = make_elastic_chain(40)
    pattern, v0, Hfun = chain_pattern(40)                              # в равновесии цепи: все ленты заполнены
    H = np.array(Hfun(v0))
    n, h1, h2 = 80, np.sqrt(EPS), 1e-4                                 # разность разностного градиента: как SciPy без jac и hess
    g_fd = lambda v: np.array([fd_forward(energy, v, i, h1) for i in range(n)])
    g0 = g_fd(v0)
    Hff = np.array([(g_fd(v0 + h2 * e) - g0) / h2 for e in np.eye(n)]).T
    err = np.abs(Hff - H).max()
    nnz_ca, nnz_fd = int(pattern.sum()), int(np.sum(np.abs(Hff) > 1e-12))
    assert nnz_ca == 472 and int(np.sum(np.abs(H) > 1e-12)) >= 460 and 5e-3 < err < 1e-1 and nnz_fd > 1000, (nnz_ca, err, nnz_fd)   # средняя пружина горизонтальна: 4 элемента y–z численно нулевые
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
    ax = axs[0]
    ax.spy(pattern, markersize=2.2, color=INK)
    ax.set(title=f"CasADi: {nnz_ca} ненулевых из 6400", xticks=[0, 40, 79], yticks=[0, 40, 79]); ax.grid(False)
    ax.text(40, 86, "блоки: $y$–$y$, $y$–$z$, $z$–$y$, $z$–$z$ — по пружине на соседей", ha="center", fontsize=10, color=GRAY)
    ax = axs[1]
    pc = ax.pcolormesh(np.maximum(np.abs(Hff), 1e-12), cmap="Greys", norm=LogNorm(vmin=1e-12, vmax=1e3))
    ax.invert_yaxis(); ax.set(aspect="equal", title=f"разность разностей:\nошибка до {err:.0e}, «ненулевых» {nnz_fd}", xticks=[0, 40, 80], yticks=[0, 40, 80]); ax.grid(False)
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"$|\nabla^2E_{ij}|$")
    save("08_sparsity.png")


if __name__ == "__main__":
    num = numbers()
    print("числа проверены; рисую")
    fig_review()
    fig_secants()
    fig_fd_ucurve(num)
    fig_fd_noise(num)
    fig_graphs()
    fig_cost(num)
    fig_sparsity()
