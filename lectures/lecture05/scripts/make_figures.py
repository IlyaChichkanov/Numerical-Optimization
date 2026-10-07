"""Иллюстрации к конспекту лекции 5 -> папка ../img/ (коммитится в репозиторий).

Запуск:  uv run python lectures/lecture05/scripts/make_figures.py

Скрипт самодостаточен. Палитра — та же, что в лекциях 1-4; размеры — по правилу
«ширина PNG в пикселях ≈ ширине показа»: dpi 100, figsize не шире 9 дюймов, шрифт 12,
так что при <img width="900"> подписи не уменьшаются. Методы (gd, newton, newton_damped,
bfgs, gauss_newton) определены здесь тем же кодом, что в build_demo05.py; числа
конспекта считаются в скрипте и проверяются ассертами.

Картинки в порядке появления в конспекте:
  00_hooks            две задачи-крючка: Розенброк и синус ДЗ 2
  01_model_1d         парабола-модель касается функции; вершина — следующая точка
  02_model_ellipses   эллипсы модели на Розенброке в x_0 и x_1
  03_newton_1d        ln cosh: сходится из 0.9, расходится из 1.2
  04_rates_newton     спуск (прямая) против чистого Ньютона (горб и обрыв)
  05_basins           Химмельблау: куда приходит Ньютон из каждой точки
  06_damped           чистый против демпфированного Ньютона: 7 с выбросом, 22 без
  07_bowl_saddle      модель-чаша и модель-седло: куда ведёт шаг
  08_secant           касательная против секущей на графике f'
  09_bfgs_ellipses    BFGS на квадратичной: B_0 = I, B_1, B_2 = Q
  10_rates_three      спуск, BFGS, Ньютон: прямая, загиб, обрыв
  11_gn_sine          Гаусс–Ньютон на синусе: кривые и ошибки
  12_chain_methods    цепь: итерации от N для четырёх методов
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy.optimize import brentq, minimize

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


# ---------------------------------------------------------------- методы (те же, что в demo05 / l5helpers.py)
def gd(f, grad, x0, alpha=None, tol=1e-6, maxit=100_000, c=1e-4):
    """Градиентный спуск (лекция 4): alpha=None — backtracking по правилу Армихо с константой c,
    иначе постоянный шаг. Возвращает (x, число итераций, путь (k+1, n))."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        if alpha is None:
            a, fx = 1.0, f(x)
            while f(x - a * g) > fx - c * a * (g @ g):
                a /= 2
        else:
            a = alpha
        x = x - a * g; path.append(x.copy())
    return x, maxit, np.array(path)


def newton(grad, hess, x0, tol=1e-10, maxit=50):
    """Чистый метод Ньютона: x <- x - H^{-1} g, без выбора длины шага. Возвращает (x, итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        x = x - np.linalg.solve(hess(x), g); path.append(x.copy())
    return x, maxit, np.array(path)


def newton_damped(f, grad, hess, x0, tol=1e-10, maxit=200, c=1e-4):
    """Демпфированный Ньютон: направление Ньютона, длина шага — по правилу Армихо.
    Если гессиан индефинитен и направление не является спуском, матрица
    сдвигается: H + (|lambda_min| + 1e-3) I. Возвращает (x, итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        H = hess(x)
        p = -np.linalg.solve(H, g)
        if g @ p >= 0:                                   # не спуск: сдвигаем собственные числа
            lam_min = np.linalg.eigvalsh(H)[0]
            p = -np.linalg.solve(H + (abs(lam_min) + 1e-3) * np.eye(len(x)), g)
        a, fx = 1.0, f(x)
        while f(x + a * p) > fx + c * a * (g @ p):
            a /= 2
        x = x + a * p; path.append(x.copy())
    return x, maxit, np.array(path)


def bfgs(f, grad, x0, tol=1e-6, maxit=10_000, c=1e-4, Q=None):
    """BFGS с H_0 = I: p = -H g, обновление обратного гессиана по паре (s, y).
    Длина шага: правило Армихо (Q=None) или точный шаг на квадратичной
    1/2 x^T Q x + b^T x (передать Q). Возвращает (x, итераций, путь, число вычислений f)."""
    x = np.asarray(x0, float).copy(); n = len(x)
    H = np.eye(n); path = [x.copy()]; g = grad(x); nfev = 1
    for k in range(maxit):
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path), nfev
        p = -H @ g
        if Q is None:
            a, fx = 1.0, f(x)
            while f(x + a * p) > fx + c * a * (g @ p):
                a /= 2; nfev += 1
            nfev += 1
        else:
            a = -(g @ p) / (p @ Q @ p)
        s = a * p; x_new = x + s; g_new = grad(x_new); y = g_new - g
        if s @ y > 1e-12:                                # условие кривизны: иначе H не обновляем
            rho = 1.0 / (s @ y); I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, g = x_new, g_new; path.append(x.copy())
    return x, maxit, np.array(path), nfev


def gauss_newton(resid, jac, x0, tol=1e-6, maxit=100, damped=False, c=1e-4):
    """Гаусс–Ньютон для f = 1/2 ||r(x)||^2: шаг — решение линейного МНК min ||J p + r||
    (np.linalg.lstsq). damped=True — длина шага по правилу Армихо. Возвращает (x, итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        r = resid(x); J = jac(x); g = J.T @ r
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        p = np.linalg.lstsq(J, -r, rcond=None)[0]
        a = 1.0
        if damped:
            fx = 0.5 * (r @ r)
            while 0.5 * np.sum(resid(x + a * p) ** 2) > fx + c * a * (g @ p) and a > 1e-10:
                a /= 2
        x = x + a * p; path.append(x.copy())
    return x, maxit, np.array(path)


# ---------------------------------------------------------------- задачи
def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])


def rosen_hess(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200.0]])


def rosen_contour(ax, xlim=(-2, 2), ylim=(-1, 3)):
    X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3.3, 14), colors=GRAY, linewidths=0.7, alpha=0.8)
    ax.plot(1, 1, "*", color=RED, ms=14, zorder=7)
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$")


X0_ROSEN = np.array([-1.2, 1.0])


def make_chain(N, D=70.0, g=9.81):
    """Цепь без пола (лекция 1, §7.3): E(v) = 1/2 v^T H v + b^T v + const, v = (y_1..y_N, z_1..z_N).
    Возвращает (energy, grad_E, H, b, v0, unpack); unpack(v) -> (y, z) с закреплёнными концами."""
    m = 4.0 / N
    K = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    H = np.kron(np.eye(2), D * K)
    b = np.zeros(2 * N)
    b[0] -= D * (-2.0); b[N - 1] -= D * 2.0
    b[N] -= D * 1.0; b[2 * N - 1] -= D * 1.0
    b[N:] += m * g
    v0 = np.r_[np.linspace(-2, 2, N + 2)[1:-1], np.ones(N)]       # старт: прямая между концами

    def unpack(v):
        return np.r_[-2.0, v[:N], 2.0], np.r_[1.0, v[N:], 1.0]

    def energy(v):
        y, z = unpack(v)
        return 0.5 * D * np.sum(np.diff(y) ** 2 + np.diff(z) ** 2) + m * g * np.sum(z[1:-1])

    def grad_E(v):
        return H @ v + b
    return energy, grad_E, H, b, v0, unpack


def sine_data():
    """Данные задачи 2(б) ДЗ 2: y = 2 sin(3t + 0.7) + шум, 60 точек, тот же генератор и порядок вызовов."""
    data_rng = np.random.default_rng(2026)
    N, omega = 60, 3.0
    t = np.sort(data_rng.uniform(0, 2 * np.pi, N))
    y = 2.0 * np.sin(3.0 * t + 0.7) + 0.2 * data_rng.standard_normal(N)
    starts = data_rng.uniform(-5, 5, (20, 3))
    return t, y, starts


def sine_problem(t, y):
    """resid, jac, f, grad, hess для модели x1 sin(x2 t + x3)."""
    def resid(x):
        return x[0] * np.sin(x[1] * t + x[2]) - y

    def jac(x):
        s, c = np.sin(x[1] * t + x[2]), np.cos(x[1] * t + x[2])
        return np.c_[s, x[0] * t * c, x[0] * c]

    def f(x):
        r = resid(x); return 0.5 * (r @ r)

    def grad(x):
        return jac(x).T @ resid(x)

    def hess(x):
        J, r = jac(x), resid(x)
        s, c = np.sin(x[1] * t + x[2]), np.cos(x[1] * t + x[2])
        S = np.zeros((3, 3))                            # сумма r_i * hess(r_i)
        S[0, 1] = S[1, 0] = r @ (t * c); S[0, 2] = S[2, 0] = r @ c
        S[1, 1] = r @ (-x[0] * t ** 2 * s); S[1, 2] = S[2, 1] = r @ (-x[0] * t * s); S[2, 2] = r @ (-x[0] * s)
        return J.T @ J + S
    return resid, jac, f, grad, hess


X0_SINE = np.array([1.0, 3.0, 0.0])


def himmelblau_newton_basins(n=300, lim=6.0, maxit=30):
    """Чистый Ньютон на функции Химмельблау из каждой точки сетки n x n на [-lim, lim]^2.
    Возвращает сетку, индекс стационарной точки (или -1), число итераций, список точек и их типы."""
    xs = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, xs); X = X.copy(); Y = Y.copy()
    its = np.full(X.shape, -1)
    for k in range(maxit):
        a, b = X ** 2 + Y - 11, X + Y ** 2 - 7
        gx, gy = 4 * X * a + 2 * b, 2 * a + 4 * Y * b
        hxx, hxy, hyy = 12 * X ** 2 + 4 * Y - 42, 4 * X + 4 * Y, 4 * X + 12 * Y ** 2 - 26
        det = hxx * hyy - hxy ** 2
        with np.errstate(all="ignore"):
            dx = (hyy * gx - hxy * gy) / det; dy = (hxx * gy - hxy * gx) / det
        done = (np.hypot(gx, gy) < 1e-8) & (its < 0)
        its[done] = k
        active = its < 0
        X = np.where(active, X - dx, X); Y = np.where(active, Y - dy, Y)
    pts = np.c_[X.ravel(), Y.ravel()]
    ok = its.ravel() >= 0
    uniq = np.unique(pts[ok].round(3), axis=0)
    idx = np.full(X.size, -1)
    for j, p in enumerate(uniq):
        idx[ok & np.all(np.isclose(pts, p, atol=1e-2), axis=1)] = j
    kinds = []
    for p in uniq:
        hxx, hxy, hyy = 12 * p[0] ** 2 + 4 * p[1] - 42, 4 * p[0] + 4 * p[1], 4 * p[0] + 12 * p[1] ** 2 - 26
        lam = np.linalg.eigvalsh([[hxx, hxy], [hxy, hyy]])
        kinds.append("min" if lam.min() > 0 else ("max" if lam.max() < 0 else "saddle"))
    return xs, idx.reshape(X.shape), its, uniq, kinds


def lncosh_newton(x0, n=4):
    xs = [x0]
    for _ in range(n):
        xs.append(xs[-1] - 0.5 * np.sinh(2 * xs[-1]))
    return np.array(xs)


# ================================================================ общие прогоны (числа конспекта)
def runs_rosen():
    xg, kg, pg = gd(rosen, rosen_grad, X0_ROSEN)
    xn, kn, pn = newton(rosen_grad, rosen_hess, X0_ROSEN)
    xd, kd, pd = newton_damped(rosen, rosen_grad, rosen_hess, X0_ROSEN)
    xb, kb, pb, nfb = bfgs(rosen, rosen_grad, X0_ROSEN)
    assert abs(kg - 13756) < 140, kg
    assert kn == 7 and np.allclose(xn, 1, atol=1e-8), (kn, xn)
    assert kd == 22 and np.allclose(xd, 1, atol=1e-8), (kd, xd)
    assert kb == 34 and nfb == 54 and np.allclose(xb, 1, atol=1e-6), (kb, nfb, xb)
    en = np.linalg.norm(pn - 1, axis=1)
    assert np.allclose(en[:7], [2.2, 2.208, 4.182, 0.4796, 0.05597, 9.62e-6, 1.85e-11], rtol=0.02), en
    assert np.allclose(pn[2], [0.763, -3.175], atol=2e-3), pn[2]
    ed = np.linalg.norm(pd - 1, axis=1)
    assert ed.max() < 2.3, ed.max()                      # демпфированный: без выброса (у чистого 4.18)
    assert all(np.linalg.eigvalsh(rosen_hess(x))[0] > 0 for x in pn), "гессиан на пути Ньютона должен быть > 0"
    # SciPy с тем же допуском: числа раздела 4.4 и таблицы раздела 6
    r_b = minimize(rosen, X0_ROSEN, jac=rosen_grad, method="BFGS", options={"gtol": 1e-6})
    r_l = minimize(rosen, X0_ROSEN, jac=rosen_grad, method="L-BFGS-B", options={"gtol": 1e-6})
    assert (r_b.nit, r_b.nfev) == (33, 40), (r_b.nit, r_b.nfev)
    assert r_l.nit == 36, r_l.nit
    return dict(gd=(kg, pg), newton=(kn, pn), damped=(kd, pd), bfgs=(kb, pb, nfb), scipy=(r_b.nit, r_b.nfev, r_l.nit))


def runs_sine():
    t, y, starts = sine_data()
    resid, jac, f, grad, hess = sine_problem(t, y)
    xgn, kgn, pgn = gauss_newton(resid, jac, X0_SINE)
    xgd, kgd, pgd = gd(f, grad, X0_SINE)
    xnw, knw, pnw = newton(grad, hess, X0_SINE, tol=1e-6)
    assert kgn == 7 and np.allclose(xgn, [2.1014, 2.9967, 0.7003], atol=1e-3), (kgn, xgn)
    assert abs(kgd - 731) < 40 and np.allclose(xgd, xgn, atol=1e-3), (kgd, xgd)
    assert abs(xnw[0]) < 1e-6 and knw == 9, (knw, xnw)          # Ньютон пришёл в точку с нулевой амплитудой
    assert abs(f(xgn) - 1.1393) < 1e-3, f(xgn)
    lamJ = np.linalg.eigvalsh(jac(xgn).T @ jac(xgn)); lamH = np.linalg.eigvalsh(hess(xgn))
    assert np.allclose(lamJ, [17.57, 31.05, 1935.6], rtol=0.01) and np.allclose(lamH, [17.89, 31.12, 1927.3], rtol=0.01), (lamJ, lamH)
    return dict(t=t, y=y, starts=starts, resid=resid, jac=jac, f=f, grad=grad, hess=hess,
                gn=(kgn, pgn, xgn), gd=(kgd, pgd), newton=(knw, pnw, xnw))


# ================================================================ 00 две задачи-крючка
def fig_hooks(R, S):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    ax = axes[0]
    kg, pg = R["gd"]; kn, pn = R["newton"]
    rosen_contour(ax, xlim=(-2.2, 2.2), ylim=(-3.5, 3.2))
    ax.plot(pg[::50, 0], pg[::50, 1], "-", color=BLUE, lw=1.4, label=f"спуск: {kg} итераций")
    ax.plot(pn[:, 0], pn[:, 1], "-o", color=ORANGE, ms=5, lw=1.8, label=f"Ньютон: {kn} итераций")
    ax.annotate("итерация 2", pn[2], (pn[2, 0] - 1.9, pn[2, 1] + 0.1), color=ORANGE)
    ax.plot(*X0_ROSEN, "o", color=INK, ms=6, zorder=7)
    ax.legend(loc="upper left")
    ax.set_title("Розенброк: 13 756 против 7 — и выброс")

    ax = axes[1]
    t, y = S["t"], S["y"]; kgn, pgn, xgn = S["gn"]; kgd, _ = S["gd"]; knw, _, xnw = S["newton"]
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(t, y, "o", ms=3.5, color=INK, alpha=0.7, label="данные ДЗ 2")
    ax.plot(tt, xgn[0] * np.sin(xgn[1] * tt + xgn[2]), color=BLUE, lw=2.2, label=f"спуск: {kgd} итерация")
    ax.plot(tt, xnw[0] * np.sin(xnw[1] * tt + xnw[2]), color=RED, lw=2.4, ls="--", label=f"Ньютон: {knw} итераций, $x_1=0$")
    ax.set(xlabel="$t$", ylabel="$y$", ylim=(-3.2, 4.2), title="Синус: $x_1\\sin(x_2t+x_3)$")
    ax.legend(loc="upper right")
    save("00_hooks.png")


# ================================================================ 01 парабола-модель касается функции
def fig_model_1d():
    f = lambda x: np.log(np.cosh(x)); df = np.tanh; d2f = lambda x: 1 / np.cosh(x) ** 2
    xk = 0.9; x_next = xk - df(xk) / d2f(xk)
    assert abs(x_next + 0.571) < 2e-3, x_next
    t = np.linspace(-2.2, 2.6, 600)
    par = f(xk) + df(xk) * (t - xk) + 0.5 * d2f(xk) * (t - xk) ** 2
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(t, f(t), color=INK, lw=2.4, label="функция $f$")
    ax.plot(t, par, color=ORANGE, lw=2, label="модель $m_k(p)$ — парабола")
    ax.plot(xk, f(xk), "o", color=INK, ms=8, zorder=7)
    ax.plot(x_next, f(xk) + df(xk) * (x_next - xk) + 0.5 * d2f(xk) * (x_next - xk) ** 2, "s", color=ORANGE, ms=8, zorder=7)
    ax.plot([x_next, x_next], [-0.3, f(x_next)], ":", color=ORANGE, lw=1.2)
    ax.plot(0, 0, "*", color=RED, ms=14, zorder=7)
    ax.annotate("$x_k$: та же высота,\nтот же наклон,\nта же кривизна", (xk, f(xk)), (1.2, -0.56), color=INK,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.annotate("$x_{k+1}=x_k+p_k$ —\nвершина параболы", (x_next, -0.3), (-2.15, -0.5), color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.annotate("$x^*$", (0, 0), (0.12, -0.22), color=RED)
    ax.set(xlim=(-2.2, 2.6), ylim=(-0.62, 1.9), xlabel="$x$", title="Квадратичная модель в точке $x_k$ и шаг Ньютона")
    ax.legend(loc="upper left")
    save("01_model_1d.png")


# ================================================================ 02 эллипсы модели в двух точках пути
def fig_model_ellipses(R):
    kn, pn = R["newton"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, k, xlim, ylim, title in (
        (axes[0], 0, (-2.2, 0.6), (-0.4, 2.4), "старт $x_0$: модель похожа на функцию"),
        (axes[1], 1, (-2.2, 2.2), (-3.6, 2.6), "$x_1$: минимум модели далеко от функции"),
    ):
        xk = pn[k]; g = rosen_grad(xk); Hk = rosen_hess(xk); pstar = -np.linalg.solve(Hk, g)
        X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
        Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2
        ax.contour(X, Y, Z, levels=np.logspace(-1, 3.3, 14), colors=GRAY, linewidths=0.7, alpha=0.8)
        P1, P2 = X - xk[0], Y - xk[1]
        M = rosen(xk) + g[0] * P1 + g[1] * P2 + 0.5 * (Hk[0, 0] * P1 ** 2 + 2 * Hk[0, 1] * P1 * P2 + Hk[1, 1] * P2 ** 2)
        mmin = rosen(xk) + 0.5 * g @ pstar
        ax.contour(X, Y, M, levels=mmin + np.geomspace(0.5, 400, 9), colors=ORANGE, linewidths=1.0, alpha=0.9)
        ax.plot(*xk, "o", color=INK, ms=7, zorder=7)
        arrow(ax, xk, xk + pstar, ORANGE, lw=2.2)
        ax.plot(*(xk + pstar), "s", color=ORANGE, ms=7, zorder=7)
        ax.plot(1, 1, "*", color=RED, ms=14, zorder=7)
        ax.set(xlim=xlim, ylim=ylim, xlabel="$x_1$", ylabel="$x_2$", title=title)
    axes[1].annotate(f"$x_2=({pn[2, 0]:.2f},\\,{pn[2, 1]:.2f})$", pn[2], (pn[2, 0] - 2.0, pn[2, 1] + 0.15), color=ORANGE)
    save("02_model_ellipses.png")


# ================================================================ 03 ln cosh: сходится из 0.9, расходится из 1.2
def fig_newton_1d():
    f = lambda x: np.log(np.cosh(x)); df = np.tanh; d2f = lambda x: 1 / np.cosh(x) ** 2
    seq_a, seq_b = lncosh_newton(0.9), lncosh_newton(1.2, 3)
    assert abs(seq_a[4]) < 1e-8 and abs(seq_a[3] + 1.557e-3) < 1e-5 and abs(seq_b[1] + 1.533) < 2e-3 and abs(seq_b[2] - 3.82) < 0.01 and abs(seq_b[3]) > 100, (seq_a, seq_b)
    thr = brentq(lambda x: 2 * x - 0.5 * np.sinh(2 * x), 0.5, 1.5)      # порог: x1 = -x0
    assert abs(thr - 1.0886) < 1e-3, thr
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    t = np.linspace(-2.6, 4.2, 600)
    for ax, seq, title, color in ((axes[0], seq_a, "старт $0.9$: четыре шага — и $10^{-8}$", AQUA),
                                  (axes[1], seq_b, "старт $1.2$: вершина перелетает всё дальше", RED)):
        ax.plot(t, f(t), color=INK, lw=2.2, label="$f=\\ln\\cosh x$")
        for i in range(2):
            xk = seq[i]; par = f(xk) + df(xk) * (t - xk) + 0.5 * d2f(xk) * (t - xk) ** 2
            ax.plot(t, par, color=color, lw=1.4, alpha=0.9 - 0.3 * i, ls="-" if i == 0 else "--",
                    label="парабола в $x_k$" if i == 0 else None)
            ax.plot(xk, f(xk), "o", color=color, ms=7, zorder=7)
            ax.annotate(f"$x_{i}$", (xk, f(xk)), (xk + 0.1, f(xk) + 0.25), color=color)
            ax.plot([seq[i + 1], seq[i + 1]], [0, f(seq[i + 1])], ":", color=color, lw=1)
        ax.plot(seq[2], f(seq[2]), "o", color=color, ms=7, zorder=7)
        ax.annotate(f"$x_2={seq[2]:.2f}$", (seq[2], f(seq[2])), (seq[2] + 0.15, 1.1) if abs(seq[2]) < 1 else (seq[2] - 1.45, 2.75), color=color)
        ax.plot(0, 0, "*", color=RED, ms=14, zorder=7)
        ax.set(xlim=(-2.6, 4.2), ylim=(-0.3, 3.3), xlabel="$x$", title=title)
        ax.legend(loc="upper left")
    axes[1].text(0.03, 0.62, f"порог $x_0\\approx{thr:.4f}$:\n2-цикл $x_1=-x_0$", transform=axes[1].transAxes, color=RED)
    save("03_newton_1d.png")
    return thr


def errors_rosen(R):
    kg, pg = R["gd"]; kn, pn = R["newton"]; kd, pd = R["damped"]; kb, pb, nfb = R["bfgs"]
    return {k: np.linalg.norm(p - 1, axis=1) + 1e-17 for k, p in (("gd", pg), ("newton", pn), ("damped", pd), ("bfgs", pb))}


# ================================================================ 04 спуск против чистого Ньютона
def fig_rates_newton(R):
    kg, _ = R["gd"]; kn, pn = R["newton"]; err = errors_rosen(R)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(np.arange(41), np.log10(err["gd"][:41]), color=BLUE, label=f"градиентный спуск ({kg} итераций)")
    ax.plot(np.arange(len(pn)), np.log10(err["newton"]), "-o", ms=6, color=ORANGE, label=f"чистый Ньютон ({kn})")
    ax.annotate("модель врёт:\nошибка растёт", (2, np.log10(err["newton"][2])), (5.5, -1.6), color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.annotate("обрыв: цифры удваиваются", (5, np.log10(err["newton"][5])), (8, -6), color=ORANGE,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.set(xlabel="итерация $k$", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$", xlim=(0, 40), ylim=(-12.5, 1.4),
           title="Розенброк, первые 40 итераций: прямая против обрыва")
    ax.legend(loc="lower left")
    save("04_rates_newton.png")


# ================================================================ 05 бассейны Ньютона на Химмельблау
def fig_basins():
    t0 = time.perf_counter()
    xs, idx, its, uniq, kinds = himmelblau_newton_basins()
    n_min = sum(k == "min" for k in kinds); n_sad = sum(k == "saddle" for k in kinds); n_max = sum(k == "max" for k in kinds)
    assert (n_min, n_sad, n_max) == (4, 4, 1), kinds
    ok = idx >= 0
    shares = {kind: np.mean([kinds[j] == kind for j in idx[ok]]) for kind in ("min", "saddle", "max")}
    assert ok.mean() > 0.999 and 0.6 < shares["min"] < 0.7 and 0.25 < shares["saddle"] < 0.36, (ok.mean(), shares)
    med = np.median(its[ok])
    palette = []
    mins = ["#2a78d6", "#1baf7a", "#4a3aa7", "#5fb8e8"]; sads = ["#eb6834", "#eda100", "#f3a35f", "#c9a227"]
    im = isd = 0
    for kd in kinds:
        if kd == "min": palette.append(mins[im]); im += 1
        elif kd == "saddle": palette.append(sads[isd]); isd += 1
        else: palette.append(RED)
    cmap = ListedColormap(palette)
    fig, ax = plt.subplots(figsize=(9, 9.2))
    ax.imshow(np.where(idx >= 0, idx, np.nan), origin="lower", extent=(xs[0], xs[-1], xs[0], xs[-1]), cmap=cmap,
              vmin=-0.5, vmax=len(uniq) - 0.5, interpolation="nearest", alpha=0.85)
    X, Y = np.meshgrid(xs, xs)
    F = (X ** 2 + Y - 11) ** 2 + (X + Y ** 2 - 7) ** 2
    ax.contour(X, Y, F, levels=np.geomspace(1, 400, 9), colors="white", linewidths=0.5, alpha=0.6)
    marker = {"min": ("*", 18, "white"), "saddle": ("s", 10, "white"), "max": ("X", 12, "white")}
    for p, kd in zip(uniq, kinds):
        mk, ms, mec = marker[kd]
        ax.plot(*p, mk, ms=ms, color=INK, mec=mec, mew=1.2, zorder=7)
    ax.plot([], [], "*", ms=13, color=INK, mec="white", label=f"минимумы — {shares['min']:.0%} стартов")
    ax.plot([], [], "s", ms=9, color=INK, mec="white", label=f"сёдла — {shares['saddle']:.0%}")
    ax.plot([], [], "X", ms=10, color=INK, mec="white", label=f"максимум — {shares['max']:.0%}")
    ax.legend(loc="lower left", facecolor="white", frameon=True, framealpha=0.9)
    ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal",
           title=f"Химмельблау: куда приходит чистый Ньютон (медиана — {med:.0f} итераций)")
    ax.grid(False)
    save("05_basins.png")
    print(f"      бассейны: {time.perf_counter() - t0:.1f} с, сошлось {ok.mean():.4%}, доли {shares}")
    return shares, med


# ================================================================ 06 чистый против демпфированного
def fig_damped(R):
    kn, pn = R["newton"]; kd, pd = R["damped"]; err = errors_rosen(R)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    rosen_contour(ax, xlim=(-2.2, 2.2), ylim=(-3.5, 3.2))
    ax.plot(pn[:, 0], pn[:, 1], "-o", color=ORANGE, ms=5, lw=1.6, label=f"чистый Ньютон: {kn}")
    ax.plot(pd[:, 0], pd[:, 1], "-s", color=VIOLET, ms=4, lw=1.6, label=f"с правилом Армихо: {kd}")
    ax.plot(*X0_ROSEN, "o", color=INK, ms=6, zorder=7)
    ax.annotate("выброс", pn[2], (pn[2, 0] - 1.6, pn[2, 1] + 0.1), color=ORANGE)
    ax.legend(loc="upper left")
    ax.set_title("Пути: с выбросом и без")
    ax = axes[1]
    ax.plot(np.arange(len(pn)), np.log10(err["newton"]), "-o", ms=5, color=ORANGE, label=f"чистый ({kn})")
    ax.plot(np.arange(len(pd)), np.log10(err["damped"]), "-s", ms=4, color=VIOLET, label=f"Армихо ({kd})")
    ax.set(xlabel="итерация $k$", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$", xlim=(0, 23), ylim=(-12.5, 1.4),
           title="Ошибка: убывает на каждом шаге")
    ax.legend(loc="lower left")
    save("06_damped.png")


# ================================================================ 07 модель-чаша и модель-седло
def fig_bowl_saddle():
    g = np.array([1.0, 2.0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    lim = 3.2
    P1, P2 = np.meshgrid(np.linspace(-lim, lim, 400), np.linspace(-lim, lim, 400))
    for ax, B, title in ((axes[0], np.diag([1.0, 2.0]), "гессиан $\\succ0$: модель — чаша"),
                         (axes[1], np.diag([1.0, -1.0]), "есть $\\lambda<0$: модель — седло")):
        M = g[0] * P1 + g[1] * P2 + 0.5 * (B[0, 0] * P1 ** 2 + B[1, 1] * P2 ** 2)
        p = -np.linalg.solve(B, g)
        levels = np.sort(M[200, 200] + np.array([-4, -2.5, -1.2, -0.4, 0.4, 1.2, 2.5, 4, 6, 9]) + (g @ p) / 2)
        ax.contour(P1, P2, M, levels=np.unique(levels), colors=ORANGE, linewidths=0.9, alpha=0.9)
        ax.plot(0, 0, "o", color=INK, ms=8, zorder=8)
        ax.annotate("$x_k$", (0, 0), (0.12, 0.18), color=INK)
        d = -g / np.linalg.norm(g) * 1.3
        arrow(ax, (0, 0), d, BLUE, lw=2)
        ax.annotate("$-g$", d, (d[0] - 0.55, d[1] - 0.1), color=BLUE)
        arrow(ax, (0, 0), p, ORANGE if g @ p < 0 else RED, lw=2.4)
        if g @ p < 0:
            assert np.allclose(p, [-1, -1]), p
            ax.plot(*p, "s", color=ORANGE, ms=8, zorder=8)
            ax.annotate("минимум модели:\n$g^\\top p<0$ — вниз", p, (p[0] - 2.0, p[1] - 1.5), color=ORANGE)
        else:
            assert np.allclose(p, [-1, 2]) and g @ p == 3, (p, g @ p)
            ax.plot(*p, "D", color=RED, ms=8, zorder=8)
            ax.annotate("седло модели:\n$g^\\top p=3>0$ — вверх", p, (p[0] - 2.1, p[1] + 0.5), color=RED)
            tau = abs(np.linalg.eigvalsh(B)[0]) + 1e-3
            preg = -np.linalg.solve(B + tau * np.eye(2), g)
            assert g @ preg < 0, preg
            dreg = preg / np.linalg.norm(preg) * 1.6
            arrow(ax, (0, 0), dreg, AQUA, lw=2.4)
            ax.annotate("$B_k=\\nabla^2f+\\tau I$:\n$g^\\top p<0$ — вниз", dreg, (dreg[0] + 0.15, dreg[1] - 0.7), color=AQUA)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="$p_1$", ylabel="$p_2$", aspect="equal", title=title)
        ax.grid(False)
    save("07_bowl_saddle.png")


# ================================================================ 08 секущая против касательной
def fig_secant():
    fig, ax = plt.subplots(figsize=(9, 4.4))
    F = lambda x: np.tanh(x); dF = lambda x: 1 / np.cosh(x) ** 2       # F = f' для f = ln cosh
    x0, x1 = 1.3, 0.8
    t = np.linspace(-1.1, 1.8, 400)
    ax.plot(t, F(t), color=INK, lw=2.2, label="$F(x)=f'(x)$ — ищем нуль")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.plot(t, F(x1) + dF(x1) * (t - x1), color=ORANGE, lw=1.8, label="касательная в $x_1$: Ньютон, нужна $f''$")
    slope = (F(x1) - F(x0)) / (x1 - x0)
    ax.plot(t, F(x1) + slope * (t - x1), color=AQUA, lw=1.8, ls="--", label="секущая через $x_0,x_1$: $f''\\approx y/s$")
    ax.plot([x0, x1], [F(x0), F(x1)], "o", color=INK, ms=7, zorder=7)
    xn_newton, xn_secant = x1 - F(x1) / dF(x1), x1 - F(x1) / slope
    ax.plot(xn_newton, 0, "s", color=ORANGE, ms=8, zorder=7); ax.plot(xn_secant, 0, "s", color=AQUA, ms=8, zorder=7)
    ax.annotate("$x_0$", (x0, F(x0)), (x0 + 0.05, F(x0) - 0.2))
    ax.annotate("$x_1$", (x1, F(x1)), (x1 + 0.05, F(x1) - 0.2))
    ax.text(0.03, 0.84, "$s=x_1-x_0$,  $y=F(x_1)-F(x_0)$,  $f''\\approx y/s$", transform=ax.transAxes, color=AQUA)
    ax.set(xlim=(-1.1, 1.8), ylim=(-0.9, 1.1), xlabel="$x$", title="Секущее уравнение: кривизна из двух градиентов")
    ax.legend(loc="lower right")
    save("08_secant.png")


# ================================================================ 09 BFGS учится эллипсу на квадратичной
def fig_bfgs_ellipses():
    Q = np.diag([1.0, 50.0])
    xq, kq, pq, _ = bfgs(lambda x: 0.5 * x @ Q @ x, lambda x: Q @ x, [50.0, 1.0], tol=1e-10, Q=Q)
    Hs = [np.eye(2)]                                      # восстанавливаем H_k по шагам
    for k in range(len(pq) - 1):
        s = pq[k + 1] - pq[k]; y = Q @ s; rho = 1 / (s @ y); I = np.eye(2)
        Hs.append((I - rho * np.outer(s, y)) @ Hs[-1] @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s))
    assert kq == 2 and np.allclose(Hs[2], np.linalg.inv(Q), atol=1e-10), (kq, Hs[2])
    th = np.linspace(0, 2 * np.pi, 300); circ = np.c_[np.cos(th), np.sin(th)]

    def ellipse(B):       # {p : p^T B p = 1}
        L = np.linalg.cholesky(np.linalg.inv(B)); return circ @ L.T

    fig, ax = plt.subplots(figsize=(9, 5.2))
    e_true = ellipse(Q)
    for k, (Hk, col, lab) in enumerate(zip(Hs, (GRAY, VIOLET, ORANGE), ("$B_0=I$: круг", "$B_1$: кривизна вдоль первого шага", "$B_2=Q$: совпадение"))):
        e = ellipse(np.linalg.inv(Hk))
        ax.plot(e[:, 0], e[:, 1], color=col, lw=2.2 if k < 2 else 2.8, ls="-" if k < 2 else "--", label=lab)
    ax.plot(e_true[:, 0], e_true[:, 1], color=INK, lw=1.2, label="истинный гессиан $Q$")
    ax.set(xlabel="$p_1$", ylabel="$p_2$", aspect="equal", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3),
           title=f"$\\frac{{1}}{{2}}(x_1^2+50x_2^2)$: BFGS за {kq} шага учит $Q^{{-1}}$")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save("09_bfgs_ellipses.png")
    return Hs


# ================================================================ 10 спуск, BFGS, Ньютон
def fig_rates_three(R):
    kg, _ = R["gd"]; kn, pn = R["newton"]; kb, pb, nfb = R["bfgs"]; err = errors_rosen(R)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(np.arange(41), np.log10(err["gd"][:41]), color=BLUE, label=f"градиентный спуск ({kg}): прямая")
    ax.plot(np.arange(len(pb)), np.log10(err["bfgs"]), "-^", ms=5, color=AQUA, label=f"BFGS ({kb}): загиб")
    ax.plot(np.arange(len(pn)), np.log10(err["newton"]), "-o", ms=5.5, color=ORANGE, label=f"чистый Ньютон ({kn}): обрыв")
    ax.set(xlabel="итерация $k$", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$", xlim=(0, 40), ylim=(-12.5, 1.4),
           title="Розенброк: линейная, сверхлинейная, квадратичная")
    ax.legend(loc="lower left")
    save("10_rates_three.png")


# ================================================================ 11 Гаусс–Ньютон на синусе
def fig_gn_sine(S):
    t, y = S["t"], S["y"]; kgn, pgn, xgn = S["gn"]; kgd, pgd = S["gd"]; knw, pnw, xnw = S["newton"]; f = S["f"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(t, y, "o", ms=3.5, color=INK, alpha=0.7, label="данные")
    ax.plot(tt, xgn[0] * np.sin(xgn[1] * tt + xgn[2]), color=AQUA, lw=2.4, label=f"Гаусс–Ньютон, {kgn}: $f^*={f(xgn):.4f}$")
    ax.plot(tt, X0_SINE[0] * np.sin(X0_SINE[1] * tt + X0_SINE[2]), color=GRAY, lw=1.2, ls=":", label="старт $(1,\\,3,\\,0)$")
    ax.plot(tt, xnw[0] * np.sin(xnw[1] * tt + xnw[2]), color=RED, lw=2, ls="--", label=f"Ньютон, {knw}: $x_1=0$, $f={f(xnw):.0f}$")
    ax.set(xlabel="$t$", ylabel="$y$", ylim=(-3.2, 4.6), title="Один старт — разные стационарные точки")
    ax.legend(loc="upper right")
    ax = axes[1]
    egn = np.linalg.norm(pgn - xgn, axis=1) + 1e-17; egd = np.linalg.norm(pgd - xgn, axis=1) + 1e-17
    ax.plot(np.arange(1, len(egd) + 1), np.log10(egd), color=BLUE, label=f"спуск: {kgd}")
    ax.plot(np.arange(1, len(egn) + 1), np.log10(egn), "-o", ms=5, color=AQUA, label=f"Гаусс–Ньютон: {kgn}")
    ax.set_xscale("log")
    ax.set(xlabel="итерация $k$ (лог. шкала)", ylabel="$\\log_{10}\\Vert x_k-x^*\\Vert$", ylim=(-7, 0.6),
           title="Невязки малы — почти Ньютон")
    ax.legend(loc="lower left")
    save("11_gn_sine.png")


# ================================================================ 12 цепь: итерации от N
def fig_chain_methods():
    Ns = (10, 20, 40, 80, 160)
    rows = []
    t_total = time.perf_counter()
    for N in Ns:
        f, grad, H, b, v0, _ = make_chain(N)
        L = np.linalg.eigvalsh(H)[-1]
        t0 = time.perf_counter(); _, kg, _ = gd(f, grad, v0, alpha=1 / L, maxit=400_000); tg = time.perf_counter() - t0
        _, kb, _, _ = bfgs(f, grad, v0, Q=H)
        r = minimize(f, v0, jac=grad, method="L-BFGS-B", options={"gtol": 1e-6, "maxiter": 100_000, "maxcor": 5})
        rs = minimize(f, v0, jac=grad, method="BFGS", options={"gtol": 1e-6, "maxiter": 100_000})
        t0 = time.perf_counter(); np.linalg.solve(H, -b); tn = time.perf_counter() - t0
        rows.append((N, kg, kb, r.nit, tg, tn, rs.nit))
        print(f"      цепь N={N:3d}: спуск {kg} ({tg:.1f} с), BFGS точный шаг {kb}, L-BFGS m=5 {r.nit}, scipy BFGS (Вольфе) {rs.nit}, solve {tn * 1e3:.1f} мс")
    rows = np.array(rows, dtype=float)
    kg40 = rows[2, 1]; kb = rows[:, 2]
    assert abs(kg40 - 10575) < 110, kg40
    assert np.array_equal(kb, [5, 10, 20, 40, 80]), kb
    assert 60 <= rows[2, 3] <= 90, rows[2, 3]                 # L-BFGS: «75»
    assert 55 <= rows[2, 6] <= 80, rows[2, 6]                 # scipy BFGS с условием Вольфе: «66»
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(Ns, rows[:, 1], "o-", color=BLUE, label="градиентный спуск, шаг $1/L$  ($\\propto\\kappa\\sim N^2$)")
    ax.plot(Ns, rows[:, 3], "^-", color=AQUA, label="L-BFGS, память 5 (SciPy)")
    ax.plot(Ns, rows[:, 2], "s-", color=VIOLET, label="BFGS, точный шаг  ($=N/2$)")
    ax.plot(Ns, np.ones(len(Ns)), "*-", color=ORANGE, ms=11, label="Ньютон: один `solve`")
    for N, kg, kb_, kl, *_ in rows:
        ax.annotate(f"{int(kg):,}".replace(",", " "), (N, kg), (0, 7), textcoords="offset points", ha="center", color=BLUE, fontsize=10)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set(xlabel="число грузов $N$ (переменных — $2N$)", ylabel="итераций до $\\Vert\\nabla E\\Vert\\leq10^{-6}$",
           xticks=Ns, xticklabels=[str(n) for n in Ns], title="Цепь: чем больше кривизны знает $B_k$, тем меньше итераций")
    ax.legend(loc="upper left")
    save("12_chain_methods.png")
    print(f"      цепь всего: {time.perf_counter() - t_total:.1f} с")
    return rows


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("рисую в", IMG)
    R = runs_rosen()
    S = runs_sine()
    fig_hooks(R, S)
    fig_model_1d()
    fig_model_ellipses(R)
    thr = fig_newton_1d()
    fig_rates_newton(R)
    shares, med = fig_basins()
    fig_damped(R)
    fig_bowl_saddle()
    fig_secant()
    Hs = fig_bfgs_ellipses()
    fig_rates_three(R)
    fig_gn_sine(S)
    rows = fig_chain_methods()
    kg, _ = R["gd"]; kn, _ = R["newton"]; kd, _ = R["damped"]; kb, _, nfb = R["bfgs"]; sb, sf, sl = R["scipy"]
    print(f"проверки: Розенброк — спуск {kg}, Ньютон {kn}, демпфированный {kd}, BFGS {kb} ({nfb} вызовов f), scipy BFGS {sb} ({sf}), L-BFGS-B {sl};")
    print(f"          синус — Гаусс–Ньютон {S['gn'][0]}, спуск {S['gd'][0]}, Ньютон {S['newton'][0]} (x1 = {S['newton'][2][0]:.1e});")
    print(f"          порог ln cosh {thr:.4f}; Химмельблау: минимумы {shares['min']:.1%}, сёдла {shares['saddle']:.1%}, максимум {shares['max']:.1%}")
    print(f"готово за {time.perf_counter() - t_start:.1f} с")
