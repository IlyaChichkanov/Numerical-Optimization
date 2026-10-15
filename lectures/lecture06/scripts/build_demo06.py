"""Собирает пошаговый ноутбук demo06.ipynb.

    uv run python lectures/lecture06/scripts/build_demo06.py
    uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture06/demo06.ipynb

Ноутбук — «урок» по лекции 6: линейный поиск (Армихо, Вольфе, глобальная сходимость), доверительная
область (rho, точка Коши, dogleg, точный шаг) и сопряжённые градиенты (CG, Newton-CG). Методы берутся
из make_figures.py дословно (блок между метками «методы» и «Химмельблау на сетке»), поэтому числа
ноутбука и картинок совпадают. Вторая ячейка (импорты, палитра, методы, задачи) становится модулем
l6helpers.py при сборке конспекта-ноутбука (build_lecture06.py).
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo06.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def methods_block() -> str:
    """Методы и задачи из make_figures.py — один код на лекцию."""
    src = (HERE / "make_figures.py").read_text(encoding="utf-8")
    start = src.index("# ---------------------------------------------------------------- методы")
    stop = src.index("# ---------------------------------------------------------------- Химмельблау на сетке")
    return src[start:stop].rstrip() + "\n"


# ============================================================ шапка
md(r"""# Демонстрации к лекции 6: линейный поиск, доверительная область, сопряжённые градиенты

Ноутбук — для самостоятельной проверки после пары: всё, что на лекции было формулой или картинкой, здесь считается кодом, а числа конспекта печатаются рядом с ожидаемыми. Читается сверху вниз; каждая часть начинается с «что проверяем и что должны увидеть».

- **(0)** три задачи введения: Ньютон на арктангенсе — обещание параболы против правды, порог старта $1.39$; Химмельблау; цепь из 400 грузов;
- **(a)** линейный поиск (раздел 2): $\varphi(\alpha)$, Армихо и Вольфе на Розенброке, backtracking $1\to\tfrac12\to\tfrac14\to\tfrac18$;
- **(b)** доверительная область (раздел 3): модельная задача, точка Коши и dogleg, $25$ итераций на Розенброке с таблицей $\Delta_k,\rho_k$, только точка Коши — $14\,110$;
- **(c)** сопряжённые градиенты (раздел 4): эллипс за $2$ шага, цепь за $N/2$, Newton-CG в $1000$ переменных;
- **(d)** биография цепи и таблица методов SciPy (разделы 4–5).
- **(e)** подробности линейного поиска (§2.2, §3.5): сильный Вольфе своими руками, $s^\top y$ у BFGS, цепь и масштаб $H_0$, ловушка седла.

Конспект: [`lecture06.md`](lecture06.md). Запуск: `uv run jupyter lab` в корне репозитория. Пакеты — `numpy`, `scipy`, `matplotlib`; CasADi — только в семинаре.""")

code(r'''%matplotlib inline

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen as rosen_nd, rosen_der, rosen_hess_prod

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


''' + methods_block())

# ============================================================ (0)
md(r"""## (0) Три задачи введения

**Что проверяем.** Числа введения. Чистый Ньютон для $f(x)=x\arctan x-\tfrac12\ln(1+x^2)$ из $x_0=2$ улетает: $2\to-3.54\to13.95\to-279\to1.2\cdot10^5\to-2.3\cdot10^{10}$. С Армихо — $5$ итераций, $\alpha=\tfrac12,1,1,1,1$. Почему так — картинка `img/00_atan_newton.png` конспекта: парабола-модель в $x_0=2$ обещает $-1.65$, а получаем $3.28$; при больших $\lvert x_k\rvert$ ошибка возводится в квадрат, $\lvert x_{k+1}\rvert\approx\tfrac{\pi}{2}x_k^2$; порог старта $1.39$.""")

code(r'''x = 2.0; seq = [x]
for _ in range(5):
    x = x - np.arctan(x) * (1 + x * x)          # f' = arctan x, f'' = 1/(1+x^2)
    seq.append(x)
print("чистый Ньютон:", [f"{v:.4g}" for v in seq])

x = np.array([2.0]); alphas = []
while abs(np.arctan(x[0])) > 1e-10:
    g = np.array([np.arctan(x[0])]); p = -g * (1 + x[0] ** 2)
    a, _, _ = armijo(lambda z: f_arctan(z[0]), x, f_arctan(x[0]), g, p)
    x = x + a * p; alphas.append(a)
print(f"с Армихо: {len(alphas)} итераций, шаги {alphas}   (в конспекте: 5 итераций, 1/2, 1, 1, 1, 1)")''')

md(r"""**Обещание модели против правды.** Парабола-модель в $x_0=2$: $m(x)=f(2)+\arctan 2\,(x-2)+\tfrac{1}{10}(x-2)^2$. Её вершина — $x_1=-3.54$. Что она там обещает, что там на самом деле и что даёт половина шага? Затем — показатель степени ошибки $\log_{10}\lvert x_k\rvert$ (удваивается, но растёт) и отношение $\lvert x_{k+1}\rvert/x_k^2$, которое стремится к $\pi/2$.""")

code(r'''x0, x1 = seq[0], seq[1]
m = lambda x: f_arctan(x0) + np.arctan(x0) * (x - x0) + 0.5 / (1 + x0**2) * (x - x0) ** 2
x_half = x0 - 0.5 * np.arctan(x0) * (1 + x0**2)
print(f"f(x0) = {f_arctan(x0):.3f}, модель обещала m(x1) = {m(x1):.3f}, на самом деле f(x1) = {f_arctan(x1):.3f}, "
      f"половина шага: x = {x_half:.3f}, f = {f_arctan(x_half):.3f}   (в конспекте: 1.41, -1.65, 3.28, 0.27)")
print("log10|x_k|:        ", [f"{np.log10(abs(v)):.2f}" for v in seq], "  (в конспекте: 0.3, 0.55, 1.1, 2.4, 5.1, 10.4)")
print("|x_{k+1}| / x_k^2: ", [f"{abs(seq[k + 1]) / seq[k]**2:.4f}" for k in range(5)], f"  -> pi/2 = {np.pi / 2:.4f}")''')

md(r"""**Порог старта.** Чистый Ньютон сходится из $\lvert x_0\rvert<x^\ast$ и расходится из $\lvert x_0\rvert>x^\ast$, где из $x^\ast$ шаг переносит ровно в $-x^\ast$: $(1+x^2)\arctan x=2x$. В конспекте $x^\ast\approx1.39$.""")

code(r'''from scipy.optimize import brentq

thr = brentq(lambda x: (1 + x * x) * np.arctan(x) - 2 * x, 1.0, 2.0)
print(f"порог x* = {thr:.4f}   (в конспекте: 1.39)")
with np.errstate(over="ignore", invalid="ignore"):
    for x0_ in [1.39, 1.40]:
        x = x0_
        for _ in range(30):
            x = x - np.arctan(x) * (1 + x * x)
        print(f"  чистый Ньютон из {x0_}: через 30 шагов x = {x:.3g}")''')

md(r"""**Предскажите до запуска.** Сколько делений пополам понадобится на первом шаге, если стартовать из $x_0=100$? Чем дальше старт, тем длиннее шаг Ньютона: $\lvert p\rvert=\arctan(x_0)(1+x_0^2)$.""")

code(r'''x0 = 100.0
x = np.array([x0]); first = None; k = 0
while abs(np.arctan(x[0])) > 1e-10:
    g = np.array([np.arctan(x[0])]); p = -g * (1 + x[0] ** 2)
    a, _, m = armijo(lambda z: f_arctan(z[0]), x, f_arctan(x[0]), g, p)
    first = first or (a, m)
    x = x + a * p; k += 1
print(f"из x0 = {x0:g}: {k} итераций; первый шаг α = {first[0]:g} после {first[1] - 1} делений пополам; |p| = {np.arctan(x0) * (1 + x0**2):.0f}")''')

md(r"""Ответ: $\alpha=1/128$ — семь делений, потому что полный шаг Ньютона длиной $\approx15\,600$ в сотни раз длиннее разумного. Дальше — полные шаги и квадратичная сходимость.

**Химмельблау и цепь.** Куда приходит чистый Ньютон из нескольких стартов (бассейны всей сетки — картинка 07 конспекта) и сколько итераций у спуска и CG на цепи из $400$ грузов (число спуска считается точно через собственные числа, как в Л5).""")

code(r'''for x0 in [(0.0, 0.0), (0.58, 2.15), (-0.5, -1.0), (4.0, 4.0)]:
    x, k, _ = newton(him_grad, him_hess, x0)
    lam = np.linalg.eigvalsh(him_hess(x))
    kind = "минимум" if lam[0] > 0 else ("максимум" if lam[1] < 0 else "седло")
    print(f"старт {x0}: чистый Ньютон за {k} итераций в ({x[0]:.3f}, {x[1]:.3f}) — {kind}")

k_gd, kappa, _ = gd_chain_count(400)
_, _, H400, b400, v400, _ = make_chain(400)
_, k_cg, _ = cg(lambda v: H400 @ v, -b400, v400)
print(f"цепь N = 400: κ = {kappa:.0f}, спуск 1/L — {k_gd} итераций, CG — {k_cg}   (в конспекте: 937 503 и 200)")''')

# ============================================================ (a)
md(r"""## (a) Линейный поиск: Армихо, Вольфе, backtracking

**Что проверяем.** Числа §2.1–2.2: на Розенброке в $x_1$ (первая итерация Ньютона из $(-1.2,1)$) шаг Ньютона длиной $4.95$ ведёт в $f=1412$; Армихо выполнено при $\alpha\le0.176$, Вольфе ($c_2=0.9$) — при $0.044\le\alpha\le0.176$; backtracking пробует $1,\tfrac12,\tfrac14$ и принимает $\tfrac18$.""")

code(r'''x1 = X0_ROSEN - np.linalg.solve(rosen_hess(X0_ROSEN), rosen_grad(X0_ROSEN))   # x1: первая итерация Ньютона из (-1.2, 1)
p1 = -np.linalg.solve(rosen_hess(x1), rosen_grad(x1))                           # p1: шаг Ньютона в x1 (длина 4.95)
phi = lambda a: rosen(x1 + a * p1)              # разрез f вдоль шага: phi(alpha) = f(x1 + alpha * p1)
dphi = lambda a: rosen_grad(x1 + a * p1) @ p1   # производная разреза по правилу сложной функции: grad f(x1 + alpha p1) . p1
print(f"x1 = {x1.round(4)}, |p1| = {np.linalg.norm(p1):.3f}, φ(0) = {phi(0):.3f}, φ(1) = {phi(1):.1f}, φ'(0) = {dphi(0):.3f}   (в конспекте: 4.95, 4.73, 1412, -8.43)")

al = np.linspace(0, 0.3, 30001)                                        # сетка длин шага alpha
arm = np.array([phi(a) <= phi(0) + 1e-4 * a * dphi(0) for a in al])   # правило Армихо: phi под прямой с наклоном c1 phi'(0), c1 = 1e-4
wol = arm & np.array([dphi(a) >= 0.9 * dphi(0) for a in al])          # плюс условие кривизны Вольфе: наклон стал положе, c2 = 0.9
print(f"Армихо: α ≤ {al[arm].max():.3f};  Вольфе: {al[wol].min():.3f} ≤ α ≤ {al[wol].max():.3f}   (в конспекте: 0.176; 0.044–0.176)")

for a in [1, 0.5, 0.25, 0.125]:                                        # backtracking: alpha = 1, затем пополам, пока Армихо не выполнено
    ok = phi(a) <= phi(0) + 1e-4 * a * dphi(0)
    print(f"  α = {a:<6g} φ = {phi(a):9.3f}  Армихо: {'да — принят' if ok else 'нет'}")''')

code(r'''fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(al, [phi(a) for a in al], color=INK, label=r"$\varphi(\alpha)$")                 # разрез phi(alpha)
ax.plot(al, phi(0) + 1e-4 * al * dphi(0), "--", color=ORANGE, label="прямая Армихо")       # прямая Армихо: phi(0) + c1 alpha phi'(0)
ax.axvspan(al[wol].min(), al[wol].max(), color=AQUA, alpha=0.2, label="Вольфе")            # допустимые по Вольфе шаги
for a, col in [(0.02, YELLOW), (0.125, AQUA), (0.25, RED)]:                                 # три кандидата из «вопроса залу»
    ax.plot(a, phi(a), "o", color=col, ms=9, mec=INK)                                        #   0.02 — только Армихо, 0.125 — оба, 0.25 — ни одного
ax.set(ylim=(3.6, 9.6), xlabel=r"$\alpha$", title="Розенброк в $x_1$ вдоль шага Ньютона")
ax.legend(); plt.show()''')

# ============================================================ (b)
md(r"""## (b) Доверительная область: $\rho$, точка Коши, dogleg

**Что проверяем.** Модельная задача $g=(3,4)$, $B=\operatorname{diag}(1,2)$: точка Ньютона $(-3,-2)$, модель в ней $-8.5$; при $\Delta=1$ точка Коши $(-0.6,-0.8)$, модель $-4.18$. Затем метод целиком на Розенброке: $25$ итераций, из них $3$ пробных шага отвергнуты.""")

code(r'''g_t, B_t = np.array([3.0, 4.0]), np.diag([1.0, 2.0])
m = lambda p: g_t @ p + 0.5 * p @ B_t @ p
pN = -np.linalg.solve(B_t, g_t); pU = -(g_t @ g_t) / (g_t @ B_t @ g_t) * g_t
print(f"точка Ньютона {pN}, |pN| = {np.linalg.norm(pN):.3f}, m = {m(pN):.3f};  минимум вдоль -g: {pU.round(3)}, |pU| = {np.linalg.norm(pU):.3f}")
for D in [1.0, 2.0, 3.0, 4.0]:
    pc, pd, pe = cauchy_point(g_t, B_t, D), dogleg(g_t, B_t, D), tr_exact(g_t, B_t, D)
    print(f"Δ = {D}: Коши {pc.round(3)} m = {m(pc):7.3f} | dogleg {pd.round(3)} m = {m(pd):7.3f} | точный {pe.round(3)} m = {m(pe):7.3f}")''')

md(r"""Точный шаг всегда не хуже dogleg, dogleg — не хуже точки Коши; при $\Delta\ge\lVert p_N\rVert=3.61$ все три — точка Ньютона. **Предскажите до запуска:** сколько итераций сделает доверительная область на Розенброке, если брать *только* точку Коши? (Подсказка: это спуск с точным шагом, обрезанным шаром.)""")

code(r'''for step in ["exact", "dogleg", "cauchy"]:
    t0 = time.perf_counter()
    x, k, path, hist = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step=step)
    rej = sum(not h[4] for h in hist)
    print(f"{step:7s}: {k:6d} итераций, отвергнуто {rej}, x = {x.round(6)}, {time.perf_counter() - t0:.1f} с")
print("в конспекте: 25 (3 отвергнуты), 24, 14 110; спуск с backtracking из Л4 — 13 756")''')

md(r"""**Таблица $\Delta_k$ и $\rho_k$** для точного шага: на итерации $1$ — тот самый выброс Л5 — шаг обрезан до $\Delta=1$, и всё равно $\rho<0$: шаг отвергнут, радиус вчетверо меньше.""")

code(r'''x, k, path, hist = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step="exact")
print(" k    Δ_k      ρ_k     шаг     |p|")
for i, (xk, p, D, rho, acc) in enumerate(hist):
    print(f"{i:2d}  {D:7.4f}  {rho:7.3f}   {'принят ' if acc else 'ОТВЕРГ.'}  {np.linalg.norm(p):.3f}")''')

md(r"""**Отрицательная кривизна.** Из ловушки $(0.58,2.15)$ доверительная область приходит в минимум $(3,2)$ — за $7$ итераций, без всякого сдвига: в шаре минимум модели есть всегда.""")

code(r'''x, k, path, hist = trust_region(him, him_grad, him_hess, (0.58, 2.15), step="exact", tol=1e-10)
print(f"доверительная область: {k} итераций → {x.round(6)}; λ_min гессиана в старте = {np.linalg.eigvalsh(him_hess(np.array([0.58, 2.15])))[0]:.1f}")''')

# ============================================================ (c)
md(r"""## (c) Сопряжённые градиенты

**Что проверяем.** На эллипсе $\tfrac12(x_1^2+50x_2^2)$ из $(50,1)$ CG сходится за $2$ шага (спуск с точным шагом — $567$, Л4); направления сопряжены: $d_1^\top Ad_0=0$. На цепи — ровно $N/2$ итераций.""")

code(r'''A = np.diag([1.0, 50.0]); x = np.array([50.0, 1.0]); r = A @ x; d = -r; ds = []
for k in range(2):
    a = (r @ r) / (d @ A @ d); x = x + a * d; r_new = r + a * A @ d
    ds.append(d.copy()); d = -r_new + (r_new @ r_new) / (r @ r) * d; r = r_new
    print(f"итерация {k + 1}: x = {x.round(10)}")
print(f"сопряжённость d1^T A d0 = {ds[1] @ A @ ds[0]:.2e}")''')

md(r"""**Предскажите до запуска:** сколько итераций CG на цепи из $400$ грузов ($800$ переменных)? А если стартовать не с прямой между концами, а со случайной точки?""")

code(r'''for N in [10, 20, 40, 80, 160, 400]:
    _, gE_N, H_N, b_N, v0_N, _ = make_chain(N)
    _, k_sym, _ = cg(lambda v: H_N @ v, -b_N, v0_N)
    v_rand = v0_N + 0.1 * np.random.default_rng(2026).standard_normal(2 * N)
    _, k_rand, _ = cg(lambda v: H_N @ v, -b_N, v_rand)
    print(f"N = {N:3d}: CG из прямой — {k_sym:3d} (N/2 = {N // 2}), из случайного старта — {k_rand}")''')

md(r"""Из прямой — ровно $N/2$: горизонтальная часть ошибки нулевая, вертикальная симметрична и задевает только $N/2$ собственных векторов (упражнение 6.9). Из случайного старта задеты все собственные векторы — итераций больше.

**Newton-CG в $1000$ переменных.** Розенброк из SciPy (`rosen`, `rosen_der`, `rosen_hess_prod`), старт $0.5\cdot\mathbf1$. Наш `newton_cg` берёт гессиан только как произведение на вектор. Ожидаем: $47$ итераций и $225$ произведений; SciPy `Newton-CG` — $54$ и $291$; `L-BFGS-B` — $88$ (с `ftol=0`).""")

code(r'''x0 = np.full(1000, 0.5)
x, k, nmv, hist = newton_cg(rosen_nd, rosen_der, rosen_hess_prod, x0)
print(f"наш Newton-CG: {k} итераций, {nmv} произведений H v, f = {rosen_nd(x):.1e}")
r = minimize(rosen_nd, x0, jac=rosen_der, hessp=rosen_hess_prod, method="Newton-CG", options={"xtol": 1e-8})
print(f"SciPy Newton-CG: {r.nit} итераций, {r.nhev} произведений H v")
for ftol in [None, 0.0]:
    opts = {"gtol": 1e-6, "maxiter": 10_000} | ({} if ftol is None else {"ftol": ftol})
    r = minimize(rosen_nd, x0, jac=rosen_der, method="L-BFGS-B", options=opts)
    print(f"L-BFGS-B, ftol = {'по умолчанию' if ftol is None else 0}: {r.nit} итераций, |∇f| = {np.linalg.norm(rosen_der(r.x)):.1e}")''')

# ============================================================ (d)
md(r"""## (d) Биография цепи и методы SciPy

**Что проверяем.** Цепь $N=40$ одним списком: спуск $10\,575$, BFGS с точным шагом $20$, CG $20$, Ньютон $1$, доверительная область с CG Стейхауга (SciPy `trust-ncg`) — $13$. Затем таблица методов SciPy на Розенброке и критерии останова.""")

code(r'''E, gE, H, b, v0, unpack = make_chain(40)
k_gd, _, _ = gd_chain_count(40)
_, k_b, _, _, _ = bfgs(E, gE, v0, line="exact", Q=H)
_, k_cg, _ = cg(lambda v: H @ v, -b, v0)
r = minimize(E, v0, jac=gE, hessp=lambda v, p: H @ p, method="trust-ncg", options={"gtol": 1e-6})
print(f"спуск 1/L {k_gd} | BFGS точный шаг {k_b} | CG {k_cg} | Ньютон 1 | trust-ncg {r.nit}")''')

code(r'''rows = []
for m in ["BFGS", "L-BFGS-B", "CG", "Newton-CG", "trust-exact", "dogleg", "trust-ncg", "trust-krylov"]:
    kw = dict(jac=rosen_grad)
    if m in ("Newton-CG", "trust-exact", "dogleg", "trust-ncg", "trust-krylov"):
        kw["hess"] = rosen_hess
    opts = {"xtol": 1e-8} if m == "Newton-CG" else {"gtol": 1e-6}
    if m == "L-BFGS-B":
        opts["ftol"] = 0.0
    r = minimize(rosen, X0_ROSEN, method=m, options=opts, **kw)
    rows.append((m, r.nit, r.nfev, np.linalg.norm(rosen_grad(r.x))))
print(f"{'метод':13s} итераций  вычислений f   |∇f| в конце")
for m, nit, nfev, gn in rows:
    print(f"{m:13s} {nit:7d}  {nfev:12d}   {gn:.1e}")
print("в конспекте: BFGS 33/40, L-BFGS-B 37, CG 37/80, Newton-CG 85, trust-exact 25, dogleg 24, trust-ncg 29, trust-krylov 38")''')

# ============================================================ (e)
md(r"""## (e) Линейный поиск: подробности для самостоятельной проверки

Четыре проверки к §2.2 и §3.5, которые на лекции не показываются: сильный Вольфе своими руками, $s^\top y$ у BFGS на Химмельблау, цепь и масштаб $H_0$ (упражнение 6.10), ловушка седла (упражнение 6.4).""")

md(r"""**Сильный Вольфе своими руками.** Функция `wolfe` из второй ячейки (Nocedal & Wright, алгоритмы 3.5–3.6) на той же прямой: какой шаг она вернёт и сколько вычислений ей понадобится?""")

code(r'''a, f_new, g_new, nf = wolfe(rosen, rosen_grad, x1, p1, rosen(x1), rosen_grad(x1))
print(f"сильный Вольфе: α = {a:.4f}, φ = {f_new:.4f}, |φ'(α)| / |φ'(0)| = {abs(g_new @ p1) / abs(dphi(0)):.3f} (≤ 0.9), вычислений {nf}")''')

md(r"""**Зачем Вольфе квази-Ньютону.** Условие кривизны гарантирует $s^\top y>0$. Проверяем на Химмельблау: BFGS из $400$ случайных стартов (`default_rng(2026)`), считаем шаги с $s^\top y\le0$. Ожидаем: с одним Армихо — $72$ запуска и $179$ таких шагов, с Вольфе — ни одного.""")

code(r'''rng = np.random.default_rng(2026)
starts = rng.uniform(-6, 6, (400, 2))
for line in ["armijo", "wolfe"]:
    runs_bad = steps_bad = 0
    for x0 in starts:
        _, _, _, _, sy = bfgs(him, him_grad, x0, line=line, maxit=5000)
        neg = int(np.sum(sy[:, 0] <= 0)); runs_bad += neg > 0; steps_bad += neg
    print(f"{line:7s}: запусков с s^T y <= 0 — {runs_bad} из 400, таких шагов — {steps_bad}")''')

md(r"""**Цепь: виноват масштаб $H_0$, а не Вольфе.** Энергия цепи — выпуклая квадратичная функция: $s^\top y=s^\top Hs>0$ при любом шаге, условие Вольфе ничего не добавляет. С $H_0=I$ BFGS застревает и с Армихо, и с Вольфе; с $H_0\leftarrow\frac{s^\top y}{y^\top y}I$ после первого шага — $27$ итераций.""")

code(r'''E, gE, H, b, v0, unpack = make_chain(40)
for line, sc in [("armijo", False), ("wolfe", False), ("armijo", True), ("wolfe", True)]:
    x, k, _, nf, _ = bfgs(E, gE, v0, line=line, scale_H0=sc, maxit=3000)
    print(f"{line:7s} H0 {'масштабирована' if sc else '= I          '}: {k:5d} итераций, {nf:6d} вычислений E, |∇E| = {np.linalg.norm(gE(x)):.1e}")
lam_max = np.linalg.eigvalsh(H)[-1]
E_star = E(np.linalg.solve(H, -b))
print(f"пол округления sqrt(2 λmax ε |E*|) = {np.sqrt(2 * lam_max * np.finfo(float).eps * abs(E_star)):.2e}   (в конспекте ≈ 1.3e-6)")''')

md(r"""**Глобальная сходимость — к стационарной точке.** Ньютон с Армихо из ловушки $(0.58,\,2.15)$ на Химмельблау: со сдвигом $\tau I$ только при подъёме (как в Л5 и ДЗ 3) — в седло за $5$ итераций, с «модифицированным Ньютоном» (сдвиг всякий раз, когда $\lambda_{\min}\le10^{-3}$) — в минимум.""")

code(r'''for shift in ["ascent", "always"]:
    x, k, _ = newton_ls(him, him_grad, him_hess, (0.58, 2.15), shift=shift)
    lam = np.linalg.eigvalsh(him_hess(x))
    print(f"сдвиг «{shift}»: {k} итераций → ({x[0]:.4f}, {x[1]:.4f}), |∇f| = {np.linalg.norm(him_grad(x)):.1e}, собственные числа {lam.round(2)}")''')

md(r"""## Шпаргалка

| Что | Формула | Где в конспекте |
|---|---|---|
| правило Армихо | $\varphi(\alpha)\le\varphi(0)+c_1\alpha\varphi'(0)$, $c_1=10^{-4}$ | §2.2 |
| условие Вольфе (кривизна) | $\varphi'(\alpha)\ge c_2\varphi'(0)$, $c_2=0.9$ | §2.4 |
| сильный Вольфе | $\lvert\varphi'(\alpha)\rvert\le c_2\lvert\varphi'(0)\rvert$ | §2.4 |
| Вольфе ⇒ кривизна | $s^\top y\ge(1-c_2)\,\alpha\,\lvert g^\top p\rvert>0$ | §2.4 |
| глобальная сходимость | $\sum_k\lVert g_k\rVert^2\le M_B(f_0-\inf f)/(c_1\alpha_{\min})$ | §2.5 |
| отношение выигрыша | $\rho=\dfrac{f(x)-f(x+p)}{m(0)-m(p)}$; $<\tfrac14$: $\Delta/4$, $>\tfrac34$ на границе: $2\Delta$ | §3.1 |
| Moré–Sorensen | $(B+\lambda I)p=-g$, $\lambda(\Delta-\lVert p\rVert)=0$, $B+\lambda I\succeq0$ | §3.2 |
| точка Коши | $p_C=-\tau\frac{\Delta}{\lVert g\rVert}g$, $\tau=\min\big(\frac{\lVert g\rVert^3}{\Delta g^\top Bg},1\big)$ | §3.3 |
| CG | $\alpha_k=\frac{r_k^\top r_k}{d_k^\top Ad_k}$, $\beta_{k+1}=\frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k}$, $d_{k+1}=-r_{k+1}+\beta_{k+1}d_k$ | §4.2 |
| масштаб $H_0$ | $H_0\leftarrow\frac{s^\top y}{y^\top y}I$ после первого шага | §2.4 |

**Три вещи, которые легко перепутать.**

1. $c_1$ и $c_2$: $c_1$ маленькое ($10^{-4}$) — «хоть немного уменьшили»; $c_2$ большое для BFGS ($0.9$) и меньше для нелинейного CG ($0.1$–$0.4$) — «склон стал положе».
2. $\rho$ в доверительной области — отношение фактического уменьшения к обещанному; $\rho_k=1/(y_k^\top s_k)$ в формуле BFGS (Л5 §4.3) — совсем другое число.
3. Линейный CG (квадратичная задача, точный шаг, конечность за $N/2$ на цепи) и нелинейный CG (линейный поиск, на цепи $266$ итераций) — разные методы с одной формулой.""")


def main() -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbf.validate(nb)
    nbf.write(nb, OUT)
    print(f"{OUT}: {len(cells)} ячеек")


if __name__ == "__main__":
    main()
