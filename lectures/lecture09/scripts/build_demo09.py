"""Собирает пошаговый ноутбук demo09.ipynb.

    uv run --group casadi python lectures/lecture09/scripts/build_demo09.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture09/demo09.ipynb

Ноутбук — «урок» по лекции 9. У каждой ячейки с кодом две подписи: «Что делает ячейка» перед ней и «Что видим»
после неё. Методы и задачи берутся из make_figures.py дословно (блок между метками «методы» и «Ньютон для уравнения»),
поэтому числа ноутбука и картинок совпадают. Вторая ячейка (импорты, палитра, методы, задачи) становится
модулем l9helpers.py при сборке конспекта-ноутбука (build_lecture09.py). Нужен CasADi: в Colab первая ячейка ставит его сама.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo09.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def methods_block() -> str:
    """Методы и задачи лекции из make_figures.py — один код на лекцию."""
    src = (HERE / "make_figures.py").read_text(encoding="utf-8")
    start = src.index("# ---------------------------------------------------------------- методы")
    stop = src.index("# ---------------------------------------------------------------- Ньютон для уравнения")
    return src[start:stop].rstrip() + "\n"


# ============================================================ шапка
md(r"""# Демонстрации к лекции 9: алгоритмы для равенств

Ноутбук — для самостоятельной проверки после пары. Всё, что на лекции было формулой или картинкой, здесь считается кодом. У каждой ячейки с кодом две подписи: **«Что делает ячейка»** перед ней и **«Что видим»** после неё. Числа конспекта печатаются рядом с ожидаемыми.

- **(0)** числа введения: старт окружности и стартовые стержни цепи;
- **(a)** метод Ньютона–Лагранжа (§2): шаг руками на окружности, итерации, первая попытка на цепи;
- **(b)** матрица ККТ и шаг по частям (§3–§4): почему цепь не стартует, цепь против IPOPT, три способа найти один шаг;
- **(c)** обобщённый Гаусс–Ньютон (§5): множитель ошибки равен $\lambda^\ast$;
- **(e)** подробности: старт из арки, выбор $\lambda^{(0)}$, лог IPOPT рядом с нашей невязкой.

Конспект: [`lecture09.md`](lecture09.md). Запуск: `uv run --group casadi jupyter lab` в корне репозитория. В Colab первая ячейка ставит CasADi сама.""")

code(r'''%matplotlib inline

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


''' + methods_block() + r'''

def unit_circle(ax, color=INK, lw=2.0):
    """Единичная окружность — ограничение задачи §2.1."""
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=color, lw=lw)


def draw_chain(ax, ch, v, color=BLUE, ls="-", label=None):
    """Цепь: стержни линией, грузы кружками, опоры квадратами."""
    y, z = chain_xy(ch, v)
    ax.plot(y, z, ls, color=color, lw=2, label=label)
    ax.plot(y[1:-1], z[1:-1], "o", color=color, ms=9)
    ax.plot([y[0], y[-1]], [z[0], z[-1]], "s", color=INK, ms=9)''')

# ============================================================ (0)
md(r"""## (0) Числа введения

**Что проверяем.** В обеих задачах старт не лежит на ограничениях.""")

md(r"""**Что делает ячейка.** Считает ограничение окружности в старте $(1,1)$. Строит цепь лекции 8 и печатает длины стержней в стартовом положении: грузы в точках $(-1,0)$, $(0,0)$, $(1,0)$.""")

code(r'''x0 = np.array([1.0, 1.0])                                    # старт задачи 1
print("окружность: g(x0) =", 0.5 * (x0 @ x0 - 1), " (на окружности g = 0)")
ch = rod_chain()                                             # 3 груза, 4 стержня по 1.25 м, f = w (z1 + z2 + z3)
y, z = chain_xy(ch, ch["v0"])                                # координаты старта вместе с опорами
print("цепь: длины стержней на старте:", np.hypot(np.diff(y), np.diff(z)).round(3), " нужно:", ch["ell"])''')

md(r"""**Что видим.** На окружности $g=0.5\ne0$. У цепи крайние стержни $1.414$ м, средние $1$ м. Метод Ньютона–Лагранжа это допускает: ограничения выполнятся в пределе.""")

# ============================================================ (a)
md(r"""## (a) Метод Ньютона–Лагранжа (раздел 2)

**Что проверяем.** Шаг из §2.1, посчитанный руками; квадратичную сходимость на окружности; первую попытку на цепи с $\lambda^{(0)}=0$.""")

md(r"""**Шаг 1. Что делает ячейка.** Пишет систему $F$ и матрицу ККТ $K$ окружности явно, по строке на уравнение. Затем решает $K\,(p,\Delta\lambda)=-F$ в точке $(1,1)$ при $\lambda=0$.""")

code(r'''def F_circle(x1, x2, lam):
    """Система задачи 1: две строки grad_x L и ограничение."""
    return np.array([x1 - 2 - lam * x1,                      # dL/dx1 = 0
                     x2 - 1 - lam * x2,                      # dL/dx2 = 0
                     0.5 * (x1**2 + x2**2 - 1)])             # g = 0

def K_circle(x1, x2, lam):
    """Матрица ККТ: производные трёх строк F по x1, x2, lambda."""
    return np.array([[1 - lam, 0,       -x1],                # H = (1 - lambda) I и столбец -J^T
                     [0,       1 - lam, -x2],
                     [x1,      x2,      0.0]])                # строка J = (x1, x2) и ноль

step = np.linalg.solve(K_circle(1, 1, 0), -F_circle(1, 1, 0))   # (p1, p2, dlambda)
print("F =", F_circle(1, 1, 0), "  шаг (p1, p2, dlam) =", step, "  в конспекте: (0.25, -0.75, -0.75)")''')

md(r"""**Что видим.** Шаг $p=(0.25,\ -0.75)$ и $\Delta\lambda=-0.75$ — как руками в §2.1.""")

md(r"""**Шаг 2. Что делает ячейка.** Печатает код метода `newton_lagrange` — он один на всю лекцию. Затем запускает его на окружности из $(1,1)$ при $\lambda^{(0)}=0$. Функция `circle_kkt(a)` возвращает $\nabla f$, $J$, $H$ и $g$ — те же формулы, что выше.""")

code(r'''import inspect
print(inspect.getsource(newton_lagrange))                    # код метода: 15 строк

hist_c = newton_lagrange(circle_kkt(A_CIRCLE), [1.0, 1.0], [0.0])
for k, (x, lam, nF) in enumerate(hist_c):
    print(f"k = {k}: x = ({x[0]:.4f}, {x[1]:.4f}), lambda = {lam[0]:+.4f}, |F| = {nF:.2e}")
print("в конспекте: |F| = 1.12, 0.67, 0.14, 0.018, 1.6e-4, 9.5e-9; x* = (0.894, 0.447), lambda* = -1.236")''')

md(r"""**Что видим.** Шесть итераций до $\lVert F\rVert<10^{-10}$. С третьей итерации показатель невязки удваивается: $10^{-2}\to10^{-4}\to10^{-8}\to10^{-16}$.""")

md(r"""**Шаг 3. Что делает ячейка.** Запускает тот же метод на цепи из прямой $z=0$ с $\lambda^{(0)}=0$, как на окружности. Функция `ch["kkt"]` берёт $\nabla f$, $J$, $H$ и $g$ у CasADi (лекция 7). Ошибку NumPy ячейка ловит и печатает.""")

code(r'''ch = rod_chain()                                             # цепь лекции 8: 3 груза, 4 стержня
ck = ch["kkt"]                                               # (grad f, J, H, g) от CasADi
try:
    newton_lagrange(ck, ch["v0"], np.zeros(4))               # lambda(0) = 0, как на окружности
except np.linalg.LinAlgError as err:
    print("первый шаг не сделан:", err)''')

md(r"""**Что видим.** `Singular matrix`: система шага не решается, матрица $K$ вырождена. Почему — §3, демо (b).""")

# ============================================================ (b)
md(r"""## (b) Матрица ККТ и шаг по частям (разделы 3–4)

**Что проверяем.** Почему цепь не стартует при $\lambda=0$; цепь за 5 итераций с $\lambda^{(0)}$ по МНК против IPOPT; три способа §4 дают один шаг.""")

md(r"""**Шаг 1. Что делает ячейка.** Собирает $K$ цепи на старте при $\lambda=0$ и печатает её ранг и $J$ в порядке конспекта $(y_1,z_1,y_2,z_2,y_3,z_3)$. В коде переменные идут как $(y_1,y_2,y_3,z_1,z_2,z_3)$, поэтому столбцы переставлены. Затем проверяет, что $d_1$ и $d_2$ из §3.2 дают $K\,(d,0)=0$.""")

code(r'''def kkt_matrix(kkt, x, lam):
    """Матрица ККТ [[H, -J^T], [J, 0]] в точке (x, lam)."""
    _, J, H, _ = kkt(x, lam)
    m = J.shape[0]
    return np.block([[H, -J.T], [J, np.zeros((m, m))]])

perm = [0, 3, 1, 4, 2, 5]                                    # (y1, y2, y3, z1, z2, z3) -> (y1, z1, y2, z2, y3, z3)
K0 = kkt_matrix(ck, ch["v0"], np.zeros(4))                   # lambda = 0
_, J0, H0, _ = ck(ch["v0"], np.zeros(4))
print("H при lambda = 0 — ноль:", np.allclose(H0, 0), "; ранг K:", np.linalg.matrix_rank(K0), "из 10")
print("J в старте (столбцы y1, z1, y2, z2, y3, z3):\n", J0[:, perm])
for name, d in (("d1", [1, 1, 1, 0, 1, -1]), ("d2", [0, 0, 0, 1, 0, 0])):
    dv = np.array(d, float)[np.argsort(perm)]                # обратно в порядок переменных кода
    print(f"{name}: J d = {J0 @ dv}, |K (d, 0)| = {np.linalg.norm(K0 @ np.concatenate([dv, np.zeros(4)])):.0f}")''')

md(r"""**Что видим.** $H=0$, ранг $K$ — 8 из 10. Матрица $J$ — как в §3.2, а $d_1$ и $d_2$ — касательные: $Jd=0$ и $K\,(d,0)=0$.""")

md(r"""**Шаг 2. Что делает ячейка.** Берёт $\lambda^{(0)}$ по МНК (`lam_ls`), печатает собственные числа $H$ и запускает метод. Затем решает ту же задачу в IPOPT (`solve_rod_chain`, множители в нашем знаке). Строит форму цепи по итерациям.""")

code(r'''lam0 = lam_ls(ck, ch["v0"], 4)                               # J^T lambda ≈ grad f по МНК
print("lambda(0) =", lam0.round(3), "  в конспекте: все -4.905")
print("собственные числа H при lambda(0):", np.linalg.eigvalsh(ck(ch["v0"], lam0)[2]).round(2), " — все > 0")
hist = newton_lagrange(ck, ch["v0"], lam0)
v_ip, lam_ip, st = solve_rod_chain(ch)                       # IPOPT
print("|F| по итерациям:", ", ".join(f"{h[2]:.2g}" for h in hist), "  в конспекте: 9.86, 2.19, 0.55, 0.019, 1.5e-5, 4.5e-12")
print(f"наш метод: {len(hist) - 1} итераций, IPOPT: {st['iter_count']}; в конспекте: 5 и 5")
print("lambda наш  :", hist[-1][1].round(4))
print("lambda IPOPT:", lam_ip.round(4), " разница", np.abs(hist[-1][1] - lam_ip).max().round(9))

fig, ax = plt.subplots(figsize=(6.5, 3.2))
for k, c in zip((0, 1, 2), (GRAY, YELLOW, AQUA)):
    draw_chain(ax, ch, hist[k][0], color=c, ls="--", label=f"k = {k}")
draw_chain(ax, ch, hist[-1][0], color=BLUE, label=f"k = {len(hist) - 1}")
ax.set_aspect("equal"); ax.legend(ncol=4, fontsize=9, loc="upper center"); ax.set_ylim(-0.8, 1.6); plt.show()''')

md(r"""**Что видим.** С множителями по МНК все собственные числа $H$ положительны — от $5.7$ до $33.5$. Пять итераций, как у IPOPT, и множители $(-7.87,\ -5.58,\ -5.58,\ -7.87)$ совпадают с IPOPT до $10^{-6}$. Совпадение лога IPOPT по итерациям — в части (e).""")

md(r"""**Шаг 3. Что делает ячейка.** Функция `three_ways` находит шаг тремя способами §4: всей матрицей $K$, способом null-space и способом range-space. Запускаем её на окружности в $(1,1)$ и на цепи на старте.""")

code(r'''def three_ways(kkt, x, lam):
    """Шаг (p, lambda+) тремя способами: вся K, null-space, range-space."""
    gf, J, H, g = kkt(x, lam)
    n, m = H.shape[0], J.shape[0]
    s = np.linalg.solve(kkt_matrix(kkt, x, lam), -np.concatenate([gf - J.T @ lam, g]))
    full = (s[:n], lam + s[n:])                                   # вся матрица (n + m) x (n + m)
    pN = -J.T @ np.linalg.solve(J @ J.T, g)                       # к ограничению: J pN = -g
    Z = null_space(J)                                             # базис касательного пространства
    u = np.linalg.solve(Z.T @ H @ Z, -Z.T @ (gf + H @ pN))        # система (n - m) x (n - m)
    p = pN + Z @ u
    null = (p, np.linalg.lstsq(J.T, H @ p + gf, rcond=None)[0])   # множители после шага
    lp = np.linalg.solve(J @ np.linalg.solve(H, J.T), J @ np.linalg.solve(H, gf) - g)   # система m x m
    rng = (np.linalg.solve(H, J.T @ lp - gf), lp)
    return full, null, rng, pN, Z @ u

for name, kkt, x, lam in (("окружность", circle_kkt(A_CIRCLE), np.array([1.0, 1.0]), np.zeros(1)),
                          ("цепь", ck, ch["v0"], lam0)):
    full, null, rng, pN, Zu = three_ways(kkt, x, lam)
    diff = max(np.abs(full[0] - null[0]).max(), np.abs(full[0] - rng[0]).max(), np.abs(full[1] - rng[1]).max())
    print(f"{name}: p = {full[0].round(4)}, lambda+ = {full[1].round(4)}, расхождение способов {diff:.0e}")
    if name == "окружность":
        print(f"   p_N = {pN.round(3)}, Zu = {Zu.round(3)}   (в конспекте: (-0.25, -0.25) и (0.5, -0.5))")''')

md(r"""**Что видим.** На окружности $p_N=(-0.25,\ -0.25)$ и $Zu=(0.5,\ -0.5)$, а $\lambda^+=-0.75$ — как в §4. На цепи три способа решают системы $10\times10$, $2\times2$ и $4\times4$. Шаг у всех один, расхождение — на уровне округления.""")

# ============================================================ (c)
md(r"""## (c) Обобщённый Гаусс–Ньютон (раздел 5)

**Что проверяем.** На окружности ошибка угла за шаг умножается на $\lambda^\ast=1-\lVert a\rVert$, а при $a=(2,1)$ метод качается.""")

md(r"""**Что делает ячейка.** Запускает `ggn_circle` из $x^{(0)}=(0.6,\ 0.8)$ для трёх точек $a$. В матрице ККТ этого метода вместо $H=(1-\lambda)I$ стоит $I$. Печатает отношение соседних ошибок угла и число итераций до $10^{-8}$. Строит ошибку по итерациям.""")

code(r'''fig, ax = plt.subplots(figsize=(6.5, 3.4))
for a, c in ((np.array([0.96, 0.72]), AQUA), (np.array([1.2, 0.9]), BLUE), (A_CIRCLE, ORANGE)):
    e = angle_err(ggn_circle(a, [0.6, 0.8], 40), a)          # ошибка угла по итерациям
    lam_star = 1 - np.linalg.norm(a)                          # множитель ближней точки
    hit = np.nonzero(np.abs(e) <= 1e-8)[0]
    its = hit[0] if len(hit) else "не сходится"
    print(f"|a| = {np.linalg.norm(a):.3f}: lambda* = {lam_star:+.3f}, e_(k+1)/e_k = {e[6] / e[5]:+.3f}, итераций до 1e-8: {its}")
    ax.semilogy(np.abs(e) + 1e-17, "o-", ms=3, color=c, label=f"|a| = {np.linalg.norm(a):.2f}")
ax.set_xlabel("итерация k"); ax.set_ylabel("ошибка угла"); ax.set_ylim(1e-12, 3); ax.legend(); plt.show()
print("в конспекте: 11 и 25 итераций; при a = (2, 1) — не сходится")''')

md(r"""**Что видим.** При $\lVert a\rVert=1.2$ и $1.5$ отношение ошибок — $-0.2$ и $-0.5$, ровно $\lambda^\ast$. При $a=(2,1)$ ошибка угла застревает около $0.25$: метод качается между двумя точками.""")

# ============================================================ (e)
md(r"""## (e) Подробности

**Старт из арки. Что делает ячейка.** Запускает метод из арки $(-1,\ 2)$, $(0,\ 2.5)$, $(1,\ 2)$ и печатает множители и энергию ответа. Затем — редуцированный гессиан в обоих ответах.""")

code(r'''v_arch = np.array([-1.0, 0.0, 1.0, 2.0, 2.5, 2.0])         # y1, y2, y3, z1, z2, z3
ha = newton_lagrange(ck, v_arch, lam_ls(ck, v_arch, 4))
for name, (x, lam) in (("арка", ha[-1][:2]), ("висящая цепь", hist[-1][:2])):
    _, J, H, _ = ck(x, lam)
    Z = null_space(J)                                         # базис касательного пространства, 6 x 2
    print(f"{name}: lambda = {lam.round(2)}, f = {ch['w'] * x[3:].sum():.2f}, Z^T H Z: собственные числа {np.linalg.eigvalsh(Z.T @ H @ Z).round(1)}")
print(f"арка: итераций {len(ha) - 1}; в конспекте: 4 итерации, f = 61.25")''')

md(r"""**Что видим.** В арке все $\lambda_j>0$, а редуцированный гессиан отрицательно определён: $-35.2$ и $-19.0$. Это максимум, и матрица ККТ там тоже невырождена. В висящей цепи те же числа со знаком плюс.""")

md(r"""**Выбор $\lambda^{(0)}$. Что делает ячейка.** Запускает метод на цепи из прямой $z=0$ с разными стартовыми множителями.""")

code(r'''for name, l0 in (("по МНК", lam0), ("все -1", -np.ones(4)), ("все -20", -20 * np.ones(4))):
    h = newton_lagrange(ck, ch["v0"], l0)
    ok = np.allclose(h[-1][0], hist[-1][0], atol=1e-6)
    print(f"lambda(0) {name:7s}: {len(h) - 1:2d} итераций, max |F| по пути {max(t[2] for t in h):8.2f}, тот же ответ: {ok}")''')

md(r"""**Что видим.** С множителями по МНК — 5 итераций и невязка только убывает. С плохими множителями первый шаг уводит далеко, и итераций больше. Множители по МНК — дешёвый и надёжный старт.""")

md(r"""**Лог IPOPT рядом с нашей невязкой. Что делает ячейка.** Берёт из `st["iterations"]` колонки `inf_pr` и `inf_du` и сравнивает с половинами нашей невязки: $\lVert g\rVert_\infty$ и $\lVert\nabla_xL\rVert_\infty$.""")

code(r'''ours = []
for x, lam, _ in hist:
    gf, J, H, g = ck(x, lam)
    ours.append((np.abs(g).max(), np.abs(gf - J.T @ lam).max()))   # половины F в норме max
it = st["iterations"]
print(" k   наш |g|     IPOPT inf_pr   наш |grad L|  IPOPT inf_du")
for k in range(len(ours)):
    print(f"{k:2d}  {ours[k][0]:10.2e}  {it['inf_pr'][k]:12.2e}  {ours[k][1]:12.2e}  {it['inf_du'][k]:12.2e}")
print("длины шагов IPOPT alpha_pr:", it["alpha_pr"][1:])''')

md(r"""**Что видим.** Числа совпадают до ошибок округления: на этой задаче IPOPT делает те же шаги, что наш метод. Стартовые множители он тоже берёт по МНК, а линейный поиск каждый раз принимает полный шаг (`alpha_pr` $=1$).""")

# ============================================================ шпаргалка
md(r"""## Шпаргалка

| Что | Формула | Где в конспекте |
|---|---|---|
| система | $F=(\nabla f-J^\top\lambda,\ g)=0$ | §2.2 |
| матрица ККТ | $K=\begin{pmatrix}H&-J^\top\\J&0\end{pmatrix}$ | §2.2 |
| когда $K$ невырождена | LICQ и $Z^\top HZ\succ0$ | §3.1 |
| null-space | $p=p_N+Zu$, система $(n-m)\times(n-m)$ | §4 |
| range-space | $JH^{-1}J^\top\lambda^+=JH^{-1}\nabla f-g$ | §4 |
| обобщённый Гаусс–Ньютон | $H\to J_r^\top J_r$ | §5 |

**Три вещи, которые легко перепутать.**

1. Метод Ньютона–Лагранжа ищет решение системы, а не минимум. Из арки он приходит в максимум.
2. Промежуточные точки нарушают ограничения: $g=0$ выполняется только в пределе.
3. У обобщённого Гаусса–Ньютона нет $\lambda$ в матрице, но множитель ошибки равен $\lambda^\ast$. Малые невязки — быстрая сходимость.""")


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
