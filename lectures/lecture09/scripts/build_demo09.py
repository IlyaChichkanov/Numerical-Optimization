"""Собирает пошаговый ноутбук demo09.ipynb.

    uv run --group casadi python lectures/lecture09/scripts/build_demo09.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture09/demo09.ipynb

Ноутбук — «урок» по лекции 9. У каждой ячейки с кодом две подписи: «Что делает ячейка» перед ней и «Что видим»
после неё. Методы и задачи берутся из make_figures.py дословно (блок между метками «методы» и «числа конспекта»),
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
    stop = src.index("# ---------------------------------------------------------------- числа конспекта")
    return src[start:stop].rstrip() + "\n"


# ============================================================ шапка
md(r"""# Демонстрации к лекции 9: алгоритмы для равенств

Ноутбук — для самостоятельной проверки после пары. Всё, что на лекции было формулой или картинкой, здесь считается кодом. У каждой ячейки с кодом две подписи: **«Что делает ячейка»** перед ней и **«Что видим»** после неё. Числа конспекта печатаются рядом с ожидаемыми.

- **(0)** числа введения: кубическое уравнение дороги и стартовый бросок;
- **(a)** метод Ньютона–Лагранжа (§2): шаг руками в задаче «дорога», итерации, бросок против IPOPT;
- **(b)** матрица ККТ и шаг по частям (§3–§4): $\det K$ в задаче «дорога», бросок с $T=0$, вершина дороги, три способа найти шаг;
- **(c)** обобщённый Гаусс–Ньютон (§5): множитель ошибки $q$ и три дома;
- **(e)** подробности: лог IPOPT рядом с нашей невязкой, выбор $\lambda^{(0)}$, дом над вершиной.

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

def draw_road(ax, x1=(-1.5, 1.75)):
    """Дорога x2 = x1^2 — ограничение задачи 1."""
    t = np.linspace(*x1, 300)
    ax.plot(t, t ** 2, color=INK, lw=2)''')

# ============================================================ (0)
md(r"""## (0) Числа введения

**Что проверяем.** У дороги система сводится к кубическому уравнению. Стартовый бросок не попадает в цель.""")

md(r"""**Что делает ячейка.** Ищет корни $2x_1^3-x_1-2=0$ функцией `np.roots` и печатает вещественный. Затем считает, куда долетает мяч из старта $v=(10,\ 10)$, $T=2$.""")

code(r'''roots = np.roots([2, 0, -1, -2])                             # 2 x1^3 + 0 x1^2 - x1 - 2
x1 = roots[np.isreal(roots)].real[0]                          # единственный вещественный корень
print(f"корни: {np.round(roots, 3)};  x1 = {x1:.4f}, x2 = x1^2 = {x1**2:.4f}, lambda = x2 - 1 = {x1**2 - 1:.4f}")
v1, v2, T = 10.0, 10.0, 2.0                                    # старт броска
print(f"старт броска: к моменту T мяч пролетел {v1 * T:.0f} м, высота {v2 * T - 4.905 * T**2:.2f} м (нужно 30 м и 0 м)")''')

md(r"""**Что видим.** Вещественный корень один: $x_1=1.1654$, два других комплексные. Его не записать простой формулой — это и есть повод для численного метода. Стартовый бросок недолетает: 20 м вместо 30.""")

# ============================================================ (a)
md(r"""## (a) Метод Ньютона–Лагранжа (раздел 2)

**Что проверяем.** Шаг из §2.3, посчитанный руками; квадратичную сходимость в задаче «дорога»; бросок своим методом против IPOPT.""")

md(r"""**Шаг 1. Что делает ячейка.** Пишет систему $F$ и матрицу ККТ $K$ задачи «дорога» явно, по строке на уравнение. Затем решает $K\,(p,\Delta\lambda)=-F$ в точке $(1,0)$ при $\lambda=0$.""")

code(r'''def F_road(x1, x2, lam):
    """Система задачи 1 при a = (2, 1): две строки grad_x L и дорога."""
    return np.array([x1 - 2 + 2 * lam * x1,                  # dL/dx1 = 0
                     x2 - 1 - lam,                           # dL/dx2 = 0
                     x2 - x1**2])                            # g = 0

def K_road(x1, x2, lam):
    """Матрица ККТ: производные трёх строк F по x1, x2, lambda."""
    return np.array([[1 + 2 * lam, 0,   2 * x1],             # H = diag(1 + 2 lambda, 1) и столбец -J^T
                     [0,           1,   -1.0],
                     [-2 * x1,     1,   0.0]])               # строка J = (-2 x1, 1) и ноль

step = np.linalg.solve(K_road(1, 0, 0), -F_road(1, 0, 0))   # (p1, p2, dlambda)
print("F =", F_road(1, 0, 0), "  шаг (p1, p2, dlam) =", step.round(4), "  в конспекте: (0.2, 1.4, 0.4)")''')

md(r"""**Что видим.** Шаг $p=(0.2,\ 1.4)$ и $\Delta\lambda=0.4$ — как руками в §2.3.""")

md(r"""**Шаг 2. Что делает ячейка.** Печатает код метода `newton_lagrange` — он один на всю лекцию. Затем запускает его в задаче «дорога» из $(1,0)$ при $\lambda^{(0)}=0$. Функция `road_kkt(a)` возвращает $\nabla f$, $J$, $H$ и $g$ — те же формулы, что выше.""")

code(r'''import inspect
print(inspect.getsource(newton_lagrange))                    # код метода: 15 строк

hist = newton_lagrange(road_kkt(A_ROAD), [1.0, 0.0], [0.0])
for k, (x, lam, nF) in enumerate(hist):
    print(f"k = {k}: x = ({x[0]:.4f}, {x[1]:.4f}), lambda = {lam[0]:.4f}, "
          f"x2 - x1^2 = {x[1] - x[0]**2:.1e}, |F| = {nF:.2e}")    # нарушение ограничения и невязка
print("в конспекте: x2 - x1^2 = -1, -0.04, -0.0011, -6e-7, -1e-13; |F| = 1.73, 0.165, 0.003, 1.1e-6, 1.6e-13")
print("             x* = (1.165, 1.358), lambda* = 0.358")''')

md(r"""**Что видим.** Четыре итерации до $\lVert F\rVert<10^{-10}$. Показатель невязки удваивается: $10^{-3}\to10^{-6}\to10^{-13}$. Промежуточные точки не на дороге: шаг идёт до касательной. Нарушение $x_2-x_1^2$ падает так же быстро, и ответ лежит на дороге.""")

md(r"""**Шаг 3. Что делает ячейка.** Решает бросок. Функция `throw_problem()` строит задачу в CasADi и даёт `kkt` с производными (лекция 7). Стартовые множители — по МНК (`lam_ls`). Затем та же задача в IPOPT (`solve_throw_ipopt`, множители в нашем знаке). Рисует полёт мяча до момента $T$ на итерациях.""")

code(r'''pr = throw_problem()                                          # x = (v1, v2, T), D = 30 м
tk = pr["kkt"]
x0 = np.array([10.0, 10.0, 2.0])                               # старт: мяч пролетает 20 м
lam0 = lam_ls(tk, x0, 2)                                       # J^T lambda ≈ grad f по МНК
hist_t = newton_lagrange(tk, x0, lam0)
x_ip, lam_ip, st = solve_throw_ipopt(pr, x0)
print("lambda(0) =", lam0.round(3), "  в конспекте: (4.903, 5.093)")
print("|F| по итерациям:", ", ".join(f"{h[2]:.2g}" for h in hist_t), "  в конспекте: 10.0, 1.47, 0.026, 6.9e-6, 1.8e-13")
v1, v2, T = hist_t[-1][0]
print(f"наш метод: {len(hist_t) - 1} итерации, IPOPT: {st['iter_count']};  скорость {np.hypot(v1, v2):.2f} м/с, "
      f"угол {np.degrees(np.arctan2(v2, v1)):.1f}°, T = {T:.3f} с, lambda = {hist_t[-1][1].round(3)}")
print("с lambda(0) = 0:", len(newton_lagrange(tk, x0, [0.0, 0.0])) - 1, "итераций")

fig, ax = plt.subplots(figsize=(6.5, 3.0))
for k, c in zip((0, 1, 2), (GRAY, YELLOW, AQUA)):
    X, Y = throw_path(*hist_t[k][0]); ax.plot(X, Y, "--", color=c, label=f"k = {k}"); ax.plot(X[-1], Y[-1], "o", color=c)
X, Y = throw_path(*hist_t[-1][0]); ax.plot(X, Y, color=BLUE, label=f"k = {len(hist_t) - 1}")
ax.plot(30, 0, "v", color=RED, ms=10); ax.axhline(0, color=GRAY, lw=1)
ax.set_xlabel("расстояние, м"); ax.set_ylabel("высота, м"); ax.legend(ncol=4, fontsize=9); plt.show()''')

md(r"""**Что видим.** Четыре итерации, столько же, сколько у IPOPT. Ответ — $45^\circ$, $17.16$ м/с, $\lambda^\ast=(4.905,\ 4.905)$. С $\lambda^{(0)}=0$ — на одну итерацию больше. Совпадение лога IPOPT по итерациям — в части (e).""")

# ============================================================ (b)
md(r"""## (b) Матрица ККТ и шаг по частям (разделы 3–4)

**Что проверяем.** Формулу $\det K$ в задаче «дорога»; бросок со старта $T=0$; вершину дороги из вопроса залу; три способа §4 дают один шаг.""")

md(r"""**Шаг 1. Что делает ячейка.** Считает $\det K$ задачи «дорога» численно в трёх точках и сравнивает с $1+2\lambda+4x_1^2$ и с $\tilde Z^\top H\tilde Z$ при $\tilde Z=(1,\ 2x_1)$.""")

code(r'''def kkt_matrix(kkt, x, lam):
    """Матрица ККТ [[H, -J^T], [J, 0]] в точке (x, lam)."""
    _, J, H, _ = kkt(x, lam)
    m = J.shape[0]
    return np.block([[H, -J.T], [J, np.zeros((m, m))]])

rk = road_kkt(A_ROAD)
for x, lam in ((np.array([1.0, 0.0]), 0.0), (np.array([1.1654, 1.3581]), 0.3581), (np.array([0.3, -2.0]), -0.7)):
    H = rk(x, [lam])[2]; Zt = np.array([1.0, 2 * x[0]])            # вектор вдоль дороги
    print(f"x = {x}, lambda = {lam:+.4f}: det K = {np.linalg.det(kkt_matrix(rk, x, [lam])):.4f}, "
          f"1 + 2 lambda + 4 x1^2 = {1 + 2 * lam + 4 * x[0]**2:.4f}, Z~^T H Z~ = {Zt @ H @ Zt:.4f}")''')

md(r"""**Что видим.** Три числа в каждой строке совпадают. В ответе $\det K=7.15$. В третьей точке $\lambda=-0.7$ и $x_1=0.3$: определитель — ноль, и $K$ вырождена.""")

md(r"""**Шаг 2. Что делает ячейка.** Печатает $J$ и $g$ задачи «бросок» в старте $v=(10,\ 10)$, $T=0$ и пробует сделать шаг.""")

code(r'''_, J0, _, g0 = tk(np.array([10.0, 10.0, 0.0]), np.zeros(2))
print("J =\n", J0, "\ng =", g0, "  ранг J:", np.linalg.matrix_rank(J0))
try:
    newton_lagrange(tk, [10.0, 10.0, 0.0], [0.0, 0.0])
except np.linalg.LinAlgError as err:
    print("шаг не сделан:", err)''')

md(r"""**Что видим.** Строки $J$ совпадают, ранг 1 — LICQ нарушено. Линеаризованные ограничения требуют $10p_T=30$ и $10p_T=0$ одновременно, и NumPy пишет `Singular matrix`.""")

md(r"""**Шаг 3. Что делает ячейка.** Дом в $(0,\ 2)$ над вершиной. Запускает метод из вершины и из $(1,0)$, печатает ответы, расстояния и $Z^\top HZ$.""")

code(r'''a_top = np.array([0.0, 2.0])
for x0_ in ([0.0, 0.0], [1.0, 0.0]):
    h = newton_lagrange(road_kkt(a_top), x0_, [0.0])
    x, lam = h[-1][0], h[-1][1]
    J = road_kkt(a_top)(x, lam)[1]; Z = null_space(J)          # направление вдоль дороги
    ZHZ = (Z.T @ road_kkt(a_top)(x, lam)[2] @ Z).item()
    print(f"старт {x0_}: {len(h) - 1} итер., x = {x.round(4)}, lambda = {lam[0]:+.2f}, "
          f"расстояние {np.linalg.norm(x - a_top):.3f}, Z^T H Z = {ZHZ:+.2f}")''')

md(r"""**Что видим.** Из вершины метод не уходит: одна итерация, $\lambda=-2$, расстояние $2$. Там $Z^\top HZ=-3<0$ — это не минимум. Из $(1,0)$ метод приходит в ближайшую точку $(1.22,\ 1.5)$: расстояние $1.32$, $Z^\top HZ>0$.""")

md(r"""**Шаг 4. Что делает ячейка.** Функция `three_ways` находит шаг тремя способами §4: всей матрицей $K$, способом null-space и способом range-space. Запускаем её в задаче «дорога» в $(1,0)$ и в задаче «бросок» в старте с $\lambda^{(0)}$ по МНК.""")

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

for name, kkt, x, lam in (("дорога", rk, np.array([1.0, 0.0]), np.zeros(1)), ("бросок", tk, x0, lam0)):
    full, null, rng, pN, Zu = three_ways(kkt, x, lam)
    print(f"{name}: p = {full[0].round(4)}, lambda+ = {full[1].round(4)}")
    print(f"   расхождение: null-space {np.abs(full[0] - null[0]).max():.0e}, range-space {np.abs(full[0] - rng[0]).max():.0e}")
    if name == "дорога":
        print(f"   p_N = {pN.round(3)}, Zu = {Zu.round(3)}   (в конспекте: (-0.4, 0.2) и (0.6, 1.2))")''')

md(r"""**Что видим.** В задаче «дорога» $p_N=(-0.4,\ 0.2)$, $Zu=(0.6,\ 1.2)$ и $\lambda^+=0.4$ — как в §4. В задаче «бросок» оба способа совпадают с полной матрицей. У range-space ошибка округления в сто раз больше: $H$ броска почти вырождена (упражнение 9.3).""")

# ============================================================ (c)
md(r"""## (c) Обобщённый Гаусс–Ньютон (раздел 5)

**Что проверяем.** В задаче «дорога» ошибка за шаг умножается на $q=-2\lambda^\ast/(1+4x_1^{\ast2})$; в задаче «бросок» метод не сходится.""")

md(r"""**Что делает ячейка.** Для трёх домов запускает Ньютон–Лагранж и обобщённый Гаусс–Ньютон (`road_ggn_kkt`: в матрице $I$ вместо $H$) из $(1,0)$. Печатает $q$ по формуле, измеренное отношение соседних ошибок и число итераций.""")

code(r'''for a in ((1.5, 2.0), (2.0, 1.0), (0.0, 0.3)):
    a = np.array(a)
    hn = newton_lagrange(road_kkt(a), [1.0, 0.0], [0.0])
    hg = newton_lagrange(road_ggn_kkt(a), [1.0, 0.0], [0.0], maxit=500)
    xs, ls = hn[-1][0], hn[-1][1][0]
    q = -2 * ls / (1 + 4 * xs[0]**2)                              # множитель ошибки по формуле §5
    e = np.array([np.linalg.norm(t[0] - xs) for t in hg])         # ошибка Гаусса–Ньютона
    print(f"дом {a}: q = {q:+.4f}, измерено |e(k+1)/e(k)| = {e[4] / e[3]:.4f}; "
          f"итераций: Гаусс–Ньютон {len(hg) - 1}, Ньютон–Лагранж {len(hn) - 1}")
print("в конспекте: 6/4, 11/4, 43/6")

ggn_throw = lambda x, lam: (*tk(x, lam)[:2], np.diag([1.0, 1.0, 0.0]), tk(x, lam)[3])   # J_r^T J_r броска
h = newton_lagrange(ggn_throw, x0, lam0, maxit=30)
print("бросок, Гаусс–Ньютон: |F| на итерациях 26–30:", [f"{t[2]:.2f}" for t in h[-5:]])''')

md(r"""**Что видим.** Измеренное отношение ошибок совпадает с $|q|$ (у первого дома ошибка падает так быстро, что отношение видно хуже). В задаче «бросок» невязка качается около 5 и не падает: $\lambda^\ast=4.905$ велико.""")

# ============================================================ (e)
md(r"""## (e) Подробности

**Лог IPOPT рядом с нашей невязкой. Что делает ячейка.** Берёт из `st["iterations"]` колонки `inf_pr` и `inf_du` и сравнивает с половинами нашей невязки: $\max\lvert g_j\rvert$ и $\max\lvert(\nabla_xL)_i\rvert$.""")

code(r'''it = st["iterations"]
print(" k   наш |g|     IPOPT inf_pr   наш |grad L|  IPOPT inf_du")
for k, (x, lam, _) in enumerate(hist_t):
    gf, J, H, g = tk(x, lam)
    print(f"{k:2d}  {np.abs(g).max():10.3e}  {it['inf_pr'][k]:12.3e}  {np.abs(gf - J.T @ lam).max():12.3e}  {it['inf_du'][k]:12.3e}")
print("длины шагов IPOPT alpha_pr:", it["alpha_pr"][1:], "  сдвиг гессиана regularization_size:", it["regularization_size"])''')

md(r"""**Что видим.** Числа совпадают до ошибок округления. IPOPT принимает полный шаг (`alpha_pr` $=1$) и не сдвигает гессиан: на этой задаче он делает ровно шаги Ньютона–Лагранжа.""")

md(r"""**Выбор $\lambda^{(0)}$. Что делает ячейка.** Запускает метод в задаче «бросок» из того же старта с разными стартовыми множителями.""")

code(r'''for name, l0 in (("по МНК", lam0), ("нули", np.zeros(2)), ("все 20", 20 * np.ones(2))):
    h = newton_lagrange(tk, x0, l0)
    print(f"lambda(0) {name:7s}: {len(h) - 1:2d} итераций, max |F| по пути {max(t[2] for t in h):8.2f}")''')

md(r"""**Что видим.** С множителями по МНК — 4 итерации, и невязка только убывает. С нулями и с плохими множителями — по 5 итераций, а невязка по пути вырастает до 17 и 44. Множители по МНК — дешёвый и надёжный старт.""")

md(r"""**Дом над вершиной. Что делает ячейка.** Для дома $(0,\ h)$ ответ — вершина, $\lambda^\ast=-h$. Ячейка печатает измеренное отношение ошибок Гаусса–Ньютона и $2h$.""")

code(r'''for h_ in (0.1, 0.2, 0.3, 0.4, 0.45):
    hg = newton_lagrange(road_ggn_kkt(np.array([0.0, h_])), [1.0, 0.0], [0.0], maxit=2000)
    e = np.array([np.linalg.norm(t[0]) for t in hg])         # ответ — вершина (0, 0)
    k = len(hg) // 2                                          # отношение в середине пути, где оно уже установилось
    print(f"h = {h_:.2f}: измерено {e[k + 1] / e[k]:.3f}, 2h = {2 * h_:.2f}, итераций {len(hg) - 1}")''')

md(r"""**Что видим.** Отношение ошибок равно $2h$, и чем выше дом, тем больше итераций. При $h=0.5$ вершина перестаёт быть ближайшей точкой (упражнение 9.4).""")

# ============================================================ шпаргалка
md(r"""## Шпаргалка

| Что | Формула | Где в конспекте |
|---|---|---|
| система | $F=(\nabla f-J^\top\lambda,\ g)=0$ | §1 |
| матрица ККТ | $K=\begin{pmatrix}H&-J^\top\\J&0\end{pmatrix}$ | §2.1 |
| когда $K$ невырождена | LICQ и $Z^\top HZ\succ0$ | §3.2 |
| null-space | $p=p_N+Zu$, система $(n-m)\times(n-m)$ | §4 |
| range-space | $JH^{-1}J^\top\lambda^+=JH^{-1}\nabla f-g$ | §4 |
| обобщённый Гаусс–Ньютон | $H\to J_r^\top J_r$, ошибка $\times q$ | §5 |

**Три вещи, которые легко перепутать.**

1. Метод Ньютона–Лагранжа ищет решение системы, а не минимум. Из вершины дороги он не уходит, хотя это не ближайшая точка.
2. Промежуточные точки нарушают ограничения: $g=0$ выполняется только в пределе.
3. У обобщённого Гаусса–Ньютона нет $\lambda$ в матрице, но множитель ошибки $q$ пропорционален $\lambda^\ast$. Малые невязки — быстрая сходимость.""")


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
