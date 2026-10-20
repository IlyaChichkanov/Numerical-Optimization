"""Собирает пошаговый ноутбук demo07.ipynb.

    uv run --group casadi python lectures/lecture07/scripts/build_demo07.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture07/demo07.ipynb

Ноутбук — «урок» по лекции 7: конечные разности и U-кривая, прямой и обратный режим алгоритмического
дифференцирования своими руками, CasADi на упругой цепи. Разности, пример из трёх операций и цепь берутся
из make_figures.py дословно (блок между метками «разности и задачи» и «числа конспекта»), поэтому числа
ноутбука и картинок совпадают. Вторая ячейка (импорты, палитра, разности, задачи) становится модулем
l7helpers.py при сборке конспекта-ноутбука (build_lecture07.py). Нужен CasADi: в Colab первая ячейка ставит его сама.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo07.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def methods_block() -> str:
    """Разности, пример и цепь из make_figures.py — один код на лекцию."""
    src = (HERE / "make_figures.py").read_text(encoding="utf-8")
    start = src.index("# ---------------------------------------------------------------- разности и задачи")
    stop = src.index("# ---------------------------------------------------------------- числа конспекта")
    return src[start:stop].rstrip() + "\n"


# ============================================================ шапка
md(r"""# Демонстрации к лекции 7: конечные разности и алгоритмическое дифференцирование

Ноутбук — для самостоятельной проверки после пары: всё, что на лекции было формулой или картинкой, здесь считается кодом, а числа конспекта печатаются рядом с ожидаемыми. Читается сверху вниз; каждая часть начинается с «что проверяем и что должны увидеть».

- **(0)** числа введения: градиент и гессиан упругой цепи разностями и через CasADi — $81$ и $25\,600$ вычислений против $1.6$ и $5.5$;
- **(a)** конечные разности (раздел 2): U-кривая на Розенброке, лучшие $h$ и наклоны веток, SciPy BFGS с `jac=None`;
- **(b)** алгоритмическое дифференцирование своими руками (разделы 3–4): программа как список операций, прямой и обратный режим в 20 строк, трасса примера $x_1x_2+\sin(x_1x_2)$;
- **(c)** CasADi на цепи (раздел 5): `n_instructions`, разреженность гессиана, `jtimes`, IPOPT с точным гессианом против L-BFGS;
- **(e)** подробности: мнимый трюк, разности от шумной функции, разностный гессиан, время при $n=2000$.

Конспект: [`lecture07.md`](lecture07.md). Запуск: `uv run --group casadi jupyter lab` в корне репозитория; в Colab первая ячейка ставит CasADi сама.""")

code(r'''%matplotlib inline

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


''' + methods_block() + r'''

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
    ax.set(xlabel="шаг $h$", ylabel="ошибка производной"); ax.legend(fontsize=8)''')

# ============================================================ (0)
md(r"""## (0) Числа введения

**Что проверяем.** Цену производных упругой цепи из ДЗ 4 ($N=40$ грузиков, $n=80$ переменных). Разностями: градиент — $n+1=81$ вычисление энергии, гессиан по значениям энергии — $4n^2=25\,600$ с ошибкой $\sim3\cdot10^{-5}$. CasADi: градиент за $1.6$ цены энергии (в операциях), гессиан за $5.5$ — и точные.""")

code(r'''energy, grad, v0 = make_elastic_chain(40)
nf, ng, nh, nhd, nnz = chain_costs(40)
print(f"CasADi, N=40: операций в E {nf}, в градиенте {ng} (x{ng / nf:.2f}), в гессиане {nh} (x{nh / nf:.1f}); ненулевых в гессиане {nnz} из {80 * 80}")
print(f"разности:     градиент — {80 + 1} вычислений E, гессиан по E — {4 * 80 * 80} вычислений")
g_fd = np.array([fd_forward(energy, v0, i, np.sqrt(EPS)) for i in range(80)])        # прямые разности с шагом sqrt(eps)
v, E = chain_casadi(40)
g_ca = np.array(ca.Function("G", [v], [ca.gradient(E, v)])(v0)).ravel()
print(f"ошибка градиента: разности {np.abs(g_fd - grad(v0)).max():.1e}, CasADi {np.abs(g_ca - grad(v0)).max():.1e}   (в конспекте: 1e-6 и 1e-15)")''')

# ============================================================ (a)
md(r"""## (a) Конечные разности: U-кривая (раздел 2)

**Что проверяем.** Для $\partial f/\partial x_1$ Розенброка в $(-1.2,1)$ (точно $-215.6$): у прямой разности лучший $h\approx6\cdot10^{-9}$ и ошибка $1.4\cdot10^{-6}$, у центральной — $h\approx2.5\cdot10^{-6}$ и ошибка $10^{-9}$. Наклоны веток в логарифмических осях: $1$ и $2$ справа, $-1$ слева. Формула $h^\ast=2\sqrt{\varepsilon\lvert f\rvert/\lvert f''\rvert}$ даёт $4\cdot10^{-9}$ и ошибку до $5\cdot10^{-6}$.""")

code(r'''exact = rosen_grad(X0_ROSEN)[0]
ef, ec = ucurve(rosen, X0_ROSEN, 0, exact)                           # ошибки по сетке HS
slope = lambda e, lo, hi: np.polyfit(np.log10(HS[(HS >= lo) & (HS <= hi)]), np.log10(e[(HS >= lo) & (HS <= hi)]), 1)[0]
for name, e, lo in (("прямая", ef, 1e-5), ("центральная", ec, 1e-4)):
    j = np.argmin(e)
    print(f"{name:12s}: лучший h = {HS[j]:.1e}, ошибка {e[j]:.1e};  наклон справа {slope(e, lo, 1e-1):.2f}, слева {slope(e, 1e-15, 1e-10):.2f}")
f11 = 2 - 400 * X0_ROSEN[1] + 1200 * X0_ROSEN[0] ** 2                 # d^2 f / d x1^2 в старте
print(f"формула: h* = 2 sqrt(eps |f| / |f''|) = {2 * np.sqrt(EPS * rosen(X0_ROSEN) / f11):.1e}, ошибка ~ {2 * np.sqrt(EPS * rosen(X0_ROSEN) * f11):.1e}   (в конспекте: 4e-9 и до 5e-6)")
fig, ax = plt.subplots(figsize=(7.5, 4))
ucurve_plot(ax, [ef, ec], ["прямая разность", "центральная разность"], [BLUE, ORANGE])
ax.set(title="U-кривая: ошибка разности по шагу h"); plt.show()''')

md(r"""**SciPy без `jac`.** BFGS на Розенброке с точным градиентом и с разностным, который SciPy берёт сам (`jac=None`, прямые разности с шагом `eps = 1.49e-8`): итераций $33$ и $33$, вызовов $f$ — $40$ и $120$, а точный градиент в найденной точке — $10^{-8}$ против $10^{-5}$.""")

code(r'''for name, jac in (("точный градиент", rosen_grad), ("jac=None (разности)", None), ("jac='3-point'", "3-point")):
    r = minimize(rosen, X0_ROSEN, jac=jac, method="BFGS", options={"gtol": 1e-6})
    print(f"{name:20s}: итераций {r.nit}, вызовов f {r.nfev:3d}, |grad f(x*)| точный = {np.linalg.norm(rosen_grad(r.x)):.1e}, |x* - (1,1)| = {np.linalg.norm(r.x - 1):.1e}")''')

# ============================================================ (b)
md(r"""## (b) Алгоритмическое дифференцирование своими руками (разделы 3–4)

**Что проверяем.** Программа $f=x_1x_2+\sin(x_1x_2)$ — три операции в списке `PROGRAM`. Прямой режим по направлению $d=(1,0)$ в точке $(2,0.5)$: $\dot v_1=0.5$, $\dot v_2=0.2702$, $\dot f=0.7702$. Обратный режим: $\bar v_2=1$, $\bar v_1=1.5403$, $\bar x=(0.7702,\ 3.0806)$ — весь градиент за один проход. Код `forward_mode` и `reverse_mode` — в ячейке настройки, по 10 строк каждый.""")

code(r'''x = {"x1": 2.0, "x2": 0.5}
val, dot = forward_mode(PROGRAM, x, {"x1": 1.0, "x2": 0.0})            # прямой режим, d = (1, 0)
print("прямой режим, d = (1, 0):", {k: round(v, 4) for k, v in dot.items()})
val, bar = reverse_mode(PROGRAM, x)                                    # обратный режим
print("обратный режим:          ", {k: round(v, 4) for k, v in bar.items()})
g = lambda x1, x2: np.array([x2 * (1 + np.cos(x1 * x2)), x1 * (1 + np.cos(x1 * x2))])
print("градиент по формуле:     ", g(2.0, 0.5).round(4), "  (в конспекте: 0.7702, 3.0806)")
assert np.allclose([bar["x1"], bar["x2"]], g(2.0, 0.5)) and np.isclose(dot["f"], g(2.0, 0.5)[0])''')

md(r"""**Трасса обратного прохода по шагам** — та же, что в §4.1: $f=v_1+v_2$ раздаёт по единице, $v_2=\sin v_1$ добавляет $v_1$ долю $\cos v_1=0.5403$, $v_1=x_1x_2$ раздаёт входам $\bar v_1x_2$ и $\bar v_1x_1$.""")

code(r'''val = dict(x)
for name, op, args in PROGRAM:
    val[name] = OPS[op][0](*[val[k] for k in args])
bar = {k: 0.0 for k in val}; bar["f"] = 1.0
for name, op, args in reversed(PROGRAM):
    a = [val[k] for k in args]
    for p, k in zip(OPS[op][1](*a), args):
        bar[k] += bar[name] * p
        print(f"{name} = {op}{args}: {k} += {bar[name]:.4f} * {p:.4f}  ->  bar {k} = {bar[k]:.4f}")''')

md(r"""**Мини-задание лекции:** та же функция в $(1,\pi)$. $v_1=\pi$, $\bar v_1=1+\cos\pi=0$ — градиент нулевой, точка стационарная.""")

code(r'''val, bar = reverse_mode(PROGRAM, {"x1": 1.0, "x2": np.pi})
print(f"в (1, pi): v1 = {val['v1']:.4f}, bar v1 = {bar['v1']:.1e}, градиент = ({bar['x1']:.1e}, {bar['x2']:.1e})")''')

# ============================================================ (c)
md(r"""## (c) CasADi на упругой цепи (раздел 5)

**Что проверяем.** Энергия цепи как граф CasADi (цикл по пружинам, `chain_casadi`). Число операций `n_instructions()`: градиент $\times1.6$, гессиан $\times5.5$–$5.7$, гессиан на вектор через `jtimes` $\times3.3$ — при $N=40$, $200$, $1000$; ненулевых в гессиане $472$ из $6400$ при $N=40$. IPOPT из прямой: $24$ итерации с точным гессианом против $232$ с L-BFGS.""")

code(r'''print(f"{'N':>5s} {'n':>5s} {'E':>7s} {'grad':>7s} {'hess':>7s} {'H d':>7s}   ненулевых в гессиане")
for N in (40, 200, 1000):
    nf, ng, nh, nhd, nnz = chain_costs(N)
    print(f"{N:5d} {2 * N:5d} {nf:7d} {ng:7d} {nh:7d} {nhd:7d}   {nnz} из {(2 * N) ** 2} = {100 * nnz / (2 * N) ** 2:.1f} %   (x{ng / nf:.2f}, x{nh / nf:.1f}, x{nhd / nf:.1f})")
print("в конспекте: x1.6, x5.5-5.7, x3.3; 472 из 6400")''')

md(r"""**Портрет гессиана.** Пружина связывает двух соседей, отсюда четыре трёхдиагональные ленты (блоки $y$–$y$, $y$–$z$, $z$–$y$, $z$–$z$). CasADi находит эти ленты по графу до всяких вычислений.""")

code(r'''pattern, v_star, Hfun = chain_pattern(40)                            # структура из графа и равновесие цепи
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.spy(pattern, markersize=2.5, color=INK); ax.grid(False)
ax.set(title=f"гессиан энергии цепи: {int(pattern.sum())} ненулевых из 6400", xticks=[0, 40, 79], yticks=[0, 40, 79]); plt.show()''')

md(r"""**Что даёт солверу точный гессиан.** IPOPT на той же цепи из прямой: `hessian_approximation="exact"` — гессиан от CasADi, `"limited-memory"` — L-BFGS внутри IPOPT.""")

code(r'''v, E = chain_casadi(40); v0 = make_elastic_chain(40)[2]                     # старт — прямая между концами
for mode in ("exact", "limited-memory"):
    solver = ca.nlpsol("s", "ipopt", {"x": v, "f": E}, {"ipopt.hessian_approximation": mode, "ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.tol": 1e-8})
    sol = solver(x0=v0); st = solver.stats()
    print(f"hessian_approximation = {mode:15s}: итераций {st['iter_count']:3d}, E* = {float(sol['f']):.4f}, {st['return_status']}")
print("в конспекте: 24 против 232")''')

# ============================================================ (e)
md(r"""## (e) Подробности

**Мнимый трюк (упражнение 7.4).** $f'(x)\approx\operatorname{Im}f(x+ih)/h$: вычитания нет, при $h=10^{-20}$ ошибка на уровне $10^{-14}$ при производной $-215.6$.""")

code(r'''h = 1e-20
z = X0_ROSEN.astype(complex); z[0] += 1j * h
print(f"мнимый трюк: ошибка {abs(np.imag(rosen(z)) / h - rosen_grad(X0_ROSEN)[0]):.1e};  прямая разность при лучшем h: 1.4e-6")''')

md(r"""**Разности от шумной функции (вопрос залу §2.4).** $f$ с относительным шумом $10^{-10}$: левая ветка поднимается на шесть порядков, дно уезжает к $h\sim10^{-6}$ (прямая) и $10^{-4}$ (центральная), ошибки $4\cdot10^{-4}$ и $10^{-6}$.""")

code(r'''rng = np.random.default_rng(0)
noisy = lambda x: rosen(x) * (1 + 1e-10 * rng.standard_normal())
exact = rosen_grad(X0_ROSEN)[0]; ef, ec = ucurve(rosen, X0_ROSEN, 0, exact)
efn, ecn = ucurve(noisy, X0_ROSEN, 0, exact)
for name, e in (("прямая", efn), ("центральная", ecn)):
    j = np.argmin(e); print(f"шумная f, {name:12s}: лучший h = {HS[j]:.1e}, ошибка {e[j]:.1e}")
fig, ax = plt.subplots(figsize=(7.5, 4))
ucurve_plot(ax, [ef, efn, ec, ecn], ["прямая, точная f", "прямая, шум 1e-10", "центральная, точная f", "центральная, шум 1e-10"], [BLUE, "#9dc3ea", ORANGE, "#f5b89d"])
ax.set(title="шумная функция: дно U уезжает вправо и вверх"); plt.show()''')

md(r"""**Разностный гессиан цепи** по значениям энергии в равновесии: $25\,600$ вычислений, ошибка до $3\cdot10^{-5}$ и шум вместо нулей между лентами.""")

code(r'''n, hh = 80, 1e-4
energy = make_elastic_chain(40)[0]
H = np.array(Hfun(v_star)); I = np.eye(n); Hff = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        ei, ej = I[i], I[j]
        Hff[i, j] = (energy(v_star + hh * (ei + ej)) - energy(v_star + hh * (ei - ej)) - energy(v_star - hh * (ei - ej)) + energy(v_star - hh * (ei + ej))) / (4 * hh * hh)
print(f"ошибка разностного гессиана: {np.abs(Hff - H).max():.1e}; «ненулевых» {int(np.sum(np.abs(Hff) > 1e-12))} вместо 472")
fig, ax = plt.subplots(figsize=(5, 4.2))
pc = ax.pcolormesh(np.maximum(np.abs(Hff), 1e-12), cmap="Greys", norm=LogNorm(vmin=1e-12, vmax=1e3)); ax.invert_yaxis(); ax.grid(False)
ax.set(aspect="equal", title="разностный гессиан: шум между лентами"); fig.colorbar(pc, ax=ax, label="$|H_{ij}|$"); plt.show()''')

md(r"""**Время при $n=2000$.** Градиент разностями по numpy-энергии против градиента CasADi: сотни миллисекунд против долей миллисекунды. При малых $n$ разница меньше — накладные расходы вызова `Function` сравнимы с самим счётом.""")

code(r'''energy1000, grad1000, v1000 = make_elastic_chain(1000)
v, E = chain_casadi(1000); G = ca.Function("G", [v], [ca.gradient(E, v)])
t0 = time.perf_counter(); [fd_forward(energy1000, v1000, i, 1e-8) for i in range(2000)]; t_fd = time.perf_counter() - t0
t0 = time.perf_counter(); [G(v1000) for _ in range(50)]; t_ad = (time.perf_counter() - t0) / 50
print(f"n = 2000: градиент разностями {t_fd * 1e3:.0f} мс, CasADi {t_ad * 1e3:.2f} мс — в {t_fd / t_ad:.0f} раз")''')

md(r"""## Шпаргалка

| Что | Формула | Где в конспекте |
|---|---|---|
| прямая разность | $\dfrac{f(x+he_i)-f(x)}{h}$, ошибка $\tfrac h2\lvert f''\rvert+\tfrac{2\varepsilon\lvert f\rvert}{h}$ | §2.1 |
| лучший шаг | $h^\ast=2\sqrt{\varepsilon\lvert f\rvert/\lvert f''\rvert}\sim\sqrt\varepsilon$, ошибка $\sim\sqrt\varepsilon$ | §2.1 |
| центральная разность | $\dfrac{f(x+he_i)-f(x-he_i)}{2h}$, $h^\ast\sim\varepsilon^{1/3}$, ошибка $\sim\varepsilon^{2/3}$ | §2.2 |
| прямой режим | $\dot v_i=\sum_j\frac{\partial\varphi_i}{\partial v_j}\dot v_j$, на выходе $\nabla f^\top d$ | §3.2 |
| обратный режим | $\bar v_j\mathrel{+}=\bar v_i\,\partial\varphi_i/\partial v_j$, на выходе $\bar x=\nabla f$ | §4.2 |
| цена | вперёд $\le2.5$ цены $f$ на направление; назад $\le4$ на весь градиент | §3.3, §4 |
| CasADi | `gradient` — назад, `jtimes` — вперёд, `hessian` — назад по прямому, `n_instructions()` — цена | §5 |

**Три вещи, которые легко перепутать.**

1. $\dot v_i$ — производная $v_i$ по направлению $d$ (считается вместе с $v_i$); $\bar v_i$ — производная $f$ по $v_i$ (считается после, от выхода).
2. Ошибка округления — не ошибка формулы: формула точна в пределе $h\to0$, а числа — нет.
3. `n_instructions()` — число операций, а не время: при малых $n$ вызов `Function` дороже самого счёта.""")


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
