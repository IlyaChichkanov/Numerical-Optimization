"""Собирает пошаговый ноутбук demo08.ipynb.

    uv run --group casadi python lectures/lecture08/scripts/build_demo08.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture08/demo08.ipynb

Ноутбук — «урок» по лекции 8: система условий первого порядка на окружности, множитель, которого нет (клюв и
повторённое ограничение), цепь на стержнях в IPOPT с проверкой трёх условий, портфель и лог IPOPT. Задачи берутся
из make_figures.py дословно (блок между метками «задачи» и «числа конспекта»), поэтому числа ноутбука и картинок
совпадают. Вторая ячейка (импорты, палитра, задачи) становится модулем l8helpers.py при сборке конспекта-ноутбука
(build_lecture08.py). Нужен CasADi: в Colab первая ячейка ставит его сама.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo08.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def methods_block() -> str:
    """Задачи лекции из make_figures.py — один код на лекцию."""
    src = (HERE / "make_figures.py").read_text(encoding="utf-8")
    start = src.index("# ---------------------------------------------------------------- задачи")
    stop = src.index("# ---------------------------------------------------------------- числа конспекта")
    return src[start:stop].rstrip() + "\n"


# ============================================================ шапка
md(r"""# Демонстрации к лекции 8: ограничения-равенства

Ноутбук — для самостоятельной проверки после пары. Всё, что на лекции было формулой или картинкой, здесь считается кодом. Числа конспекта печатаются рядом с ожидаемыми. Каждая часть начинается с того, что проверяем и что должны увидеть.

- **(0)** числа введения: два решения на окружности и натяжения стержней цепи;
- **(a)** условие первого порядка (§2): окружность на плоскости и в пространстве, касательная часть $\nabla f$, два множителя;
- **(b)** когда множителей нет (§3): клюв и ограничение, записанное дважды;
- **(c)** цепь на стержнях в IPOPT (§5): три проверки и силы в стержнях;
- **(e)** подробности: цена доходности, цена длины цепи, лог IPOPT из двух стартов, знак множителей у солверов.

Конспект: [`lecture08.md`](lecture08.md). Запуск: `uv run --group casadi jupyter lab` в корне репозитория. В Colab первая ячейка ставит CasADi сама.""")

code(r'''%matplotlib inline

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.optimize import fsolve

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


''' + methods_block() + r'''

def level_circles(ax, a, levels, color=BLUE, lw=1.2, alpha=0.8):
    """Линии уровня f = 1/2 |x - a|^2 — окружности вокруг a радиуса sqrt(2 f)."""
    th = np.linspace(0, 2 * np.pi, 400)
    for f in levels:
        r = np.sqrt(2 * f)
        ax.plot(a[0] + r * np.cos(th), a[1] + r * np.sin(th), color=color, lw=lw, alpha=alpha)


def unit_circle(ax, color=INK, lw=2.2):
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=color, lw=lw)''')

# ============================================================ (0)
md(r"""## (0) Числа введения

**Что проверяем.** Задача 1: у системы касания два решения, $\lambda=-1.236$ и $\lambda=3.236$. Задача 2: IPOPT решает цепь за 13 итераций. Множители — натяжения: 23.9 Н у опор и 13.6 Н в середине.""")

code(r'''for z0 in ([1, 0.5, 0], [-1, -0.5, 0]):                               # два старта для fsolve: справа и слева
    z = fsolve(circle_system, z0, xtol=1e-14)                          # решаем (x - a) - lam x = 0, 1/2(|x|^2 - 1) = 0
    print(f"x = ({z[0]:+.3f}, {z[1]:+.3f}),  lambda = {z[2]:+.3f},  f = {0.5 * np.sum((z[:2] - A_CIRCLE) ** 2):.3f}")
print(f"в конспекте: lambda = 1 - sqrt5 = {1 - np.sqrt(5):+.3f} (минимум) и 1 + sqrt5 = {1 + np.sqrt(5):+.3f} (максимум)")''')

code(r'''ch = rod_chain()                                                       # N = 40 грузов, 41 стержень длины 5/41, f = w sum z
v, lam, st = solve_rod_chain(ch)                                       # IPOPT; lam — в нашем знаке L = f - lam^T g
chk = chain_checks(ch, v, lam)                                         # натяжения F_j = 2|lam_j| ell и их составляющие
print(f"IPOPT: {st['iter_count']} итераций (в конспекте 13), f* = {ch['w'] * chain_xy(ch, v)[1][1:-1].sum():.3f} Дж (5.756)")
print(f"натяжение: концы {chk['F'][0]:.1f} Н, середина {chk['F'][20]:.1f} Н (в конспекте 23.9 и 13.6)")
print(f"горизонтальная составляющая: от {np.abs(chk['Fh']).min():.4f} до {np.abs(chk['Fh']).max():.4f} Н — одна и та же во всех 41 стержнях")''')

# ============================================================ (a)
md(r"""## (a) Условие первого порядка (раздел 2)

**Что проверяем.** На окружности система $x-a=\lambda x$, $\lVert x\rVert=1$ имеет два решения. В обоих $\nabla f=\lambda\nabla g$ с точностью машинного нуля. В точке $(0.5,\ 0.866)$ у $\nabla f$ есть касательная часть: $1.232$.

Касательная часть — это $Z^\top\nabla f$, где $Z$ — базис касательного пространства (`null_space(J)`).""")

code(r'''a = A_CIRCLE                                                           # a = (2, 1)
grad_f = lambda x: x - a                                               # grad f для f = 1/2 |x - a|^2
grad_g = lambda x: x                                                   # grad g для g = 1/2 (|x|^2 - 1)
sols = [fsolve(circle_system, z0, xtol=1e-14) for z0 in ([1, 0.5, 0], [-1, -0.5, 0])]
for z in sols:
    x, lam = z[:2], z[2]
    print(f"x = ({x[0]:+.4f}, {x[1]:+.4f}), lambda = {lam:+.4f}:  |grad f - lambda grad g| = {np.linalg.norm(grad_f(x) - lam * grad_g(x)):.1e}")''')

code(r'''def tangent_part(x):
    """Касательная часть grad f: Z^T grad f, где Z — базис ker J (здесь J = grad g^T — одна строка)."""
    J = grad_g(x).reshape(1, -1)                                       # якобиан 1 x 2
    Z = null_space(J)                                                  # 2 x 1 — базис касательного пространства
    return (Z.T @ grad_f(x)).item()

for name, x in (("минимум", sols[0][:2]), ("максимум", sols[1][:2]), ("точка (0.5, 0.866)", np.array([0.5, np.sqrt(3) / 2]))):
    print(f"{name:18s}: касательная часть grad f = {tangent_part(x):+.4f}")''')

md(r"""**Два ограничения.** Окружность в пространстве — пересечение сферы $\tfrac12(\lVert x\rVert^2-1)=0$ и плоскости $x_3=0$; точка $a=(2,1,1)$. Якобиан $2\times3$, касательное пространство — прямая. Ожидаем $\lambda=(-1.236,\ -1)$.""")

code(r'''xs, gf, J, lam, T = sphere_plane()                                   # окружность = сфера ∩ плоскость, a = (2, 1, 1)
print(f"x* = {xs.round(3)},  grad f = {gf.round(3)}")
print(f"J =\n{J.round(3)}\nrank J = {np.linalg.matrix_rank(J)} — LICQ выполнено")
print(f"lambda = {lam.round(3)}   (в конспекте -1.236 и -1);  |grad f - J^T lambda| = {np.linalg.norm(gf - J.T @ lam):.1e}")
print(f"базис T = {T.ravel().round(3)} — касательная к окружности;  |J T| = {np.abs(J @ T).max():.1e}")''')

code(r'''fig, ax = plt.subplots(figsize=(6, 4.2))
unit_circle(ax)                                                         # g = 0
level_circles(ax, a, [0.5 * np.sum((z[:2] - a) ** 2) for z in sols] + [0.2, 2.0], color=BLUE)   # линии уровня f, две — касаются
for z, col in zip(sols, (RED, VIOLET)):
    x = z[:2]
    ax.annotate("", x + 0.35 * grad_f(x), x, arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))       # grad f
    ax.annotate("", x + 0.35 * grad_g(x), x, arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=2))      # grad g
    ax.plot(*x, "o", color=col, ms=7); ax.annotate(f"$\\lambda={z[2]:+.3f}$", x, (8, -14), textcoords="offset points", color=col)
ax.plot(*a, "o", color=INK); ax.set_aspect("equal"); ax.set_xlim(-2, 3.4); ax.set_ylim(-1.8, 2.4)
ax.set_title("Две точки касания: $\\nabla f$ (красный) $\\parallel\\nabla g$ (зелёный)"); plt.show()''')

# ============================================================ (b)
md(r"""## (b) Когда множителей нет (раздел 3)

**Что проверяем.** Клюв $x_2^3=x_1^2$ при $f=x_2$: минимум в нуле, но там $\nabla g=0$. Число $\hat\lambda=\nabla f^\top\nabla g/\lVert\nabla g\rVert^2$ — лучший множитель в точке кривой. К острию $\hat\lambda$ растёт как $x_1^{-2/3}$, а невязка $\lVert\nabla f-\hat\lambda\nabla g\rVert$ стремится к $1$.""")

code(r'''print(f"{'x1':>8s} {'|grad g|':>10s} {'lambda^':>9s} {'невязка':>9s} {'угол':>7s}")
for x1 in (0.5, 0.1, 0.01, 0.001, 1e-4):
    gg, lam_hat, ang = cusp_grad(x1)                                    # grad g, оценка lambda, угол к grad f
    res = np.linalg.norm(np.array([0.0, 1.0]) - lam_hat * gg)          # |grad f - lambda^ grad g|
    print(f"{x1:8.0e} {np.linalg.norm(gg):10.4f} {lam_hat:9.2f} {res:9.4f} {ang:6.1f}°")
print("в конспекте: при x1 = 0.1, 0.01, 0.001 — lambda^ = 2.3, 14.6, 73 и угол 55°, 72°, 81°")''')

md(r"""**Ограничение, записанное дважды.** Ближайшая к $a=(2,1)$ точка прямой $x_1+x_2=1$: $x^\ast=(1,0)$, $\lambda=-1$. Если записать ограничение дважды, то $\operatorname{rank}J=1$. Подходит любая пара с $\lambda_1+\lambda_2=-1$, и IPOPT возвращает одну из них.""")

code(r'''x = ca.SX.sym("x", 2)
f = 0.5 * ca.sumsqr(x - ca.DM([2, 1]))                                 # f = 1/2 |x - a|^2
g1 = x[0] + x[1] - 1                                                   # одно ограничение
for name, g in (("одно ограничение", g1), ("то же дважды", ca.vertcat(g1, g1)), ("g и 2g", ca.vertcat(g1, 2 * g1))):
    s = ca.nlpsol("s", "ipopt", {"x": x, "f": f, "g": g}, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
    sol = s(x0=[0, 0], lbg=0, ubg=0)                                   # равенство: lbg = ubg = 0
    J = ca.Function("J", [x], [ca.jacobian(g, x)])(sol["x"]).full()    # якобиан в решении
    print(f"{name:18s}: x* = {np.array(sol['x']).ravel()}, rank J = {np.linalg.matrix_rank(J)}, lam (наш знак) = {-np.array(sol['lam_g']).ravel()}")''')

# ============================================================ (c)
md(r"""## (c) Цепь на стержнях в IPOPT (раздел 5)

**Что проверяем.** IPOPT из прямой $z=0$ делает 13 итераций. Три проверки из §5.1: невязка $\lVert\nabla f-J^\top\lambda\rVert\approx10^{-13}$, $\operatorname{rank}J=41$, собственные числа редуцированного гессиана $39\times39$ от 2.2 до 681. Горизонтальная составляющая $2\lambda_j\Delta y_j$ одинакова во всех стержнях.

**Предскажите до запуска:** сколько ньютонов дают вертикальные составляющие на двух опорах вместе? Вес цепи — $40\cdot0.981=39.24$ Н.""")

code(r'''ch = rod_chain()                                                       # задача: v = (y_1..y_40, z_1..z_40), g_j = |p_{j+1} - p_j|^2 - ell^2
v, lam, st = solve_rod_chain(ch)                                       # IPOPT из прямой z = 0
chk = chain_checks(ch, v, lam)                                         # невязка, ранг J, редуцированный гессиан, силы
print(f"итераций {st['iter_count']} (в конспекте 13);  |grad f - J^T lam| = {chk['res']:.1e}")
print(f"rank J = {chk['rank']} из 41 — LICQ выполнено;  Z: {chk['Z'].shape} — базис касательного пространства размерности 80 - 41 = 39")
print(f"редуцированный гессиан Z^T H_L Z: собственные числа от {chk['eig'].min():.2f} до {chk['eig'].max():.1f} — все > 0, строгий минимум")
print(f"множители: концы {lam[0]:.2f}, середина {lam[20]:.2f} (в конспекте -97.9 и -55.8)")''')

code(r'''Fh, Fv = chk["Fh"], chk["Fv"]                                          # горизонтальная и вертикальная составляющие силы стержня j
print(f"горизонтальная составляющая 2 lam_j dy_j: {Fh.min():.4f} … {Fh.max():.4f} Н — одинакова во всех стержнях")
print(f"вертикальные на опорах: {abs(Fv[0]):.2f} + {abs(Fv[-1]):.2f} = {abs(Fv[0]) + abs(Fv[-1]):.2f} Н = вес цепи {ch['w'] * ch['N']:.2f} Н")
y, z = chain_xy(ch, v)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
axes[0].plot(y, z, "o-", color=BLUE, ms=3); axes[0].set_aspect("equal"); axes[0].set_title(f"цепь: нижняя точка z = {z.min():.3f}")
axes[1].plot(chk["F"], "o-", color=ORANGE, ms=3, label="натяжение $F_j=2|\\lambda_j|\\ell$")
axes[1].plot(np.abs(Fh), color=AQUA, label="горизонтальная составляющая"); axes[1].set_xlabel("стержень $j$"); axes[1].set_ylabel("Н"); axes[1].legend(fontsize=9)
plt.show()''')

# ============================================================ (e)
md(r"""## (e) Подробности

**Цена доходности.** Портфель при цели $\rho=10\,\%$: $\lambda_\rho=0.257$ совпадает с численной производной $f^\ast(\rho)$. При $\rho>10.64\,\%$ доля первого актива отрицательна. Тогда в задаче лекции 3 включается запрет $x\ge0$, и цена доходности другая: $0.47$ при $12\,\%$.""")

code(r'''x, f_star, lam_pf = portfolio_eq(0.10)                                 # система ККТ линейна: решается np.linalg.solve
d = 1e-5
fd = (portfolio_eq(0.10 + d)[1] - portfolio_eq(0.10 - d)[1]) / (2 * d)  # центральная разность по rho
print(f"x* = {np.round(x, 4)}, f* = {f_star:.6f}, lambda_rho = {lam_pf[1]:.4f}, df*/drho (разность) = {fd:.4f}")
rhos = np.linspace(0.06, 0.14, 801)
xa = np.array([portfolio_eq(r)[0][0] for r in rhos])                   # доля первого актива как функция цели
print(f"x_1 = 0 при rho = {rhos[np.argmin(np.abs(xa))] * 100:.2f} %")''')

md(r"""**Цена длины цепи.** Удлиняем все стержни. По множителям $\Delta f^\ast\approx\sum_j\lambda_j\,\Delta(\ell^2)$ — это первый порядок. При $+1\,\%$ предсказание $4.873$, пересчёт $4.886$. При $+5\,\%$ — $1.25$ против $1.56$: растёт вклад второго порядка.""")

code(r'''ch = rod_chain(); v, lam, _ = solve_rod_chain(ch); f0 = ch["w"] * chain_xy(ch, v)[1][1:-1].sum()
for k in (1.01, 1.05):
    ch2 = rod_chain(L_total=5.0 * k)                                   # стержни длиннее на 1 % и 5 %
    v2, _, _ = solve_rod_chain(ch2, v0=v)                              # старт — старое решение
    f2 = ch2["w"] * chain_xy(ch2, v2)[1][1:-1].sum()
    pred = f0 + lam.sum() * (ch2["ell"] ** 2 - ch["ell"] ** 2)         # первый порядок: sum lam_j d(ell^2)
    print(f"длина цепи × {k}: f* = {f2:.3f}, предсказание по множителям {pred:.3f}, расхождение {f2 - pred:.3f}")''')

md(r"""**Лог IPOPT из двух стартов.** `inf_pr` $=\lVert g(x_k)\rVert_\infty$ и `inf_du` $=\lVert\nabla_xL(x_k,\lambda_k)\rVert_\infty$ берём из `solver.stats()["iterations"]`. Из опущенной прямой $z=0$ — 13 итераций. Из горизонтальной прямой $z=1$ — 32, и `inf_du` по дороге доходит до $10^{13}$. На горизонтальной прямой все стержни лежат на одной линии. Вертикальный сдвиг не меняет их длин в первом порядке, и якобиан «не видит» это направление.""")

code(r'''N = ch["N"]; t = np.linspace(-2, 2, N + 2)[1:-1]
fig, ax = plt.subplots(figsize=(7, 3.6))
for name, v0, col in (("старт z = 0", np.r_[t, np.zeros(N)], BLUE), ("старт z = 1 (горизонталь)", np.r_[t, np.ones(N)], ORANGE)):
    _, _, st = solve_rod_chain(ch, v0=v0)                              # IPOPT из двух стартов
    it = st["iterations"]
    ax.semilogy(np.maximum(it["inf_pr"], 1e-17), "-", color=col, label=f"{name}: inf_pr, {st['iter_count']} итераций")
    ax.semilogy(np.maximum(it["inf_du"], 1e-17), "--", color=col, label=f"{name}: inf_du")
ax.set_xlabel("итерация"); ax.set_ylabel("невязка"); ax.legend(fontsize=8); plt.show()''')

md(r"""**Знак множителей у солверов.** IPOPT и SciPy `trust-constr` пишут функцию Лагранжа как $f+\lambda^\top g$. Поэтому `lam_g` и `res.v` равны $-\lambda$ в соглашении конспекта $L=f-\lambda^\top g$. Проверка на цепи и на окружности.""")

code(r'''sol_lam_g = -lam                                                       # это то, что вернул IPOPT (solve_rod_chain меняет знак)
J = ca.Function("J", [ch["v"]], [ca.jacobian(ch["g"], ch["v"])])(v).full()
gf = ca.Function("gf", [ch["v"]], [ca.gradient(ch["f"], ch["v"])])(v).full().ravel()
print(f"|grad f - J^T lam|       = {np.linalg.norm(gf - J.T @ lam):.1e}   (наш знак — ноль)")
print(f"|grad f - J^T lam_g|     = {np.linalg.norm(gf - J.T @ sol_lam_g):.1e}   (знак IPOPT без поправки — не ноль)")
print(f"|grad f + J^T lam_g|     = {np.linalg.norm(gf + J.T @ sol_lam_g):.1e}   (L = f + lam_g^T g — ноль)")''')

code(r'''from scipy.optimize import minimize, NonlinearConstraint
a = A_CIRCLE
con = NonlinearConstraint(lambda x: 0.5 * (x @ x - 1), 0, 0, jac=lambda x: x.reshape(1, -1))   # g(x) = 1/2(|x|^2 - 1) = 0
res = minimize(lambda x: 0.5 * np.sum((x - a) ** 2), [1.0, 0.2], jac=lambda x: x - a, method="trust-constr",
               constraints=[con], options={"gtol": 1e-12, "xtol": 1e-14})
print(f"trust-constr: x = {res.x.round(4)}, res.v = {res.v[0][0]:+.4f}  ->  наш lambda = {-res.v[0][0]:+.4f} (в конспекте -1.236)")''')

# ============================================================ шпаргалка
md(r"""## Шпаргалка

| Что | Формула | Где в конспекте |
|---|---|---|
| касательное пространство | $T=\{p:\ Jp=0\}$, размерность $n-m$ | §2.2–2.3 |
| первый порядок | $\nabla_xL=0$ и $g=0$ — $n+m$ уравнений | §2.3 |
| LICQ | $\operatorname{rank}J=m$ | §3.2 |
| второй порядок | $Z^\top\nabla^2_{xx}L\,Z\succ0$ | §4.2 |
| множитель — цена | $df^\ast/db=\lambda^\ast$ | §5.2 |
| знак у солверов | `lam_g` $=-\lambda$, `res.v` $=-\lambda$ | §6 |

**Три вещи, которые легко перепутать.**

1. Кривизну проверяют у $L$, а не у $f$. Слагаемое $-\lambda\nabla^2g$ — кривизна ограничений. У цепи на стержнях $\nabla^2f=0$, а минимум есть.
2. Знак $\lambda$ не говорит, минимум это или максимум. У окружности с $a=(0.3,0.1)$ в минимуме $\lambda>0$. Говорит редуцированный гессиан.
3. У IPOPT и SciPy множители с противоположным знаком: их функция Лагранжа $f+\lambda^\top g$.""")


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
