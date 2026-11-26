"""Собирает пошаговый ноутбук demo10.ipynb.

    uv run --group casadi python lectures/lecture10/scripts/build_demo10.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture10/demo10.ipynb

Ноутбук — «урок» по лекции 10. У каждой ячейки с кодом две подписи: «Что делает ячейка» перед ней и «Что видим»
после неё. Задачи берутся из make_figures.py дословно (блок между метками «методы» и «числа конспекта»),
поэтому числа ноутбука и картинок совпадают. Вторая ячейка (импорты, палитра, задачи) становится
модулем l10helpers.py при сборке конспекта-ноутбука (build_lecture10.py). Нужен CasADi: в Colab первая ячейка ставит его сама.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "demo10.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def methods_block() -> str:
    """Задачи лекции из make_figures.py — один код на лекцию."""
    src = (HERE / "make_figures.py").read_text(encoding="utf-8")
    start = src.index("# ---------------------------------------------------------------- методы")
    stop = src.index("# ---------------------------------------------------------------- числа конспекта")
    return src[start:stop].rstrip() + "\n"


# ============================================================ шапка
md(r"""# Демонстрации к лекции 10: условия ККТ

Ноутбук — для самостоятельной проверки после пары. Всё, что на лекции было формулой или картинкой, здесь считается кодом. У каждой ячейки с кодом две подписи: **«Что делает ячейка»** перед ней и **«Что видим»** после неё. Числа конспекта печатаются рядом с ожидаемыми.

- **(a)** условия ККТ в задаче «парк» (§2): оба дома и самая дальняя точка;
- **(b)** активное множество (§3): портфель руками за две догадки, сверка с IPOPT;
- **(c)** уголок и гипербола (§4): точки задачи «парк» при $\mu h=\tau$;
- **(e)** подробности: перебор всех активных множеств портфеля.

Конспект: [`lecture10.md`](lecture10.md). Запуск: `uv run --group casadi jupyter lab` в корне репозитория. В Colab первая ячейка ставит CasADi сама.""")

code(r'''%matplotlib inline

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

from itertools import combinations

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import brentq

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


''' + methods_block() + r'''

def draw_park(ax):
    """Парк — круг радиуса 1 с центром в начале координат."""
    ax.add_patch(Circle((0, 0), 1, facecolor=AQUA, alpha=0.18, edgecolor=AQUA, lw=2))
    ax.set_aspect("equal")''')

# ============================================================ (a)
md(r"""## (a) Условия ККТ в задаче «парк»

**Что проверяем.** Ответы для двух домов удовлетворяют одной системе ККТ: стационарность, $h\ge0$, $\mu\ge0$, $\mu h=0$. У самой дальней точки границы $\mu<0$ — это не минимум.""")

md(r"""**Что делает ячейка.** Для дома внутри, $a=(0.5,\ 0.3)$, и дома снаружи, $a=(2,\ 1)$, берёт ответ из `park_solution` (формулы §2.1). Печатает четыре условия ККТ. Затем проверяет вторую точку границы $x=-a/\lVert a\rVert$ и рисует оба ответа.""")

code(r'''def kkt_report(a, x, mu):
    a, x = np.asarray(a, float), np.asarray(x, float)
    stat = x - a + 2 * mu * x                       # стационарность: grad f - mu grad h = x - a + 2 mu x
    h = 1 - x @ x                                   # допустимость: h >= 0
    print(f"a = {a}: x = {x.round(3)}, mu = {mu:.3f} | |стационарность| = {np.abs(stat).max():.1e}, "
          f"h = {h:.3f}, mu*h = {mu * h:.1e}")

for a in (HOUSE_IN, HOUSE_OUT):
    x, mu = park_solution(a)                        # ответ и множитель по формулам §2.1
    kkt_report(a, x, mu)
print("в конспекте: дом внутри — x* = a, mu = 0, h = 0.66; дом снаружи — x* = (0.894, 0.447), mu = 0.618")

x_far = -HOUSE_OUT / np.linalg.norm(HOUSE_OUT)      # второе решение системы на границе — самая дальняя точка
mu_far = -(np.linalg.norm(HOUSE_OUT) + 1) / 2
kkt_report(HOUSE_OUT, x_far, mu_far)
print("в конспекте: самая дальняя точка (-0.894, -0.447), mu = -1.618 < 0 — не минимум")

fig, ax = plt.subplots(figsize=(5, 4))
draw_park(ax)
for a in (HOUSE_IN, HOUSE_OUT):
    x, _ = park_solution(a)
    ax.plot(*a, "s", color=ORANGE, ms=9)            # дом
    ax.plot(*x, "o", color=INK, ms=6)               # ближайшая точка парка
    ax.plot([a[0], x[0]], [a[1], x[1]], color=GRAY, lw=1)
ax.plot(*x_far, "x", color=RED, ms=9, mew=2)        # самая дальняя точка границы
ax.set(xlim=(-1.3, 2.4), ylim=(-1.2, 1.4), title="Ответы задачи «парк»")
plt.show()''')

md(r"""**Что видим.** У обоих ответов все четыре условия выполнены: стационарность — до машинной точности, $h\ge0$, $\mu\ge0$, $\mu h=0$. Внутри нулю равен $\mu$, снаружи — $h$. Дальняя точка удовлетворяет стационарности и $\mu h=0$, но $\mu=-1.618<0$: условие знака отсекает её.""")

# ============================================================ (b)
md(r"""## (b) Активное множество: портфель руками

**Что проверяем.** Алгоритм «угадай и проверь» из §3.1 находит ответ за две догадки. IPOPT даёт те же доли и те же множители со знаком минус.""")

md(r"""**Что делает ячейка.** `portfolio_active(rho, A)` решает систему лекции 8, где доли из `A` равны нулю. Ячейка делает две догадки при $\rho=12\,\%$: $\mathcal A=\varnothing$ и $\mathcal A=\{1\}$. После каждой проверяет $x\ge0$ и $\mu\ge0$. Номера активов в коде — с нуля, поэтому $\{1\}$ в конспекте — это `[0]` в коде.""")

code(r'''rho = 0.12
for A in ([], [0]):                                  # догадки: пусто, потом первая доля равна нулю
    x, lam1, lam_rho, mu = portfolio_active(rho, A)
    print(f"A = {[i + 1 for i in A]}: x = {x.round(4)}, lam_rho = {lam_rho:.4f}, mu = {mu.round(4)}")
    print(f"   x >= 0: {bool(x.min() >= -1e-12)},  mu >= 0: {bool(mu.size == 0 or mu.min() >= 0)}")
print("в конспекте: догадка 1 — x1 = -0.2015 < 0; догадка 2 — x = (0, 0.1414, 0.5026, 0.3560), mu1 = 0.0082")

x0 = portfolio_active(rho, [])[0]
x1 = portfolio_active(rho, [0])[0]
fig, ax = plt.subplots(figsize=(6, 3.2))
pos = np.arange(4)
ax.bar(pos - 0.18, x0, width=0.36, color=GRAY, label="догадка 1: без границ")
ax.bar(pos + 0.18, x1, width=0.36, color=BLUE, label="догадка 2: $x_1=0$")
ax.axhline(0, color=INK, lw=0.8)
ax.set(xticks=pos, xticklabels=[f"актив {i + 1}" for i in pos], ylabel="доля", title="Портфель при $\\rho=12\\,\\%$")
ax.legend()
plt.show()''')

md(r"""**Что видим.** Первая догадка даёт $x_1=-0.20$ — проверка $x\ge0$ не проходит. Вторая догадка проходит обе проверки: доли неотрицательны, $\mu_1=0.0082\ge0$. Цена доходности $\lambda_\rho$ выросла с $0.351$ до $0.473$.""")

md(r"""**Что делает ячейка.** Решает ту же задачу IPOPT через CasADi. Равенства идут в `g` с границами `lbg=ubg=0`, неравенства $x\ge0$ — в границы переменных `lbx=0`, `ubx=inf`. Множители равенств IPOPT возвращает в `lam_g`, множители границ — в `lam_x`.""")

code(r'''xs = ca.SX.sym("x", 4)                                         # доли
nlp = {"x": xs,
       "f": 0.5 * ca.mtimes([xs.T, ca.DM(PF_SIGMA), xs]),       # риск
       "g": ca.vertcat(ca.sum1(xs) - 1,                         # весь капитал вложен
                       ca.dot(ca.DM(PF_R), xs) - rho)}          # доходность rho
solver = ca.nlpsol("pf", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
res = solver(x0=np.ones(4) / 4, lbx=0, ubx=ca.inf, lbg=0, ubg=0)
x_ip = np.array(res["x"]).ravel()
print("IPOPT: итераций", solver.stats()["iter_count"], "| x =", x_ip.round(4))
print("lam_g =", np.array(res["lam_g"]).ravel().round(4), "| наши -lambda =", -np.round([-0.0265, 0.4731], 4))
print("lam_x =", np.array(res["lam_x"]).ravel().round(4), "| наши -mu =", -np.round([0.0082, 0, 0, 0], 4))
print("в конспекте: 9 итераций, lam_x = (-0.0082, 0, 0, 0), lam_g = (0.0265, -0.4731)")''')

md(r"""**Что видим.** IPOPT приходит к тем же долям за 9 итераций. Его множители — наши со знаком минус: и у равенств, как в лекции 8, и у границ. Отрицательный `lam_x` значит активную нижнюю границу.""")

# ============================================================ (c)
md(r"""## (c) Уголок и гипербола: точки при $\mu h=\tau$

**Что проверяем.** Если заменить $\mu h=0$ на $\mu h=\tau$, точка задачи «парк» лежит внутри парка. При уменьшении $\tau$ она приближается к ответу.""")

md(r"""**Что делает ячейка.** Для дома $(2,\ 1)$ и $\tau=1,\ 0.1,\ 0.01,\ 0.001$ решает систему $x-a+2\mu x=0$, $\mu h=\tau$ функцией `park_tau`. Печатает $\mu$, $h$ и расстояние до ответа, рисует расстояние от $\tau$.""")

code(r'''taus = [1.0, 0.1, 0.01, 0.001]
dist = []
x_star, mu_star = park_solution(HOUSE_OUT)
for tau in taus:
    x, mu = park_tau(HOUSE_OUT, tau)                # точка, где mu*h = tau
    h = 1 - x @ x
    dist.append(np.linalg.norm(x - x_star))
    print(f"tau = {tau:<6}: x = ({x[0]:.4f}, {x[1]:.4f}), mu = {mu:.4f}, h = {h:.4f}, до ответа {dist[-1]:.4f}")
print("в конспекте: при tau = 0.1 расстояние 0.07, при tau = 0.01 — 0.008; mu -> 0.618")

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.loglog(taus, dist, "o-", color=BLUE)
ax.set(xlabel="$\\tau$", ylabel="расстояние до $x^\\ast$", title="Чем меньше $\\tau$, тем ближе")
plt.show()''')

md(r"""**Что видим.** Все точки внутри парка: $h>0$. Расстояние до ответа уменьшается примерно в 10 раз при каждом уменьшении $\tau$ в 10 раз, а $\mu$ стремится к $0.618$. Каждая точка — решение системы из одних уравнений, её находит метод Ньютона. Это идея метода внутренней точки (лекция 12).""")

# ============================================================ (e)
md(r"""## (e) Подробности: перебор всех активных множеств портфеля

**Что делает ячейка.** Перебирает все активные множества из не больше чем двух границ при $\rho=12\,\%$. Три и четыре границы вместе с двумя равенствами — это 5–6 ограничений на 4 переменные, такие системы вырождены. Для каждого множества печатает, проходят ли проверки $x\ge0$ и $\mu\ge0$.""")

code(r'''for k in range(3):
    for A in combinations(range(4), k):
        x, lam1, lam_rho, mu = portfolio_active(0.12, A)
        ok_x = x.min() >= -1e-12                                  # допустимость
        ok_mu = mu.size == 0 or mu.min() >= -1e-12                # знаки
        mark = "  <- ответ" if ok_x and ok_mu else ""
        print(f"A = {str([i + 1 for i in A]):8s} x >= 0: {str(ok_x):5s} mu >= 0: {str(ok_mu):5s}{mark}")''')

md(r"""**Что видим.** Из 11 множеств обе проверки проходит одно — $\{1\}$. Алгоритм §3.1 нашёл его за две догадки, перебор проверяет 11.""")

# ============================================================ шпаргалка
md(r"""## Шпаргалка

| Что | Формула | Где в конспекте |
|---|---|---|
| условия ККТ | $\nabla f-J_g^\top\lambda-J_h^\top\mu=0$, $g=0$, $h\ge0$, $\mu\ge0$, $\mu_ih_i=0$ | §2.2 |
| активное множество | $\mathcal A(x)=\{i:\ h_i(x)=0\}$ | §2.2 |
| «угадай и проверь» | $h_i\ge0$ вне $\mathcal A$, $\mu_i\ge0$ в $\mathcal A$ | §3.1 |
| сглаживание | $\mu h=\tau$ вместо $\mu h=0$ | §4 |

**Три вещи, которые легко перепутать.**

1. Неактивное ограничение — это $\mu_i=0$, а не «ограничение нарушено». Нарушенное ограничение — это $h_i<0$.
2. На границе может быть и $\mu_i=0$: тогда равны нулю оба числа (упражнение 10.1).
3. Знак множителей у солверов свой: IPOPT возвращает наши $\mu$ и $\lambda$ со знаком минус.""")


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
