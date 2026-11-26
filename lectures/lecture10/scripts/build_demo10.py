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

- **(a)** условия ККТ в задаче «такси» (§2): четыре догадки и «обрезка»;
- **(b)** активное множество (§3): портфель руками за две догадки, сверка с IPOPT;
- **(c)** уголок и гипербола (§4): точки задачи «такси» при $\mu_ih_i=\tau$;
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
from scipy.optimize import brentq

BLUE, ORANGE, AQUA, YELLOW, RED, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"
GRAY, INK = "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


''' + methods_block() + r'''

def draw_quadrant(ax, a_max=1.6, b_max=2.6):
    """Допустимая область задачи «такси»: четверть a >= 0, b >= 0."""
    ax.fill_between([0, a_max], 0, b_max, color=AQUA, alpha=0.15)
    ax.axhline(0, color=AQUA, lw=2)
    ax.axvline(0, color=AQUA, lw=2)''')

# ============================================================ (a)
md(r"""## (a) Условия ККТ в задаче «такси»

**Что проверяем.** Из четырёх догадок об активных ограничениях все условия ККТ проходит одна: работает только $b\ge0$. «Обрезка» ответа без ограничений не удовлетворяет стационарности.""")

md(r"""**Что делает ячейка.** Задача: $y\approx at+b$ по четырём поездкам, $x=(a,b)$, $a\ge0$, $b\ge0$. `taxi_guess(A)` решает систему, где ограничения из `A` — равенства: `0` — это $a\ge0$, `1` — это $b\ge0$. Множители — градиент $f$ в найденной точке. Ячейка печатает четыре догадки, потом «обрезку», и рисует их на плоскости $(a,b)$.""")

code(r'''print("grad f(a, b) = (30a + 10b - 29.4, 10a + 4b - 9.4)")                 # суммы по данным из введения
names = {(): "не работает ни одно", (0, 1): "работают оба", (0,): "только a >= 0", (1,): "только b >= 0"}
for A, name in names.items():
    x, mu = taxi_guess(A)                                    # система с ограничениями A как равенствами
    ok_x = x.min() >= -1e-12                                 # допустимость: a >= 0, b >= 0
    ok_mu = all(mu[i] >= -1e-12 for i in A)                  # знак у работающих ограничений
    print(f"{name:20s}: (a, b) = {x.round(3) + 0.0}, mu = {mu.round(3) + 0.0}, x >= 0: {ok_x}, mu >= 0: {ok_mu}")   # + 0.0 убирает «-0.»
x_clip = np.array([taxi_guess(())[0][0], 0.0])               # «обрезка»: b заменили нулём, a оставили
print("обрезка (1.18, 0): grad f =", taxi_grad(x_clip).round(3), f"| f = {taxi_f(x_clip):.3f}, у ответа f = {taxi_f(taxi_guess((1,))[0]):.3f}")
print("в конспекте: ответ (0.98, 0), mu2 = 0.4; оба как равенства mu = (-29.4, -9.4); только a = 0: mu1 = -5.9")

fig, ax = plt.subplots(figsize=(5.2, 4))
draw_quadrant(ax)
for A, col in (((), RED), ((0, 1), VIOLET), ((0,), VIOLET), ((1,), INK)):
    ax.plot(*taxi_guess(A)[0], "*" if A == (1,) else "o", color=col, ms=12 if A == (1,) else 8)
ax.plot(*x_clip, "D", color=ORANGE, ms=8)                     # «обрезка»
ax.set(xlim=(-0.3, 1.6), ylim=(-1.0, 2.6), xlabel="a", ylabel="b", title="Четыре догадки и «обрезка»")
plt.show()''')

md(r"""**Что видим.** Без ограничений $b=-0.60<0$. «Оба как равенства» и «только $a\ge0$» дают отрицательные множители. Ответ — $(0.98,\ 0)$ с $\mu_2=0.4$: у каждого ограничения одно из чисел $h_i$, $\mu_i$ равно нулю. У «обрезки» первая компонента градиента равна $6$ при неработающем $a\ge0$, и $f=0.724$ против $0.124$.""")

md(r"""## (b) Активное множество: портфель руками

**Что проверяем.** Схема «угадай и проверь» из §3.1 находит ответ за две догадки. IPOPT даёт те же доли и те же множители со знаком минус.""")

md(r"""**Что делает ячейка.** `portfolio_active(rho, A)` решает систему Лагранжа, где доли из `A` равны нулю. Ячейка делает две догадки при $\rho=12\,\%$: $\mathcal A=\varnothing$ и $\mathcal A=\{1\}$. После каждой проверяет $x\ge0$ и $\mu\ge0$. Номера активов в коде — с нуля, поэтому $\{1\}$ в конспекте — это `[0]` в коде.""")

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

md(r"""**Что видим.** IPOPT приходит к тем же долям за 9 итераций. Его множители — наши со знаком минус: и у равенств, и у границ. Отрицательный `lam_x` значит активную нижнюю границу.""")

# ============================================================ (c)
md(r"""## (c) Уголок и гипербола: точки при $\mu_ih_i=\tau$

**Что проверяем.** Заменим $\mu_ih_i=0$ на $\mu_ih_i=\tau$ и возьмём ветвь с $\mu_i>0$, $h_i>0$. Точка задачи «такси» тогда лежит внутри четверти $a>0$, $b>0$. При уменьшении $\tau$ она приближается к ответу.""")

md(r"""**Что делает ячейка.** Для $\tau=1,\ 0.1,\ 0.01,\ 0.001$ решает систему $\nabla f=(\mu_1,\mu_2)$, $\mu_1a=\tau$, $\mu_2b=\tau$ функцией `taxi_tau`. Корень она ищет там, где $a>0$ и $b>0$, — это и есть выбор ветви. Ячейка печатает точку, множители и расстояние до ответа и рисует расстояние от $\tau$.""")

code(r'''taus = [1.0, 0.1, 0.01, 0.001]
dist = []
x_star = taxi_guess((1,))[0]                                 # ответ (0.98, 0)
for tau in taus:
    x, mu = taxi_tau(tau)                                    # точка, где mu_i * h_i = tau
    dist.append(np.linalg.norm(x - x_star))
    print(f"tau = {tau:<6}: (a, b) = ({x[0]:.4f}, {x[1]:.4f}), mu = ({mu[0]:.4f}, {mu[1]:.4f}), до ответа {dist[-1]:.4f}")
print("в конспекте: при tau = 0.1 точка (0.924, 0.180), до ответа 0.19; при tau = 0.01 — 0.025; mu -> (0, 0.4)")

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.loglog(taus, dist, "o-", color=BLUE)
ax.set(xlabel="$\\tau$", ylabel="расстояние до ответа", title="Чем меньше $\\tau$, тем ближе")
plt.show()''')

md(r"""**Что видим.** Все точки внутри: $a>0$ и $b>0$. Расстояние до ответа уменьшается примерно в 10 раз при каждом уменьшении $\tau$ в 10 раз. Множители стремятся к $(0,\ 0.4)$ — к множителям ответа. Каждая точка — решение системы из одних уравнений, её находит метод Ньютона. Это идея метода внутренней точки (лекция 12).""")

md(r"""## (e) Подробности: перебор всех активных множеств портфеля

**Что делает ячейка.** Перебирает все активные множества из не больше чем двух границ при $\rho=12\,\%$. Три и четыре границы вместе с двумя равенствами — это 5–6 ограничений на 4 переменные, такие системы вырождены. Для каждого множества печатает, проходят ли проверки $x\ge0$ и $\mu\ge0$.""")

code(r'''for k in range(3):
    for A in combinations(range(4), k):
        x, lam1, lam_rho, mu = portfolio_active(0.12, A)
        ok_x = x.min() >= -1e-12                                  # допустимость
        ok_mu = mu.size == 0 or mu.min() >= -1e-12                # знаки
        mark = "  <- ответ" if ok_x and ok_mu else ""
        print(f"A = {str([i + 1 for i in A]):8s} x >= 0: {str(ok_x):5s} mu >= 0: {str(ok_mu):5s}{mark}")''')

md(r"""**Что видим.** Из 11 множеств обе проверки проходит одно — $\{1\}$. Схема §3.1 нашла его за две догадки, перебор проверяет 11.""")

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
2. На границе может быть и $\mu_i=0$: тогда равны нулю оба числа (упражнение 10.1(б)).
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
