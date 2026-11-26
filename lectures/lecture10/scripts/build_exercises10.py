"""Собирает разбор упражнений лекции 10: exercises10.ipynb (с проверкой кодом) и exercises10.md (кратко).

    uv run --group casadi python lectures/lecture10/scripts/build_exercises10.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture10/exercises10.ipynb

Задачи «такси» и «портфель» и ближайшая точка круга импортируются из l10helpers.py (собирается build_lecture10.py) — тот же код, что в демо.
Условия задач берутся из раздела 9 lecture10.md при сборке и ставятся перед каждым решением (блок «Условие»),
чтобы разбор не расходился с конспектом. Пять задач: разминка, ближайшая точка круга, квадрат, портфель при других
доходностях, пример без LICQ.
"""

import re
from pathlib import Path

import nbformat as nbf

LECTURE = Path(__file__).resolve().parent.parent
OUT_NB = LECTURE / "exercises10.ipynb"
OUT_MD = LECTURE / "exercises10.md"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def statements() -> dict[str, tuple[str, str, str]]:
    """Условия задач из раздела 9 конспекта: номер -> (заголовок, вводная строка с сутью, текст условия)."""
    src = (LECTURE / "lecture10.md").read_text(encoding="utf-8")
    block = src[src.index("## 9. Упражнения"):src.index("## Вопросы для самопроверки")]
    out: dict[str, tuple[str, str, str]] = {}
    for chunk in re.split(r"\n(?=\*\*10\.\d\. )", block)[1:]:
        first, _, rest = chunk.strip().partition("\n")
        m = re.match(r"\*\*10\.(\d)\. (.+?)\*\*\s*(.*)$", first)
        assert m, first
        out[m.group(1)] = (m.group(2).rstrip("."), m.group(3).strip(), rest.strip())
    return out


def put_statements() -> None:
    """Перед каждым решением «## 10.N. …» вставляет условие из конспекта и подпись «Решение»."""
    st = statements()
    seen = set()
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        m = re.match(r"## 10\.(\d)\. ([^\n]+)\n?(.*)$", cell.source, re.S)
        if not m:
            continue
        n, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        assert n in st and title == st[n][0], (n, title, st[n][0])
        _, intro, text = st[n]
        cell.source = (f"## 10.{n}. {title}\n\n**Условие.** {intro}\n\n{text}\n\n**Решение.**"
                       + (f"\n\n{body}" if body else ""))
        seen.add(n)
    assert seen == set(st), (sorted(seen), sorted(st))


md(r"""# Разбор упражнений к лекции 10

Условия — раздел 9 конспекта [`lecture10.md`](lecture10.md); краткие ответы без кода — [`exercises10.md`](exercises10.md). Все упражнения — домашние; семинар — у доски, задачи в §8 конспекта. **Сначала попробуйте сами**, потом сверяйтесь. Каждый разбор: условие → решение по шагам → проверка кодом там, где она помогает. Нужен CasADi: `uv run --group casadi jupyter lab`; в Colab первая ячейка ставит его сама.""")

code(r'''%matplotlib inline

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from l10helpers import *  # задачи лекции и park_solution — ближайшая точка круга (собираются из build_demo10.py)''')

# ============================================================ 10.0
md(r"""## 10.0. Разминка — счёт руками

**Шаг 1. Условия ККТ.** Здесь $f=x_1^2+x_2^2$, $h=x_1+x_2-c$, $\nabla f=(2x_1,\ 2x_2)$, $\nabla h=(1,\ 1)$:

$$
2x_1=\mu,\qquad 2x_2=\mu,\qquad x_1+x_2-c\ge0,\qquad \mu\ge0,\qquad \mu\,(x_1+x_2-c)=0 .
$$

**Шаг 2. Симметрия.** Из первых двух уравнений $x_1=x_2=\mu/2$.

**Шаг 3. $c=2$.** Пусть $\mu=0$. Тогда $x=(0,\ 0)$ и $h=-2<0$ — недопустимо. Значит, $\mu>0$ и ограничение активно:

$$
x_1+x_2=\mu=2,\qquad x^\ast=(1,\ 1),\qquad \mu=2\ge0 .
$$

**Шаг 4. $c=-2$.** Пусть $\mu=0$. Тогда $x=(0,\ 0)$ и $h=0+0+2=2>0$ — допустимо. Ограничение неактивно, $\mu=0$, $x^\ast=(0,\ 0)$.""")

# ============================================================ 10.1
md(r"""## 10.1. Ближайшая точка круга

**Шаг 1. Формула ответа.** Стационарность: $\nabla f=\mu\nabla h$, то есть $x-a=-2\mu x$. Отсюда $x=a/(1+2\mu)$ — точка на отрезке от центра круга к точке $a$. Если $\lVert a\rVert\le1$, ответ — сама точка $a$ и $\mu=0$. Иначе условие $\lVert x\rVert=1$ даёт $1+2\mu=\lVert a\rVert$:

$$
\mu=\frac{\lVert a\rVert-1}{2},\qquad x^\ast=\frac{a}{\lVert a\rVert} .
$$

**Шаг 2. (а) $a=(0.5,\ 0.3)$.** $\lVert a\rVert=0.583<1$ — точка внутри. Ответ $x^\ast=a$, $\mu=0$, запас $h=1-0.25-0.09=0.66$. Ограничение не работает.

**Шаг 3. (а) $a=(2,\ 1)$.** $\lVert a\rVert=\sqrt5$, поэтому $\mu=(\sqrt5-1)/2=0.618$ и $x^\ast=(0.894,\ 0.447)$. Ограничение работает, и ответ совпадает с ответом задачи с равенством $h=0$. При одном ограничении иначе и не бывает: оно либо не работает, либо работает как равенство.

**Шаг 4. (б) $a=(0.6,\ 0.8)$.** $\lVert a\rVert=1$ — точка на границе. Ответ — сама точка, $x^\ast=a$. Тогда $h=1-0.36-0.64=0$ и $x-a=0$, откуда $\mu=0$. Условие $\mu h=0$ выполнено: равны нулю оба числа. Ограничение активно, но его множитель равен нулю.""")

code(r'''for a in ((0.5, 0.3), (2.0, 1.0), (0.6, 0.8)):
    x, mu = park_solution(a)                                   # формула ответа из шага 1
    h = 1 - x @ x                                              # запас ограничения
    print(f"a = {a}: x* = {x.round(3)}, mu = {mu:.3f}, h = {h:.3f}, mu*h = {mu * h:.1e}")''')

md(r"""**Что видим.** Внутри $\mu=0$ и $h=0.66$, снаружи $h=0$ и $\mu=0.618$, на границе оба числа равны нулю.""")

# ============================================================ 10.2
md(r"""## 10.2. Квадрат: два активных ограничения

**Шаг 1. Ограничения.** Четыре неравенства: $h_1=x_1\ge0$, $h_2=x_2\ge0$, $h_3=1-x_1\ge0$, $h_4=1-x_2\ge0$. Градиенты: $e_1$, $e_2$, $-e_1$, $-e_2$.

**Шаг 2. $a=(2,\ 2)$.** Угадаем угол $(1,\ 1)$: активны $h_3$ и $h_4$. Стационарность:

$$
x-a=(-1,\ -1)=\mu_3(-1,\ 0)+\mu_4(0,\ -1)\quad\Longrightarrow\quad \mu_3=\mu_4=1\ge0 .
$$

Обе проверки проходят: $x_1=x_2=1\ge0$, оба множителя неотрицательны. Ответ $x^\ast=(1,\ 1)$, $\mathcal A=\{3,\ 4\}$.

**Шаг 3. $a=(2,\ 0.5)$.** Угадаем $\mathcal A=\{3\}$: $x_1=1$, вторая координата свободна. Стационарность по $x_2$: $x_2-0.5=0$, то есть $x_2=0.5$. По $x_1$: $1-2=-\mu_3$, $\mu_3=1\ge0$. Ответ $x^\ast=(1,\ 0.5)$.

**Шаг 4. (б) IPOPT.** Ячейка ниже решает обе задачи с границами `lbx=0`, `ubx=1`.""")

code(r'''x = ca.SX.sym("x", 2)
for a in ((2.0, 2.0), (2.0, 0.5)):
    nlp = {"x": x, "f": 0.5 * ca.sumsqr(x - ca.DM(a))}                  # половина квадрата расстояния до a
    solver = ca.nlpsol("box", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
    res = solver(x0=[0.5, 0.5], lbx=0, ubx=1)                           # квадрат [0, 1]^2 — границы переменных
    print(f"a = {a}: x* = {np.array(res['x']).ravel().round(4)}, lam_x = {np.array(res['lam_x']).ravel().round(4)}")''')

md(r"""**Что видим.** У активной верхней границы `lam_x` $=+1$ — это наш $\mu$ с плюсом. У нижней границы в портфеле `lam_x` был отрицателен. Правило: знак `lam_x` говорит, какая граница активна, модуль — наш множитель.""")

# ============================================================ 10.3
md(r"""## 10.3. Портфель при других доходностях

**Шаг 1. Схема.** Начинаем с $\mathcal A=\varnothing$. Если какая-то доля отрицательна, добавляем в $\mathcal A$ самую отрицательную. Если какой-то $\mu_i<0$, убираем это ограничение из $\mathcal A$. Ячейка ниже делает это и печатает все догадки.""")

code(r'''def guess_and_check(rho, maxit=10):
    A = []                                                              # стартовая догадка: ничего не активно
    for k in range(1, maxit + 1):
        x, lam1, lam_rho, mu = portfolio_active(rho, A)                 # система лекции 8 с x_i = 0 для i из A
        print(f"   догадка {k}: A = {[i + 1 for i in A]}, x = {x.round(4)}, lam_rho = {lam_rho:.4f}, mu = {mu.round(4)}")
        neg = [i for i in range(4) if i not in A and x[i] < -1e-12]     # нарушенные x_i >= 0
        if neg:
            A = sorted(A + [min(neg, key=lambda i: x[i])])              # добавить самую отрицательную долю
            continue
        bad = [i for i, m in zip(A, mu) if m < -1e-12]                  # отрицательные множители
        if bad:
            A.remove(bad[0])
            continue
        return A, lam_rho

for rho in (0.08, 0.14):
    print(f"rho = {rho:.0%}:")
    A, lam_rho = guess_and_check(rho)
    lam_free = portfolio_active(rho, [])[2]                             # цена доходности без запрета
    print(f"   ответ: A = {[i + 1 for i in A]}; lam_rho с запретом {lam_rho:.4f}, без запрета {lam_free:.4f}")''')

md(r"""**Шаг 2. $\rho=8\,\%$.** Первая догадка $\mathcal A=\varnothing$ проходит: все доли положительны. Ограничения $x\ge0$ не работают, и $\lambda_\rho=0.163$ — как без запрета.

**Шаг 3. $\rho=14\,\%$.** Нужны три догадки:

$$
\varnothing\ \xrightarrow{\ x_1=-0.50\ }\ \{1\}\ \xrightarrow{\ x_2=-0.27\ }\ \{1,\ 2\} .
$$

Ответ $x=(0,\ 0,\ 0.25,\ 0.75)$, $\mu=(0.061,\ 0.023)$. Цена доходности $\lambda_\rho=1.215$ против $0.445$ без запрета — почти втрое выше.""")

# ============================================================ 10.4
md(r"""## 10.4. Для желающих: когда множителей нет

**Шаг 1. (а) Решение.** Из двух ограничений $0\le x_2\le(1-x_1)^3$. Это возможно, только если $(1-x_1)^3\ge0$, то есть $x_1\le1$. Минимум $-x_1$ — при наибольшем $x_1$:

$$
x_1=1,\qquad 0\le x_2\le0\quad\Longrightarrow\quad x^\ast=(1,\ 0) .
$$

**Шаг 2. (б) Градиенты в $(1,\ 0)$.** Оба ограничения активны:

$$
\nabla f=(-1,\ 0),\qquad \nabla h_1=\big(-3(1-x_1)^2,\ -1\big)=(0,\ -1),\qquad \nabla h_2=(0,\ 1) .
$$

**Шаг 3. Множителей нет.** Условие $\nabla f=\mu_1\nabla h_1+\mu_2\nabla h_2$ покомпонентно:

$$
-1=\mu_1\cdot0+\mu_2\cdot0,\qquad 0=-\mu_1+\mu_2 .
$$

Первое равенство $-1=0$ не выполняется ни при каких $\mu$.

**Шаг 4. Что нарушено.** LICQ: $\nabla h_1=-\nabla h_2$, градиенты активных ограничений линейно зависимы. Теорема ККТ к этой точке не применима.""")

code(r'''x = ca.SX.sym("x", 2)
nlp = {"x": x, "f": -x[0], "g": ca.vertcat((1 - x[0]) ** 3 - x[1], x[1])}   # оба неравенства в g >= 0
solver = ca.nlpsol("kt", "ipopt", nlp, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
res = solver(x0=[0.5, 0.05], lbg=0, ubg=ca.inf)
st = solver.stats()
print("статус:", st["return_status"], "| итераций:", st["iter_count"])
print("x =", np.array(res["x"]).ravel().round(4), "| lam_g =", np.array(res["lam_g"]).ravel().round(0))''')

md(r"""**Что видим.** IPOPT сообщает об успехе и останавливается рядом с $(1,\ 0)$. Но множители огромны — больше $10^4$, а $x_1\approx1.003$ вместо $1$. Солвер пытается выполнить условие, которое в этой точке не выполняется. Огромные множители — признак нарушенного LICQ, как в лекции 8.""")

MD_TEXT = r"""# Упражнения к лекции 10 — краткие ответы

Разбор с проверкой кодом — [`exercises10.ipynb`](exercises10.ipynb). Условия — раздел 9 [конспекта](lecture10.md).

**10.0.** ККТ: $2x_1=2x_2=\mu$, $x_1+x_2\ge c$, $\mu\ge0$, $\mu(x_1+x_2-c)=0$. При $c=2$ ограничение активно: $x^\ast=(1,\ 1)$, $\mu=2$. При $c=-2$ неактивно: $x^\ast=(0,\ 0)$, $\mu=0$.

**10.1.** (а) $a=(0.5,\ 0.3)$ внутри круга: $x^\ast=a$, $\mu=0$, $h=0.66$ — ограничение не работает. $a=(2,\ 1)$ снаружи: $x^\ast=(0.894,\ 0.447)$, $\mu=(\sqrt5-1)/2=0.618$ — ответ задачи с равенством $h=0$. (б) $x^\ast=a$, $h=0$ и $\mu=0$: ограничение активно, но множитель равен нулю; $\mu h=0$ выполнено.

**10.2.** (а) $a=(2,\ 2)$: $x^\ast=(1,\ 1)$, активны обе верхние границы, $\mu_3=\mu_4=1$. $a=(2,\ 0.5)$: $x^\ast=(1,\ 0.5)$, активна граница $x_1\le1$, $\mu_3=1$. (б) У активной верхней границы `lam_x` $=+1$, у нижней он отрицателен; модуль — наш $\mu$.

**10.3.** $\rho=8\,\%$: одна догадка $\mathcal A=\varnothing$, $\lambda_\rho=0.163$ с запретом и без. $\rho=14\,\%$: три догадки, $\varnothing\to\{1\}\to\{1,\ 2\}$; $x=(0,\ 0,\ 0.25,\ 0.75)$, $\mu=(0.061,\ 0.023)$; $\lambda_\rho=1.215$ против $0.445$ без запрета.

**10.4.** (а) Из $0\le x_2\le(1-x_1)^3$ следует $x_1\le1$, ответ $(1,\ 0)$. (б) $\nabla f=(-1,\ 0)$, $\nabla h_1=(0,\ -1)$, $\nabla h_2=(0,\ 1)$: первая компонента даёт $-1=0$. Нарушено LICQ: $\nabla h_1=-\nabla h_2$. IPOPT останавливается рядом с ответом, $x_1\approx1.003$, с множителями больше $10^4$.
"""


def main() -> None:
    put_statements()
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbf.validate(nb)
    nbf.write(nb, OUT_NB)
    OUT_MD.write_text(MD_TEXT, encoding="utf-8")
    print(f"{OUT_NB}: {len(cells)} ячеек; {OUT_MD}")


if __name__ == "__main__":
    main()
