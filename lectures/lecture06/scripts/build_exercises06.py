"""Собирает разбор упражнений лекции 6: exercises06.ipynb (с проверкой кодом) и exercises06.md (кратко).

    uv run python lectures/lecture06/scripts/build_exercises06.py
    uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture06/exercises06.ipynb

Методы и задачи импортируются из l6helpers.py (собирается build_lecture06.py) — тот же код, что в демо.
Условия задач берутся из раздела 9 lecture06.md при сборке и ставятся перед каждым решением
(блок «Условие»), чтобы разбор не расходился с конспектом.
Шесть задач (15.10: было одиннадцать) — разминка, теорема §2.3 для желающих, седло, точка Коши, dogleg,
Левенберг–Марквардт; сопряжённые градиенты убраны вместе с разделом.
"""

import re
from pathlib import Path

import nbformat as nbf

LECTURE = Path(__file__).resolve().parent.parent
OUT_NB = LECTURE / "exercises06.ipynb"
OUT_MD = LECTURE / "exercises06.md"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def statements() -> dict[str, tuple[str, str, str]]:
    """Условия задач из раздела 9 конспекта: номер -> (заголовок, вводная строка с сутью, текст условия)."""
    src = (LECTURE / "lecture06.md").read_text(encoding="utf-8")
    block = src[src.index("## 9. Упражнения"):src.index("## Вопросы для самопроверки")]
    out: dict[str, tuple[str, str, str]] = {}
    for chunk in re.split(r"\n(?=\*\*6\.\d\. )", block)[1:]:
        first, _, rest = chunk.strip().partition("\n")
        m = re.match(r"\*\*6\.(\d)\. (.+?)\*\*\s*(.*)$", first)
        assert m, first
        out[m.group(1)] = (m.group(2).rstrip("."), m.group(3).strip(), rest.strip())
    return out


def put_statements() -> None:
    """Перед каждым решением «## 6.N. …» вставляет условие из конспекта и подпись «Решение»."""
    st = statements()
    seen = set()
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        m = re.match(r"## 6\.(\d)\. ([^\n]+)\n(?:\n\*Суть:[^\n]*\*\n?)?(.*)$", cell.source, re.S)
        if not m:
            continue
        n, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        assert n in st and title == st[n][0], (n, title, st[n][0])
        _, intro, text = st[n]
        cell.source = (f"## 6.{n}. {title}\n\n**Условие.** {intro}\n\n{text}\n\n**Решение.**"
                       + (f"\n\n{body}" if body else ""))
        seen.add(n)
    assert seen == set(st), (sorted(seen), sorted(st))


md(r"""# Разбор упражнений к лекции 6

Условия — раздел 9 конспекта [`lecture06.md`](lecture06.md); краткие ответы без кода — [`exercises06.md`](exercises06.md). Все задачи домашние. **Сначала попробуйте сами**, потом сверяйтесь. Каждый разбор: условие → решение по пунктам → проверка кодом и картинка там, где она помогает.""")

code(r'''%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from l6helpers import *  # методы и задачи лекции (собираются из build_demo06.py)''')

# ---------------------------------------------------------------- 6.0
md(r"""## 6.0. Разминка — счёт руками

*Суть: Армихо, $\rho$ и точка Коши — на бумаге.*

**(а)** $\varphi(\alpha)=(1-2\alpha)^2$, $\varphi(0)=1$, $\varphi'(0)=-4$. Армихо с $c_1=\tfrac12$: $(1-2\alpha)^2\le1-2\alpha\iff4\alpha^2\le2\alpha\iff\alpha\le\tfrac12$. Backtracking: $\alpha=1$ даёт $\varphi=1>-1$ — отказ; $\alpha=\tfrac12$ даёт $\varphi=0\le0$ — принят (это точный минимум).

**(б)** $\rho=-1/4$: шаг отвергается, $\Delta\leftarrow\Delta/4$, подзадача решается заново из той же точки.

**(в)** $g^\top Bg=1$, $\lVert g\rVert=1$. При $\Delta=0.5$: $\tau=\min(1/(0.5\cdot1),1)=1$, $p_C=(-0.5,0)$ — на границе. При $\Delta=2$: $\tau=\min(1/2,1)=\tfrac12$, $p_C=(-1,0)$ — минимум вдоль $-g$ внутри шара.""")

code(r'''phi = lambda a: (1 - 2 * a) ** 2
print("6.0(а): backtracking", [(a, phi(a) <= 1 - 0.5 * a * 4) for a in (1, 0.5)])   # (α, Армихо выполнено?)
g, B = np.array([1.0, 0.0]), np.diag([1.0, 10.0])
print("6.0(в): точка Коши при Δ=0.5", cauchy_point(g, B, 0.5), " при Δ=2", cauchy_point(g, B, 2.0))''')

# ---------------------------------------------------------------- 6.1
md(r"""## 6.1. Теорема о глобальной сходимости

*Суть: каждый шаг съедает кусок $\lVert g\rVert^2$, а съесть можно лишь конечное.*

**(а)** Доказательство целиком — четыре шага (Nocedal & Wright, теорема 3.2); $c_1$ — постоянная Армихо, backtracking делит шаг пополам.

*Шаг 1. Направление.* Собственные числа $B_k^{-1}$ лежат в $[1/M_B,\,1/m_B]$, поэтому для любого $v$: $\lVert v\rVert^2/M_B\le v^\top B_k^{-1}v\le\lVert v\rVert^2/m_B$. Для $p_k=-B_k^{-1}g_k$ отсюда $-g_k^\top p_k=g_k^\top B_k^{-1}g_k\ge\lVert g_k\rVert^2/M_B$ и $\lVert p_k\rVert\le\lVert g_k\rVert/m_B$; поделив квадрат первого неравенства на квадрат второго — условие угла $(-g_k^\top p_k)^2/\lVert p_k\rVert^2\ge(m_B/M_B)^2\lVert g_k\rVert^2$.

*Шаг 2. Лемма о спуске и оценка шага.* $f(x_k+\alpha p_k)-f_k=\int_0^1\nabla f(x_k+t\alpha p_k)^\top\alpha p_k\,dt=\alpha g_k^\top p_k+\int_0^1\big(\nabla f(x_k+t\alpha p_k)-g_k\big)^\top\alpha p_k\,dt$, а последний интеграл по липшицевости не больше $\int_0^1Lt\alpha^2\lVert p_k\rVert^2\,dt=\tfrac L2\alpha^2\lVert p_k\rVert^2$. Значит, Армихо $f(x_k+\alpha p_k)\le f_k+c_1\alpha g_k^\top p_k$ заведомо выполнено, когда $\alpha g_k^\top p_k+\tfrac L2\alpha^2\lVert p_k\rVert^2\le c_1\alpha g_k^\top p_k$, то есть при $\alpha\le\bar\alpha_k=2(1-c_1)(-g_k^\top p_k)/(L\lVert p_k\rVert^2)$. Backtracking от $\alpha=1$ остановится не позже, чем шаг впервые окажется $\le\bar\alpha_k$; предыдущий кандидат был $>\bar\alpha_k$, значит $\alpha_k\ge\min(1,\bar\alpha_k/2)$.

*Шаг 3. Убывание порядка $\lVert g_k\rVert^2$.* По Армихо $f_k-f_{k+1}\ge c_1\alpha_k(-g_k^\top p_k)$. Если $\bar\alpha_k/2\ge1$, то $\alpha_k=1$ и $f_k-f_{k+1}\ge c_1(-g_k^\top p_k)\ge c_1\lVert g_k\rVert^2/M_B$. Иначе $\alpha_k\ge\bar\alpha_k/2$ и $f_k-f_{k+1}\ge\tfrac{c_1}{2}\bar\alpha_k(-g_k^\top p_k)=\tfrac{c_1(1-c_1)}{L}\cdot\tfrac{(-g_k^\top p_k)^2}{\lVert p_k\rVert^2}\ge\tfrac{c_1(1-c_1)}{L}\big(\tfrac{m_B}{M_B}\big)^2\lVert g_k\rVert^2$. В обоих случаях $f_k-f_{k+1}\ge C\lVert g_k\rVert^2$ с $C=\min\big\{c_1/M_B,\ c_1(1-c_1)(m_B/M_B)^2/L\big\}>0$.

*Шаг 4. Суммирование.* $C\sum_{k=0}^N\lVert g_k\rVert^2\le\sum_{k=0}^N(f_k-f_{k+1})=f_0-f_{N+1}\le f_0-\inf f$. Правая часть не зависит от $N$, значит, ряд $\sum_k\lVert g_k\rVert^2$ сходится, а члены сходящегося ряда стремятся к нулю: $\lVert g_k\rVert\to0$. $\blacksquare$

**(б)** Для спуска $B=I$, $m_B=M_B=1$: $\lVert\nabla f\rVert\to0$ из любого старта. На Розенброке это гарантирует сходимость к $(1,1)$ — единственной стационарной точке, — но ничего не говорит о скорости: $13\,756$ итераций.""")

# ---------------------------------------------------------------- 6.2
md(r"""## 6.2. Сломайте Ньютона: сходимость в седло

*Суть: глобальная сходимость — к стационарной точке, а не к минимуму.*""")

code(r'''TRAP = (0.58, 2.15)
x, k, _ = newton_ls(him, him_grad, him_hess, TRAP)                   # (а) сдвиг tau I только когда шаг не спуск
print(f"(а) сдвиг только при подъёме: {k} итераций → {x.round(4)}, собственные числа гессиана {np.linalg.eigvalsh(him_hess(x)).round(2)} — седло")
xa, ka, _ = newton_ls(him, him_grad, him_hess, TRAP, shift="always")  # (б) сдвиг всякий раз, когда lambda_min <= 1e-3
print(f"(б) сдвиг «всегда»: {ka} итераций → {xa.round(4)}, собственные числа {np.linalg.eigvalsh(him_hess(xa)).round(2)} — минимум")''')

md(r"""**(б)** При сдвиге «всегда, когда $\lambda_{\min}\le10^{-3}$» матрица $B_k\succ0$ на каждом шаге. Рядом с седлом компонента шага вдоль собственного вектора с отрицательным $\lambda$ направлена *от* седла (вдоль него функция убывает), поэтому седло отталкивает. Без сдвига направление Ньютона и так спуск — и метод честно идёт в седловую точку модели.""")

md(r"""**Картинка.** Оба пути из одной точки: со сдвигом только при подъёме метод честно идёт в седло, со сдвигом «всегда» — в минимум $(3,2)$.""")

code(r"""x_a, _, path_a = newton_ls(him, him_grad, him_hess, TRAP)                   # сдвиг tau I только при подъёме
x_b, _, path_b = newton_ls(him, him_grad, him_hess, TRAP, shift="always")    # сдвиг всегда, когда lambda_min <= 1e-3
fig, ax = plt.subplots(figsize=(6.5, 5))
him_contour(ax, lim=4.5)                                                       # линии уровня Химмельблау
ax.plot(path_a[:, 0], path_a[:, 1], "o-", color=ORANGE, ms=5, label="сдвиг только при подъёме → седло")
ax.plot(path_b[:, 0], path_b[:, 1], "s-", color=BLUE, ms=5, label="сдвиг всегда → минимум")
ax.plot(*TRAP, "o", color=INK, ms=8, zorder=8)                                 # старт
ax.plot(*x_a, "X", color=ORANGE, mec=INK, ms=11, zorder=8)                     # седло
ax.plot(*x_b, "*", color=RED, ms=14, zorder=8)                                 # минимум
ax.set(xlim=(-1.5, 4.5), ylim=(-0.5, 4.5), title="Химмельблау из (0.58, 2.15)")
ax.legend(loc="lower right")
plt.show()""")

# ---------------------------------------------------------------- 6.3
md(r"""## 6.3. Точка Коши — спуск в маскировке

*Суть: точка Коши даёт сходимость, но скорость спуска.*

**(а)** $m(-tg)=f-t\lVert g\rVert^2+\tfrac12t^2g^\top Bg$ при $0\le t\le\Delta/\lVert g\rVert$. Если $g^\top Bg\le0$, функция убывает — берём границу ($\tau=1$). Иначе минимум параболы $t^\ast=\lVert g\rVert^2/g^\top Bg$ — обрезаем границей: $\tau=\min\big(\lVert g\rVert^3/(\Delta\,g^\top Bg),\,1\big)$.

**(б)** Точка Коши — это спуск с точным шагом (Л4 §4.4), обрезанным шаром, — та же линейная скорость $\sim\kappa$.""")

code(r'''for step in ["cauchy", "dogleg"]:
    x, k, _, h = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step=step)
    print(f"{step}: {k} итераций")
x, k, _ = gd(rosen, rosen_grad, X0_ROSEN)
print(f"спуск с backtracking (Л4): {k} итераций")''')

md(r"""**Картинка.** Норма градиента по итерациям: у точки Коши та же линейная скорость, что у спуска, у dogleg — ньютоновская.""")

code(r"""fig, ax = plt.subplots(figsize=(7, 4))
for step, col in [("cauchy", ORANGE), ("dogleg", BLUE)]:
    _, k, path, _ = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step=step)
    ax.loglog(np.arange(1, len(path) + 1), [np.linalg.norm(rosen_grad(p)) for p in path], color=col, lw=1.8, label=f"{step}: {k}")
_, k, path = gd(rosen, rosen_grad, X0_ROSEN)
ax.loglog(np.arange(1, len(path) + 1), [np.linalg.norm(rosen_grad(p)) for p in path], color=GRAY, lw=1.8, label=f"спуск с backtracking: {k}")
ax.axhline(1e-6, color=GRAY, ls=":", lw=1)
ax.set(xlabel="итерация", ylabel="$\\|\\nabla f\\|$", title="точка Коши идёт со скоростью спуска")
ax.legend()
plt.show()""")

# ---------------------------------------------------------------- 6.4
md(r"""## 6.4. Dogleg руками

*Суть: ломаная Пауэлла — компромисс между точкой Коши и точкой Ньютона.*

$g=(4,2)$, $B=\operatorname{diag}(2,1)$: точка Ньютона $p_B=-B^{-1}g=(-2,-2)$, $\lVert p_B\rVert=2.83$; минимум вдоль $-g$: $p_U=-\frac{g^\top g}{g^\top Bg}g=-\frac{20}{36}(4,2)=(-2.22,-1.11)$, $\lVert p_U\rVert=2.48$.

- $\Delta=1<\lVert p_U\rVert$: первое колено, $p=-\Delta\,g/\lVert g\rVert=(-0.89,-0.45)$.
- $\Delta=2.6$: между $\lVert p_U\rVert$ и $\lVert p_B\rVert$ — второе колено, $p=p_U+t(p_B-p_U)$ с $\lVert p\rVert=2.6$: $t\approx0.43$, $p\approx(-2.13,-1.50)$.
- $\Delta=3>\lVert p_B\rVert$: шаг Ньютона $(-2,-2)$ — так при любом $\Delta\ge2.83$.""")

code(r'''g, B = np.array([4.0, 2.0]), np.diag([2.0, 1.0])
for D in (1.0, 2.6, 3.0):
    p = dogleg(g, B, D)
    print(f"Δ = {D}: dogleg {p.round(3)}, |p| = {np.linalg.norm(p):.3f}, модель {g @ p + 0.5 * p @ B @ p:.3f}")''')

md(r"""**Картинка.** Эллипсы модели, ломаная и три круга: при $\Delta=1$ шаг на первом звене, при $\Delta=2.6$ — на втором, при $\Delta=3$ — точка Ньютона.""")

code(r"""pB = -np.linalg.solve(B, g); pU = -(g @ g) / (g @ B @ g) * g              # точка Ньютона и минимум вдоль -g
P1, P2 = np.meshgrid(np.linspace(-3.5, 1.5, 300), np.linspace(-3.5, 1.5, 300))
M = g[0] * P1 + g[1] * P2 + 0.5 * (B[0, 0] * P1 ** 2 + B[1, 1] * P2 ** 2)   # модель m(p)
fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(P1, P2, M, levels=np.sort(0.5 * g @ pB + np.array([0.2, 0.8, 2, 4, 7, 11])), colors=GRAY, linewidths=0.7)
ax.plot([0, pU[0], pB[0]], [0, pU[1], pB[1]], "--", color=ORANGE, lw=2, label="ломаная dogleg")
for D, col in [(1.0, VIOLET), (2.6, BLUE), (3.0, AQUA)]:
    ax.add_patch(plt.Circle((0, 0), D, fill=False, color=col, lw=1.2))
    p = dogleg(g, B, D)
    ax.plot(*p, "s", color=col, mec=INK, ms=9, label=f"Δ = {D:g}: шаг ({p[0]:.2f}, {p[1]:.2f})")
ax.plot(*pU, "D", color=YELLOW, mec=INK, ms=9, label="минимум вдоль −g")
ax.plot(*pB, "o", color=RED, mec=INK, ms=9, label="точка Ньютона")
ax.plot(0, 0, "o", color=INK, ms=7)
ax.set(xlim=(-3.5, 1.5), ylim=(-3.5, 1.5), aspect="equal", xlabel="$p_1$", ylabel="$p_2$")
ax.legend(loc="upper left", fontsize=9)
plt.show()""")

# ---------------------------------------------------------------- 6.5
md(r"""## 6.5. Левенберг–Марквардт = доверительная область

*Суть: $\lambda$ и $\Delta$ — две ручки одного механизма.*

$p(\lambda)=-\big(\tfrac1{1+\lambda},\ \tfrac1{100+\lambda}\big)$ — длины ниже. Каждому $\lambda$ соответствует $\Delta=\lVert p(\lambda)\rVert$, при котором этот шаг — точное решение подзадачи (условия Moré–Sorensen). Длина монотонно убывает: $\frac{d}{d\lambda}\lVert p(\lambda)\rVert^2=-2\sum_i\frac{g_i^2}{(\lambda_i+\lambda)^3}<0$, поэтому «уменьшить $\Delta$» и «увеличить $\lambda$» — одно и то же.""")

code(r'''JTJ, JTr = np.diag([1.0, 100.0]), np.array([1.0, 1.0])
for lam in [0, 1, 10, 100]:
    p = -np.linalg.solve(JTJ + lam * np.eye(2), JTr)        # шаг Левенберга–Марквардта
    D = np.linalg.norm(p)
    print(f"λ = {lam:3d}: p = {p.round(4)}, Δ = |p| = {D:.4f}; точный шаг подзадачи при этом Δ: {tr_exact(JTr, JTJ, D).round(4)}")''')

md(r"""**Картинка.** Длина шага Левенберга–Марквардта как функция $\lambda$: монотонно убывает, поэтому каждому $\Delta$ отвечает ровно один $\lambda$.""")

code(r"""lam = np.geomspace(1e-2, 1e3, 200)
norms = [np.linalg.norm(np.linalg.solve(JTJ + l * np.eye(2), JTr)) for l in lam]
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(lam, norms, color=BLUE, lw=2)
for l in [1, 10, 100]:
    D = np.linalg.norm(np.linalg.solve(JTJ + l * np.eye(2), JTr))
    ax.plot(l, D, "o", color=RED, mec=INK, ms=8)
    ax.annotate(f"λ = {l}: Δ = {D:.3f}", (l, D), (l * 1.4, D * 1.4), fontsize=10)
ax.set(xlabel="λ", ylabel="длина шага = радиус Δ", title="больше λ — короче шаг — меньше Δ")
plt.show()""")

MD_TEXT = r"""# Упражнения к лекции 6 — краткие ответы

Разбор с проверкой кодом — [`exercises06.ipynb`](exercises06.ipynb). Условия — раздел 9 [конспекта](lecture06.md).

**6.0.** (а) Армихо при $\alpha\le\tfrac12$; backtracking: $1$ — отказ, $\tfrac12$ — принят. (б) $\rho=-0.25$: отвергнуть, $\Delta/4$. (в) $\Delta=0.5$: $p_C=(-0.5,0)$; $\Delta=2$: $p_C=(-1,0)$.

**6.1.** (а) $\alpha_k\ge\min(1,\bar\alpha_k/2)$, $\bar\alpha_k=2(1-c_1)\lvert g_k^\top p_k\rvert/(L\lVert p_k\rVert^2)$; $f_k-f_{k+1}\ge C\lVert g_k\rVert^2$, $C=\min\{c_1/M_B,\ c_1(1-c_1)(m_B/M_B)^2/L\}$. (б) Спуск сходится к $(1,1)$ из любого старта; скорости теорема не даёт — $13\,756$ итераций.

**6.2.** (а) седло $(0.087,2.884)$, $\lambda=(-31.7,\ 75.5)$, $5$ итераций. (б) со сдвигом «всегда» — в минимум $(3,2)$ за $6$ итераций: рядом с седлом сдвинутый шаг направлен от него.

**6.3.** (а) $\tau=\min(\lVert g\rVert^3/(\Delta g^\top Bg),1)$, при $g^\top Bg\le0$ — $1$. (б) $14\,110$ против $24$ у dogleg и $13\,756$ у спуска: точка Коши — спуск с точным шагом.

**6.4.** $p_B=(-2,-2)$, $p_U=(-2.22,-1.11)$; $\Delta=1$: $(-0.89,-0.45)$; $\Delta=2.6$: $\approx(-2.13,-1.50)$; $\Delta=3$: $(-2,-2)$ — точка Ньютона при любом $\Delta\ge\lVert p_B\rVert=2.83$.

**6.5.** $\lambda=0,1,10,100$: $\lVert p\rVert=1.000,\ 0.500,\ 0.091,\ 0.011$; длина монотонно убывает по $\lambda$, поэтому каждому $\Delta$ отвечает свой $\lambda$.
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
