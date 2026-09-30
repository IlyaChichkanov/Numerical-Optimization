"""Собирает ноутбук с разбором упражнений лекции 4: exercises04.ipynb.

    uv run python lectures/lecture04/build_exercises04.py
    uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture04/exercises04.ipynb

Формат каждого упражнения: условие с пометкой «суть» -> разбор по пунктам
(а), (б), ... -> проверка кодом, по одной идее на ячейку. Краткие ответы
без кода — exercises04.md.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "exercises04.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


# ============================================================ шапка
md(r"""# Лекция 4. Разбор упражнений

Упражнения из раздела 12 конспекта [`lecture04.md`](lecture04.md). Формат одинаковый: условие и его суть → разбор по пунктам → проверка кодом, по одной идее на ячейку.

На семинаре у доски: 12.1, 12.3, 12.4, 12.7, 12.5 — в этом порядке. Дома: 12.2, 12.6, 12.8, 12.9, 12.10. Сначала попробуйте решить сами; краткие ответы без кода — [`exercises04.md`](exercises04.md).

В проверках используются две функции из демо: `gd` — градиентный спуск (постоянный шаг или backtracking) и `newton` — чистый метод Ньютона без линейного поиска.""")

code('''%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

BLUE, ORANGE, AQUA, RED, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8985", "#0b0b0b"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.2, "legend.frameon": False})


def gd(f, grad, x0, alpha=None, tol=1e-6, maxit=100_000, c=1e-4):
    """Градиентный спуск. alpha=None — backtracking: с 1 делим пополам, пока f не
    уменьшится хотя бы на c*alpha*|grad|^2. Иначе — постоянный шаг alpha.
    Возвращает (x, число итераций, путь)."""
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
    """Чистый метод Ньютона без линейного поиска. Возвращает (x, число итераций, путь)."""
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    for k in range(maxit):
        g = grad(x)
        if np.linalg.norm(g) <= tol:
            return x, k, np.array(path)
        x = x - np.linalg.solve(hess(x), g); path.append(x.copy())
    return x, maxit, np.array(path)''')

# ============================================================ 12.0
md(r"""## 12.0. Разминка — счёт руками

> Шесть пунктов на повторение, каждый решается на бумаге за минуту. (а) $\nabla f$ и $\nabla^2f$ для $3x_1^2-2x_1x_2+x_2^2+5x_1$. (б) Два шага спуска на $\tfrac12(x_1^2+9x_2^2)$ из $(2,1)$, $\alpha=0.1$. (в) Собственные числа $\begin{pmatrix}6&-2\\-2&2\end{pmatrix}$ и $\kappa$. (г) Безопасные шаги и $1/L$ для неё. (д) Выпуклы ли $e^{x_1}+x_2^2$ и $x_1x_2$? (е) Стационарные точки $x^3-3x$.
>
> *Суть: держать руки тёплыми — градиенты, гессианы и шаги без компьютера.*

**(а)** По координатам: $\nabla f=(6x_1-2x_2+5,\ -2x_1+2x_2)$, гессиан постоянный: $\nabla^2f=\begin{pmatrix}6&-2\\-2&2\end{pmatrix}$.

**(б)** $\nabla f=(x_1,\ 9x_2)$, шаг даёт $\big(0.9\,x_1,\ 0.1\,x_2\big)$: из $(2,\,1)$ в $(1.8,\ 0.1)$, затем в $(1.62,\ 0.01)$. Координаты сжимаются в $1-\alpha=0.9$ и $1-9\alpha=0.1$ раза за шаг — те самые множители $|1-\alpha\lambda_i|$ из 12.3.

**(в)** След $8$, определитель $6\cdot2-(-2)^2=8$: $\lambda^2-8\lambda+8=0$, $\lambda=4\pm2\sqrt2\approx6.83$ и $1.17$. Число обусловленности $\kappa=\dfrac{4+2\sqrt2}{4-2\sqrt2}=3+2\sqrt2\approx5.8$.

**(г)** $L=\lambda_{\max}=4+2\sqrt2\approx6.83$: безопасны шаги $0<\alpha<2/L\approx0.29$, а лучший гарантированный — $1/L\approx0.15$.

**(д)** $\nabla^2(e^{x_1}+x_2^2)=\operatorname{diag}(e^{x_1},\,2)\succ0$ — выпукла. У $x_1x_2$ гессиан $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ с собственными числами $\pm1$ — индефинитен, не выпукла (седло в нуле).

**(е)** $g'=3x^2-3=0$ даёт $x=\pm1$; $g''=6x$: в $x=1$ положительна — локальный минимум, в $x=-1$ отрицательна — локальный максимум. Глобальных экстремумов нет: $x^3$ уходит в $\pm\infty$.""")

code("""# (б)-(в): проверяем арифметику разминки
A = np.array([[6.0, -2.0], [-2.0, 2.0]])
lam = np.linalg.eigvalsh(A)
print(f"(в) собственные числа {lam.round(3)}; 4 -/+ 2 sqrt(2) = {4 - 2 * np.sqrt(2):.3f}, {4 + 2 * np.sqrt(2):.3f};  kappa = {lam[1] / lam[0]:.3f} = 3 + 2 sqrt(2) = {3 + 2 * np.sqrt(2):.3f}")

x = np.array([2.0, 1.0])
for _ in range(2):
    x = x - 0.1 * np.array([x[0], 9 * x[1]])
    print(f"(б) шаг: x = {x.round(4)}")
assert np.allclose(x, [1.62, 0.01])""")

# ============================================================ 12.1
md(r"""## 12.1. Стационарные точки и седло

> $f(x)=(x_1^2-1)^2+x_2^2$. (а) Найдите все точки с $\nabla f=0$. (б) Классифицируйте по гессиану. (в) Куда придёт спуск из $(0,\,0.5)$? Из $(0.01,\,0.5)$?
>
> *Суть: спуск гарантирует только $\nabla f=0$, и седло его ловит.*

**(а)** $\nabla f=\big(4x_1(x_1^2-1),\ 2x_2\big)$. Вторая компонента — нуль только при $x_2=0$; первая — при $x_1\in\{0,\,1,\,-1\}$. Три точки: $(0,0)$, $(1,0)$, $(-1,0)$.

**(б)** Смешанных производных нет: $\nabla^2f=\operatorname{diag}(12x_1^2-4,\ 2)$.

- $(0,0)$: $\operatorname{diag}(-4,\,2)$, знаки разные — **седло**.
- $(\pm1,0)$: $\operatorname{diag}(8,\,2)\succ0$ — **строгие минимумы**; там $f=0$, а $f\ge0$ всюду, так что оба глобальные.
- Максимумов нет: $f$ не ограничена сверху.

**(в) Старт $(0,\,0.5)$.** При $x_1=0$ первая компонента градиента — нуль, и шаг её не меняет: вся траектория живёт на прямой $x_1=0$. На этой прямой $f=1+x_2^2$ — парабола с минимумом в $x_2=0$, и метод сходится к **седлу** $(0,0)$. Критерий $\|\nabla f\|\le\varepsilon$ подвоха не заметит: в седле градиент тоже нулевой.

**Старт $(0.01,\,0.5)$.** Теперь покоординатно $x_1\leftarrow x_1\big(1+4\alpha(1-x_1^2)\big)$: при $|x_1|<1$ скобка больше единицы, и модуль $x_1$ **растёт**. А $x_2\leftarrow(1-2\alpha)x_2$ затухает. Точка уходит от седла и приходит в минимум $(1,0)$.

**Мораль.** Спуск сходится к стационарной точке, не обязательно к минимуму, — но к седлу лишь со множества стартов меры нуль (здесь — прямая $x_1=0$): вдоль направления с $\lambda<0$ ошибка растёт, и любое возмущение уводит. «Мера нуль» держится и в float: $4\cdot0\cdot(0-1)$ — нуль точно, без ошибок округления.""")

code('''f1 = lambda x: (x[0] ** 2 - 1) ** 2 + x[1] ** 2
g1 = lambda x: np.array([4 * x[0] * (x[0] ** 2 - 1), 2 * x[1]])
h1 = lambda x: np.diag([12 * x[0] ** 2 - 4, 2.0])

# (а)-(б): три стационарные точки и их классификация по собственным числам
for p in ([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]):
    lam = np.linalg.eigvalsh(h1(p))
    kind = "минимум" if lam.min() > 0 else ("максимум" if lam.max() < 0 else "седло")
    print(f"x = {p}: grad = {g1(p)}, lambda = {lam} -> {kind}")''')

code('''# (в): два старта на расстоянии 0.01 — две судьбы
xa, ka, pa = gd(f1, g1, [0.0, 0.5])
xb, kb, pb = gd(f1, g1, [0.01, 0.5])
print(f"из (0, 0.5):    пришли в {xa.round(6)} за {ka} итераций — седло")
print(f"из (0.01, 0.5): пришли в {xb.round(6)} за {kb} итераций — минимум")
assert np.allclose(xa, [0, 0], atol=1e-6) and np.allclose(xb, [1, 0], atol=1e-5)''')

code('''xx, yy = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-0.8, 0.8, 200))
zz = (xx ** 2 - 1) ** 2 + yy ** 2
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.contour(xx, yy, zz, levels=np.r_[0.02, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5], colors=GRAY, linewidths=0.8)
ax.plot(pa[:, 0], pa[:, 1], "-o", ms=4, color=BLUE, label="из $(0,\\,0.5)$: к седлу")
ax.plot(pb[:, 0], pb[:, 1], "-o", ms=4, color=ORANGE, label="из $(0.01,\\,0.5)$: к минимуму")
ax.plot([1, -1], [0, 0], "*", ms=13, color=RED, mec="white", ls="none", label="минимумы")
ax.plot([0], [0], "s", ms=8, color=INK, ls="none", label="седло")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8); ax.set_title("12.1: два старта — две судьбы", fontsize=10)
plt.show()''')

# ============================================================ 12.2
md(r"""## 12.2. Зазор между необходимым и достаточным

> (а) $x^4$, $x^3$, $-x^4$ в нуле: что говорят условия и что на самом деле? (б) $f=x_1^2+x_2^4$ и $g=x_1^2-x_2^4$: гессианы в нуле одинаковы. А судьбы? (в) Пеано, $f=(x_2-x_1^2)(x_2-2x_1^2)$: проверьте $\nabla f(0)=0$, $\nabla^2f(0)=\operatorname{diag}(0,2)$. (г) Покажите: на каждой прямой через нуль — строгий минимум. (д) Найдите $f<0$ сколь угодно близко к нулю.
>
> *Суть: при нулевом собственном числе гессиана условия второго порядка молчат — решают старшие члены.*

**(а)** У всех трёх $f'(0)=f''(0)=0$: необходимые условия выполнены, достаточное ($f''>0$) — нет. На деле $x^4$ — минимум, $x^3$ — перегиб (не экстремум), $-x^4$ — максимум: решают члены выше второго порядка, которых условия не видят.

**(б)** Оба гессиана в нуле равны $\operatorname{diag}(2,\,0)\succeq0$. Но $f=x_1^2+x_2^4\ge0$ — минимум, а $g(0,x_2)=-x_2^4<0$ — **седло**: вдоль вырожденного направления ($\lambda=0$) всё решает четвёртая степень, и она у них разная.

**(в)** Раскроем скобки: $f=x_2^2-3x_1^2x_2+2x_1^4$. Тогда $\nabla f=\big(-6x_1x_2+8x_1^3,\ 2x_2-3x_1^2\big)$ — в нуле нуль. Вторые производные в нуле дают $\nabla^2f(0)=\operatorname{diag}(0,\,2)\succeq0$: необходимые условия выполнены, достаточное — нет, направление $e_1$ вырождено.

**(г)** Прямая $x_2=mx_1$: $f=x_1^2(m-x_1)(m-2x_1)$. При $m\ne0$ и малых $x_1$ обе скобки близки к $m$, так что $f\approx m^2x_1^2>0$. На оси $x_1$ ($m=0$): $f=2x_1^4>0$; на оси $x_2$: $f=x_2^2>0$. На **каждой** прямой в нуле строгий локальный минимум.

**(д)** $f<0$ ровно там, где скобки разных знаков: между параболами $x_1^2<x_2<2x_1^2$. На средней параболе $x_2=\tfrac32x_1^2$ выходит $f=-\tfrac14x_1^4<0$ при любом $x_1\ne0$ — сколь угодно близко к нулю. Значит, нуль — **не минимум**, хотя на каждой прямой выглядел минимумом: к точке можно подходить и по кривым. Гарантию даёт только $\nabla^2f(0)\succ0$ — равномерная по направлениям оценка $\tfrac{\lambda_{\min}}2\|d\|^2$, которой здесь нет.""")

code('''fp = lambda x1, x2: (x2 - x1 ** 2) * (x2 - 2 * x1 ** 2)

# (г): вдоль прямых через нуль f > 0 при малых t
t = np.r_[-np.logspace(-3, -1, 7), np.logspace(-3, -1, 7)]
worst = min(fp(t * np.cos(th), t * np.sin(th)).min() for th in np.linspace(0, np.pi, 12)[:-1])
print(f"(г) минимум f вдоль 11 прямых через нуль при 0 < |t| <= 0.1: {worst:.2e} > 0")
assert worst > 0

# (д): на параболе x2 = 1.5 x1^2 функция отрицательна сколь угодно близко к нулю
for x1 in (0.1, 0.01, 0.001):
    print(f"(д) f({x1}, 1.5*{x1}^2) = {fp(x1, 1.5 * x1 ** 2):.3e}  (= -x1^4/4 = {-x1 ** 4 / 4:.3e})")
assert all(fp(x1, 1.5 * x1 ** 2) < 0 for x1 in (0.1, 0.01, 0.001))''')

code('''xx, yy = np.meshgrid(np.linspace(-0.6, 0.6, 400), np.linspace(-0.15, 0.6, 400))
zz = (yy - xx ** 2) * (yy - 2 * xx ** 2)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.contourf(xx, yy, zz, levels=[zz.min(), 0], colors=[RED], alpha=0.25)
ax.contour(xx, yy, zz, levels=np.r_[0.003, 0.01, 0.03, 0.08, 0.15], colors=GRAY, linewidths=0.8)
xs = np.linspace(-0.6, 0.6, 200)
ax.plot(xs, xs ** 2, color=RED, lw=1.2); ax.plot(xs, 2 * xs ** 2, color=RED, lw=1.2)
ax.plot(xs, 1.5 * xs ** 2, "--", color=INK, lw=1, label="$x_2=\\\\frac{3}{2}x_1^2$: $f=-x_1^4/4<0$")
for th in np.linspace(0, np.pi, 7)[:-1]:
    ax.plot([-0.6 * np.cos(th), 0.6 * np.cos(th)], [-0.6 * np.sin(th), 0.6 * np.sin(th)], color=BLUE, lw=0.6, alpha=0.6)
ax.plot(0, 0, "o", color=INK, ms=5)
ax.text(0.33, 0.13, "$f<0$", color=RED, fontsize=10)
ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.15, 0.6)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("12.2: на каждой прямой (синие) минимум, но между параболами $f<0$", fontsize=9.5)
plt.show()''')

# ============================================================ 12.3
md(r"""## 12.3. Спуск на вытянутой квадратичной

> $f(x)=\tfrac12(x_1^2+\kappa x_2^2)$, $\kappa\ge1$, постоянный шаг $\alpha$. (а) Итерация покоординатно. (б) При каких $\alpha$ сходимость из любой точки? (в) Нарисуйте $|1-\alpha|$ и $|1-\alpha\kappa|$; какой шаг лучший? (г) $\kappa=50$: итераций на цифру при $\alpha=1/\kappa$ и при лучшем шаге; во сколько сжимается $f$? (д) Метры → километры: гессиан и $\kappa$? (е) Шаг с $D^{-1}$ — это спуск в метрах; $D=\nabla^2f$, $\alpha=1$ — Ньютон.
>
> *Суть: вся скорость спуска видна в двух числах $|1-\alpha|$ и $|1-\alpha\kappa|$.*

**(а)** $\nabla f=(x_1,\ \kappa x_2)$, и шаг расщепляется по координатам:
$$x_1\leftarrow(1-\alpha)\,x_1,\qquad x_2\leftarrow(1-\alpha\kappa)\,x_2.$$
Две независимые геометрические прогрессии. Минимум в нуле, так что $x$ — это и есть ошибка.

**(б)** Обе прогрессии затухают $\iff$ $|1-\alpha|<1$ и $|1-\alpha\kappa|<1$. Второе условие строже: $0<\alpha<2/\kappa$. При $\alpha>2/\kappa$ координата $x_2$ раскачивается с растущей амплитудой — расходимость из любого старта.

**(в)** Ошибка сжимается за шаг не хуже, чем в $\rho(\alpha)=\max\big(|1-\alpha|,\ |1-\alpha\kappa|\big)$ раз. На картинке ниже это две ломаные: пологая с нулём в $\alpha=1$ и крутая с нулём в $\alpha=1/\kappa$. Максимум двух минимален там, где они пересекаются: $1-\alpha=\alpha\kappa-1$, откуда
$$\alpha^\ast=\frac2{\kappa+1},\qquad \rho(\alpha^\ast)=\frac{\kappa-1}{\kappa+1}.$$

**(г)** Итераций на одну верную цифру — $n$ из $\rho^n=0.1$, то есть $n=\ln10/(-\ln\rho)$. При $\kappa=50$:

| шаг | множитель $\rho$ | итераций на цифру |
|---|---|---|
| $1/\kappa=0.02$ | $0.98$ | $\approx114$ |
| $\alpha^\ast=2/51\approx0.039$ | $49/51\approx0.961$ | $\approx58$ |

Лучший шаг вдвое быстрее, но оба $\sim\kappa$. Функция $f=\tfrac12x^\top Qx$ **квадратична** по ошибке, поэтому сама она сжимается в $\rho^2$ раз за шаг: $0.98^2=0.9604$ и $(49/51)^2\approx0.923$. Проверка ниже: из $(50,\,1)$ до $\|\nabla f\|\le10^{-8}$ — $1106$ итераций при $\alpha=1/\kappa$ и $567$ при $\alpha^\ast$; точный линейный поиск даёт те же $567$: знать $\kappa$ ему не нужно, но и быстрее лучшего постоянного шага он не бывает.

**(д)** Замена $x_1=1000\,u_1$: $\tilde f(u)=\tfrac12\big(10^6u_1^2+u_2^2\big)$, гессиан $D=\operatorname{diag}(10^6,\,1)$, $\kappa=10^6$ — из круглой задачи ($\kappa=1$) сделали вытянутую **одной сменой линейки**. С шагом $1/\lambda_{\max}=10^{-6}$ по пункту (г): множитель $1-10^{-6}$, около $2.3\cdot10^6$ итераций на одну цифру. В метрах та же задача решается шагом $\alpha=1$ за одну итерацию.

**(е)** $\nabla\tilde f(u)=Du$, поэтому $D^{-1}\nabla\tilde f(u)=u$, и предложенный шаг — это $u\leftarrow(1-\alpha)u$. Умножим первую координату на $1000$: $x\leftarrow(1-\alpha)x=x-\alpha\nabla f(x)$ — **тот же спуск в метрах**, где $\kappa=1$. Умножение градиента на $D^{-1}$ называют предобуславливанием; ошибка масштаба в $10^3$ раз стоит $10^6$ раз по $\kappa$ и по итерациям, поэтому переменные нормируют до запуска. А при $D=\nabla^2\tilde f$ и $\alpha=1$ шаг $u-D^{-1}Du=0$ попадает в минимум сразу — это шаг Ньютона из раздела 7: ему смена линейки безразлична.""")

code('''# (в)-(г): множители и цена одной цифры при kappa = 50
kappa = 50.0
Q = np.diag([1.0, kappa])
fq = lambda x: 0.5 * x @ Q @ x
gq = lambda x: Q @ x

rho = lambda a: max(abs(1 - a), abs(1 - a * kappa))
per_digit = lambda r: np.log(10) / (-np.log(r))
a_best = 2 / (kappa + 1)
for name, a in (("1/kappa     ", 1 / kappa), ("2/(kappa+1) ", a_best)):
    print(f"шаг {name}= {a:.4f}: множитель {rho(a):.4f}, итераций на цифру {per_digit(rho(a)):.0f}, множитель для f {rho(a) ** 2:.4f}")
assert np.isclose(rho(a_best), (kappa - 1) / (kappa + 1))''')

code('''# запуски из (50, 1) до ||grad|| <= 1e-8: 1106, 567 и 567
x0 = np.array([kappa, 1.0])
_, k1, p_const = gd(fq, gq, x0, alpha=1 / kappa, tol=1e-8)
_, k2, _ = gd(fq, gq, x0, alpha=a_best, tol=1e-8)

def gd_exact(x0, tol=1e-8):                        # точный линейный поиск: alpha = g^T g / g^T Q g
    x = np.asarray(x0, float).copy(); path = [x.copy()]
    while np.linalg.norm(gq(x)) > tol:
        g = gq(x); x = x - (g @ g) / (g @ Q @ g) * g; path.append(x.copy())
    return x, len(path) - 1, np.array(path)

_, k3, p_ex = gd_exact(x0)
print(f"alpha = 1/kappa: {k1} итераций;  alpha = 2/(kappa+1): {k2};  точный шаг: {k3}")
assert (k1, k2, k3) == (1106, 567, 567)

with np.errstate(over="ignore", invalid="ignore"):
    _, _, p_div = gd(fq, gq, x0, alpha=0.041, maxit=2000)
print(f"alpha = 0.041 > 2/kappa: через 2000 итераций |x2| = {abs(p_div[-1, 1]):.2e} — расходится")''')

code('''# (в): оба множителя как функции alpha
al = np.linspace(0, 2.4 / kappa, 500)
fig, ax = plt.subplots(figsize=(6.2, 3.3))
ax.plot(al, np.abs(1 - al), color=BLUE, label="$|1-\\\\alpha|$ (пологое направление)")
ax.plot(al, np.abs(1 - al * kappa), color=ORANGE, label="$|1-\\\\alpha\\\\kappa|$ (жёсткое направление)")
ax.axhline(1, color=GRAY, ls="--", lw=1)
ax.axvline(a_best, color=RED, ls=":", lw=1); ax.axvline(2 / kappa, color=GRAY, ls=":", lw=1)
ax.plot(a_best, rho(a_best), "o", color=RED, ms=6)
ax.annotate(f"$\\\\alpha^*=2/51$, множитель $49/51={rho(a_best):.3f}$", xy=(a_best, rho(a_best)), xytext=(0.0165, 0.55), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=RED), color=RED)
ax.text(2 / kappa, 0.05, " $2/\\\\kappa$: граница\\n сходимости", fontsize=8.5, color=GRAY)
ax.text(1 / kappa, 0.05, "$1/\\\\kappa$", fontsize=8.5, ha="center", color=GRAY)
ax.set_xlim(0, 2.4 / kappa); ax.set_ylim(0, 1.45)
ax.set_xlabel("$\\\\alpha$"); ax.set_ylabel("множитель за итерацию"); ax.legend(fontsize=8, loc="upper left")
ax.set_title("12.3(в): лучший шаг — где ломаные пересекаются", fontsize=9.5)
plt.show()''')

code('''fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
xx, yy = np.meshgrid(np.linspace(-5, 55, 300), np.linspace(-2, 2, 200))
zz = 0.5 * (xx ** 2 + kappa * yy ** 2)
for ax, p, title in ((axs[0], p_const, "постоянный шаг $\\\\alpha=1/\\\\kappa$: прыжок и долгое ползание"),
                     (axs[1], p_ex, "точный линейный поиск: зигзаг")):
    ax.contour(xx, yy, zz, levels=0.5 * np.array([2, 5, 10, 20, 30, 40, 50]) ** 2, colors=GRAY, linewidths=0.7)
    ax.plot(p[:12, 0], p[:12, 1], "-o", ms=3.5, color=BLUE)
    ax.plot(0, 0, "*", ms=12, color=RED, mec="white")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(title, fontsize=9.5)
    ax.set_xlim(-5, 55); ax.set_ylim(-1.6, 1.6)
axs[0].annotate("$x_2\\\\to0$ за 1 шаг", xy=(49, 0), xytext=(30, 1.0), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=GRAY))
plt.tight_layout(); plt.show()''')

code('''# (д)-(е): километры, предобуславливание, Ньютон
D = np.diag([1e6, 1.0])
gu = lambda u: D @ u
u0 = np.array([1.0, 1.0])
print(f"(д) kappa в переменных u: 1e6;  итераций на цифру с шагом 1e-6: {np.log(10) / (-np.log(1 - 1e-6)):.2e}")

S = np.diag([1000.0, 1.0])                          # x = S u: обратно к метрам
u, x = u0.copy(), S @ u0
for _ in range(5):
    u = u - 0.7 * np.linalg.solve(D, gu(u))         # предобусловленный шаг в километрах
    x = x - 0.7 * x                                 # обычный спуск в метрах (grad f(x) = x)
    assert np.allclose(S @ u, x)
print(f"(е) 5 шагов: S u = {(S @ u).round(6)} = x = {x.round(6)} — траектории совпали")

u1 = u0 - np.linalg.solve(D, gu(u0))                # D = hess, alpha = 1: шаг Ньютона
print(f"    D = hess, alpha = 1: из {u0} сразу в {u1} — минимум за один шаг")
assert np.allclose(u1, 0)''')

# ============================================================ 12.4
md(r"""## 12.4. Розенброк: цена одной верной цифры

> (а) Гессиан в $(1,1)$. (б) Собственные числа и $\kappa$ — через след и определитель. (в) Оценка итераций на уменьшение ошибки в $10^6$ раз. (г) Сравнение с демо (c) и Ньютоном.
>
> *Суть: одно число $\kappa$ предсказывает тридцать тысяч итераций.*

**(а)** Дифференцируем $\nabla f=\big(-2(1-x_1)-400x_1(x_2-x_1^2),\ 200(x_2-x_1^2)\big)$ ещё раз:
$$\nabla^2f(1,1)=\begin{pmatrix}802&-400\\-400&200\end{pmatrix}.$$

**(б)** След $1002$, определитель $802\cdot200-400^2=400$. Когда собственные числа сильно разнесены, корень извлекать не нужно:
$$\lambda_{\max}\approx\operatorname{tr}=1002,\qquad \lambda_{\min}\approx\frac{\det}{\operatorname{tr}}\approx0.40,\qquad \kappa\approx\frac{\operatorname{tr}^2}{\det}\approx2510.$$
Точные значения: $1001.6$, $0.3994$, $\kappa=2508$ — приём ошибается на доли процента. Малое $\lambda$ — вдоль дна долины, большое — поперёк.

**(в)** Вблизи минимума работает картина 12.3: вдоль пологого направления множитель $1-1/\kappa$, и
$$n=\frac{\ln10^6}{-\ln(1-1/\kappa)}\approx\kappa\ln10^6\approx2508\cdot13.8\approx3.5\cdot10^4.$$

**(г)** Факты из демо: старт $(-1.2,\,1)$, стоп $\|\nabla f\|\le10^{-6}$.

| метод | итераций |
|---|---|
| шаг $0.001\approx1/\lambda_{\max}$ | $32\,076$ |
| шаг $0.0019$ (почти $2/\lambda_{\max}$) | $16\,851$ |
| backtracking | $13\,756$ |
| Ньютон | $7$ до $10^{-10}$, $6$ до $10^{-6}$ |

Оценка $3.5\cdot10^4$ и факт $32\,076$ — **одного порядка**; большего требовать нельзя: оценка — про сжатие вблизи минимума, факт — про весь путь из $(-1.2,1)$. Шаг вдвое ближе к границе — вдвое быстрее, как и предсказывает 12.3(в). При $\alpha>2/\lambda_{\max}\approx0.002$ метод **не сходится**: точка не улетает (изогнутая долина держит), но садится на 2-цикл поперёк долины, и $f$ перестаёт убывать. Ньютону $\kappa$ безразлично.""")

code('''fr = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
gr = lambda x: np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)])
hr = lambda x: np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]], [-400 * x[0], 200.0]])

H = hr(np.array([1.0, 1.0]))
lam = np.linalg.eigvalsh(H)
kappa_r = lam[1] / lam[0]
tr, det = np.trace(H), np.linalg.det(H)
print(f"(а) гессиан в (1,1): {H.tolist()}")
print(f"(б) точно: lambda = {lam.round(4)}, kappa = {kappa_r:.0f};  приём: tr = {tr:.0f}, det/tr = {det / tr:.2f}, tr^2/det = {tr ** 2 / det:.0f}")
print(f"(в) оценка: kappa ln(1e6) = {kappa_r * np.log(1e6):.0f} итераций")''')

code('''# (г): четыре запуска из (-1.2, 1)
x0 = np.array([-1.2, 1.0])
for name, a in (("шаг 0.001   ", 0.001), ("шаг 0.0019  ", 0.0019), ("backtracking", None)):
    _, k, _ = gd(fr, gr, x0, alpha=a)
    print(f"{name}: {k} итераций до ||grad|| <= 1e-6")
x_n, k_n, p_n = newton(gr, hr, x0)
_, k_n6, _ = newton(gr, hr, x0, tol=1e-6)
print(f"Ньютон      : {k_n} итераций до 1e-10, {k_n6} до 1e-6")
assert (k_n, k_n6) == (7, 6)''')

code('''# за границей 2/lambda_max ~ 0.002: не расходимость, а 2-цикл поперёк долины
_, _, p_d = gd(fr, gr, x0, alpha=0.002, maxit=40_000)
print("шаг 0.002: последние f:", " <-> ".join(f"{fr(q):.4e}" for q in p_d[-4:]), f";  |x| <= {np.abs(p_d).max():.2f} — точка не улетела")''')

# ============================================================ 12.5
md(r"""## 12.5. Прочитайте график

> Спуск с шагом $1/L$, старт $\|\nabla f(x_0)\|=1$, стоп $\|\nabla f\|\le10^{-8}$; $\log_{10}\|\nabla f(x_k)\|$ — прямая с наклоном $-0.0004$ на итерацию. (а) Множитель $q$ и $\kappa$? (б) Сколько итераций? (в) Масштабировать ли переменные? (г) Как выглядел бы Ньютон?
>
> *Суть: по одному наклону читаются множитель, $\kappa$ и цена цифры.*

**(а)** Прямая в полулогарифмических осях — это геометрическая прогрессия, и её наклон равен $\log_{10}q$. Отсюда $q=10^{-0.0004}\approx0.99908$. Для шага $1/L$ множитель равен $1-1/\kappa$, значит $\kappa=1/(1-q)\approx1090$. Удобная цепочка: наклон $-0.0004$ ⇒ одна цифра стоит $1/0.0004=2500$ итераций ⇒ $\kappa\approx2500/\ln10\approx1090$.

**(б)** От $1$ до $10^{-8}$ — восемь порядков: $8/0.0004=20\,000$ итераций.

**(в)** Да. При $\kappa\sim10^3$ каждая цифра стоит $\sim2500$ итераций, и если часть $\kappa$ пришла из масштабов переменных (12.3(д)), масштабирование заберёт её даром. График останется прямой — метод тот же, — но наклон станет круче: при $\kappa\approx3$ уже около $-0.2$, цифра за пять итераций.

**(г)** Не прямая, а «обрыв»: число верных цифр удваивается от шага к шагу, и расстояние между соседними точками графика растёт. От $1$ до $10^{-8}$ Ньютон доходит за $3$–$5$ итераций; вдали от минимума возможно плато и лишь потом срыв — как на Розенброке в 12.4.

**Проверка.** Квадратичная с $\kappa=1090$, старт с $\|\nabla f\|=1$, шаг $1/\lambda_{\max}$: наклон хвоста получается $\log_{10}(1-1/1090)\approx-0.000399$, итераций около $19$ тысяч. Расхождение с $20\,000$ в несколько процентов — нормально: прямая — это асимптотика; первые итерации быстро гасят жёсткие компоненты градиента, и линия стартует чуть ниже нуля.""")

code('''# квадратичная с kappa = 1090: наклон хвоста и число итераций
kappa_5 = 1090.0
lam5 = np.logspace(0, np.log10(kappa_5), 10)        # диагональ гессиана от 1 до kappa
f5 = lambda x: 0.5 * x @ (lam5 * x)
g5 = lambda x: lam5 * x
x0_5 = (np.ones(10) / np.sqrt(10)) / lam5           # так ||grad f(x0)|| = 1
assert np.isclose(np.linalg.norm(g5(x0_5)), 1)

x5, k5, p5 = gd(f5, g5, x0_5, alpha=1 / lam5.max(), tol=1e-8)
lg5 = np.log10(np.linalg.norm(p5 * lam5, axis=1))
slope = (lg5[-1] - lg5[-5001]) / 5000
print(f"итераций: {k5} (условие даёт 20000);  наклон хвоста {slope:.6f}, log10(1 - 1/kappa) = {np.log10(1 - 1 / kappa_5):.6f}")
assert abs(slope - np.log10(1 - 1 / kappa_5)) < 1e-5 and abs(k5 - 20_000) / 20_000 < 0.1''')

code('''# (г): Ньютон на гладкой сильно выпуклой (не квадратичной) функции — обрыв
gn_ = lambda x: np.tanh(x) + 0.1 * x
hn = lambda x: np.diag(1 / np.cosh(x) ** 2 + 0.1)
_, kn, pn = newton(gn_, hn, np.full(10, 0.9), tol=1e-8)
gnn = np.array([np.linalg.norm(gn_(q_)) for q_ in pn])
print(f"Ньютон: {kn} итераций, ||grad f_k|| = {np.array2string(gnn, precision=2)} — цифры удваиваются")''')

code('''fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3))
ax = axs[0]
ax.plot(np.arange(len(lg5)), lg5, color=BLUE, lw=1.5, label="GD, шаг $1/L$, $\\\\kappa=1090$")
ax.plot([0, 20000], [0, -8], "--", color=GRAY, lw=1, label="прямая из условия: наклон $-0.0004$")
ax.axhline(-8, color=GRAY, ls=":", lw=1)
ax.set_xlabel("итерация $k$"); ax.set_ylabel("$\\\\log_{10}\\\\|\\\\nabla f(x_k)\\\\|$")
ax.set_title(f"12.5: прямая — линейная сходимость, {k5} итераций", fontsize=9.5); ax.legend(fontsize=8, loc="upper right")
ax = axs[1]
ax.plot(np.arange(len(gnn)), np.log10(gnn), "-o", color=ORANGE, ms=5, label="Ньютон")
ax.axhline(-8, color=GRAY, ls=":", lw=1)
ax.set_xlabel("итерация $k$"); ax.set_ylabel("$\\\\log_{10}\\\\|\\\\nabla f(x_k)\\\\|$")
ax.set_title("(г) Ньютон: не прямая, а обрыв", fontsize=9.5); ax.legend(fontsize=8, loc="lower left")
ax.set_xticks(np.arange(len(gnn)))
plt.tight_layout(); plt.show()''')

# ============================================================ 12.6
md(r"""## 12.6. Лемма о спуске

> Дано $\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$. (а) Докажите $f(y)\le f(x)+\nabla f(x)^\top(y-x)+\tfrac L2\|y-x\|^2$. (б) Подставьте $y=x-\alpha\nabla f(x)$. (в) Когда убывание гарантировано и какой шаг лучший? (г) $L$ для $\tfrac12x^\top Qx$?
>
> *Суть: главная формула раздела 4 выводится в три строки из одного интеграла.*

**(а)** Идём по отрезку $\gamma(t)=x+t(y-x)$. По формуле Ньютона–Лейбница
$$f(y)-f(x)=\int_0^1\nabla f(\gamma(t))^\top(y-x)\,dt.$$
Вычтем $\nabla f(x)^\top(y-x)$ — это тот же интеграл от постоянного вектора:
$$f(y)-f(x)-\nabla f(x)^\top(y-x)=\int_0^1\big(\nabla f(\gamma(t))-\nabla f(x)\big)^\top(y-x)\,dt.$$
Скалярное произведение не больше произведения норм, а по условию Липшица $\|\nabla f(\gamma(t))-\nabla f(x)\|\le L\,t\,\|y-x\|$. Остаётся проинтегрировать $L\,t\,\|y-x\|^2$ по $t$ от $0$ до $1$ — получается $\tfrac L2\|y-x\|^2$. Это верхняя парабола из раздела 4.3.

**(б)** При $y=x-\alpha\nabla f(x)$: линейный член равен $-\alpha\|\nabla f\|^2$, квадратичный — $\tfrac{L\alpha^2}2\|\nabla f\|^2$. Вместе:
$$f(x-\alpha\nabla f)\le f(x)-\alpha\Big(1-\frac{\alpha L}2\Big)\|\nabla f(x)\|^2.$$

**(в)** Гарантированное убывание $\psi(\alpha)=\alpha(1-\alpha L/2)\|\nabla f\|^2$ положительно при $0<\alpha<2/L$. Как парабола по $\alpha$, оно максимально в вершине $\alpha=1/L$, где убывание не меньше $\|\nabla f\|^2/(2L)$.

**(г)** $\nabla f=Qx$, и $\|Q(x-y)\|\le\lambda_{\max}(Q)\,\|x-y\|$ с равенством на старшем собственном векторе: $L=\lambda_{\max}(Q)$. Для дважды дифференцируемой $f$ вообще «$L$-липшицев градиент» $\iff$ $-LI\preceq\nabla^2f\preceq LI$, причём лемме достаточно верхней половины. На практике глобальная $L$ пессимистична: у Розенброка на прямоугольнике $[-2,2]\times[-1,3]$ она $\approx5327$, а у дна долины backtracking принимает шаг $2^{-9}$ — на порядок больше $1/L$: он находит локальную константу сам.""")

code('''# Розенброк: L по сетке на прямоугольнике и проверка неравенства на случайных парах
lo, hi = np.array([-2.0, -1.0]), np.array([2.0, 3.0])
grid = np.array([[a, b] for a in np.linspace(lo[0], hi[0], 101) for b in np.linspace(lo[1], hi[1], 101)])
L = max(np.linalg.eigvalsh(hr(q))[-1] for q in grid)
print(f"L = max lambda_max(hess) на прямоугольнике = {L:.0f};  1/L = {1 / L:.1e}")

rng = np.random.default_rng(0)
X = rng.uniform(lo, hi, size=(10_000, 2)); Y = rng.uniform(lo, hi, size=(10_000, 2))
lhs = np.array([fr(y) - fr(x) - gr(x) @ (y - x) for x, y in zip(X, Y)])
rhs = 0.5 * L * np.sum((Y - X) ** 2, axis=1)
print(f"max (f(y) - f(x) - grad^T(y-x)) / (L/2 |y-x|^2) по 10 000 парам = {np.max(lhs / rhs):.3f}  (должно быть <= 1)")
assert np.max(lhs / rhs) <= 1 + 1e-9

dec = [(fr(x) - fr(x - gr(x) / L)) / (gr(x) @ gr(x) / (2 * L)) for x in X[:2000]]
print(f"шаг 1/L: фактическое убывание / гарантированное по 2000 точкам >= {min(dec):.2f}  (должно быть >= 1)")
assert min(dec) >= 1 - 1e-9''')

code('''# гарантия и факт как функции alpha в точке (-1.2, 1)
x = np.array([-1.2, 1.0]); g = gr(x); L_loc = np.linalg.eigvalsh(hr(x))[-1]
alphas = np.linspace(0, 2.4 / L, 300)
actual = np.array([fr(x) - fr(x - a * g) for a in alphas])
guaranteed = alphas * (1 - alphas * L / 2) * (g @ g)
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(alphas * L, actual, color=BLUE, label="фактическое $f(x)-f(x-\\\\alpha\\\\nabla f)$")
ax.plot(alphas * L, guaranteed, color=ORANGE, label="гарантия леммы $\\\\alpha(1-\\\\alpha L/2)\\\\|\\\\nabla f\\\\|^2$")
ax.axvline(1, color=GRAY, ls="--", lw=1); ax.axvline(2, color=GRAY, ls="--", lw=1)
ax.text(1, ax.get_ylim()[1] * 0.9, "$\\\\alpha=1/L$", ha="center", fontsize=8.5); ax.text(2, ax.get_ylim()[1] * 0.9, "$\\\\alpha=2/L$", ha="center", fontsize=8.5)
ax.set_xlabel("$\\\\alpha L$"); ax.set_ylabel("убывание $f$"); ax.legend(fontsize=8.5, loc="lower left")
ax.set_title(f"12.6: гарантия и факт; глобальная $L={L:.0f}$, локальная $\\\\lambda_{{\\\\max}}={L_loc:.0f}$", fontsize=9.5)
plt.show()''')

# ============================================================ 12.7
md(r"""## 12.7. Тип сходимости на слух

> Ошибки четырёх методов: (а) $10^{-1},10^{-2},10^{-3},\dots$; (б) $10^{-1},10^{-2},10^{-4},10^{-8},\dots$; (в) $1/k$; (г) $1/k!$. Тип сходимости каждой и число итераций от $\le10^{-1}$ до $\le10^{-12}$?
>
> *Суть: тип сходимости читается по отношению соседних ошибок.*

Смотрим на $\|e_{k+1}\|/\|e_k\|$: отделено от единицы одним $q<1$ — линейная; стремится к нулю — сверхлинейная; $\|e_{k+1}\|\le C\|e_k\|^2$ — квадратичная.

- **(а)** $e_k=10^{-k}$: отношение всегда $0.1$ — **линейная с множителем $0.1$**, цифра за итерацию. От $10^{-1}$ до $10^{-12}$ — **11** итераций.
- **(б)** $e_k=10^{-2^k}$: $e_{k+1}=e_k^2$ — **квадратичная**, число цифр удваивается: $1\to2\to4\to8\to16$. До $10^{-12}$ (первое значение — уже $10^{-16}$) — **4** итерации.
- **(в)** $e_k=1/k$: отношение $k/(k+1)\to1$ — не отделено от единицы, **не линейная** (сублинейная). Нужно $k\ge10^{12}$: **$\sim10^{12}$** итераций, на практике никогда.
- **(г)** $e_k=1/k!$: отношение $1/(k+1)\to0$ — **сверхлинейная**, но не квадратичная ($e_{k+1}/e_k^2\to\infty$). Первый раз $\le10^{-1}$ при $k=4$, $\le10^{-12}$ при $k=15$ — **11** итераций, как у (а): на коротком отрезке выигрыш сверхлинейности не успевает проявиться.

Порядок по скорости: (б) $\gg$ (г) $>$ (а) $\gg$ (в). Ньютон — это (б), квазиньютоновские методы — типа (г), спуск на сильно выпуклой функции — (а).""")

code('''from math import factorial

seqs = {"(а) 10^-k  ": [10.0 ** -k for k in range(1, 6)],
        "(б) 10^-2^k": [10.0 ** -(2 ** k) for k in range(5)],
        "(в) 1/k    ": [1 / k for k in range(10, 15)],
        "(г) 1/k!   ": [1 / factorial(k) for k in range(3, 8)]}
for name, e in seqs.items():
    r = np.array(e[1:]) / np.array(e[:-1])
    print(f"{name} e_(k+1)/e_k = {np.array2string(r, precision=3)}")

first = lambda e, tol: next(k for k in range(10 ** 6) if e(k) <= tol)
for name, e in (("(а)", lambda k: 10.0 ** -k if k else 1.0), ("(б)", lambda k: 10.0 ** -(2 ** k)), ("(г)", lambda k: 1 / factorial(k))):
    print(f"{name} от 1e-1 до 1e-12: {first(e, 1e-12) - first(e, 1e-1)} итераций")
print("(в) от 1e-1 до 1e-12: 1e12 - 10 итераций")''')

# ============================================================ 12.8
md(r"""## 12.8. Наискорейший спуск и ортогональность

> (а) Минимум $\nabla f^\top p$ по единичным $p$ — на антиградиенте. (б) $\nabla f$ ортогонален линии уровня. (в) При точном шаге $\nabla f(x_{k+1})\perp\nabla f(x_k)$.
>
> *Суть: три коротких доказательства о том, куда смотрит градиент.*

**(а)** Пусть $g=\nabla f(x)\ne0$. Для единичного $p$: $g^\top p=\|g\|\cos\theta\ge-\|g\|$, и равенство достигается только при $\theta=180^\circ$, то есть $p=-g/\|g\|$. Минимум производной по направлению равен $-\|g\|$ и достигается только на антиградиенте.

**(б)** Пусть $\gamma(t)$ — кривая на линии уровня, $\gamma(0)=x$: тождество $f(\gamma(t))=\mathrm{const}$. Дифференцируем по $t$ в нуле: $\nabla f(x)^\top\gamma'(0)=0$. Вектор $\gamma'(0)$ — любая касательная к линии уровня, значит $\nabla f$ перпендикулярен им всем.

**(в)** Точный шаг минимизирует $\varphi(\alpha)=f(x_k-\alpha g_k)$, и во внутреннем минимуме $\varphi'(\alpha_k)=-\nabla f(x_{k+1})^\top g_k=0$: следующий градиент перпендикулярен предыдущему, путь поворачивает на $90^\circ$ каждый шаг. По (б) это значит: остановились там, где луч шага коснулся линии уровня. В $\mathbb R^2$ направление через шаг повторяется точно — отсюда правильный зигзаг на картинке 04.""")

code('''# (а): случайные направления не круче антиградиента; (в): ортогональность на зигзаге из 12.3
rng = np.random.default_rng(1)
x = np.array([-1.2, 1.0]); g = gr(x)
th = rng.uniform(0, 2 * np.pi, 10_000)
dd = np.c_[np.cos(th), np.sin(th)] @ g
print(f"(а) min g^T p по 10 000 случайным единичным p: {dd.min():.3f} >= -|g| = {-np.linalg.norm(g):.3f}")
assert dd.min() >= -np.linalg.norm(g)

gs = [gq(q) for q in p_ex[:7]]
cos = [gs[k] @ gs[k + 1] / (np.linalg.norm(gs[k]) * np.linalg.norm(gs[k + 1])) for k in range(6)]
print("(в) cos между соседними градиентами при точном шаге:", np.round(cos, 12))''')

# ============================================================ 12.9
md(r"""## 12.9. Цепь: $\kappa$ растёт с числом грузов

> (а) Покажите $\nabla^2E=\operatorname{diag}(DK,\,DK)$ с трёхдиагональной $K$. (б) По $\lambda_j=2-2\cos\frac{j\pi}{N+1}$ найдите $\kappa(N)$ и покажите $\kappa\approx(2(N+1)/\pi)^2$. (в) $N=40$: итераций спуска и шагов Ньютона? (г) Почему Ньютон — это `np.linalg.solve`?
>
> *Суть: чем мельче дробим задачу, тем хуже она обусловлена — $\kappa\sim N^2$.*

**(а)** Координата $y_i$ входит ровно в две пружины, $(y_i-y_{i-1})^2$ и $(y_{i+1}-y_i)^2$, поэтому
$$\frac{\partial E}{\partial y_i}=D\,(2y_i-y_{i-1}-y_{i+1}),\qquad \frac{\partial^2E}{\partial y_i^2}=2D,\qquad \frac{\partial^2E}{\partial y_i\,\partial y_{i\pm1}}=-D.$$
Это матрица $DK$ с трёхдиагональной $K$. По $z$ то же самое: вес $mg\sum z_i$ линеен и вторых производных не даёт. Членов $y_iz_j$ в энергии нет, так что $\nabla^2E=\operatorname{diag}(DK,\,DK)$ — постоянная матрица: $E$ квадратична.

**(б)** $\lambda_{\min}(K)=2-2\cos\frac\pi{N+1}$, а $\lambda_{\max}(K)=2+2\cos\frac\pi{N+1}$ (у $j=N$ косинус поменял знак). Общий множитель $D$ сокращается:
$$\kappa(N)=\frac{1+\cos\frac\pi{N+1}}{1-\cos\frac\pi{N+1}}.$$
При больших $N$ угол $\theta=\pi/(N+1)$ мал, $\cos\theta\approx1-\theta^2/2$: числитель $\approx2$, знаменатель $\approx\theta^2/2$, и
$$\kappa\approx\frac4{\theta^2}=\Big(\frac{2(N+1)}\pi\Big)^2\ \sim\ N^2.$$
Для $N=10,20,40,80$: $\kappa=48.4,\ 178.1,\ 680.6,\ 2658.4$; формула даёт $49.0,\ 178.7,\ 681.3,\ 2659.1$.

**(в)** $N=40$: $\kappa\approx681$, $L=D\,\lambda_{\max}(K)\approx279.6$. Стартовый градиент $\|\nabla E(v_0)\|\approx6.2$, стоп по $10^{-6}$ — уменьшить в $6.2\cdot10^6$ раз: оценка $\kappa\ln(6.2\cdot10^6)\approx10\,640$ итераций, факт — $10\,575$. Функция квадратична, и асимптотика работает с первого шага. Ньютону нужен **один** шаг.

**(г)** Для квадратичной $E$ квадратичная модель Ньютона — тождество, и его шаг из любой точки решает $\nabla E=Hv+b=0$. Это линейная система: `np.linalg.solve(H, -grad(0))`. Минимальная энергия $E^\ast=13.4417$.""")

code('''# (а)-(б): гессиан собирается из K, kappa растёт как N^2
D, g_acc = 70.0, 9.81
p_left, p_right = np.array([-2.0, 1.0]), np.array([2.0, 1.0])

def make_chain(N):
    m = 4.0 / N
    def unpack(v):
        return np.r_[p_left[0], v[:N], p_right[0]], np.r_[p_left[1], v[N:], p_right[1]]
    def energy(v):
        y, z = unpack(v)
        return 0.5 * D * np.sum(np.diff(y) ** 2 + np.diff(z) ** 2) + m * g_acc * np.sum(z[1:-1])
    def grad_E(v):
        y, z = unpack(v)
        return np.r_[D * (2 * y[1:-1] - y[:-2] - y[2:]), D * (2 * z[1:-1] - z[:-2] - z[2:]) + m * g_acc]
    K = 2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
    H = np.kron(np.eye(2), D * K)
    v0 = np.r_[np.linspace(*[p_left[0], p_right[0]], N + 2)[1:-1], np.linspace(*[p_left[1], p_right[1]], N + 2)[1:-1]]
    return energy, grad_E, unpack, K, H, v0

N = 40
energy, grad_E, unpack, K, H, v0 = make_chain(N)
H_num = np.column_stack([grad_E(e) - grad_E(np.zeros(2 * N)) for e in np.eye(2 * N)])   # градиент линеен — разности точны
print(f"(а) гессиан = kron(I2, D K): max |H_num - H| = {np.abs(H_num - H).max():.1e}")

print("(б)  N   kappa(K)   (2(N+1)/pi)^2")
for n in (10, 20, 40, 80):
    ev = np.linalg.eigvalsh(2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1))
    print(f"   {n:3d}  {ev[-1] / ev[0]:8.1f}   {(2 * (n + 1) / np.pi) ** 2:10.1f}")''')

code('''# (в)-(г): спуск против одного шага Ньютона при N = 40
lamH = np.linalg.eigvalsh(H)
L, kappa_c = lamH[-1], lamH[-1] / lamH[0]
gn0 = np.linalg.norm(grad_E(v0))
print(f"L = {L:.1f}, kappa = {kappa_c:.1f};  ||grad E(v0)|| = {gn0:.2f};  оценка kappa*ln(||grad||/1e-6) = {kappa_c * np.log(gn0 / 1e-6):.0f}")

v_gd, k_gd, _ = gd(energy, grad_E, v0, alpha=1 / L)
v_newton = v0 - np.linalg.solve(H, grad_E(v0))
print(f"GD с шагом 1/L: {k_gd} итераций, E = {energy(v_gd):.4f}")
print(f"один шаг Ньютона: E = {energy(v_newton):.4f}, |grad E| = {np.linalg.norm(grad_E(v_newton)):.1e}, расстояние до GD {np.linalg.norm(v_gd - v_newton):.1e}")
assert abs(k_gd - 10_575) <= 5 and np.linalg.norm(grad_E(v_newton)) < 1e-9''')

code('''y0, z0 = unpack(v0); y1, z1 = unpack(v_newton)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.plot(y0, z0, "--", color=GRAY, label="старт: прямая между концами")
ax.plot(y1, z1, "-o", ms=3, color=BLUE, label="один шаг Ньютона = np.linalg.solve")
ax.plot([-2, 2], [1, 1], "o", color=INK, ms=7)
ax.set_xlabel("$y$"); ax.set_ylabel("$z$"); ax.legend(fontsize=8.5, loc="center")
ax.set_title(f"12.9: цепь $N={N}$, $E^*={energy(v_newton):.4f}$, $\\\\kappa={kappa_c:.0f}$", fontsize=10)
plt.show()''')

# ============================================================ 12.10
md(r"""## 12.10. Логистическая регрессия: гессиан и сдвиг

> (а) Найдите $\nabla f$ и покажите $\nabla^2f=\frac1nX^\top SX$, $S\succeq0$ диагональная. (б) Выведите выпуклость. (в) Почему «возраст $40\pm10$» со столбцом единиц даёт большое $\kappa$? (г) Почему масштабирование не лечит, а стандартизация лечит?
>
> *Суть: большое $\kappa$ приходит из среднего признака, и лечит его центровка.*

**(а)** Пусть $\sigma(t)=1/(1+e^{-t})$. Слагаемое $\log(1+e^{-t})$ имеет производную $-\sigma(-t)$, и по цепному правилу
$$\nabla f(w)=-\frac1n\sum_{i=1}^n\sigma\big({-y_ix_i^\top w}\big)\,y_i\,x_i.$$
Множитель $\sigma(-y_ix_i^\top w)$ — вероятность, которую модель даёт **неправильному** классу: чем грубее ошибка, тем сильнее объект тянет $w$. Дифференцируем ещё раз ($\sigma'=\sigma(1-\sigma)$, $y_i^2=1$):
$$\nabla^2f(w)=\frac1n\sum_{i=1}^n\sigma(z_i)\big(1-\sigma(z_i)\big)\,x_ix_i^\top=\frac1nX^\top SX,\qquad z_i=x_i^\top w.$$
Заметьте: $y_i$ из гессиана исчез — кривизна зависит от уверенности модели, а не от её правоты.

**(б)** Для любого $v$: $v^\top\nabla^2f\,v=\frac1n\|S^{1/2}Xv\|^2\ge0$ — гессиан неотрицателен всюду, $f$ выпукла (строго, если столбцы $X$ линейно независимы). Любая стационарная точка — глобальный минимум.

**(в)** С точностью до весов $\frac1nX^\top X$ — матрица **вторых моментов**, а второй момент — это среднее$^2$ + дисперсия: у возраста $40^2+10^2=1700$, почти весь от среднего. Косинус угла между столбцом возраста $a$ и столбцом единиц
$$\cos\angle(a,\mathbf1)=\frac{\bar a}{\sqrt{\bar a^2+\operatorname{var}a}}=\frac{40}{\sqrt{1700}}\approx0.97:$$
столбцы почти коллинеарны — «возраст» почти константа. Вдоль разности этих столбцов гессиан почти вырожден (маленькое $\lambda_{\min}$), вдоль суммы — велик, и выходит $\kappa\approx7.5\cdot10^4$.

**(г)** Масштабирование делит столбец на число — угол со столбцом единиц не меняется, коллинеарность остаётся: $\kappa$ падает лишь до $\sim2\cdot10^3$. Стандартизация ещё и вычитает среднее — столбцы признаков становятся **ортогональны** единицам ($\sum_i(x_{ij}-\mu_j)=0$), и $\kappa\approx2.75$. Это аффинная замена переменных: задача та же, ответ пересчитывается, а спуск сходится за $195$ итераций вместо «не сходится за $20\,000$». Ньютону всё равно: $6$ итераций в любом варианте.""")

code('''# (а)-(б): формулы и проверка градиента конечными разностями
f_log = lambda w, X, y: np.mean(np.logaddexp(0, -y * (X @ w)))
g_log = lambda w, X, y: -X.T @ (expit(-y * (X @ w)) * y) / len(y)
h_log = lambda w, X, y: (X.T * (expit(X @ w) * (1 - expit(X @ w)))) @ X / len(y)

rng = np.random.default_rng(10)
Xr = np.c_[np.ones(30), rng.normal(size=(30, 2))]; yr = rng.choice([-1.0, 1.0], 30); wr = rng.normal(size=3)
g_fd = np.array([(f_log(wr + 1e-6 * e, Xr, yr) - f_log(wr - 1e-6 * e, Xr, yr)) / 2e-6 for e in np.eye(3)])
print(f"градиент vs конечные разности: max|diff| = {np.abs(g_log(wr, Xr, yr) - g_fd).max():.1e}")
ev = np.linalg.eigvalsh(h_log(wr, Xr, yr))
print(f"собственные числа гессиана: {ev.round(4)} — все >= 0, выпукла")
assert ev.min() >= 0''')

code('''# (в)-(г): данные демо (возраст, стаж) и три варианта координат
rng = np.random.default_rng(4); n = 200
z1, z2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
y = 2 * (rng.uniform(size=n) < expit(1.5 * z1 - z2 + 0.3)) - 1.0
X = np.c_[np.ones(n), 40 + 10 * z1, 5 + z2]
mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0)
cosang = lambda u, v: u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
print(f"(в) cos(возраст, единицы) = {cosang(X[:, 1], X[:, 0]):.3f}  (формула: 40/sqrt(1700) = {40 / np.sqrt(1700):.3f})")

Xm = X.copy(); Xm[:, 1:] = X[:, 1:] / sd                 # только масштаб
Xs = X.copy(); Xs[:, 1:] = (X[:, 1:] - mu) / sd          # стандартизация: сдвиг + масштаб
for name, XX in (("сырые признаки", X), ("только масштаб", Xm), ("стандартизация", Xs)):
    w_star, k_new, _ = newton(lambda w: g_log(w, XX, y), lambda w: h_log(w, XX, y), np.zeros(3))
    ev = np.linalg.eigvalsh(h_log(w_star, XX, y))
    _, k_gd, _ = gd(lambda w: f_log(w, XX, y), lambda w: g_log(w, XX, y), np.zeros(3), maxit=20_000)
    tail = " (не сошёлся)" if k_gd == 20_000 else ""
    print(f"(г) {name}: cos(возраст, 1) = {cosang(XX[:, 1], XX[:, 0]):.2f}, kappa = {ev[-1] / ev[0]:.3g};  GD {k_gd} ит.{tail}, Ньютон {k_new} ит.")
print(f"точность классификации: {np.mean((Xs @ w_star > 0) == (y > 0)):.3f}")''')

# ============================================================ итог
md(r"""## Что запомнить

- $\nabla f=0$ — необходимое условие: спуск останавливается в любой стационарной точке, в том числе в седле, но к седлу приходит только со множества меры нуль (12.1). Нулевое собственное число гессиана — условия молчат, решают старшие члены; проверка «по направлениям» не заменяет $\nabla^2f\succ0$ (12.2).
- На квадратичной с постоянным шагом ошибка по каждому собственному направлению сжимается в $|1-\alpha\lambda_i|$ раз; сходимость при $0<\alpha<2/\lambda_{\max}$, лучший шаг $2/(\lambda_{\max}+\lambda_{\min})$ с множителем $(\kappa-1)/(\kappa+1)$; итераций на цифру $\sim\kappa$ (12.3, 12.4, 12.9).
- $\kappa$ — свойство координат: масштабирование — самый дешёвый предобусловливатель, а при признаках с большим средним нужна ещё и центровка (12.3(д)–(е), 12.10).
- Наклон прямой $\log_{10}\|\nabla f_k\|$ равен $\log_{10}(1-1/\kappa)$: по графику читаются $\kappa$ и цена цифры (12.5).
- Лемма о спуске: при $\nabla^2f\preceq LI$ шаг $1/L$ гарантирует убывание на $\|\nabla f\|^2/(2L)$; для квадратичной $L=\lambda_{\max}$; глобальная $L$ пессимистична, backtracking находит локальную (12.6).
- Тип сходимости — по отношению соседних ошибок: отделено от единицы — линейная, $\to0$ — сверхлинейная, $\|e_{k+1}\|\le C\|e_k\|^2$ — квадратичная (12.7).
- Антиградиент — самый крутой спуск и перпендикуляр к линии уровня; при точном шаге соседние градиенты ортогональны — зигзаг (12.8).
- Для квадратичной функции Ньютон — один шаг, то есть линейная система (12.9); для логистической регрессии — несколько (12.10).""")

# ============================================================ сборка
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
nbf.validate(nb)
nbf.write(nb, OUT)
print(f"{OUT}: {len(cells)} ячеек")
