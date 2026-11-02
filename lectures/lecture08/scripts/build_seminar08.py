"""Собирает семинар к лекции 8: seminar08.ipynb — множитель как цена ограничения.

    uv run python lectures/lecture08/scripts/build_seminar08.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture08/seminar08.ipynb

~30 минут за ноутбуком: (1) формула цены df*/db = lambda* и её доказательство, (2) портфель лекции 3
с двумя равенствами — система условий линейна, (3) проверка формулы численной производной,
(4) тот же портфель в IPOPT и знак lam_g. Данные портфеля и portfolio_eq — из l8helpers.py.
Раздел появился 02.11 (круг 19 ревью): §5.2 конспекта перенесён сюда.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "seminar08.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(r"""# Семинар 8. Множитель как цена ограничения

Весь семинар — за ноутбуком, около 30 минут. На лекции (§5) множитель был силой — натяжением стержня. Здесь — его второй смысл: цена ограничения.

| Мин | Шаг |
|---|---|
| 8 | **1. Формула цены** $df^\ast/db=\lambda^\ast$ и её доказательство |
| 10 | **2. Портфель лекции 3.** Линейная система и цена доходности $\lambda_\rho$ |
| 7 | **3. Проверка формулы** численной производной |
| 5 | **4. Тот же портфель в IPOPT** и знак множителей |

Конспект: [`lecture08.md`](lecture08.md). Установка: `uv sync --group casadi` в корне репозитория; в Colab первая ячейка поставит CasADi сама.""")

code(r'''import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import numpy as np
import casadi as ca
from l8helpers import *          # палитра, данные портфеля PF_SIGMA, PF_R и portfolio_eq (лекция 8)

print("CasADi", ca.__version__)''')

# ============================================================ шаг 1
md(r"""## Шаг 1. Формула цены (8 мин)

**Главная мысль.** Множитель — цена ограничения. На столько изменится оптимум, если сдвинуть правую часть ограничения на единицу.

Сдвинем правую часть ограничений: $g(x)=b$, здесь $b\in\mathbb R^m$. Оптимум $f^\ast$ становится функцией $b$, и

$$
\frac{df^\ast}{db}=\lambda^\ast ,
$$

то есть $\partial f^\ast/\partial b_j=\lambda^\ast_j$ для каждого ограничения.

Лекция 3 обещала доказательство. Для равенств оно занимает три строки. На решении $g(x^\ast(b))=b$, поэтому

$$
f^\ast(b)=L\big(x^\ast(b),\lambda^\ast(b),b\big),\qquad L=f-\lambda^\top\big(g(x)-b\big) .
$$

Продифференцируем по $b$. Слагаемые с $dx^\ast/db$ умножаются на $\nabla_xL=0$. Слагаемые с $d\lambda^\ast/db$ умножаются на $g-b=0$. Остаётся

$$
\frac{df^\ast}{db}=\frac{\partial L}{\partial b}=\lambda^\ast .
$$

Почему $x^\ast(b)$ и $\lambda^\ast(b)$ можно дифференцировать — упражнение 8.6.

<img src="img/10_portfolio_price.png" width="650" alt="минимальный риск f* как функция цели доходности ρ — выпуклая кривая; в точке ρ₀ касательная, её наклон — множитель λ_ρ">

*Минимальный риск портфеля как функция цели доходности $\rho$. Наклон касательной в точке $\rho_0$ — множитель $\lambda_\rho$. Числа — в шагах 2 и 3.*""")

# ============================================================ шаг 2
md(r"""## Шаг 2. Портфель лекции 3 (10 мин)

**Главная мысль.** У портфеля с одними равенствами система условий линейна. Её решает один `np.linalg.solve`.

Портфель из лекции 3 (§2.2): четыре актива, $x\in\mathbb R^4$ — вектор долей капитала в них. Минимизируем риск при заданной доходности $\rho$:

$$
\min_{x\in\mathbb R^4}\ \tfrac12x^\top\Sigma x\qquad\text{при}\qquad\mathbf 1^\top x=1,\quad r^\top x=\rho .
$$

Здесь $\Sigma$ — ковариационная матрица доходностей размера $4\times4$. Вектор $r\in\mathbb R^4$ — ожидаемые доходности: $4$, $8$, $11$ и $15\,\%$. Запрета коротких позиций нет, ограничений только два. Множитель бюджета — $\lambda_1$, множитель доходности — $\lambda_\rho$.

Условие первого порядка $\nabla f=J^\top\lambda$ здесь — $\Sigma x=\lambda_1\mathbf 1+\lambda_\rho r$. Вместе с ограничениями это 6 уравнений на 6 неизвестных:

$$
\begin{pmatrix}\Sigma&-\mathbf 1&-r\\ \mathbf 1^\top&0&0\\ r^\top&0&0\end{pmatrix}\begin{pmatrix}x\\ \lambda_1\\ \lambda_\rho\end{pmatrix}=\begin{pmatrix}0\\ 1\\ \rho\end{pmatrix} .
$$

Система линейна: градиент квадратичной $f$ линеен, ограничения линейны.

**Предскажите до запуска.** Какой знак у $\lambda_\rho$ при $\rho=10\,\%$? Подсказка: что станет с минимальным риском, если поднять цель доходности?

**Что делает ячейка.** Собирает матрицу $6\times6$ блоками, решает систему при $\rho=10\,\%$ и печатает доли, риск и множители.""")

code(r'''rho = 0.10                                                             # цель доходности, 10 %
one = np.ones(4)                                                       # вектор из единиц: бюджет 1^T x = 1
K = np.block([[PF_SIGMA, -one[:, None], -PF_R[:, None]],               # 4 строки: Sigma x - lambda_1 1 - lambda_rho r = 0
              [one[None, :], np.zeros((1, 2))],                        # строка бюджета: 1^T x = 1
              [PF_R[None, :], np.zeros((1, 2))]])                      # строка доходности: r^T x = rho
rhs = np.concatenate([np.zeros(4), [1.0, rho]])                       # правая часть системы
sol = np.linalg.solve(K, rhs)                                          # 6 неизвестных: x, lambda_1, lambda_rho
x, lam1, lam_rho = sol[:4], sol[4], sol[5]
f_star = 0.5 * x @ PF_SIGMA @ x                                        # минимальный риск
print(f"доли x* = {x.round(4)}   (сумма {x.sum():.3f}, доходность {PF_R @ x:.3f})")
print(f"риск f* = {f_star:.6f}")
print(f"множители: lambda_1 = {lam1:.4f}, lambda_rho = {lam_rho:.4f}")
x_h, f_h, lam_h = portfolio_eq(rho)                                    # то же из l8helpers — для сверки
assert np.allclose(x, x_h) and np.allclose([lam1, lam_rho], lam_h) and np.isclose(f_star, f_h)''')

md(r"""**Что видим.** Все четыре доли положительны: при цели $10\,\%$ коротких позиций нет. Поэтому запрет шортов из лекции 3 здесь ничего бы не изменил. $\lambda_\rho=0.2571>0$: чем выше цель доходности, тем больше минимальный риск. Множитель бюджета $\lambda_1=-0.0093$ почти ноль.""")

# ============================================================ шаг 3
md(r"""## Шаг 3. Проверка формулы цены (7 мин)

**Главная мысль.** Множитель $\lambda_\rho$ равен наклону кривой $f^\ast(\rho)$. Проверим это численной производной.

**Что делает ячейка.** Решает систему при $\rho\pm h$ и считает центральную разность из лекции 7: $\big(f^\ast(\rho+h)-f^\ast(\rho-h)\big)/(2h)$. Сравнивает её с $\lambda_\rho$.""")

code(r'''h = 1e-5                                                               # шаг разности
fd = (portfolio_eq(rho + h)[1] - portfolio_eq(rho - h)[1]) / (2 * h)   # центральная разность df*/drho
print(f"df*/drho численно = {fd:.6f}")
print(f"lambda_rho        = {lam_rho:.6f}")
assert np.isclose(fd, lam_rho, atol=1e-8)''')

md(r"""**Что делает ячейка.** Поднимает цель на один процентный пункт, с $10$ до $11\,\%$. Сравнивает рост риска с предсказанием по множителю $\lambda_\rho\cdot0.01$.""")

code(r'''f_11 = portfolio_eq(0.11)[1]                                           # минимальный риск при цели 11 %
print(f"рост риска на деле:          {f_11 - f_star:.5f}")
print(f"по множителю lambda_rho·0.01: {lam_rho * 0.01:.5f}")
assert np.isclose(f_11 - f_star, lam_rho * 0.01, rtol=0.15)           # первый порядок: близко, но не точно''')

md(r"""**Что видим.** Численная производная совпадает с $\lambda_\rho$ до шести знаков: множитель — цена доходности. Лишний процентный пункт по множителю стоит $0.0026$ риска, а на деле — $0.0028$. Формула даёт производную, то есть первый порядок. Кривая $f^\ast(\rho)$ — выпуклая парабола, поэтому настоящий рост чуть больше.""")

# ============================================================ шаг 4
md(r"""## Шаг 4. Тот же портфель в IPOPT (5 мин)

**Главная мысль.** IPOPT возвращает те же множители с обратным знаком: его функция Лагранжа — $f+\lambda^\top g$ (§6 конспекта).

**Что делает ячейка.** Записывает портфель в CasADi. Ограничения — $g_1=\mathbf 1^\top x-1$ и $g_2=r^\top x-\rho$, аргументы `lbg=ubg=0` делают оба равенствами. Ячейка печатает `lam_g` и сравнивает его с $\lambda$ из шага 2.""")

code(r'''xs = ca.SX.sym("x", 4)                                                 # доли — символьный вектор из R^4
f = 0.5 * ca.mtimes([xs.T, ca.DM(PF_SIGMA), xs])                       # риск 1/2 x^T Sigma x
g = ca.vertcat(ca.sum1(xs) - 1, ca.dot(ca.DM(PF_R), xs) - rho)         # бюджет и доходность
solver = ca.nlpsol("s", "ipopt", {"x": xs, "f": f, "g": g}, {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
res = solver(x0=one / 4, lbg=0, ubg=0)                                 # старт — равные доли; lbg = ubg = 0 — равенства
lam_g = np.array(res["lam_g"]).ravel()                                 # множители в знаке IPOPT
print(f"IPOPT: {solver.stats()['iter_count']} итерация, x = {np.array(res['x']).ravel().round(4)}")
print(f"lam_g = {lam_g.round(4)},   наш lambda = {np.round([lam1, lam_rho], 4)}")
assert np.allclose(-lam_g, [lam1, lam_rho], atol=1e-6)''')

md(r"""**Что видим.** IPOPT нашёл те же доли за одну итерацию. Задача квадратичная, ограничения линейны, и один шаг метода Ньютона решает систему шага 2 точно. `lam_g` $=(0.0093,\ -0.2571)$ — наши множители с обратным знаком. Прежде чем читать цену из солвера, проверьте его знак.""")

# ============================================================ итог
md(r"""## Итог

- Множитель — цена ограничения: $df^\ast/db=\lambda^\ast$. В доказательстве слагаемые с производными $x^\ast$ и $\lambda^\ast$ умножаются на нули.
- У портфеля с равенствами система условий линейна. При цели $10\,\%$ цена доходности $\lambda_\rho=0.2571$.
- Формула даёт производную: на целом процентном пункте она ошибается во втором порядке, $0.0026$ против $0.0028$.
- IPOPT отдаёт `lam_g` $=-\lambda$.

**Дома:** упражнение 8.5 — портфель при целях от $6$ до $14\,\%$; 8.4(б) — цена длины цепи; 8.6 — почему $x^\ast(b)$ можно дифференцировать. Разбор — [`exercises08.ipynb`](exercises08.ipynb).""")


def main() -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbf.validate(nb)
    nbf.write(nb, OUT)
    n_code = sum(c.cell_type == "code" for c in cells)
    print(f"{OUT}: {len(cells)} ячеек, из них {n_code} с кодом")


if __name__ == "__main__":
    main()
