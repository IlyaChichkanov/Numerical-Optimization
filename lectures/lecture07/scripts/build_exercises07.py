"""Собирает разбор упражнений лекции 7: exercises07.ipynb (с проверкой кодом) и exercises07.md (кратко).

    uv run --group casadi python lectures/lecture07/scripts/build_exercises07.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture07/exercises07.ipynb

Разности, мини-АД и цепь импортируются из l7helpers.py (собирается build_lecture07.py) — тот же код, что в демо.
Условия задач берутся из раздела 9 lecture07.md при сборке и ставятся перед каждым решением
(блок «Условие»), чтобы разбор не расходился с конспектом. Шесть задач: разминка, U-кривая, прямой режим,
обратный режим и мини-задание лекции, мнимый трюк (для желающих), разреженность и jtimes.
"""

import re
from pathlib import Path

import nbformat as nbf

LECTURE = Path(__file__).resolve().parent.parent
OUT_NB = LECTURE / "exercises07.ipynb"
OUT_MD = LECTURE / "exercises07.md"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def statements() -> dict[str, tuple[str, str, str]]:
    """Условия задач из раздела 9 конспекта: номер -> (заголовок, вводная строка с сутью, текст условия)."""
    src = (LECTURE / "lecture07.md").read_text(encoding="utf-8")
    block = src[src.index("## 9. Упражнения"):src.index("## Вопросы для самопроверки")]
    out: dict[str, tuple[str, str, str]] = {}
    for chunk in re.split(r"\n(?=\*\*7\.\d\. )", block)[1:]:
        first, _, rest = chunk.strip().partition("\n")
        m = re.match(r"\*\*7\.(\d)\. (.+?)\*\*\s*(.*)$", first)
        assert m, first
        out[m.group(1)] = (m.group(2).rstrip("."), m.group(3).strip(), rest.strip())
    return out


def put_statements() -> None:
    """Перед каждым решением «## 7.N. …» вставляет условие из конспекта и подпись «Решение»."""
    st = statements()
    seen = set()
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        m = re.match(r"## 7\.(\d)\. ([^\n]+)\n(?:\n\*Суть:[^\n]*\*\n?)?(.*)$", cell.source, re.S)
        if not m:
            continue
        n, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        assert n in st and title == st[n][0], (n, title, st[n][0])
        _, intro, text = st[n]
        cell.source = (f"## 7.{n}. {title}\n\n**Условие.** {intro}\n\n{text}\n\n**Решение.**"
                       + (f"\n\n{body}" if body else ""))
        seen.add(n)
    assert seen == set(st), (sorted(seen), sorted(st))


md(r"""# Разбор упражнений к лекции 7

Условия — раздел 9 конспекта [`lecture07.md`](lecture07.md); краткие ответы без кода — [`exercises07.md`](exercises07.md). Все задачи домашние. **Сначала попробуйте сами**, потом сверяйтесь. Каждый разбор: условие → решение по пунктам → проверка кодом и картинка там, где она помогает. Нужен CasADi: `uv run --group casadi jupyter lab`; в Colab первая ячейка ставит его сама.""")

code(r'''%matplotlib inline

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from l7helpers import *  # разности, мини-АД, цепь (собираются из build_demo07.py)''')

# ---------------------------------------------------------------- 7.0
md(r"""## 7.0. Разминка — счёт руками

*Суть: две ошибки и счёт проходов — на бумаге.*

**(а)** $f=e^x$, $f''=1$, $\lvert f\rvert=1$ в нуле. Усечение $\tfrac h2$, округление $\tfrac{2\varepsilon}{h}$. При $h=10^{-4}$: усечение $5\cdot10^{-5}$, округление $4\cdot10^{-12}$ — главная ошибка усечения. При $h=10^{-12}$: усечение $5\cdot10^{-13}$, округление $4\cdot10^{-4}$ — главная ошибка округления, и она в десять раз больше, чем при $h=10^{-4}$.

**(б)** Якобиан $3\times1000$: прямому режиму нужен проход на каждый из $1000$ столбцов, обратному — на каждую из $3$ строк: $1000$ против $3$. Якобиан $1000\times3$ — наоборот: вперёд $3$, назад $1000$.

**(в)** Гессиан $n\times n$ — $n$ проходов «назад по прямому» (по одному на столбец $\nabla^2f\,e_i$), каждый ценой нескольких $f$; CasADi сокращает это за счёт разреженности.""")

code(r'''fexp = lambda x: np.exp(x[0])
for h in (1e-4, 1e-12):
    print(f"7.0(а) h = {h:.0e}: ошибка прямой разности {abs(fd_forward(fexp, np.array([0.0]), 0, h) - 1):.1e}  (усечение h/2 = {h / 2:.0e}, округление 2 eps / h = {2 * EPS / h:.0e})")''')

# ---------------------------------------------------------------- 7.1
md(r"""## 7.1. U-кривая своими руками

*Суть: наклоны веток — это показатели степени $h$ и $1/h$.*

**(а)** Розенброк, $\partial f/\partial x_1$ в $(-1.2,1)$, точно $-215.6$: прямая разность — лучший $h\approx6\cdot10^{-9}$, ошибка $1.4\cdot10^{-6}$; центральная — $h\approx2.5\cdot10^{-6}$, ошибка $10^{-9}$. Наклоны: справа $1$ и $2$, слева $-1$ — степени $h$ в ошибке усечения и $1/h$ в ошибке округления. Ошибка в минимуме здесь на три порядка больше, чем у $\sin x$ в лекции, потому что оценка $2\sqrt{\varepsilon|f||f''|}$ растёт с масштабом функции: $|f|=24$, $|f''|=1330$ дают $5\cdot10^{-6}$ против $2.5\cdot10^{-8}$ при $|f|=|f''|=0.84$.

**(б)** У квадратичной третья производная равна нулю, поэтому ошибка усечения центральной разности равна нулю при любом $h$: остаётся одна ветка — округление, растущее при $h\to0$; при $h=0.1$ ошибка уже на уровне $10^{-15}$.""")

code(r'''exact = rosen_grad(X0_ROSEN)[0]
ef, ec = ucurve(rosen, X0_ROSEN, 0, exact)
slope = lambda e, lo, hi: np.polyfit(np.log10(HS[(HS >= lo) & (HS <= hi)]), np.log10(e[(HS >= lo) & (HS <= hi)]), 1)[0]
for name, e, lo in (("прямая", ef, 1e-5), ("центральная", ec, 1e-4)):
    j = np.argmin(e); print(f"7.1(а) {name:12s}: лучший h {HS[j]:.1e}, ошибка {e[j]:.1e}, наклоны {slope(e, lo, 1e-1):.2f} / {slope(e, 1e-15, 1e-10):.2f}")
f11 = 2 - 400 * X0_ROSEN[1] + 1200 * X0_ROSEN[0] ** 2                 # d^2 f / d x1^2 в старте
print(f"7.1(а) оценка ошибки в минимуме 2 sqrt(eps |f| |f''|): Розенброк {2 * np.sqrt(EPS * rosen(X0_ROSEN) * f11):.1e}, sin x в 1: {2 * np.sqrt(EPS) * np.sin(1):.1e}")
fq = lambda x: 3 * x[0] ** 2 + x[0]
eq = np.array([abs(fd_central(fq, np.array([1.0]), 0, h) - 7) for h in HS])
print(f"7.1(б) квадратичная, центральная: при h = 0.1 ошибка {eq[-1]:.1e}, при h = 1e-12 — {eq[np.argmin(abs(HS - 1e-12))]:.1e}")
fig, ax = plt.subplots(figsize=(7.5, 4))
ucurve_plot(ax, [ef, ec, eq], ["Розенброк: прямая", "Розенброк: центральная", "3x²+x: центральная"], [BLUE, ORANGE, AQUA])
ax.set(title="7.1: U-кривые; у квадратичной — одна ветка"); plt.show()''')

# ---------------------------------------------------------------- 7.2
md(r"""## 7.2. Прямой режим руками

*Суть: $\dot v$ течёт вместе с $v$.*

В $(0,1)$ по $d=(1,1)$: $v_1=e^{0}=1$, $\dot v_1=e^{x_1}\dot x_1=1$; $v_2=v_1x_2=1$, $\dot v_2=\dot v_1x_2+v_1\dot x_2=1+1=2$; $v_3=x_1^2=0$, $\dot v_3=2x_1\dot x_1=0$; $f=v_2+v_3=1$, $\dot f=\dot v_2+\dot v_3=2$. Проверка: $\nabla f=(e^{x_1}x_2+2x_1,\ e^{x_1})=(1,1)$, $\nabla f^\top d=2$.""")

code(r'''PROGRAM2 = [("v1", "exp", ("x1",)), ("v2", "mul", ("v1", "x2")), ("v3", "sq", ("x1",)), ("f", "add", ("v2", "v3"))]
val, dot = forward_mode(PROGRAM2, {"x1": 0.0, "x2": 1.0}, {"x1": 1.0, "x2": 1.0})
print("7.2 значения:", {k: round(float(v), 4) for k, v in val.items()}, " прямой режим:", {k: round(float(v), 4) for k, v in dot.items()})
assert dot["f"] == 2.0''')

# ---------------------------------------------------------------- 7.3
md(r"""## 7.3. Обратный режим руками

*Суть: доли чувствительности складываются там, где у величины два потребителя.*

**(а)** $\bar f=1$; $f=v_2+v_3$: $\bar v_2=1$, $\bar v_3=1$; $v_3=x_1^2$: $\bar x_1\mathrel{+}=\bar v_3\cdot2x_1=0$; $v_2=v_1x_2$: $\bar v_1=\bar v_2x_2=1$, $\bar x_2=\bar v_2v_1=1$; $v_1=e^{x_1}$: $\bar x_1\mathrel{+}=\bar v_1e^{x_1}=1$. Итого $\nabla f=(1,1)$ — совпало с 7.2, и $\bar x_1$ собран из двух долей ($0$ через $x_1^2$ и $1$ через $e^{x_1}$).

**(б)** В $(1,\pi)$: $v_1=\pi$, $\bar v_2=1$, $\bar v_1=1+\cos\pi=0$, $\nabla f=(\bar v_1x_2,\ \bar v_1x_1)=(0,0)$ — стационарная точка: на кривой $x_1x_2=\pi$ производная $f$ по $v_1$ обращается в ноль.""")

code(r'''val, bar = reverse_mode(PROGRAM2, {"x1": 0.0, "x2": 1.0})
print("7.3(а) обратный режим:", {k: round(float(v), 4) for k, v in bar.items()})
x2 = ca.SX.sym("x", 2)
for name, prog, expr, pt in (("7.3(а)", PROGRAM2, ca.exp(x2[0]) * x2[1] + x2[0] ** 2, [0.0, 1.0]),
                             ("7.3(б)", PROGRAM, x2[0] * x2[1] + ca.sin(x2[0] * x2[1]), [1.0, np.pi])):
    _, bar = reverse_mode(prog, {"x1": pt[0], "x2": pt[1]})
    g_ca = np.array(ca.Function("g", [x2], [ca.gradient(expr, x2)])(pt)).ravel()
    print(f"{name}: руками ({bar['x1']:.4f}, {bar['x2']:.4f}), CasADi {g_ca.round(4)}")
    assert np.allclose([bar["x1"], bar["x2"]], g_ca, atol=1e-12)''')

# ---------------------------------------------------------------- 7.4
md(r"""## 7.4. Мнимый трюк

*Суть: вычитания нет — нет и ошибки округления.*

Для аналитической $f$: $f(x+ih)=f(x)+ihf'(x)-\tfrac{h^2}2f''(x)-i\tfrac{h^3}6f'''(x)+\dots$, мнимая часть $hf'(x)-\tfrac{h^3}{6}f'''(x)+\dots$, значит $\operatorname{Im}f(x+ih)/h=f'(x)+O(h^2)$. Вычитать нечего: мнимая часть считается с относительной точностью $\varepsilon$, и $h$ можно брать $10^{-20}$ — ошибка усечения $10^{-40}$, округления нет; на Розенброке ошибка $3\cdot10^{-14}$ при $\lvert f'\rvert=215.6$, то есть $10^{-16}$ относительно. Не работает там, где $f$ не аналитична: $\lvert x\rvert$ (модуль комплексного числа — вещественное, мнимая часть пропадает), `if x > 0` (сравнение комплексных чисел не определено), `max`, `round`.""")

code(r'''h = 1e-20
z = X0_ROSEN.astype(complex); z[0] += 1j * h
print(f"7.4 мнимый трюк: ошибка {abs(np.imag(rosen(z)) / h - rosen_grad(X0_ROSEN)[0]):.1e}; лучшая прямая разность: {ucurve(rosen, X0_ROSEN, 0, rosen_grad(X0_ROSEN)[0])[0].min():.1e}")''')

# ---------------------------------------------------------------- 7.5
md(r"""## 7.5. Разреженность и гессиан на вектор

*Суть: гессиан цепи — ленты, и CasADi это знает.*

Доля ненулевых падает как $1/N$: у каждой строки гессиана не больше шести ненулевых соседей при $2N$ столбцах ($28\,\%$ при $N=10$, $7\,\%$ при $40$, $2\,\%$ при $160$). Произведение $\nabla^2E\,d$ — один проход «назад по прямому» ценой $\approx3.3$ операций $E$ независимо от $N$: это прямой режим, применённый к программе обратного прохода, а она сама стоит $1.6$. Полный гессиан стоит $5$–$6$ цен $E$ только благодаря разреженности: CasADi собирает несколько столбцов за один проход.""")

code(r'''print(f"{'N':>5s} {'E':>6s} {'H':>6s} {'H d':>6s}   ненулевых")
for N in (10, 40, 160):
    nf, ng, nh, nhd, nnz = chain_costs(N)
    print(f"{N:5d} {nf:6d} {nh:6d} {nhd:6d}   {nnz} из {(2 * N) ** 2} = {100 * nnz / (2 * N) ** 2:.1f} %   (гессиан x{nh / nf:.1f}, H d x{nhd / nf:.1f})")''')

MD_TEXT = r"""# Упражнения к лекции 7 — краткие ответы

Разбор с проверкой кодом — [`exercises07.ipynb`](exercises07.ipynb). Условия — раздел 9 [конспекта](lecture07.md).

**7.0.** (а) $h=10^{-4}$: усечение $5\cdot10^{-5}$ главная; $h=10^{-12}$: округление $4\cdot10^{-4}$ главная. (б) $3\times1000$: вперёд $1000$, назад $3$; $1000\times3$: вперёд $3$, назад $1000$. (в) $n$ проходов «назад по прямому».

**7.1.** (а) Розенброк в $(-1.2,1)$: прямая — $h\approx6\cdot10^{-9}$, ошибка $1.4\cdot10^{-6}$; центральная — $h\approx2.5\cdot10^{-6}$, ошибка $10^{-9}$; наклоны $1$, $2$ справа и $-1$ слева; ошибка больше, чем у $\sin x$, из-за масштаба: $2\sqrt{\varepsilon|f||f''|}=5\cdot10^{-6}$ при $|f|=24$, $|f''|=1330$. (б) третья производная нулевая: усечения нет, одна ветка округления.

**7.2.** $\dot v_1=1$, $\dot v_2=2$, $\dot v_3=0$, $\dot f=2=\nabla f^\top d$ при $\nabla f=(1,1)$.

**7.3.** (а) $\bar v_2=\bar v_3=1$, $\bar v_1=1$, $\bar x_1=0+1=1$, $\bar x_2=1$. (б) $\bar v_1=1+\cos\pi=0$, $\nabla f=(0,0)$.

**7.4.** $\operatorname{Im}f(x+ih)/h=f'(x)+O(h^2)$ без вычитания; ошибка на Розенброке $3\cdot10^{-14}$ при $h=10^{-20}$; не работает для $\lvert x\rvert$, сравнений, `max`.

**7.5.** Ненулевых $28\,\%$, $7\,\%$, $2\,\%$ при $N=10,40,160$ — не больше шести на строку, доля $\sim1/N$; $\nabla^2E\,d$ — $\approx3.3$ цены $E$ при любом $N$ (один проход назад по прямому); полный гессиан $5$–$6$ цен благодаря разреженности.
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
