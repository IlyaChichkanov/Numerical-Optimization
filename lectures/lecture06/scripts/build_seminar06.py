"""Собирает семинар к лекции 6: seminar06.ipynb — глобализация в CasADi и IPOPT.

    uv run python lectures/lecture06/scripts/build_seminar06.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture06/seminar06.ipynb

~30 минут за ноутбуком: (1) производные из CasADi и наш метод доверительной области на них,
(2) что IPOPT делает с гессианом и длиной шага — тот же сдвиг tau I и тот же backtracking,
(3) соревнование. Методы — из l6helpers.py. 15.10 семинар сокращён с семи шагов до четырёх.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "seminar06.ipynb"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(r"""# Семинар 6. Глобализация в CasADi и IPOPT

Весь семинар — за ноутбуком, ~30 минут. Основы CasADi — семинар 4 ([`seminar04.ipynb`](../lecture04/seminar04.ipynb), Л4 §11.1): символьная переменная, `Function`, `nlpsol`.

| Мин | Часть |
|---|---|
| 8 | **1. Производные из CasADi и наш метод доверительной области.** $\nabla f$ и $\nabla^2f$ одной строкой; `trust_region` из лекции на этих производных; ловушка седла |
| 12 | **2. Что делает IPOPT.** Сдвиг гессиана `regularization_size` и длина шага `alpha_pr` — это $\tau I$ и backtracking из лекции |
| 10 | **3. Соревнование.** Химмельблау из трудного старта: кто придёт в минимум за меньшее число вычислений $f$ |

Установка: `uv sync --group casadi` в корне репозитория; в Colab первая ячейка поставит CasADi сама.""")

code(r'''import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import casadi as ca
import numpy as np
from scipy.optimize import minimize
from l6helpers import *          # newton, newton_ls, trust_region, rosen, him, ... (лекция 6)

print("CasADi", ca.__version__)''')

# ============================================================ часть 1
md(r"""## Часть 1. Производные из CasADi и наш метод доверительной области

**Шаг 1. Розенброк и Химмельблау символьно.** Записываем $f$ один раз — CasADi строит градиент и гессиан сам (алгоритмическое дифференцирование, лекция 7). Проверка: гессиан Розенброка в $(1,1)$ — $\begin{pmatrix}802&-400\\-400&200\end{pmatrix}$ (Л4).""")

code(r'''x = ca.SX.sym("x", 2)
f_r = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2                 # Розенброк
f_h = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2       # Химмельблау

def derivatives(f):
    """f, grad f, hess f как функции numpy-векторов — из одного символьного выражения."""
    F, G, H = (ca.Function(n, [x], [e]) for n, e in [("F", f), ("G", ca.gradient(f, x)), ("H", ca.hessian(f, x)[0])])
    return (lambda z: float(F(z)), lambda z: np.array(G(z)).ravel(), lambda z: np.array(H(z)))

fr, gr, hr = derivatives(f_r)
fh, gh, hh = derivatives(f_h)
print(hr([1, 1]))
assert np.allclose(hr([1, 1]), [[802, -400], [-400, 200]])
assert np.allclose(gr(X0_ROSEN), rosen_grad(X0_ROSEN)) and np.allclose(hh([3, 2]), him_hess(np.array([3.0, 2.0])))''')

md(r"""**Шаг 2. Наш метод доверительной области на производных CasADi.** `trust_region` из лекции принимает любые `f`, `grad`, `hess` — подставляем CasADi. Ожидаем числа конспекта: $25$ итераций на Розенброке, из них $3$ пробных шага отвергнуты.""")

code(r'''x_tr, k_tr, _, hist_tr = trust_region(fr, gr, hr, X0_ROSEN, step="exact")
print(f"доверительная область: {k_tr} итераций, отвергнуто {sum(not h[4] for h in hist_tr)}, x = {x_tr.round(6)}")
assert k_tr == 25''')

md(r"""**Ловушка седла.** Химмельблау из $(0.58,\,2.15)$. Вопрос залу до запуска: куда придут Ньютон с Армихо (сдвиг $\tau I$ только когда шаг не спуск, как в ДЗ 3) и доверительная область?""")

code(r'''TRAP = np.array([0.58, 2.15])
x_ls, k_ls, _ = newton_ls(fh, gh, hh, TRAP)                                  # Ньютон + Армихо
x_tr, k_tr, _, _ = trust_region(fh, gh, hh, TRAP, step="exact", tol=1e-10)   # доверительная область
for name, xx, kk in [("Ньютон + Армихо", x_ls, k_ls), ("доверительная область", x_tr, k_tr)]:
    lam = np.linalg.eigvalsh(hh(xx))
    print(f"{name}: {kk} итераций → {xx.round(4)}, собственные числа {lam.round(1)} — {'минимум' if lam[0] > 0 else 'седло'}")
assert np.allclose(x_tr, [3, 2]) and np.linalg.eigvalsh(hh(x_ls))[0] < 0''')

# ============================================================ часть 2
md(r"""## Часть 2. Что делает IPOPT

IPOPT — промышленный солвер ньютоновского типа (семинар 4: Розенброк примерно за 20 итераций). Внутри — те же два механизма, что в лекции: **сдвиг гессиана** $\nabla^2f+\delta I$, когда он не положительно определён, и **линейный поиск** с делением шага пополам. `stats()["iterations"]` хранит по итерациям сдвиг `regularization_size` и длину шага `alpha_pr`.

**Шаг 3. Химмельблау из ловушки: сдвиг гессиана.** Наш Ньютон с Армихо пришёл отсюда в седло; в старте $\lambda_{\min}(\nabla^2f)=-31.3$. Вопрос до запуска: куда придёт IPOPT?""")

code(r'''opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
S_h = ca.nlpsol("S_h", "ipopt", {"x": x, "f": f_h}, opts)
sol_h = S_h(x0=TRAP); st_h = S_h.stats(); it_h = st_h["iterations"]
print(f"IPOPT: {st_h['iter_count']} итераций → {np.array(sol_h['x']).ravel().round(6)}")
print("сдвиг гессиана по итерациям:", [round(r, 2) for r in it_h["regularization_size"]])
print(f"λ_min гессиана в старте: {np.linalg.eigvalsh(hh(TRAP))[0]:.1f}")
assert np.allclose(np.array(sol_h["x"]).ravel(), [3, 2], atol=1e-6)
assert it_h["regularization_size"][1] > abs(np.linalg.eigvalsh(hh(TRAP))[0])''')

md(r"""IPOPT сдвигает гессиан **всякий раз**, когда тот не положительно определён (а не только когда направление ведёт вверх): $\delta=100$, затем $33$ — больше $\lvert\lambda_{\min}\rvert=31.3$. Это строка «сдвиг всегда» из таблицы §3.5 конспекта — $100\,\%$ минимумов на Химмельблау: седло не притягивает.

**Шаг 4. Розенброк: длина шага.** Вопрос до запуска: какой `alpha_pr` будет на второй итерации? (В лекции backtracking в точке $x_1$ принимает $\tfrac18$.)""")

code(r'''S = ca.nlpsol("S", "ipopt", {"x": x, "f": f_r}, opts)
sol = S(x0=X0_ROSEN); st = S.stats(); it = st["iterations"]
print(f"IPOPT: {st['iter_count']} итераций, x = {np.array(sol['x']).ravel().round(6)}, вычислений f: {st['n_call_nlp_f']}")
print(" k   f            alpha_pr   сдвиг")
for k in range(len(it["obj"])):
    print(f"{k:2d}  {it['obj'][k]:11.6f}  {it['alpha_pr'][k]:8.4f}   {it['regularization_size'][k]:g}")
assert st["iter_count"] == 21 and it["alpha_pr"][2] == 0.125''')

md(r"""На второй итерации `alpha_pr` $=0.125$ — три деления пополам, ровно как у нашего backtracking в §2.1. У IPOPT вместо одного правила Армихо — фильтр, свой останов и масштабирование задачи, но на задаче без ограничений механизм тот же: **полный шаг Ньютона, если функция с ним согласна, иначе — пополам**.""")

# ============================================================ часть 3
md(r"""## Часть 3. Соревнование

Химмельблау из старта $(0,0)$ — оттуда чистый Ньютон приходит прямо в **максимум** $(-0.27,-0.92)$. Задача: прийти в **минимум** за наименьшее число вычислений $f$. Ниже — таблица для нескольких методов; добавьте свой (любой метод SciPy, свои параметры доверительной области, IPOPT с другими опциями) и побейте лидера.""")

code(r'''X0 = np.array([0.0, 0.0])
results = []

class Counter:
    """Считает вызовы f."""
    def __init__(self, f): self.f, self.n = f, 0
    def __call__(self, z): self.n += 1; return self.f(z)

def kind(z):
    lam = np.linalg.eigvalsh(hh(np.asarray(z, float)))
    return "минимум" if lam[0] > 0 else ("максимум" if lam[1] < 0 else "седло")

x_n, k_n, _ = newton(gh, hh, X0)
results.append(("чистый Ньютон", k_n, k_n, kind(x_n)))
for shift in ["ascent", "always"]:
    fc = Counter(fh); x_l, k_l, _ = newton_ls(fc, gh, hh, X0, shift=shift)
    results.append((f"Ньютон + Армихо ({shift})", k_l, fc.n, kind(x_l)))
fc = Counter(fh); x_t, k_t, _, _ = trust_region(fc, gh, hh, X0, step="exact")
results.append(("наша доверительная область", k_t, fc.n, kind(x_t)))
for m_ in ["BFGS", "trust-exact", "Newton-CG"]:
    kw = {"hess": hh} if m_ != "BFGS" else {}
    r = minimize(fh, X0, jac=gh, method=m_, **kw)
    results.append((f"SciPy {m_}", r.nit, r.nfev, kind(r.x)))
s_ = S_h(x0=X0); st_ = S_h.stats()
results.append(("IPOPT", st_["iter_count"], st_["n_call_nlp_f"], kind(np.array(s_["x"]).ravel())))

print(f"{'метод':30s} итераций  вычислений f   куда пришёл")
for name, k, nf, kd in sorted(results, key=lambda r: (r[3] != "минимум", r[2])):
    print(f"{name:30s} {k:7d}  {nf:12d}   {kd}")''')

md(r"""## Итог

- CasADi даёт $\nabla f$ и $\nabla^2f$ по записи функции; наш метод доверительной области на них повторяет числа конспекта: $25$ итераций на Розенброке, минимум $(3,2)$ из ловушки седла, куда Ньютон с Армихо не попадает.
- **IPOPT внутри — та же глобализация:** `regularization_size` — сдвиг гессиана $\delta I$ всякий раз, когда гессиан не положительно определён (поэтому IPOPT не приходит в сёдла Химмельблау), `alpha_pr` — backtracking (на Розенброке $0.125$ на второй итерации, как у нас).
- Соревнование: из $(0,0)$ чистый Ньютон идёт в максимум; кто из остальных приходит в минимум дешевле всех — видно в таблице выше.

**Дома:** упражнения 6.0–6.6 (разбор — [`exercises06.ipynb`](exercises06.ipynb)).""")


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
