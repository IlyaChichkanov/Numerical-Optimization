"""Собирает семинар к лекции 6: seminar06.ipynb — глобализация в CasADi и IPOPT.

    uv run python lectures/lecture06/scripts/build_seminar06.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture06/seminar06.ipynb

~30 минут за ноутбуком: (1) CasADi как источник производных для наших методов, (2) что IPOPT делает
с длиной шага и с гессианом — те же Армихо и сдвиг tau I, (3) соревнование. Методы — из l6helpers.py.
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
| 10 | **1. CasADi как источник производных.** $\nabla f$, $\nabla^2 f$ и произведение $\nabla^2f\,v$ без матрицы; наш метод доверительной области на этих производных |
| 12 | **2. Глобализация внутри IPOPT.** Длина шага `alpha_pr` и сдвиг гессиана `regularization_size` — это правило Армихо и $\tau I$ из лекций 5–6; лог IPOPT построчно; L-BFGS внутри IPOPT |
| 8 | **3. Соревнование.** Химмельблау из трудного старта: кто придёт в минимум за меньшее число вычислений $f$ |

Установка: `uv sync --group casadi` в корне репозитория; в Colab первая ячейка поставит CasADi сама.""")

code(r'''import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import casadi as ca
import numpy as np
from scipy.optimize import minimize
from l6helpers import *          # armijo, newton_ls, trust_region, him, rosen, ... (лекция 6)

print("CasADi", ca.__version__)''')

# ============================================================ часть 1
md(r"""## Часть 1. CasADi как источник производных

**Шаг 1. Розенброк символьно.** Записываем $f$ один раз — CasADi строит градиент и гессиан сам (алгоритмическое дифференцирование, лекция 7). Проверка: гессиан в $(1,1)$ — $\begin{pmatrix}802&-400\\-400&200\end{pmatrix}$ (Л4).""")

code(r'''x = ca.SX.sym("x", 2)
f_r = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
F_r = ca.Function("F", [x], [f_r])
G_r = ca.Function("G", [x], [ca.gradient(f_r, x)])
H_r = ca.Function("H", [x], [ca.hessian(f_r, x)[0]])

H11 = np.array(H_r([1, 1]))
print(H11)
assert np.allclose(H11, [[802, -400], [-400, 200]])
assert np.allclose(np.array(G_r(X0_ROSEN)).ravel(), rosen_grad(X0_ROSEN))''')

md(r"""**Шаг 2. Гессиан на вектор — без матрицы.** `jtimes(e, x, v)` — производная выражения `e` по `x` в направлении `v`. Если `e` — градиент, получаем $\nabla^2f\,v$. Это то, что нужно Newton-CG и CG Стейхауга (раздел 4.3 конспекта): матрица $n\times n$ не строится, а стоит такое произведение как несколько градиентов. Проверяем на цепи из $400$ грузов, записанной **циклом** — как записали бы её в обычной программе.""")

code(r'''N = 400
E_py, gE, H_chain, b_chain, v0, unpack = make_chain(N)
w = ca.SX.sym("w", 2 * N)
D, m, g = 70.0, 4.0 / N, 9.81
yy = ca.vertcat(-2.0, w[:N], 2.0); zz = ca.vertcat(1.0, w[N:], 1.0)
E_ca = 0
for i in range(N + 1):                                  # энергия пружин и грузов — циклом
    E_ca += 0.5 * D * ((yy[i + 1] - yy[i]) ** 2 + (zz[i + 1] - zz[i]) ** 2)
for i in range(1, N + 1):
    E_ca += m * g * zz[i]

v = ca.SX.sym("v", 2 * N)
Hv = ca.Function("Hv", [w, v], [ca.jtimes(ca.gradient(E_ca, w), w, v)])
u = np.random.default_rng(2026).standard_normal(2 * N)
err = np.abs(np.array(Hv(v0, u)).ravel() - H_chain @ u).max()
print(f"|Hv(CasADi) - H v| = {err:.1e}")
assert err < 1e-9''')

md(r"""**Предскажите до запуска:** сколько итераций сделает CG на этой цепи, если вместо матрицы ему дать только функцию `Hv`? (Ответ из лекции — $N/2$.)""")

code(r'''Av = lambda vec: np.array(Hv(v0, vec)).ravel()           # для квадратичной функции Hv не зависит от точки
v_cg, k_cg, _ = cg(Av, -b_chain, v0)
print(f"CG с CasADi-произведениями: {k_cg} итераций, E = {E_py(v_cg):.4f}")
assert k_cg == N // 2
assert np.allclose(v_cg, np.linalg.solve(H_chain, -b_chain), atol=1e-6)''')

md(r"""**Шаг 3. Наш метод доверительной области на производных CasADi.** Функция `trust_region` из лекции принимает любые `f`, `grad`, `hess` — подставляем CasADi. На Розенброке должно получиться то же, что в конспекте: $25$ итераций, из них $3$ пробных шага отвергнуты.""")

code(r'''f_np = lambda z: float(F_r(z))
g_np = lambda z: np.array(G_r(z)).ravel()
h_np = lambda z: np.array(H_r(z))
x_tr, k_tr, path_tr, hist_tr = trust_region(f_np, g_np, h_np, X0_ROSEN, step="exact")
print(f"доверительная область: {k_tr} итераций, отвергнуто {sum(not h[4] for h in hist_tr)}, x = {x_tr.round(6)}")
assert k_tr == 25''')

md(r"""**Ловушка седла.** Химмельблау из $(0.58,\,2.15)$. Вопрос залу до запуска: куда придут Ньютон с Армихо (сдвиг $\tau I$ только при подъёме, как в ДЗ 3) и доверительная область?""")

code(r'''f_h = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
F_h, G_h, H_h = (ca.Function(n, [x], [e]) for n, e in
                 [("F", f_h), ("G", ca.gradient(f_h, x)), ("H", ca.hessian(f_h, x)[0])])
fh = lambda z: float(F_h(z)); gh = lambda z: np.array(G_h(z)).ravel(); hh = lambda z: np.array(H_h(z))
TRAP = np.array([0.58, 2.15])
x_ls, k_ls, _ = newton_ls(fh, gh, hh, TRAP)
x_tr, k_tr, _, _ = trust_region(fh, gh, hh, TRAP, step="exact", tol=1e-10)
for name, xx, kk in [("Ньютон + Армихо", x_ls, k_ls), ("доверительная область", x_tr, k_tr)]:
    lam = np.linalg.eigvalsh(hh(xx))
    print(f"{name}: {kk} итераций → {xx.round(4)}, собственные числа {lam.round(1)} — {'минимум' if lam[0] > 0 else 'седло'}")
assert np.allclose(x_tr, [3, 2]) and np.linalg.eigvalsh(hh(x_ls))[0] < 0''')

# ============================================================ часть 2
md(r"""## Часть 2. Глобализация внутри IPOPT

IPOPT — промышленный солвер ньютоновского типа (семинар 4: Розенброк примерно за 20 итераций). Внутри — те же два механизма, что в лекции: **линейный поиск** с делением шага пополам и **сдвиг гессиана** $\nabla^2f+\delta I$, когда он не положительно определён. `stats()["iterations"]` хранит по итерациям длину шага `alpha_pr` и сдвиг `regularization_size`.

**Шаг 4. Розенброк: длина шага.** Вопрос до запуска: какой `alpha_pr` будет на второй итерации? (В лекции: Армихо в точке $x_1$ выполнено при $\alpha\le0.176$, backtracking принимает $\tfrac18$.)""")

code(r'''opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
S = ca.nlpsol("S", "ipopt", {"x": x, "f": f_r}, opts)
sol = S(x0=X0_ROSEN)
st = S.stats(); it = st["iterations"]
print(f"IPOPT: {st['iter_count']} итераций, x = {np.array(sol['x']).ravel().round(6)}, вычислений f: {st['n_call_nlp_f']}")
print(" k   f            alpha_pr   сдвиг")
for k in range(len(it["obj"])):
    print(f"{k:2d}  {it['obj'][k]:11.6f}  {it['alpha_pr'][k]:8.4f}   {it['regularization_size'][k]:g}")
assert st["iter_count"] == 21 and it["alpha_pr"][2] == 0.125''')

md(r"""Сравним с нашим Ньютоном с Армихо: длины шагов на первых итерациях.""")

code(r'''xk = X0_ROSEN.copy(); ours = []
for k in range(12):
    gk = rosen_grad(xk); p = -np.linalg.solve(rosen_hess(xk), gk)
    a, _, _ = armijo(rosen, xk, rosen(xk), gk, p)
    ours.append(a); xk = xk + a * p
print("наш Ньютон + Армихо:", ours)
print("IPOPT alpha_pr:     ", [round(a, 4) for a in it["alpha_pr"][1:13]])
assert np.allclose(ours, it["alpha_pr"][1:13])''')

md(r"""Совпадают все двенадцать длин шага: $1,\ 0.125,\ 1,\ 1,\ 1,\ 0.25,\ 1,\ 1,\ 1,\ 0.5,\ 1,\ 1$. У IPOPT вместо одного правила Армихо — фильтр, свой останов и масштабирование задачи, но на задаче без ограничений механизм тот же: **полный шаг Ньютона, если функция с ним согласна, иначе — пополам**.

**Шаг 5. Химмельблау: сдвиг гессиана.** Из ловушки $(0.58,2.15)$ наш метод с Армихо пришёл в седло. В старте $\lambda_{\min}(\nabla^2f)=-31.3$. Что сделает IPOPT?""")

code(r'''S_h = ca.nlpsol("S_h", "ipopt", {"x": x, "f": f_h}, opts)
sol_h = S_h(x0=TRAP); st_h = S_h.stats(); it_h = st_h["iterations"]
print(f"IPOPT: {st_h['iter_count']} итераций → {np.array(sol_h['x']).ravel().round(6)}")
print("сдвиг гессиана по итерациям:", [round(r, 2) for r in it_h["regularization_size"]])
print(f"λ_min в старте: {np.linalg.eigvalsh(hh(TRAP))[0]:.1f}")
assert np.allclose(np.array(sol_h["x"]).ravel(), [3, 2], atol=1e-6)
assert it_h["regularization_size"][1] > abs(np.linalg.eigvalsh(hh(TRAP))[0])''')

md(r"""IPOPT сдвигает гессиан **всякий раз**, когда тот не положительно определён (а не только когда направление ведёт вверх): $\delta=100$, затем $33$ — больше $\lvert\lambda_{\min}\rvert=31.3$. Это «модифицированный Ньютон» из таблицы §2.5 конспекта — $100\,\%$ минимумов на Химмельблау. Проверим на сетке $25\times25$ стартов:""")

code(r'''count = {"минимум": 0, "седло": 0, "максимум": 0, "не сошёлся": 0}
for a in np.linspace(-6, 6, 25):
    for b_ in np.linspace(-6, 6, 25):
        s_ = S_h(x0=[a, b_])
        if not S_h.stats()["success"]:
            count["не сошёлся"] += 1; continue
        lam = np.linalg.eigvalsh(hh(np.array(s_["x"]).ravel()))
        count["минимум" if lam[0] > 0 else ("максимум" if lam[1] < 0 else "седло")] += 1
print(count)
assert count["минимум"] == 625''')

md(r"""**Шаг 6. Лог IPOPT построчно.** Тот же запуск с подробным логом (пишем в файл и печатаем таблицу итераций). Столбцы, которые теперь понятны:

- `alpha_pr` — принятая длина шага (`1` — полный ньютоновский, `1.25e-01` — три деления пополам);
- `ls` — сколько длин шага попробовал линейный поиск;
- `lg(rg)` — $\log_{10}$ сдвига гессиана $\delta$ (наш $\tau$); прочерк — сдвига не было.""")

code(r'''import os, tempfile
log = os.path.join(tempfile.mkdtemp(), "ipopt.log")
S_log = ca.nlpsol("S_log", "ipopt", {"x": x, "f": f_h},
                  {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes",
                   "ipopt.output_file": log, "ipopt.file_print_level": 5})
S_log(x0=TRAP)
lines = open(log, encoding="utf-8").read().splitlines()
start = next(i for i, l in enumerate(lines) if l.lstrip().startswith("iter"))
print("\n".join(lines[start:start + 10]))''')

md(r"""Попросите кого-нибудь прочитать вслух строку итерации 1 и перевести каждый столбец: значение $f$, невязки, $\lVert d\rVert$ — длина шага Ньютона, `lg(rg)` $=2$ — сдвиг $10^2=100$, `alpha_pr` — шаг принят целиком.

**Шаг 7. L-BFGS внутри IPOPT.** Если гессиана нет, IPOPT строит квази-ньютоновскую модель: `hessian_approximation="limited-memory"`. Вопрос до запуска: больше или меньше итераций, чем $21$?""")

code(r'''S_lb = ca.nlpsol("S_lb", "ipopt", {"x": x, "f": f_r}, opts | {"ipopt.hessian_approximation": "limited-memory"})
sol_lb = S_lb(x0=X0_ROSEN)
print(f"IPOPT с L-BFGS: {S_lb.stats()['iter_count']} итераций, x = {np.array(sol_lb['x']).ravel().round(6)}")''')

md(r"""Итераций больше (в нашей среде $47$ против $21$), но ни одной второй производной — как у BFGS из Л5: $34$ против $7$ у Ньютона.""")

# ============================================================ часть 3
md(r"""## Часть 3. Соревнование

Химмельблау из старта $(0,0)$ — оттуда чистый Ньютон приходит прямо в **максимум** $(-0.27,-0.92)$. Задача: прийти в **минимум** за наименьшее число вычислений $f$. Ниже — таблица для нескольких методов; добавьте свой (любой метод SciPy, свои параметры доверительной области, IPOPT с другими опциями) и побейте лидера.""")

code(r'''X0 = np.array([0.0, 0.0])
results = []

class Counter:
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

- CasADi даёт $\nabla f$, $\nabla^2f$ и **произведение $\nabla^2f\,v$ без матрицы** по записи функции — даже записанной циклом; на цепи из $400$ грузов CG с такими произведениями сходится за $200$ итераций.
- Наш метод доверительной области на CasADi-производных повторяет числа конспекта: $25$ итераций на Розенброке, минимум $(3,2)$ из ловушки седла.
- **IPOPT внутри — та же глобализация:** `alpha_pr` — backtracking (на Розенброке $0.125$ на второй итерации, как у нас), `regularization_size` / `lg(rg)` — сдвиг гессиана $\delta I$, который делается всякий раз, когда гессиан не положительно определён, поэтому IPOPT не приходит в сёдла Химмельблау.
- Квази-ньютоновская модель (`limited-memory`) — больше итераций, но без вторых производных.

**Дома:** упражнения 6.0–6.5, 6.7, 6.9, 6.10 (разбор — [`exercises06.ipynb`](exercises06.ipynb)); попробуйте в шаге 7 `ipopt.limited_memory_max_history` = 1, 5, 20 — как меняется число итераций?""")


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
