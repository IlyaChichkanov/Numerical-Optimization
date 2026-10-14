"""Собирает разбор упражнений лекции 6: exercises06.ipynb (с проверкой кодом) и exercises06.md (кратко).

    uv run python lectures/lecture06/scripts/build_exercises06.py
    uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture06/exercises06.ipynb

Методы и задачи импортируются из l6helpers.py (собирается build_lecture06.py) — тот же код, что в демо.
"""

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


md(r"""# Разбор упражнений к лекции 6

Условия — раздел 9 конспекта [`lecture06.md`](lecture06.md); краткие ответы без кода — [`exercises06.md`](exercises06.md). На семинаре разбираем 6.6 и 6.8, остальное — дома. **Сначала попробуйте сами**, потом сверяйтесь. Каждый разбор: условие с сутью → решение по пунктам → проверка кодом.""")

code(r'''%matplotlib inline

import numpy as np
from scipy.optimize import minimize
from l6helpers import *  # методы и задачи лекции (собираются из build_demo06.py)''')

# ---------------------------------------------------------------- 6.0
md(r"""## 6.0. Разминка — счёт руками

*Суть: Армихо, Вольфе, $\rho$, точка Коши и CG — на бумаге.*

**(а)** $\varphi(\alpha)=(1-2\alpha)^2$, $\varphi(0)=1$, $\varphi'(0)=-4$. Армихо с $c_1=\tfrac12$: $(1-2\alpha)^2\le1-2\alpha\iff4\alpha^2\le2\alpha\iff\alpha\le\tfrac12$. Backtracking: $\alpha=1$ даёт $\varphi=1>-1$ — отказ; $\alpha=\tfrac12$ даёт $\varphi=0\le0$ — принят (это точный минимум).

**(б)** Вольфе: $\varphi'(\alpha)=-4(1-2\alpha)\ge0.9\cdot(-4)\iff\alpha\ge0.05$. Вместе с Армихо — $[0.05,\ 0.5]$. Сильное условие $\lvert1-2\alpha\rvert\le0.9$ даёт $\alpha\in[0.05,0.95]$, с Армихо — то же $[0.05,0.5]$.

**(в)** $\rho=-1/4$: шаг отвергается, $\Delta\leftarrow\Delta/4$, подзадача решается заново из той же точки.

**(г)** $g^\top Bg=1$, $\lVert g\rVert=1$. При $\Delta=0.5$: $\tau=\min(1/(0.5\cdot1),1)=1$, $p_C=(-0.5,0)$ — на границе. При $\Delta=2$: $\tau=\min(1/2,1)=\tfrac12$, $p_C=(-1,0)$ — минимум вдоль $-g$ внутри шара.

**(д)** Два различных собственных числа — не больше $2$ итераций; пять различных — не больше $5$.

**(е)** $p=-B^{-1}g=(-1,\ 0.25)$, $g^\top p=-0.75<0$ — спуск, хотя $B$ индефинитна (ведёт вниз «по случайности» $g$). $B+\tau I\succ0$ при $\tau>4$.""")

code(r'''phi = lambda a: (1 - 2 * a) ** 2
print("6.0(а): backtracking", [(a, phi(a) <= 1 - 0.5 * a * 4) for a in (1, 0.5)])
g, B = np.array([1.0, 0.0]), np.diag([1.0, 10.0])
print("6.0(г): Коши при Δ=0.5", cauchy_point(g, B, 0.5), " при Δ=2", cauchy_point(g, B, 2.0))
for A in (np.diag([1, 1, 5, 5, 5.0]), np.diag([1, 2, 3, 4, 5.0])):
    _, k, _ = cg(lambda v, A=A: A @ v, np.ones(5), np.zeros(5), tol=1e-12)
    print("6.0(д): CG итераций", k)
B = np.diag([1.0, -4.0]); g = np.array([1.0, 1.0]); p = -np.linalg.solve(B, g)
print("6.0(е): p =", p, " g^T p =", g @ p)''')

# ---------------------------------------------------------------- 6.1
md(r"""## 6.1. Зачем $c_1$

*Суть: «просто уменьшение» не гарантирует сходимости.*

**(а)** $f(x_k)=(1+2^{-k})^2$ убывает, но $x_k$ прыгает между окрестностями $\pm1$, где $f'=\pm2\ne0$: предел — не стационарная точка.

**(б)** Направление $p_k=-\operatorname{sign}(x_k)$ (спуск, $\lvert f'\rvert>0$), длина $\alpha_k=\lvert x_k\rvert+\lvert x_{k+1}\rvert\approx2$. Обещание Армихо: уменьшить хотя бы на $c_1\alpha_k\lvert f'(x_k)\rvert\approx4c_1$. Фактическое уменьшение $(1+2^{-k})^2-(1+2^{-k-1})^2\approx2^{-k}\to0$. При $2^{-k}<4c_1$ правило нарушено — такие шаги Армихо не пропустит.""")

code(r'''c1 = 1e-4
for k in [5, 10, 12, 13, 15]:
    xk, xk1 = (-1) ** k * (1 + 2.0 ** -k), (-1) ** (k + 1) * (1 + 2.0 ** -(k + 1))
    alpha = abs(xk) + abs(xk1); slope = -abs(2 * xk)
    print(f"k = {k:2d}: уменьшение {xk**2 - xk1**2:.2e}, требуется ≥ {-c1 * alpha * slope:.2e} → Армихо {'да' if xk1**2 <= xk**2 + c1 * alpha * slope else 'нет'}")''')

# ---------------------------------------------------------------- 6.2, 6.3
md(r"""## 6.2. Вольфе и BFGS

*Суть: условие кривизны — это ровно то, что нужно для $s^\top y>0$.*

**(а)** $s^\top y=\alpha\,(g_{k+1}-g_k)^\top p_k=\alpha\big(\varphi'(\alpha)-\varphi'(0)\big)\ge\alpha(c_2-1)\varphi'(0)=(1-c_2)\,\alpha\,\lvert g_k^\top p_k\rvert>0$.

**(б)** Прямая Армихо $\ell(\alpha)=\varphi(0)+c_1\alpha\varphi'(0)$ уходит вниз без ограничения, а $\varphi$ ограничена снизу — они пересекаются; пусть $\alpha'$ — первое пересечение. По теореме о среднем есть $\alpha''\in(0,\alpha')$ с $\varphi'(\alpha'')=c_1\varphi'(0)>c_2\varphi'(0)$, и в его окрестности выполнены оба условия (Nocedal & Wright, лемма 3.1).

**(в)** Нелинейному CG нужна почти точная минимизация вдоль направления, иначе следующее направление теряет сопряжённость и может даже не быть спуском — отсюда $c_2=0.1$–$0.4$. BFGS довольно $s^\top y>0$ и дешёвого шага — $c_2=0.9$.

## 6.3. Глобальная сходимость

*Суть: каждый шаг съедает кусок $\lVert g\rVert^2$, а съесть можно лишь конечное.*

**(а)** Лемма о спуске: $\varphi(\alpha)\le\varphi(0)+\alpha\varphi'(0)+\tfrac L2\alpha^2\lVert p\rVert^2$. Армихо выполнено, если $\tfrac L2\alpha\lVert p\rVert^2\le(1-c_1)\lvert\varphi'(0)\rvert$, т. е. при $\alpha\le\bar\alpha$. Отвергнутый шаг $2\alpha$ был больше $\bar\alpha$, значит $\alpha\ge\min(1,\bar\alpha/2)$. Подставляем $\lvert g^\top p\rvert\ge\lVert g\rVert^2/M_B$ и $\lVert p\rVert\le\lVert g\rVert/m_B$ — получаем $\alpha_{\min}$ из конспекта.

**(б)** При $m_B\to0$ направление почти ортогонально градиенту ($\cos\theta\ge m_B/M_B\to0$), $\alpha_{\min}\to0$ — оценка вырождается. При индефинитной $B_k$ шаг может быть подъёмом, и правило Армихо вообще не применимо.

**(в)** Для спуска $B=I$, $m_B=M_B=1$: $\lVert\nabla f\rVert\to0$ из любого старта. На Розенброке это гарантирует сходимость к $(1,1)$ (единственная стационарная точка), но ничего не говорит о скорости — $13\,756$ итераций.""")

# ---------------------------------------------------------------- 6.4
md(r"""## 6.4. Сломайте Ньютона: сходимость в седло

*Суть: глобальная сходимость — к стационарной точке, а не к минимуму.*""")

code(r'''x, k, _ = newton_ls(him, him_grad, him_hess, (0.58, 2.15))
print(f"(а) из (0.58, 2.15): {k} итераций → {x.round(4)}, собственные числа гессиана {np.linalg.eigvalsh(him_hess(x)).round(2)} — седло")

rng = np.random.default_rng(2026)
found = {}
for x0 in rng.uniform(-6, 6, (300, 2)):
    xs, ks, _ = newton_ls(him, him_grad, him_hess, x0)
    lam = np.linalg.eigvalsh(him_hess(xs))
    if lam[0] < 0 < lam[1]:
        key = tuple(xs.round(3))
        if key not in found and key != (0.087, 2.884):
            found[key] = x0
    if len(found) >= 2:
        break
for saddle, x0 in found.items():
    xa, ka, _ = newton_ls(him, him_grad, him_hess, x0, shift="always")
    print(f"(б) старт {x0.round(2)} → седло {saddle};  (в) со сдвигом «всегда» → {xa.round(3)}")''')

md(r"""**(в)** При сдвиге «всегда, когда $\lambda_{\min}\le10^{-3}$» матрица $B_k\succ0$ на каждом шаге. Рядом с седлом компонента шага вдоль собственного вектора с отрицательным $\lambda$ направлена *от* седла (вдоль него функция убывает), поэтому седло отталкивает. Без сдвига (направление Ньютона — спуск и так) метод честно идёт в седловую точку модели.""")

# ---------------------------------------------------------------- 6.5
md(r"""## 6.5. Точка Коши

*Суть: точка Коши даёт сходимость, но скорость спуска.*

**(а)** $m(-tg)=f-t\lVert g\rVert^2+\tfrac12t^2g^\top Bg$ при $0\le t\le\Delta/\lVert g\rVert$. Если $g^\top Bg\le0$, функция убывает — берём границу ($\tau=1$). Иначе минимум параболы $t^\ast=\lVert g\rVert^2/g^\top Bg$ — обрезаем границей: $\tau=\min\big(\lVert g\rVert^3/(\Delta\,g^\top Bg),\,1\big)$.

**(б)** Точка Коши — это спуск с точным шагом (Л4 §4.4), обрезанным шаром, — та же линейная скорость $\sim\kappa$.""")

code(r'''for step in ["cauchy", "dogleg"]:
    x, k, _, h = trust_region(rosen, rosen_grad, rosen_hess, X0_ROSEN, step=step)
    print(f"{step}: {k} итераций")
x, k, _ = gd(rosen, rosen_grad, X0_ROSEN)
print(f"спуск с backtracking (Л4): {k} итераций")''')

# ---------------------------------------------------------------- 6.6
md(r"""## 6.6. Dogleg руками

*Суть: ломаная Пауэлла — компромисс между точкой Коши и точкой Ньютона.*

$g=(4,2)$, $B=\operatorname{diag}(2,1)$: $p_B=-B^{-1}g=(-2,-2)$, $\lVert p_B\rVert=2.83$; $p_U=-\frac{g^\top g}{g^\top Bg}g=-\frac{20}{36}(4,2)=(-2.22,-1.11)$, $\lVert p_U\rVert=2.48$.

- $\Delta=1<\lVert p_U\rVert$: первое колено, $p=-\Delta\,g/\lVert g\rVert=(-0.89,-0.45)$.
- $\Delta=2.6$: между $\lVert p_U\rVert$ и $\lVert p_B\rVert$ — второе колено, $p=p_U+t(p_B-p_U)$ с $\lVert p\rVert=2.6$: $t\approx0.43$, $p\approx(-2.13,-1.50)$.
- $\Delta=3>\lVert p_B\rVert$: шаг Ньютона $(-2,-2)$.""")

code(r'''g, B = np.array([4.0, 2.0]), np.diag([2.0, 1.0])
for D in (1.0, 2.6, 3.0):
    p = dogleg(g, B, D)
    print(f"Δ = {D}: dogleg {p.round(3)}, |p| = {np.linalg.norm(p):.3f}, модель {g @ p + 0.5 * p @ B @ p:.3f}")''')

# ---------------------------------------------------------------- 6.7
md(r"""## 6.7. Левенберг–Марквардт = доверительная область

*Суть: $\lambda$ и $\Delta$ — две ручки одного механизма.*

**(а)** $p(\lambda)=-\big(\tfrac1{1+\lambda},\ \tfrac1{100+\lambda}\big)$ — длины ниже; каждому $\lambda$ соответствует $\Delta=\lVert p(\lambda)\rVert$, при котором этот шаг — решение подзадачи (условия Moré–Sorensen).

**(б)** $\frac{d}{d\lambda}\lVert p(\lambda)\rVert^2=-2\sum_i\frac{\tilde g_i^2}{(\lambda_i+\lambda)^3}<0$: длина монотонно убывает, поэтому «уменьшить $\Delta$» и «увеличить $\lambda$» — одно и то же.""")

code(r'''JTJ, JTr = np.diag([1.0, 100.0]), np.array([1.0, 1.0])
for lam in [0, 1, 10, 100]:
    p = -np.linalg.solve(JTJ + lam * np.eye(2), JTr)
    D = np.linalg.norm(p)
    print(f"λ = {lam:3d}: p = {p.round(4)}, Δ = |p| = {D:.4f}; точный шаг подзадачи при этом Δ: {tr_exact(JTr, JTJ, D).round(4)}")''')

# ---------------------------------------------------------------- 6.8
md(r"""## 6.8. CG руками на эллипсе

*Суть: два шага — и точный минимум.*

$r_0=(50,50)$, $d_0=-(50,50)$, $\alpha_0=\frac{5000}{127\,500}=0.0392$, $x_1=(48.04,\,-0.961)$; $r_1=(48.04,\,-48.04)$, $\beta_1=0.923$, $d_1=(-94.2,\ 1.88)$, $\alpha_1=0.510$, $x_2=(0,0)$. Сопряжённость: $d_1^\top Ad_0=0$; невязки ортогональны: $r_1^\top r_0=0$. Спуск с точным шагом после двух шагов — в $(46.2,\ 0.92)$: первый шаг тот же, второй уже зигзаг.""")

code(r'''A = np.diag([1.0, 50.0]); x = np.array([50.0, 1.0]); r = A @ x; d = -r; hist = [(x.copy(), r.copy(), d.copy())]
for _ in range(2):
    a = (r @ r) / (d @ A @ d); x = x + a * d; rn = r + a * A @ d; d = -rn + (rn @ rn) / (r @ r) * d; r = rn
    hist.append((x.copy(), r.copy(), d.copy()))
print("x1 =", hist[1][0].round(3), " x2 =", hist[2][0].round(10))
print("d1^T A d0 =", round(hist[1][2] @ A @ hist[0][2], 8), "  r1^T r0 =", round(hist[1][1] @ hist[0][1], 8))
x = np.array([50.0, 1.0])
for _ in range(2):
    g = A @ x; x = x - (g @ g) / (g @ A @ g) * g
print("спуск с точным шагом после двух шагов:", x.round(3))''')

# ---------------------------------------------------------------- 6.9
md(r"""## 6.9. Почему на цепи $N/2$

*Суть: CG сходится за число «задетых» собственных чисел.*""")

code(r'''E, gE, H, b, v0, unpack = make_chain(40)
lam, V = np.linalg.eigh(H)
e0 = V.T @ (v0 - np.linalg.solve(H, -b))
hit = np.abs(e0) > 1e-12
print(f"(а) компонент начальной ошибки > 1e-12: {hit.sum()} из {len(e0)}; различных задетых собственных чисел: {len(np.unique(lam[hit].round(8)))}")
y_err = (v0 - np.linalg.solve(H, -b))[:40]
z_err = (v0 - np.linalg.solve(H, -b))[40:]
print(f"(б) горизонтальная ошибка max|.| = {np.abs(y_err).max():.1e}; вертикальная симметрична: {np.allclose(z_err, z_err[::-1])}")
v_rand = v0 + 0.1 * np.random.default_rng(2026).standard_normal(80)
_, k, _ = cg(lambda v: H @ v, -b, v_rand)
print(f"(в) из несимметричного старта CG делает {k} итераций (= N)")''')

md(r"""**(б)** Горизонтальные координаты в старте — равномерная сетка между концами — это и есть решение (по горизонтали на грузы действуют только пружины). Вертикальная ошибка симметрична относительно середины цепи, а у матрицы $K$ ровно $N/2$ симметричных собственных векторов $\sin\frac{jk\pi}{N+1}$ с нечётным $k$.""")

# ---------------------------------------------------------------- 6.10
md(r"""## 6.10. Цепь, BFGS и предел точности

*Суть: ниже $\sqrt{\lambda_{\max}\varepsilon\lvert f\rvert}$ значения функции не различают «лучше» и «хуже», а плохой масштаб $H_0$ портит BFGS сильнее, чем отсутствие Вольфе.*""")

code(r'''E, gE, H, b, v0, unpack = make_chain(20)
r = minimize(E, v0, jac=gE, method="BFGS", options={"gtol": 1e-6})
print(f"(а) SciPy BFGS, N = 20: {r.message!r}, |∇E| = {np.linalg.norm(gE(r.x)):.1e} после {r.nit} итераций")
lam_max = np.linalg.eigvalsh(H)[-1]; E_star = E(np.linalg.solve(H, -b))
print(f"(б) пол sqrt(2 λmax ε |E*|) = {np.sqrt(2 * lam_max * np.finfo(float).eps * abs(E_star)):.1e}")
for N in [10, 20, 40, 80]:
    EN, gEN, _, _, v0N, _ = make_chain(N)
    print(f"(г) N = {N}: BFGS с масштабированной H0 — {bfgs(EN, gEN, v0N, scale_H0=True)[1]} итераций, "
          f"с H0 = I — {bfgs(EN, gEN, v0N, maxit=3000)[1]}")''')

md(r"""**(в)** Точный шаг на квадратичной считается формулой $\alpha=-g^\top p/p^\top Hp$, CG — тоже: значения $E$ не сравниваются, точность ограничена только округлением в $Hv+b$, а это $\sim\varepsilon\lVert H\rVert\lVert v\rVert$ — порядка $10^{-13}$.

**(г)** На выпуклой квадратичной $s^\top y>0$ всегда — Вольфе ничего не добавляет. Мешает масштаб: $H_0=I$ в сотни раз «длиннее» обратного гессиана, первые шаги режутся в крошки, пары $(s,y)$ крошечные. Масштаб $\frac{s^\top y}{y^\top y}$ — оценка $1/\lambda$ вдоль шага — сразу ставит $H_0$ в правильный порядок.""")

MD_TEXT = r"""# Упражнения к лекции 6 — краткие ответы

Разбор с проверкой кодом — [`exercises06.ipynb`](exercises06.ipynb). Условия — раздел 9 [конспекта](lecture06.md).

**6.0.** (а) Армихо при $\alpha\le\tfrac12$; backtracking: $1$ — отказ, $\tfrac12$ — принят. (б) Вольфе: $\alpha\ge0.05$, вместе — $[0.05,0.5]$; сильное — то же. (в) $\rho=-0.25$: отвергнуть, $\Delta/4$. (г) $\Delta=0.5$: $p_C=(-0.5,0)$; $\Delta=2$: $p_C=(-1,0)$. (д) $2$ и $5$ итераций. (е) $p=(-1,0.25)$, $g^\top p=-0.75$ — спуск; $\tau>4$.

**6.1.** (а) $f$ убывает к $1$, $x_k\to\pm1$, $f'\to\pm2$. (б) $p_k=-\operatorname{sign}x_k$, $\alpha_k\approx2$; обещание $\approx4c_1$, факт $\approx2^{-k}$ — Армихо нарушено при $2^{-k}<4c_1$ ($k\ge12$ при $c_1=10^{-4}$).

**6.2.** (а) $s^\top y\ge(1-c_2)\alpha\lvert g^\top p\rvert$. (б) пересечение $\varphi$ с прямой Армихо + теорема о среднем. (в) CG нужна почти точная минимизация вдоль направления, BFGS — только $s^\top y>0$.

**6.3.** (а) $\alpha\ge\min(1,\bar\alpha/2)$, $\bar\alpha=2(1-c_1)\lvert g^\top p\rvert/(L\lVert p\rVert^2)$. (б) $m_B\to0$ — угол к градиенту стремится к $90^\circ$; индефинитная $B$ — шаг может быть подъёмом. (в) Спуск сходится из любого старта, скорость теорема не даёт.

**6.4.** (а) седло $(0.087,2.884)$, $\lambda=(-31.7,\ 75.5)$, 5 итераций. (б) из других стартов — сёдла $(-0.128,-1.954)$, $(-3.073,-0.081)$, $(3.385,0.074)$. (в) со сдвигом «всегда» — минимумы: седло отталкивает.

**6.5.** (а) $\tau=\min(\lVert g\rVert^3/(\Delta g^\top Bg),1)$, при $g^\top Bg\le0$ — $1$. (б) $14\,110$ против $24$ у dogleg и $13\,756$ у спуска: точка Коши — спуск с точным шагом.

**6.6.** $p_B=(-2,-2)$, $p_U=(-2.22,-1.11)$; $\Delta=1$: $(-0.89,-0.45)$; $\Delta=2.6$: $\approx(-2.13,-1.50)$; $\Delta=3$: $(-2,-2)$.

**6.7.** $\lambda=0,1,10,100$: $\lVert p\rVert=1.000,\ 0.500,\ 0.091,\ 0.011$; длина монотонно убывает по $\lambda$.

**6.8.** $x_1=(48.04,-0.961)$, $x_2=(0,0)$; $d_1^\top Ad_0=0$, $r_1^\top r_0=0$; спуск после двух шагов — $(46.2,0.92)$.

**6.9.** (а) $20$ задетых собственных чисел из $80$. (б) горизонтальная ошибка нулевая, вертикальная симметрична. (в) из несимметричного старта — $N=40$ итераций.

**6.10.** (а) «precision loss» при $\lVert\nabla E\rVert\approx6.5\cdot10^{-6}$ после $38$ итераций. (б) пол $\approx2.4\cdot10^{-6}$ при $N=20$ ($1.3\cdot10^{-6}$ при $N=40$). (в) точный шаг и CG не сравнивают значения $E$. (г) с масштабированной $H_0$ — $10,\ 17,\ 27,\ 47$ итераций при $N=10,20,40,80$; с $H_0=I$ — $42$ при $N=10$, застревает при $N=20$ и $40$, $132$ при $N=80$: успех с $H_0=I$ — дело случая.
"""


def main() -> None:
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
