"""Собирает разбор упражнений лекции 9: exercises09.ipynb (с проверкой кодом) и exercises09.md (кратко).

    uv run --group casadi python lectures/lecture09/scripts/build_exercises09.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture09/exercises09.ipynb

Метод Ньютона–Лагранжа, дом у дороги и бросок импортируются из l9helpers.py (собирается build_lecture09.py) — тот же код,
что в демо. Условия задач берутся из раздела 10 lecture09.md при сборке и ставятся перед каждым решением
(блок «Условие»), чтобы разбор не расходился с конспектом. Семь задач: разминка, шаг как QP, «сломай метод»,
три способа на броске, скорость обобщённого Гаусса–Ньютона, прямая по точкам, знаки собственных чисел K.
"""

import re
from pathlib import Path

import nbformat as nbf

LECTURE = Path(__file__).resolve().parent.parent
OUT_NB = LECTURE / "exercises09.ipynb"
OUT_MD = LECTURE / "exercises09.md"
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


def statements() -> dict[str, tuple[str, str, str]]:
    """Условия задач из раздела 10 конспекта: номер -> (заголовок, вводная строка с сутью, текст условия)."""
    src = (LECTURE / "lecture09.md").read_text(encoding="utf-8")
    block = src[src.index("## 10. Упражнения"):src.index("## Вопросы для самопроверки")]
    out: dict[str, tuple[str, str, str]] = {}
    for chunk in re.split(r"\n(?=\*\*9\.\d\. )", block)[1:]:
        first, _, rest = chunk.strip().partition("\n")
        m = re.match(r"\*\*9\.(\d)\. (.+?)\*\*\s*(.*)$", first)
        assert m, first
        out[m.group(1)] = (m.group(2).rstrip("."), m.group(3).strip(), rest.strip())
    return out


def put_statements() -> None:
    """Перед каждым решением «## 9.N. …» вставляет условие из конспекта и подпись «Решение»."""
    st = statements()
    seen = set()
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        m = re.match(r"## 9\.(\d)\. ([^\n]+)\n?(.*)$", cell.source, re.S)
        if not m:
            continue
        n, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        assert n in st and title == st[n][0], (n, title, st[n][0])
        _, intro, text = st[n]
        cell.source = (f"## 9.{n}. {title}\n\n**Условие.** {intro}\n\n{text}\n\n**Решение.**"
                       + (f"\n\n{body}" if body else ""))
        seen.add(n)
    assert seen == set(st), (sorted(seen), sorted(st))


md(r"""# Разбор упражнений к лекции 9

Условия — раздел 10 конспекта [`lecture09.md`](lecture09.md); краткие ответы без кода — [`exercises09.md`](exercises09.md). Все упражнения — домашние; семинар — ноутбук [`seminar09.ipynb`](seminar09.ipynb). **Сначала попробуйте сами**, потом сверяйтесь. Каждый разбор: условие → решение по пунктам → проверка кодом и картинка там, где она помогает. Нужен CasADi: `uv run --group casadi jupyter lab`; в Colab первая ячейка ставит его сама.""")

code(r'''%matplotlib inline

import importlib.util, subprocess, sys
if importlib.util.find_spec("casadi") is None:          # Colab: поставить CasADi один раз
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "casadi"], check=True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from l9helpers import *  # newton_lagrange, дом у дороги, бросок (собираются из build_demo09.py)''')

# ---------------------------------------------------------------- 9.0
md(r"""## 9.0. Разминка — счёт руками

**(а)** Здесь $\nabla f=x$, $g=x_1+x_2-1$, $J=(1,\ 1)$, $\nabla^2g=0$. Поэтому $H=I$ при любом $\lambda$:

$$
F=\begin{pmatrix}x_1-\lambda\\ x_2-\lambda\\ x_1+x_2-1\end{pmatrix},\qquad
K=\begin{pmatrix}1&0&-1\\0&1&-1\\1&1&0\end{pmatrix}.
$$

**(б)** В точке $(0,0)$ при $\lambda=0$ невязка $F=(0,\ 0,\ -1)$. Система $K\,(p_1,p_2,\Delta\lambda)=(0,\ 0,\ 1)$ даёт $p_1=p_2=\Delta\lambda$ и $2\Delta\lambda=1$. Шаг $p=(0.5,\ 0.5)$, $\Delta\lambda=0.5$. В новой точке $F=(0,\ 0,\ 0)$.

Почему за один шаг: $f$ квадратичная, а $g$ линейная, поэтому $F$ линейна по $(x,\lambda)$. Линейная модель $F$ совпадает с самой $F$, и её нуль — точное решение.

**(в)** *Null-space.* $p_N=-J^\top(JJ^\top)^{-1}g=(1,1)\cdot\tfrac12=(0.5,\ 0.5)$. Базис касательной $Z=(1,-1)/\sqrt2$, правая часть $-Z^\top(\nabla f+Hp_N)=-Z^\top(0.5,\ 0.5)=0$, поэтому $u=0$. Множитель: $J^\top\lambda^+=Hp+\nabla f=(0.5,\ 0.5)$, то есть $\lambda^+=0.5$.

*Range-space.* $JH^{-1}J^\top=2$ и $JH^{-1}\nabla f-g=0+1=1$, поэтому $\lambda^+=0.5$. Затем $p=H^{-1}(J^\top\lambda^+-\nabla f)=(0.5,\ 0.5)$. Оба способа дают шаг из (б).

**Что делает ячейка.** Запускает метод Ньютона–Лагранжа из $(0,0)$ при $\lambda=0$ и проверяет, что хватило одного шага.""")

code(r'''def kkt_00(x, lam):
    """Задача 9.0: f = 1/2 |x|^2, g = x1 + x2 - 1."""
    return x.copy(), np.array([[1.0, 1.0]]), np.eye(2), np.array([x[0] + x[1] - 1])

h = newton_lagrange(kkt_00, [0.0, 0.0], [0.0])
print("итерации:", [(x.round(3), lam.round(3), f"{nF:.0e}") for x, lam, nF in h])
assert len(h) - 1 == 1 and np.allclose(h[-1][0], [0.5, 0.5]) and np.isclose(h[-1][1][0], 0.5)''')

md(r"""**Что видим.** Одна итерация: $x=(0.5,\ 0.5)$, $\lambda=0.5$, и невязка — машинный ноль.""")

# ---------------------------------------------------------------- 9.1
md(r"""## 9.1. Шаг как квадратичная задача

Функция Лагранжа задачи о шаге — с множителем $\mu\in\mathbb R^m$ и нашим знаком:

$$
\ell(p,\mu)=\nabla f^\top p+\tfrac12\,p^\top Hp-\mu^\top\big(g+Jp\big).
$$

Условие первого порядка лекции 8 — производная по $p$ равна нулю, и выполнено ограничение:

$$
\nabla f+Hp-J^\top\mu=0,\qquad g+Jp=0 .
$$

Это ровно система шага из условия: $Hp-J^\top\lambda^+=-\nabla f$ и $Jp=-g$, где $\mu=\lambda^+$.

У задачи о шаге ограничения линейны, а её якобиан — та же $J$. Её редуцированный гессиан — тот же $Z^\top HZ$. При условиях теоремы о матрице ККТ (LICQ и $Z^\top HZ\succ0$) система имеет единственное решение, и это строгий минимум задачи о шаге. Поэтому метод Ньютона–Лагранжа можно читать как «решай на каждом шаге квадратичную задачу с линейными ограничениями». Это и есть SQP (*sequential quadratic programming*, лекция 13).""")

# ---------------------------------------------------------------- 9.2
md(r"""## 9.2. Сломайте метод

**(а)** Решим систему условий для дома $a=(0,2)$ и посмотрим, сколько у неё решений.

**Шаг 1. Система.** Градиенты: $\nabla f=(x_1,\ x_2-2)$, $\nabla g=(-2x_1,\ 1)$. Условие $\nabla f=\lambda\nabla g$ и дорога:

$$
x_1=-2\lambda x_1,\qquad x_2-2=\lambda,\qquad x_2=x_1^2 .
$$

**Шаг 2. Первое уравнение.** Его можно записать как $x_1\,(1+2\lambda)=0$. Значит, либо $x_1=0$, либо $\lambda=-\tfrac12$.

**Шаг 3. Вершина.** При $x_1=0$ дорога даёт $x_2=0$, а второе уравнение — $\lambda=-2$:

$$
x=(0,\ 0),\qquad\lambda=-2 .
$$

**Шаг 4. Ещё две точки.** При $\lambda=-\tfrac12$ второе уравнение даёт $x_2=1.5$, а дорога — $x_1=\pm\sqrt{1.5}$:

$$
x=\big(\pm1.225,\ 1.5\big),\qquad\lambda=-0.5 .
$$

**Шаг 5. Где минимум.** В вершине $J=(0,\ 1)$, $Z=(1,\ 0)$ и $Z^\top HZ=1+2\lambda=-3<0$: это максимум расстояния вдоль дороги. В точках $(\pm1.225,\ 1.5)$ гессиан $H=\operatorname{diag}(0,\ 1)$, $Z\parallel(1,\ 2x_1)$ и $Z^\top HZ>0$: это минимумы, расстояние $1.32$ против $2$.

**Что делает ячейка.** Запускает метод из четырёх стартов при $\lambda^{(0)}=0$ и рисует пути.""")

code(r'''a_top = np.array([0.0, 2.0])
fig, ax = plt.subplots(figsize=(6, 4.2))
draw_road(ax, (-1.6, 1.6)); ax.plot(*a_top, "ks", ms=9)
for x0, c in (([0.0, 0.0], ORANGE), ([0.3, 0.09], BLUE), ([1.0, 0.0], AQUA), ([-1.0, 1.0], VIOLET)):
    h = newton_lagrange(road_kkt(a_top), x0, [0.0]); P = np.array([t[0] for t in h])
    print(f"из {x0}: {len(h) - 1} итер., x = {P[-1].round(4)}, lambda = {h[-1][1][0]:+.2f}, расстояние {np.linalg.norm(P[-1] - a_top):.3f}")
    ax.plot(P[:, 0], P[:, 1], "o-", ms=4, color=c, label=f"из {tuple(x0)}")
ax.set_aspect("equal"); ax.legend(fontsize=9, loc="upper center"); ax.set_ylim(-0.4, 2.6); plt.show()''')

md(r"""**Что видим.** Из вершины метод не уходит: одна итерация, $\lambda=-2$. Из остальных стартов он приходит в одну из ближайших точек $(\pm1.225,\ 1.5)$ с $\lambda=-0.5$. Метод находит любое решение системы — и максимум тоже.

**(б)** Для дома $(2,1)$ в точке $(0,0)$ определитель $\det K=1+2\lambda+4x_1^2=1+2\lambda$. Он равен нулю при $\lambda^{(0)}=-\tfrac12$. Тогда $H=\operatorname{diag}(0,\ 1)$, вдоль дороги $Z=(1,\ 0)$ и $Z^\top HZ=0$: кривизны вдоль дороги нет, LICQ при этом выполнено.

**Что делает ячейка.** Запускает метод для дома $(2,1)$ из $(0,0)$ при $\lambda^{(0)}=-0.5$ и ловит ошибку NumPy.""")

code(r'''try:
    newton_lagrange(road_kkt(A_ROAD), [0.0, 0.0], [-0.5])
except np.linalg.LinAlgError as err:
    print("lambda(0) = -0.5:", err)''')

md(r"""**Что видим.** `Singular matrix`: первый шаг не сделан, как и предсказывает $\det K=0$.""")

# ---------------------------------------------------------------- 9.3
md(r"""## 9.3. Три способа на броске

**(а)** **Шаг 1. Вторые производные.** Ограничение $g_1=v_1T-30$ даёт одну смешанную производную $\partial^2g_1/\partial v_1\partial T=1$. У $g_2=v_2T-4.905\,T^2$ — смешанная по $v_2$ и $T$, равная $1$, и $\partial^2g_2/\partial T^2=-9.81$. У $f$ вторые производные — $1$ по $v_1$ и по $v_2$.

**Шаг 2. Гессиан функции Лагранжа.** $H=\nabla^2f-\lambda_1\nabla^2g_1-\lambda_2\nabla^2g_2$:

$$
H=\begin{pmatrix}1&0&-\lambda_1\\0&1&-\lambda_2\\-\lambda_1&-\lambda_2&9.81\,\lambda_2\end{pmatrix}.
$$

**Шаг 3. Определитель.** Раскроем по первой строке:

$$
\det H=1\cdot\big(9.81\,\lambda_2-\lambda_2^2\big)-\lambda_1\cdot\lambda_1=9.81\,\lambda_2-\lambda_1^2-\lambda_2^2 .
$$

**Шаг 4. В ответе.** При $\lambda^\ast=(4.905,\ 4.905)$:

$$
\det H=4.905\cdot9.81-2\cdot4.905^2=4.905\,(9.81-9.81)=0 .
$$

В ответе $H$ вырождена, и range-space, которому нужна $H^{-1}$, там не работает. Null-space нужна только положительность $Z^\top HZ$ — она есть: это минимум.

**(б)** **Что делает ячейка.** В старте $(10,\ 10,\ 2)$ с $\lambda^{(0)}$ по МНК и в ответе печатает $\det H$, обусловленность $H$ и $Z^\top HZ$. Затем делает первый шаг тремя способами.""")

code(r'''pr = throw_problem(); tk = pr["kkt"]; x0 = np.array([10.0, 10.0, 2.0])
l0 = lam_ls(tk, x0, 2)
hs = newton_lagrange(tk, x0, l0)
for name, x, lam in (("старт", x0, l0), ("ответ", *hs[-1][:2])):
    gf, J, H, g = tk(x, lam); Z = null_space(J)
    print(f"{name}: lambda = {np.round(lam, 3)}, det H = {np.linalg.det(H):+.4f}, "
          f"обусловленность H = {np.linalg.cond(H):.1e}, Z^T H Z = {(Z.T @ H @ Z).item():.3f}")
gf, J, H, g = tk(x0, l0)
K = np.block([[H, -J.T], [J, np.zeros((2, 2))]])
p_full = np.linalg.solve(K, -np.concatenate([gf - J.T @ l0, g]))[:3]                          # 5 x 5
pN = -J.T @ np.linalg.solve(J @ J.T, g); Z = null_space(J)
p_null = pN + Z @ np.linalg.solve(Z.T @ H @ Z, -Z.T @ (gf + H @ pN))                           # 1 x 1
lp = np.linalg.solve(J @ np.linalg.solve(H, J.T), J @ np.linalg.solve(H, gf) - g)              # 2 x 2
p_rng = np.linalg.solve(H, J.T @ lp - gf)
print(f"расхождение с полной матрицей: null-space {np.abs(p_full - p_null).max():.0e}, range-space {np.abs(p_full - p_rng).max():.0e}")''')

md(r"""**Что видим.** В старте $\det H=-0.02$: $H$ знаконеопределена и почти вырождена, её обусловленность — порядка $10^5$. В ответе обусловленность $H$ огромна, $\det H$ — ноль с точностью округления. При этом $Z^\top HZ>0$ в обеих точках. Поэтому null-space надёжен, а range-space теряет точность — и в ответе не работает вовсе.""")

# ---------------------------------------------------------------- 9.4
md(r"""## 9.4. Скорость обобщённого Гаусса–Ньютона

**(а)** **Шаг 1. Вершина — решение.** Для дома $(0,h)$ система такая: $x_1=-2\lambda x_1$, $x_2-h=\lambda$, $x_2=x_1^2$. В точке $(0,0)$ первое и третье уравнения выполнены, а второе даёт

$$
\lambda^\ast=-h .
$$

**Шаг 2. Кривизна вдоль дороги.** В вершине $J=(0,\ 1)$, поэтому $Z=(1,\ 0)$. Гессиан $H=\operatorname{diag}(1+2\lambda^\ast,\ 1)$, и

$$
Z^\top HZ=1+2\lambda^\ast=1-2h .
$$

При $h<\tfrac12$ это число положительно: вершина — ближайшая точка дороги.

**(б)** **Шаг 3. Множитель ошибки.** У обобщённого Гаусса–Ньютона в матрице $I$, поэтому $Z^\top Z=1$ вместо $Z^\top HZ$:

$$
q=1-\frac{Z^\top HZ}{Z^\top Z}=1-(1-2h)=2h .
$$

**Шаг 4. Граница $h=\tfrac12$.** Там $Z^\top HZ=0$ и $q=1$: ошибка за шаг почти не уменьшается. Вершина перестаёт быть строгим минимумом — дом стоит в центре кривизны дороги, на высоте её радиуса кривизны $\tfrac12$. При $h>\tfrac12$ вершина становится максимумом, а ближайшие точки уходят в стороны.

**Что делает ячейка.** Для трёх высот дома печатает измеренное отношение соседних ошибок и $2h$. Затем запускает метод при $h=0.5$ на 2000 итераций.""")

code(r'''for h_ in (0.2, 0.4, 0.49):
    hg = newton_lagrange(road_ggn_kkt(np.array([0.0, h_])), [1.0, 0.0], [0.0], maxit=3000)
    e = np.array([np.linalg.norm(t[0]) for t in hg]); k = len(hg) // 2      # ответ — вершина (0, 0)
    print(f"h = {h_}: q = 2h = {2 * h_:.2f}, измерено {e[k + 1] / e[k]:.3f}, итераций {len(hg) - 1}")
hg = newton_lagrange(road_ggn_kkt(np.array([0.0, 0.5])), [1.0, 0.0], [0.0], maxit=2000)
print(f"h = 0.5: после 2000 итераций расстояние до вершины {np.linalg.norm(hg[-1][0]):.3f}, |F| = {hg[-1][2]:.1e}")''')

md(r"""**Что видим.** Отношение ошибок равно $2h$. При $h=0.5$ за 2000 итераций метод так и не дошёл до вершины: ошибка падает медленнее любой геометрической прогрессии.""")

# ---------------------------------------------------------------- 9.5
md(r"""## 9.5. Прямая по точкам

Невязки $r_i=n_1t_i+n_2s_i-d$ линейны по $x$, поэтому $J_r^\top J_r$ — точный $\nabla^2f$. Обобщённый Гаусс–Ньютон выбрасывает только кривизну ограничения $-\lambda\nabla^2g=-2\lambda\operatorname{diag}(1,1,0)$.

**(а)** **Что делает ячейка.** Генерирует 30 точек около прямой $s=0.5t+1$ при двух уровнях шума. Затем решает задачу двумя методами из одного старта: $n=(0,1)$, $d$ — среднее $s_i$.""")

code(r'''rng = np.random.default_rng(2026)
t = rng.uniform(0, 10, 30)                                    # абсциссы точек
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, sigma in zip(axes, (0.1, 1.0)):
    s = 0.5 * t + 1 + sigma * rng.standard_normal(30)         # точки около прямой
    Jr = np.column_stack([t, s, -np.ones(30)])                # якобиан невязок, 30 x 3

    def kkt_exact(x, lam):
        n1, n2, d = x
        return Jr.T @ (Jr @ x), np.array([[2 * n1, 2 * n2, 0.0]]), Jr.T @ Jr - lam[0] * np.diag([2.0, 2.0, 0.0]), np.array([n1**2 + n2**2 - 1])

    def kkt_ggn(x, lam):                                      # то же, но в матрице J_r^T J_r вместо H
        gf, J, _, g = kkt_exact(x, lam)
        return gf, J, Jr.T @ Jr, g

    x0 = np.array([0.0, 1.0, s.mean()])
    hg = newton_lagrange(kkt_ggn, x0, [0.0], maxit=200); hn = newton_lagrange(kkt_exact, x0, [0.0])
    P = np.column_stack([t, s]); C = (P - P.mean(0)).T @ (P - P.mean(0))   # матрица разброса 2 x 2
    ev, V = np.linalg.eigh(C); n_pca = V[:, 0] * np.sign(V[1, 0])
    x_g, lam_g = hg[-1][0], hg[-1][1][0]
    assert np.allclose(x_g[:2], n_pca, atol=1e-8) and np.isclose(lam_g, ev[0] / 2)
    print(f"sigma = {sigma}: обобщённый Гаусс–Ньютон {len(hg) - 1} итераций, Ньютон–Лагранж {len(hn) - 1}; "
          f"lambda* = {lam_g:.4f} = s_min/2; s_min/s_max = {ev[0] / ev[1]:.4f}")
    ax.plot(t, s, "o", ms=4, color=GRAY); tt = np.array([0, 10])
    ax.plot(tt, (x_g[2] - x_g[0] * tt) / x_g[1], color=BLUE); ax.set_title(f"шум {sigma}: {len(hg) - 1} итераций")
plt.show()''')

md(r"""**Что видим.** При шуме $0.1$ обобщённый Гаусс–Ньютон делает 5 итераций, как Ньютон–Лагранж. При шуме $1$ — уже 13, а Ньютон–Лагранж — по-прежнему 5.

**(б)** Ответ совпадает с собственным вектором матрицы разброса $C=\sum_i(p_i-\bar p)(p_i-\bar p)^\top$ для меньшего собственного числа $s_{\min}$. Здесь $p_i=(t_i,s_i)$, $\bar p$ — их среднее. При оптимальном $d=n^\top\bar p$ функция $f=\tfrac12n^\top Cn$, поэтому минимум на окружности $\lVert n\rVert=1$ — собственный вектор.

Множитель: умножим $\nabla f=J^\top\lambda$ скалярно на $x$. Слева $x^\top J_r^\top r=\lVert r\rVert^2$, справа $\lambda\,(2n_1^2+2n_2^2)=2\lambda$. Значит, $\lambda^\ast=\tfrac12\lVert r\rVert^2=f^\ast=s_{\min}/2$. Малый шум — малый $\lambda^\ast$, и метод почти ньютоновский. Множитель ошибки здесь — отношение $s_{\min}/s_{\max}$: $0.0013$ против $0.0995$.""")

# ---------------------------------------------------------------- 9.6
md(r"""## 9.6. Для желающих: знаки собственных чисел матрицы ККТ

Обозначим $K_s=\begin{pmatrix}H&J^\top\\ J&0\end{pmatrix}$ и $Y=J^\top(JJ^\top)^{-1}$ — матрицу $n\times m$, для которой $JY=I$. Столбцы $Z$ и $Y$ вместе дают базис $\mathbb R^n$: $JZ=0$, а $Y$ — комбинации строк $J$.

**Шаг 1. Смена базиса.** Возьмём невырожденную матрицу $T=\begin{pmatrix}Z&Y&0\\0&0&I\end{pmatrix}$ размера $(n+m)\times(n+m)$. Тогда

$$
T^\top K_sT=\begin{pmatrix}Z^\top HZ&Z^\top HY&0\\ Y^\top HZ&Y^\top HY&I\\ 0&I&0\end{pmatrix}.
$$

**Шаг 2. Убираем связь с первым блоком.** Блок $A=Z^\top HZ$ положительно определён. Вычтем из второй блочной строки первую, умноженную на $Y^\top HZA^{-1}$, и так же со столбцами. Получится блочно-диагональная матрица $\operatorname{diag}\Big(A,\ \begin{pmatrix}B&I\\ I&0\end{pmatrix}\Big)$, где $B=Y^\top HY-Y^\top HZA^{-1}Z^\top HY$.

**Шаг 3. Считаем знаки.** У $A$ ровно $n-m$ положительных собственных чисел. У $\begin{pmatrix}B&I\\ I&0\end{pmatrix}$ определитель $(-1)^m\ne0$ при любой $B$. Будем непрерывно уменьшать $B$ до нуля: собственные числа не проходят через ноль. При $B=0$ они равны $\pm1$, по $m$ штук. Итого $n-m+m=n$ положительных и $m$ отрицательных.

**Шаг 4. Переносим на $K_s$.** Все преобразования — вида $M\mapsto S^\top MS$ с невырожденной $S$. По закону инерции Сильвестра число положительных и отрицательных собственных чисел при этом не меняется. $\blacksquare$

Поэтому $K_s$ знаконеопределена при любой задаче с ограничениями, и разложение Холецкого к ней неприменимо. IPOPT проверяет эти числа после разложения $LDL^\top$. Если положительных меньше $n$, он прибавляет к $H$ матрицу $\delta I$.

**Что делает ячейка.** Считает собственные числа симметричной матрицы ККТ в ответах обеих задач лекции.""")

code(r'''for name, kkt, x0, m in (("дорога", road_kkt(A_ROAD), [1.0, 0.0], 1), ("бросок", throw_problem()["kkt"], [10.0, 10.0, 2.0], 2)):
    h = newton_lagrange(kkt, x0, lam_ls(kkt, np.array(x0), m))
    _, J, H, _ = kkt(*h[-1][:2])
    Ks = np.block([[H, J.T], [J, np.zeros((m, m))]])                # симметричная матрица ККТ
    ev = np.linalg.eigvalsh(Ks)
    print(f"{name} в ответе: положительных {(ev > 0).sum()}, отрицательных {(ev < 0).sum()}  (n = {H.shape[0]}, m = {m})")''')

md(r"""**Что видим.** У дороги два положительных и одно отрицательное, у броска три и два: ровно $n$ и $m$. У броска это так, хотя сама $H$ в ответе вырождена (упражнение 9.3): важна только $Z^\top HZ$.""")

MD_TEXT = r"""# Упражнения к лекции 9 — краткие ответы

Разбор с проверкой кодом — [`exercises09.ipynb`](exercises09.ipynb). Условия — раздел 10 [конспекта](lecture09.md).

**9.0.** (а) $F=(x_1-\lambda,\ x_2-\lambda,\ x_1+x_2-1)$, $K=\begin{pmatrix}1&0&-1\\0&1&-1\\1&1&0\end{pmatrix}$. (б) $p=(0.5,\ 0.5)$, $\Delta\lambda=0.5$, после шага $F=0$: $F$ линейна по $(x,\lambda)$, и линейная модель точна. (в) null-space: $p_N=(0.5,\ 0.5)$, $u=0$; range-space: $2\lambda^+=1$; шаг тот же.

**9.1.** Условие первого порядка задачи о шаге: $\nabla f+Hp-J^\top\mu=0$, $g+Jp=0$ — система шага с $\mu=\lambda^+$. При LICQ и $Z^\top HZ\succ0$ это единственный строгий минимум задачи о шаге — основа SQP.

**9.2.** (а) У системы три решения: вершина $(0,0)$ с $\lambda=-2$ — максимум расстояния вдоль дороги ($Z^\top HZ=-3$), и $(\pm1.225,\ 1.5)$ с $\lambda=-0.5$ — минимумы. Из вершины метод не уходит, из остальных стартов приходит в ближайшие точки. (б) $\det K=1+2\lambda$ в $(0,0)$, шаг не определён при $\lambda^{(0)}=-0.5$.

**9.3.** (а) $H=\begin{pmatrix}1&0&-\lambda_1\\0&1&-\lambda_2\\-\lambda_1&-\lambda_2&9.81\lambda_2\end{pmatrix}$, $\det H=9.81\lambda_2-\lambda_1^2-\lambda_2^2=0$ в ответе. (б) В старте $\det H=-0.02$; null-space совпадает с полной матрицей, range-space теряет точность и в ответе не работает, а $Z^\top HZ>0$ всюду.

**9.4.** (а) $\lambda^\ast=-h$, $Z=(1,0)$, $Z^\top HZ=1-2h$. (б) $q=2h$; при $h=\tfrac12$ кривизны вдоль дороги нет, $q=1$, метод почти стоит; при $h>\tfrac12$ вершина — максимум.

**9.5.** (а) Шум $0.1$ — 5 итераций, шум $1$ — 13; Ньютон–Лагранж — 5 в обоих случаях. (б) Ответ — собственный вектор матрицы разброса для $s_{\min}$; $\lambda^\ast=f^\ast=s_{\min}/2$; множитель ошибки — $s_{\min}/s_{\max}$: $0.0013$ и $0.0995$.

**9.6.** Замена базиса на $(Z,\ Y)$ и исключение блока $Z^\top HZ\succ0$ дают $\operatorname{diag}\big(Z^\top HZ,\ \begin{pmatrix}B&I\\I&0\end{pmatrix}\big)$: $n-m$ положительных и ещё $m$ положительных и $m$ отрицательных. По закону инерции Сильвестра у $K_s$ — $n$ положительных и $m$ отрицательных собственных чисел. Проверка: дорога — 2 и 1, бросок — 3 и 2.
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
