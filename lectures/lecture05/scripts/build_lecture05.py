"""Собирает конспект-ноутбук lecture05.ipynb: текст из lecture05.md плюс код демонстраций.

    uv run python lectures/lecture05/scripts/build_lecture05.py
    uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture05/lecture05.ipynb

Текст режется на ячейки так же, как в tools/md2nb.py (по заголовкам ## и ###).
Ячейки демонстраций берутся из build_demo05.py (тот же код, что в demo05.ipynb)
и вставляются в разделы, к которым относятся — только две, чтобы конспект
не перегружать кодом:

    (a) Ньютон и демпфирование        -> после раздела 3, перед 4
    (b) BFGS                          -> после раздела 4, перед 5

Части (0), (c), (d) остаются только в demo05.ipynb.

Заголовки частей «## (a) …» становятся «### Демо (a). …», чтобы не ломать
нумерацию разделов конспекта. Вторая ячейка demo05 (импорты, методы, задачи,
рисовальщики) выносится в модуль l5helpers.py, который импортируют разбор
упражнений и семинар. Отдельный demo05.ipynb остаётся как есть.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent            # scripts/
LECTURE = HERE.parent                             # lectures/lecture05/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LECTURE.parent.parent / "tools"))

from md2nb import split_cells  # noqa: E402
import build_demo05  # noqa: E402  (импорт собирает список ячеек, файл не пишет)

MD = LECTURE / "lecture05.md"
OUT = LECTURE / "lecture05.ipynb"

# где вставлять какую часть: часть -> префикс заголовка ячейки конспекта, ПЕРЕД которой она идёт
PLACEMENT = [
    ("a", "## 4."),
    ("b", "## 5."),
]
PART_HEAD = re.compile(r"^## \((\w)\) (.*)$", re.M)


def demo_parts() -> tuple[nbf.NotebookNode, dict[str, list[nbf.NotebookNode]]]:
    """Возвращает ячейку с импортами и словарь часть -> её ячейки (без шпаргалки)."""
    cells = build_demo05.cells
    assert cells[0].cell_type == "markdown" and cells[0].source.startswith("# Демонстрации")
    assert cells[1].cell_type == "code" and "def gd(" in cells[1].source
    parts: dict[str, list[nbf.NotebookNode]] = {}
    current = None
    for cell in cells[2:]:
        if cell.cell_type == "markdown":
            m = PART_HEAD.match(cell.source)
            if m:
                current = m.group(1)
                parts[current] = []
                cell = nbf.v4.new_markdown_cell(
                    PART_HEAD.sub(lambda mm: f"### Демо ({mm.group(1)}). {mm.group(2)}", cell.source, count=1)
                )
            elif cell.source.startswith("## Шпаргалка"):
                current = None
        if current is not None:
            parts[current].append(cell)
    assert set(parts) == {"0", "a", "b", "c", "d"}, sorted(parts)
    return cells[1], parts


def main() -> None:
    setup, parts = demo_parts()
    text_cells = split_cells(MD.read_text(encoding="utf-8"))

    # вспомогательный код (палитра, gd, newton) — в модуль рядом с ноутбуком
    helper_body = setup.source.replace("%matplotlib inline\n\n", "", 1)
    (LECTURE / "l5helpers.py").write_text(
        '"""Вспомогательный код демонстраций лекции 5.\n\n'
        'Методы: gd(f, grad, x0, alpha=None, tol, maxit, c) — градиентный спуск (лекция 4);\n'
        '        newton(grad, hess, x0, tol, maxit) — чистый Ньютон;\n'
        '        newton_damped(f, grad, hess, x0, tol, maxit, c) — Ньютон с backtracking по Армихо;\n'
        '        bfgs(f, grad, x0, tol, maxit, c, Q=None) — BFGS, H_0 = I; Q — точный шаг на квадратичной;\n'
        '        gauss_newton(resid, jac, x0, tol, maxit, damped) — Гаусс–Ньютон для 1/2 ||r||^2.\n'
        'Задачи: rosen, rosen_grad, rosen_hess, X0_ROSEN; make_chain(N) -> (energy, grad_E, H, b, v0, unpack);\n'
        '        sine_data() -> (t, y, starts); sine_problem(t, y) -> (resid, jac, f, grad, hess); X0_SINE.\n'
        'Рисовальщики (принимают ax=None, возвращают ax): plot_rosen_path(paths, labels, ax, ...),\n'
        '        plot_errors(errs, labels, ax, logx).\n\n'
        'Файл генерируется скриптом scripts/build_lecture05.py из scripts/build_demo05.py — не правьте руками.\n"""\n\n'
        + helper_body + "\n",
        encoding="utf-8",
    )

    out: list[nbf.NotebookNode] = []
    for i, src in enumerate(text_cells):
        if i == 1:  # после шапки конспекта, перед «## 1.» — короткая ячейка с импортом
            out.append(nbf.v4.new_markdown_cell(
                "> **Код в этом ноутбуке.** Две демонстрации из [`demo05.ipynb`](demo05.ipynb) вставлены "
                "в те разделы, к которым относятся: «Демо (a)» — Ньютон и демпфирование — после раздела 3, "
                "«Демо (b)» — BFGS — после раздела 4; остальные части (задачи, Гаусс–Ньютон, цепь) — только в "
                "`demo05.ipynb`. При чтении демо можно пропускать, на лекции — запускать по ходу. Вспомогательный код (палитра, методы "
                "`gd`, `newton`, `newton_damped`, `bfgs`, `gauss_newton`, задачи и рисовальщики) вынесен в "
                "[`l5helpers.py`](l5helpers.py); его подключает ячейка ниже."
            ))
            out.append(nbf.v4.new_code_cell(
                "%matplotlib inline\n\nfrom l5helpers import *  # палитра, методы, задачи, рисовальщики (см. файл рядом)"
            ))
        for part, prefix in PLACEMENT:
            if src.startswith(prefix):
                out.extend(nbf.v4.new_code_cell(c.source) if c.cell_type == "code"
                           else nbf.v4.new_markdown_cell(c.source) for c in parts.pop(part))
        out.append(nbf.v4.new_markdown_cell(src))
    assert not set(parts) & {part for part, _ in PLACEMENT}, f"не вставлены части {sorted(parts)}"

    nb = nbf.v4.new_notebook(cells=out)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbf.validate(nb)
    nbf.write(nb, OUT)
    n_code = sum(c.cell_type == "code" for c in out)
    print(f"{OUT}: {len(out)} ячеек, из них {n_code} с кодом")


if __name__ == "__main__":
    main()
