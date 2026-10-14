"""Собирает конспект-ноутбук lecture06.ipynb: текст из lecture06.md плюс код демонстраций.

    uv run python lectures/lecture06/scripts/build_lecture06.py
    uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture06/lecture06.ipynb

Текст режется на ячейки так же, как в tools/md2nb.py (по заголовкам ## и ###).
Ячейки демонстраций берутся из build_demo06.py (тот же код, что в demo06.ipynb)
и вставляются в разделы, к которым относятся:

    (a) линейный поиск               -> после раздела 2, перед 3
    (b) доверительная область        -> после раздела 3, перед 4
    (c) сопряжённые градиенты        -> после раздела 4, перед 5

Части (0) и (d) остаются только в demo06.ipynb.

Заголовки частей «## (a) …» становятся «### Демо (a). …», чтобы не ломать
нумерацию разделов конспекта. Вторая ячейка demo06 (импорты, палитра, методы, задачи)
выносится в модуль l6helpers.py, который импортируют разбор упражнений и семинар.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent            # scripts/
LECTURE = HERE.parent                             # lectures/lecture06/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LECTURE.parent.parent / "tools"))

from md2nb import split_cells  # noqa: E402
import build_demo06  # noqa: E402  (импорт собирает список ячеек, файл не пишет)

MD = LECTURE / "lecture06.md"
OUT = LECTURE / "lecture06.ipynb"

# где вставлять какую часть: часть -> префикс заголовка ячейки конспекта, ПЕРЕД которой она идёт
PLACEMENT = [
    ("a", "## 3."),
    ("b", "## 4."),
    ("c", "## 5."),
]
PART_HEAD = re.compile(r"^## \((\w)\) (.*)$", re.M)


def demo_parts() -> tuple[nbf.NotebookNode, dict[str, list[nbf.NotebookNode]]]:
    """Возвращает ячейку с импортами и словарь часть -> её ячейки (без шпаргалки)."""
    cells = build_demo06.cells
    assert cells[0].cell_type == "markdown" and cells[0].source.startswith("# Демонстрации")
    assert cells[1].cell_type == "code" and "def armijo(" in cells[1].source
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

    helper_body = setup.source.replace("%matplotlib inline\n\n", "", 1)
    (LECTURE / "l6helpers.py").write_text(
        '"""Вспомогательный код демонстраций лекции 6.\n\n'
        'Линейный поиск: armijo(f, x, fx, g, p, c1, alpha) — backtracking; wolfe(f, grad, x, p, fx, g, c1, c2) —\n'
        '        сильные условия Вольфе.\n'
        'Методы: newton_ls(f, grad, hess, x0, shift="ascent"|"always") — Ньютон с Армихо и сдвигом tau I;\n'
        '        bfgs(f, grad, x0, line="armijo"|"wolfe"|"exact", Q, scale_H0) — BFGS (Л5) с выбором поиска;\n'
        '        trust_region(f, grad, hess, x0, step="exact"|"dogleg"|"cauchy") — доверительная область;\n'
        '        cauchy_point, dogleg, tr_exact — шаги подзадачи; cg(Av, b, x0) — линейные сопряжённые градиенты;\n'
        '        newton_cg(f, grad, hessp, x0) — неточный Ньютон; gd, newton — из Л4–Л5.\n'
        'Задачи: rosen, rosen_grad, rosen_hess, X0_ROSEN; him, him_grad, him_hess (Химмельблау);\n'
        '        make_chain(N) -> (energy, grad_E, H, b, v0, unpack); gd_chain_count(N); f_arctan.\n\n'
        'Файл генерируется скриптом scripts/build_lecture06.py из scripts/build_demo06.py — не правьте руками.\n"""\n\n'
        + helper_body + "\n",
        encoding="utf-8",
    )

    out: list[nbf.NotebookNode] = []
    for i, src in enumerate(text_cells):
        if i == 1:  # после шапки конспекта — короткая ячейка с импортом
            out.append(nbf.v4.new_markdown_cell(
                "> **Код в этом ноутбуке.** Три демонстрации из [`demo06.ipynb`](demo06.ipynb) вставлены "
                "в те разделы, к которым относятся: «Демо (a)» — линейный поиск — после раздела 2, «Демо (b)» — "
                "доверительная область — после раздела 3, «Демо (c)» — сопряжённые градиенты — после раздела 4; "
                "крючки и биография цепи — только в `demo06.ipynb`. Вспомогательный код (палитра, методы "
                "`armijo`, `wolfe`, `newton_ls`, `bfgs`, `trust_region`, `cg`, `newton_cg`, задачи) вынесен в "
                "[`l6helpers.py`](l6helpers.py); его подключает ячейка ниже."
            ))
            out.append(nbf.v4.new_code_cell(
                "%matplotlib inline\n\nfrom l6helpers import *  # палитра, методы, задачи (см. файл рядом)"
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
