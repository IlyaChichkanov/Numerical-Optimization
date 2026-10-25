"""Собирает конспект-ноутбук lecture07.ipynb: текст из lecture07.md плюс код демонстраций.

    uv run --group casadi python lectures/lecture07/scripts/build_lecture07.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture07/lecture07.ipynb

Текст режется на ячейки так же, как в tools/md2nb.py (по заголовкам ## и ###).
Ячейки демонстраций берутся из build_demo07.py (тот же код, что в demo07.ipynb)
и вставляются в разделы, к которым относятся:

    (a) конечные разности                  -> после раздела 2, перед 3
    (b) прямой и обратный режим руками     -> после раздела 4, перед 5
    (c) CasADi на цепи                     -> после раздела 5, перед 6

Части (0) и (e) остаются только в demo07.ipynb. Заголовки частей «## (a) …» становятся
«### Демо (a). …», чтобы не ломать нумерацию разделов конспекта. Вторая ячейка demo07
(импорты, палитра, разности, задачи, мини-АД) выносится в модуль l7helpers.py, который импортирует разбор упражнений.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent            # scripts/
LECTURE = HERE.parent                             # lectures/lecture07/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LECTURE.parent.parent / "tools"))

from md2nb import split_cells  # noqa: E402
import build_demo07  # noqa: E402  (импорт собирает список ячеек, файл не пишет)

MD = LECTURE / "lecture07.md"
OUT = LECTURE / "lecture07.ipynb"

# где вставлять какую часть: часть -> префикс заголовка ячейки конспекта, ПЕРЕД которой она идёт
PLACEMENT = [
    ("a", "## 3."),
    ("b", "## 5."),
    ("c", "## 6."),
]
PART_HEAD = re.compile(r"^## \((\w)\) (.*)$", re.M)


def demo_parts() -> tuple[nbf.NotebookNode, dict[str, list[nbf.NotebookNode]]]:
    """Возвращает ячейку с импортами и словарь часть -> её ячейки (без шпаргалки)."""
    cells = build_demo07.cells
    assert cells[0].cell_type == "markdown" and cells[0].source.startswith("# Демонстрации")
    assert cells[1].cell_type == "code" and "def fd_forward(" in cells[1].source
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
    assert set(parts) == {"0", "a", "b", "c", "e"}, sorted(parts)
    return cells[1], parts


def main() -> None:
    setup, parts = demo_parts()
    text_cells = split_cells(MD.read_text(encoding="utf-8"))

    helper_body = setup.source.replace("%matplotlib inline\n\n", "", 1)
    (LECTURE / "l7helpers.py").write_text(
        '"""Вспомогательный код демонстраций лекции 7.\n\n'
        'Разности: fd_forward(f, x, i, h), fd_central(f, x, i, h), ucurve(f, x, i, exact) -> (ошибки прямой, центральной) по HS.\n'
        'Пример: tape_example() — f = x1 x2 + sin(x1 x2) в три операции: значения, прямой и обратный режим;\n'
        '        OPS, PROGRAM, forward_mode(program, x, d), reverse_mode(program, x) — мини-АД по списку операций.\n'
        'Цепь: make_elastic_chain(N) -> (energy, grad, v0) на numpy; chain_casadi(N) -> (v, E) — граф CasADi;\n'
        '        chain_costs(N) -> (операций E, grad, hess, H d, ненулевых); chain_pattern(N) -> (портрет, равновесие, Function H).\n'
        'Задачи: rosen, rosen_grad, X0_ROSEN; ucurve_plot(ax, curves, labels, colors). Нужен CasADi (--group casadi).\n\n'
        'Файл генерируется скриптом scripts/build_lecture07.py из scripts/build_demo07.py — не правьте руками.\n"""\n\n'
        + helper_body + "\n",
        encoding="utf-8",
    )

    out: list[nbf.NotebookNode] = []
    for i, src in enumerate(text_cells):
        if i == 1:  # после шапки конспекта — короткая ячейка с импортом
            out.append(nbf.v4.new_markdown_cell(
                "> **Код в этом ноутбуке.** Три демонстрации из [`demo07.ipynb`](demo07.ipynb) вставлены "
                "в те разделы, к которым относятся: «Демо (a)» — конечные разности — после раздела 2, «Демо (b)» — "
                "прямой и обратный режим своими руками — после раздела 4, «Демо (c)» — CasADi на цепи — после раздела 5; "
                "числа введения и подробности — только в `demo07.ipynb`. Вспомогательный код (палитра, разности, "
                "мини-АД `forward_mode` / `reverse_mode`, цепь) вынесен в [`l7helpers.py`](l7helpers.py); "
                "его подключает ячейка ниже. Нужен CasADi: `uv run --group casadi jupyter lab`, в Colab он ставится сам."
            ))
            out.append(nbf.v4.new_code_cell(
                "%matplotlib inline\n\nfrom l7helpers import *  # палитра, разности, мини-АД, цепь (см. файл рядом)"
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
