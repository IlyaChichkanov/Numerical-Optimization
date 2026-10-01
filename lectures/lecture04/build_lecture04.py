"""Собирает конспект-ноутбук lecture04.ipynb: текст из lecture04.md плюс код демонстраций.

    uv run python lectures/lecture04/build_lecture04.py
    uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture04/lecture04.ipynb

Текст режется на ячейки так же, как в tools/md2nb.py (по заголовкам ## и ###).
Ячейки демонстраций берутся из build_demo04.py (тот же код, что в demo04.ipynb)
и вставляются в разделы, к которым относятся:

    (0) три задачи-крючка          -> после 1.2, перед 1.3
    (a) стационарные точки, гессиан -> после раздела 3, перед 4
    (b) спуск на квадратичной       -> после раздела 5, перед 6
    (d) регрессия и масштабирование -> после (b), перед 6
    (c) Розенброк: GD против Ньютона -> после раздела 7, перед 8
    (e) цепь: kappa растёт с N       -> после (c), перед 8

Заголовки частей «## (a) …» становятся «### Демо (a). …», чтобы не ломать
нумерацию разделов конспекта. Отдельный demo04.ipynb остаётся как есть — для
самостоятельной работы после пары.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "tools"))

from md2nb import split_cells  # noqa: E402
import build_demo04  # noqa: E402  (импорт собирает список ячеек, файл не пишет)

MD = HERE / "lecture04.md"
OUT = HERE / "lecture04.ipynb"

# где вставлять какую часть: часть -> префикс заголовка ячейки конспекта, ПЕРЕД которой она идёт
PLACEMENT = [
    ("0", "### 1.3."),
    ("a", "## 4."),
    ("b", "## 6."),
    ("d", "## 6."),
    ("c", "## 8."),
    ("e", "## 8."),
]
PART_HEAD = re.compile(r"^## \((\w)\) (.*)$", re.M)


def demo_parts() -> tuple[nbf.NotebookNode, dict[str, list[nbf.NotebookNode]]]:
    """Возвращает ячейку с импортами и словарь часть -> её ячейки (без шпаргалки)."""
    cells = build_demo04.cells
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
    assert set(parts) == {"0", "a", "b", "c", "d", "e"}, sorted(parts)
    return cells[1], parts


def main() -> None:
    setup, parts = demo_parts()
    text_cells = split_cells(MD.read_text(encoding="utf-8"))

    # вспомогательный код (палитра, gd, newton) — в модуль рядом с ноутбуком
    helper_body = setup.source.replace("%matplotlib inline\n\n", "", 1)
    (HERE / "l4helpers.py").write_text(
        '"""Вспомогательный код демонстраций лекции 4: палитра, настройки matplotlib,\n'
        'градиентный спуск `gd` и метод Ньютона `newton`.\n\n'
        'Файл генерируется скриптом build_lecture04.py из build_demo04.py — не правьте руками.\n"""\n\n'
        + helper_body + "\n",
        encoding="utf-8",
    )

    out: list[nbf.NotebookNode] = []
    for i, src in enumerate(text_cells):
        if i == 1:  # после шапки конспекта, перед «## 1.» — короткая ячейка с импортом
            out.append(nbf.v4.new_markdown_cell(
                "> **Код в этом ноутбуке.** Демонстрации из [`demo04.ipynb`](demo04.ipynb) вставлены "
                "в те разделы, к которым относятся; заголовки — «Демо (0)»–«Демо (e)». При чтении их "
                "можно пропускать, на лекции — запускать по ходу. Вспомогательный код (палитра, функции "
                "`gd` и `newton`) вынесен в [`l4helpers.py`](l4helpers.py); его подключает ячейка ниже."
            ))
            out.append(nbf.v4.new_code_cell(
                "%matplotlib inline\n\nfrom l4helpers import *  # палитра, настройки matplotlib, gd, newton"
            ))
        for part, prefix in PLACEMENT:
            if src.startswith(prefix):
                out.extend(nbf.v4.new_code_cell(c.source) if c.cell_type == "code"
                           else nbf.v4.new_markdown_cell(c.source) for c in parts.pop(part))
        out.append(nbf.v4.new_markdown_cell(src))
    assert not parts, f"не вставлены части {sorted(parts)}"

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
