"""Собирает конспект-ноутбук lecture08.ipynb: текст из lecture08.md плюс код демонстраций.

    uv run --group casadi python lectures/lecture08/scripts/build_lecture08.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture08/lecture08.ipynb

Текст режется на ячейки так же, как в tools/md2nb.py (по заголовкам ## и ###).
Ячейки демонстраций берутся из build_demo08.py (тот же код, что в demo08.ipynb)
и вставляются в разделы, к которым относятся:

    (a) две точки Лагранжа на окружности     -> перед разделом 2
    (b) множитель, которого нет              -> перед разделом 3
    (c) цепь на стержнях в IPOPT             -> перед разделом 5

Части (0) и (e) остаются только в demo08.ipynb. Заголовки частей «## (a) …» становятся
«### Демо (a). …», чтобы не ломать нумерацию разделов конспекта. Вторая ячейка demo08
(импорты, палитра, задачи) выносится в модуль l8helpers.py, который импортирует разбор упражнений.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent            # scripts/
LECTURE = HERE.parent                             # lectures/lecture08/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LECTURE.parent.parent / "tools"))

from md2nb import split_cells  # noqa: E402
import build_demo08  # noqa: E402  (импорт собирает список ячеек, файл не пишет)

MD = LECTURE / "lecture08.md"
OUT = LECTURE / "lecture08.ipynb"

# где вставлять какую часть: часть -> префикс заголовка ячейки конспекта, ПЕРЕД которой она идёт
PLACEMENT = [
    ("a", "## 2."),
    ("b", "## 3."),
    ("c", "## 5."),
]
PART_HEAD = re.compile(r"^## \((\w)\) (.*)$", re.M)


def demo_parts() -> tuple[nbf.NotebookNode, dict[str, list[nbf.NotebookNode]]]:
    """Возвращает ячейку с импортами и словарь часть -> её ячейки (без шпаргалки)."""
    cells = build_demo08.cells
    assert cells[0].cell_type == "markdown" and cells[0].source.startswith("# Демонстрации")
    assert cells[1].cell_type == "code" and "def circle_system(" in cells[1].source
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
    (LECTURE / "l8helpers.py").write_text(
        '"""Вспомогательный код демонстраций лекции 8.\n\n'
        'Окружность: A_CIRCLE, circle_system(z) — система условий первого порядка, circle_points() -> ((x, lam) ближней, дальней).\n'
        'Клюв: cusp_grad(x1) -> (grad g, оценка lambda, угол к grad f) на кривой x2 = x1^(2/3).\n'
        'Цепь на стержнях: rod_chain(N, L_total) -> словарь (v, f, g — CasADi; w, ell, N, v0); solve_rod_chain(ch, v0) -> (v*, lam*, stats)\n'
        '        в нашем знаке L = f - lam^T g; chain_xy(ch, v) -> (y, z) с опорами; chain_checks(ch, v, lam) -> невязка, rank J,\n'
        '        собственные числа редуцированного гессиана, Z, натяжения F и составляющие Fh, Fv.\n'
        'Портфель: PF_R, PF_SIGMA, portfolio_eq(rho) -> (x, f*, lam) — два равенства, система ККТ линейна.\n'
        'Рисовалки: level_circles(ax, a, levels), unit_circle(ax). Нужен CasADi (--group casadi).\n\n'
        'Файл генерируется скриптом scripts/build_lecture08.py из scripts/build_demo08.py — не правьте руками.\n"""\n\n'
        + helper_body + "\n",
        encoding="utf-8",
    )

    out: list[nbf.NotebookNode] = []
    for i, src in enumerate(text_cells):
        if i == 1:  # после шапки конспекта — короткая ячейка с импортом
            out.append(nbf.v4.new_markdown_cell(
                "> **Код в этом ноутбуке.** Три демонстрации из [`demo08.ipynb`](demo08.ipynb) вставлены "
                "в те разделы, к которым относятся: «Демо (a)» — две точки Лагранжа на окружности — перед разделом 2, "
                "«Демо (b)» — множитель, которого нет, — перед разделом 3, «Демо (c)» — цепь на стержнях в IPOPT — перед разделом 5; "
                "числа введения и подробности — только в `demo08.ipynb`. Вспомогательный код (палитра, окружность, клюв, "
                "цепь на стержнях, портфель) вынесен в [`l8helpers.py`](l8helpers.py); "
                "его подключает ячейка ниже. Нужен CasADi: `uv run --group casadi jupyter lab`, в Colab он ставится сам."
            ))
            out.append(nbf.v4.new_code_cell(
                "%matplotlib inline\n\nfrom l8helpers import *  # палитра, окружность, клюв, цепь на стержнях, портфель (см. файл рядом)"
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
