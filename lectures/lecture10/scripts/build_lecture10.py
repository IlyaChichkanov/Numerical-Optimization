"""Собирает конспект-ноутбук lecture10.ipynb: текст из lecture10.md плюс код демонстраций.

    uv run --group casadi python lectures/lecture10/scripts/build_lecture10.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture10/lecture10.ipynb

Текст режется на ячейки так же, как в tools/md2nb.py (по заголовкам ## и ###) и идёт подряд, без кода.
Ячейки демонстраций берутся из build_demo10.py (тот же код, что в demo10.ipynb) и собраны в конце,
в разделе «## Демо», после вопросов для самопроверки:

    (a) условия ККТ в задаче «такси»                       -> к разделу 2
    (b) активное множество: портфель руками, сверка с IPOPT -> к разделу 3
    (c) одна стенка: Ньютон на mu*h = 0 и на mu*h = tau      -> к разделу 4

Часть (e) остаётся только в demo10.ipynb. Заголовки частей «## (a) …» становятся
«### Демо (a). …», чтобы не ломать нумерацию разделов конспекта. Вторая ячейка demo10
(импорты, палитра, задачи) выносится в модуль l10helpers.py, который импортирует разбор упражнений.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent            # scripts/
LECTURE = HERE.parent                             # lectures/lecture10/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LECTURE.parent.parent / "tools"))

from md2nb import split_cells  # noqa: E402
import build_demo10  # noqa: E402  (импорт собирает список ячеек, файл не пишет)

MD = LECTURE / "lecture10.md"
OUT = LECTURE / "lecture10.ipynb"

DEMOS = ("a", "b", "c")                           # части демо в конце ноутбука, по порядку разделов
PART_HEAD = re.compile(r"^## \((\w)\) (.*)$", re.M)


def demo_parts() -> tuple[nbf.NotebookNode, dict[str, list[nbf.NotebookNode]]]:
    """Возвращает ячейку с импортами и словарь часть -> её ячейки (без шпаргалки)."""
    cells = build_demo10.cells
    assert cells[0].cell_type == "markdown" and cells[0].source.startswith("# Демонстрации")
    assert cells[1].cell_type == "code" and "def taxi_guess(" in cells[1].source
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
    assert set(parts) == {"a", "b", "c", "e"}, sorted(parts)
    return cells[1], parts


def main() -> None:
    setup, parts = demo_parts()
    text_cells = split_cells(MD.read_text(encoding="utf-8"))

    helper_body = setup.source.replace("%matplotlib inline\n\n", "", 1)
    (LECTURE / "l10helpers.py").write_text(
        '"""Вспомогательный код демонстраций лекции 10.\n\n'
        'Задача «такси»: TAXI_T, TAXI_Y, TAXI_A; taxi_f(x), taxi_grad(x); taxi_guess(A) -> (x, mu) при ограничениях из A как равенствах;\n'
        '        одна стенка (§4): wall_newton(c, x, mu, tau, inside) -> (x, mu, итераций). Ближайшая точка круга (упражнение 10.1): park_solution(a) -> (x*, mu*).\n'
        'Портфель: PF_R, PF_SIGMA; portfolio_active(rho, A) -> (x, lam1, lam_rho, mu) при x_i = 0 для i из A;\n'
        '        portfolio_bounded(rho) -> (A, x, lam1, lam_rho, mu) с x >= 0 перебором активных множеств.\n'
        'Рисовалка: draw_quadrant(ax). Нужен CasADi (--group casadi).\n\n'
        'Файл генерируется скриптом scripts/build_lecture10.py из scripts/build_demo10.py — не правьте руками.\n"""\n\n'
        + helper_body + "\n",
        encoding="utf-8",
    )

    out: list[nbf.NotebookNode] = []
    for i, src in enumerate(text_cells):
        if i == 1:  # после шапки конспекта — где код
            out.append(nbf.v4.new_markdown_cell(
                "> **Код в этом ноутбуке.** Текст лекции идёт подряд, без кода. Три демо из [`demo10.ipynb`](demo10.ipynb) "
                "собраны в конце, в разделе «Демо»: (a) к §2 — условия ККТ в задаче «такси», (b) к §3 — портфель руками и IPOPT, "
                "(c) к §4 — точки при $\\mu h=\\tau$. Подробности — только в `demo10.ipynb`. "
                "Нужен CasADi: `uv run --group casadi jupyter lab`, в Colab он ставится сам."
            ))
        out.append(nbf.v4.new_markdown_cell(src))

    # все демо — в конце, после вопросов для самопроверки
    out.append(nbf.v4.new_markdown_cell(
        "## Демо\n\nТри демо к разделам лекции: (a) к §2, (b) к §3, (c) к §4. "
        "Ячейка ниже подключает вспомогательный код из [`l10helpers.py`](l10helpers.py): палитру и задачи «такси» и «портфель»."
    ))
    out.append(nbf.v4.new_code_cell(
        "%matplotlib inline\n\nfrom l10helpers import *  # палитра, задачи «такси» и «портфель» (см. файл рядом)"
    ))
    for part in DEMOS:
        out.extend(nbf.v4.new_code_cell(c.source) if c.cell_type == "code"
                   else nbf.v4.new_markdown_cell(c.source) for c in parts.pop(part))

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
