"""Собирает конспект-ноутбук lecture09.ipynb: текст из lecture09.md плюс код демонстраций.

    uv run --group casadi python lectures/lecture09/scripts/build_lecture09.py
    uv run --group casadi jupyter nbconvert --to notebook --execute --inplace lectures/lecture09/lecture09.ipynb

Текст режется на ячейки так же, как в tools/md2nb.py (по заголовкам ## и ###) и идёт подряд, без кода.
Ячейки демонстраций берутся из build_demo09.py (тот же код, что в demo09.ipynb) и собраны в конце,
в разделе «## Демо», после вопросов для самопроверки:

    (a) метод Ньютона–Лагранжа: шаг руками, окружность, цепь против IPOPT     -> к разделу 2
    (b) матрица ККТ и шаг по частям: вырожденные K, три способа                -> к разделам 3–4
    (c) обобщённый Гаусс–Ньютон: множитель ошибки равен lambda*               -> к разделу 5

Части (0) и (e) остаются только в demo09.ipynb. Заголовки частей «## (a) …» становятся
«### Демо (a). …», чтобы не ломать нумерацию разделов конспекта. Вторая ячейка demo09
(импорты, палитра, методы, задачи) выносится в модуль l9helpers.py, который импортируют разбор упражнений и семинар.
"""

import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent            # scripts/
LECTURE = HERE.parent                             # lectures/lecture09/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LECTURE.parent.parent / "tools"))

from md2nb import split_cells  # noqa: E402
import build_demo09  # noqa: E402  (импорт собирает список ячеек, файл не пишет)

MD = LECTURE / "lecture09.md"
OUT = LECTURE / "lecture09.ipynb"

DEMOS = ("a", "b", "c")                           # части демо в конце ноутбука, по порядку разделов
PART_HEAD = re.compile(r"^## \((\w)\) (.*)$", re.M)


def demo_parts() -> tuple[nbf.NotebookNode, dict[str, list[nbf.NotebookNode]]]:
    """Возвращает ячейку с импортами и словарь часть -> её ячейки (без шпаргалки)."""
    cells = build_demo09.cells
    assert cells[0].cell_type == "markdown" and cells[0].source.startswith("# Демонстрации")
    assert cells[1].cell_type == "code" and "def rod_chain(" in cells[1].source
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
    (LECTURE / "l9helpers.py").write_text(
        '"""Вспомогательный код демонстраций лекции 9.\n\n'
        'Метод Ньютона–Лагранжа: newton_lagrange(kkt, x0, lam0) -> список (x_k, lam_k, |F_k|); kkt(x, lam) -> (grad f, J, H, g),\n'
        '        знак L = f - lam^T g, шаг [[H, -J^T], [J, 0]] (p, dlam) = -(grad_x L, g).\n'
        'Окружность: A_CIRCLE = (2, 1), circle_kkt(a) -> kkt; ggn_circle(a, x0, iters) -> точки обобщённого Гаусса–Ньютона;\n'
        '        angle_err(xs, a) -> угол до луча к a.\n'
        'Цепь на стержнях: rod_chain(N, L_total, m) -> словарь (v, f, g — CasADi; kkt; w, ell, N, v0); lam_ls(kkt, x, m) -> множители по МНК;\n'
        '        solve_rod_chain(ch, v0) -> (v*, lam* в нашем знаке, stats IPOPT); chain_xy(ch, v) -> (y, z) с опорами.\n'
        'Рисовалки: unit_circle(ax), draw_chain(ax, ch, v). Нужен CasADi (--group casadi).\n\n'
        'Файл генерируется скриптом scripts/build_lecture09.py из scripts/build_demo09.py — не правьте руками.\n"""\n\n'
        + helper_body + "\n",
        encoding="utf-8",
    )

    out: list[nbf.NotebookNode] = []
    for i, src in enumerate(text_cells):
        if i == 1:  # после шапки конспекта — где код
            out.append(nbf.v4.new_markdown_cell(
                "> **Код в этом ноутбуке.** Текст лекции идёт подряд, без кода. Три демо из [`demo09.ipynb`](demo09.ipynb) "
                "собраны в конце, в разделе «Демо»: (a) к §2 — метод Ньютона–Лагранжа, (b) к §3–§4 — матрица ККТ и шаг по частям, "
                "(c) к §5 — обобщённый Гаусс–Ньютон. Числа введения и подробности — только в `demo09.ipynb`. "
                "Нужен CasADi: `uv run --group casadi jupyter lab`, в Colab он ставится сам."
            ))
        out.append(nbf.v4.new_markdown_cell(src))

    # все демо — в конце, после вопросов для самопроверки
    out.append(nbf.v4.new_markdown_cell(
        "## Демо\n\nТри демо к разделам лекции: (a) к §2, (b) к §3–§4, (c) к §5. "
        "Ячейка ниже подключает вспомогательный код из [`l9helpers.py`](l9helpers.py): палитру, метод `newton_lagrange` и задачи."
    ))
    out.append(nbf.v4.new_code_cell(
        "%matplotlib inline\n\nfrom l9helpers import *  # палитра, метод Ньютона–Лагранжа, окружность, цепь на стержнях (см. файл рядом)"
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
