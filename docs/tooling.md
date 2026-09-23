# Инструменты и устройство репозитория

Страница для преподавателя и для тех, кто хочет собирать материалы сам. Студентам достаточно [README](../README.md).

## Структура

```
syllabus.md                 программа курса
pyproject.toml, uv.lock     зависимости Python (uv); requirements.txt — то же для pip
lectures/lectureNN/
    lectureNN.md            конспект лекции (читать на GitHub)
    lectureNN.ipynb         тот же конспект ячейками (собирается из md: tools/md2nb.py)
    demoNN.ipynb            демонстрации: пошаговый ноутбук с пояснениями по scipy.optimize
                            (собирается скриптом build_demoNN.py, выполняется nbconvert)
    make_figures.py         скрипт, строящий иллюстрации конспекта в img/
    img/                    картинки лекции (коммитятся)
    exercisesNN.ipynb       разбор упражнений конспекта по шагам, с проверкой кодом
                            (собирается скриптом build_exercisesNN.py); exercisesNN.md — краткие ответы
    boardNN.md              для преподавателя: раскадровка доски (что рисовать, что показывать)
lectures/extra/             заготовки, отложенные до подходящего места в программе (см. README там);
                            собираются теми же скриптами
homeworks/hwNN.md, .ipynb   домашние задания: md — краткое описание и правила, ipynb — рабочий шаблон
homeworks/solutions/        не публикуются в репозитории (студенты учатся по нему); преподаватель ведёт
                            решения и заметки к лекциям отдельно
submissions/hwNN/           сданные работы <фамилия>.ipynb — приходят pull request'ами, см. CONTRIBUTING.md
tools/                      вспомогательные скрипты
```

## Окружение

Основной способ — [uv](https://docs.astral.sh/uv/): `uv sync` создаёт `.venv` и ставит зависимости из `uv.lock`; `uv run <команда>` выполняет её в этом окружении, активировать вручную не нужно.

Группы зависимостей в `pyproject.toml`:

- `notebooks` (по умолчанию) — JupyterLab, ipykernel, nbformat;
- `casadi` — CasADi, понадобится с лекции 7: `uv sync --group casadi`;
- `docs` (по умолчанию) — пакет `markdown` для сборки HTML/PDF.

Альтернатива без uv: `pip install -r requirements.txt` в виртуальном окружении.

## Сборка материалов лекции

```bash
uv run python lectures/lecture01/make_figures.py          # иллюстрации конспекта -> img/
uv run python tools/md2nb.py lectures/lecture01/lecture01.md   # конспект md -> ipynb
uv run python lectures/lecture01/build_demo01.py          # demo01.ipynb (без выводов)
uv run python lectures/lecture01/build_exercises01.py     # exercises01.ipynb (без выводов)
uv run jupyter nbconvert --to notebook --execute --inplace lectures/lecture01/demo01.ipynb lectures/lecture01/exercises01.ipynb
```

Правило: конспект редактируется в `.md`, ноутбук-версия пересобирается скриптом и не правится руками. Демо и разбор упражнений редактируются в `build_*.py`, ноутбуки пересобираются и выполняются `nbconvert`, чтобы выводы и картинки лежали в репозитории. Иллюстрации 07–10 повторяют демонстрации ноутбука с теми же данными и seed, чтобы конспект и демо совпадали.

Проверить, что ноутбуки выполняются без ошибок:

```bash
uv run jupyter nbconvert --to notebook --execute --output /tmp/out.ipynb lectures/lecture01/demo01.ipynb
```

## PDF

Самый простой способ — открыть `.md` на GitHub или `.ipynb` в JupyterLab и напечатать в PDF из браузера (`File → Print`). Для ноутбуков есть `File → Save and Export Notebook As → HTML`, затем печать в PDF. Собранные PDF в git не попадают (`.gitignore`); чтобы выложить — `git add -f`.
