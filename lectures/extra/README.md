# Заготовки

Материалы, написанные для курса и отложенные до подходящего места в программе. Не входят в календарь занятий и в README курса; хранятся здесь, чтобы не потеряться.

| Файл | Что это | Куда планируется |
|---|---|---|
| [`duality.md`](duality.md) | Двойственность Лагранжа: функция Лагранжа как «цена вместо запрета», двойственная функция, слабая и сильная двойственность, плоскость $(h,f)$, зазор (неделимый завод и $-x_1x_2$), условие Слейтера, двойственность LP и QP, теневые цены | лекции 10–11 (ККТ; LP/QP), часть IV |
| [`demo_duality.ipynb`](demo_duality.ipynb) (сборка `build_demo_duality.py`) | демонстрации к `duality.md`: $d(\mu)$ руками и кодом, двойственность QP на ящике, LP с `marginals` против явной двойственной, зазоры и скрытая выпуклость | вместе с `duality.md` |
| [`exercises_duality.md`](exercises_duality.md) | краткие ответы к упражнениям `duality.md` | вместе с `duality.md` |
| `make_figures_duality.py` → `img/` | иллюстрации к `duality.md` | — |

Сборка и выполнение — как для лекций (см. [`docs/tooling.md`](../../docs/tooling.md)):

```bash
uv run python lectures/extra/make_figures_duality.py
uv run python lectures/extra/build_demo_duality.py
uv run jupyter nbconvert --to notebook --execute --inplace lectures/extra/demo_duality.ipynb
```
