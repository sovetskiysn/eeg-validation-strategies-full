# Приведение фигур статьи к габаритам PLOS ONE

Три фигуры рукописи (`transfer_matrix`, `scenario_accuracy_by_decoder`,
`baseline_vs_cross_subject`) генерировались шириной 11–13 дюймов, тогда как PLOS ONE
принимает растровые фигуры шириной не более 7.5 дюйма при разрешении не ниже 300 dpi.
В таком виде файлы отбиваются автоматической проверкой Editorial Manager / NAAS ещё на
этапе загрузки. Документ фиксирует рабочее решение: все три фигуры приведены к габаритам
журнала **без потери данных** — ни одна колонка, строка или панель не удалена. Ключевым
приёмом оказался поворот заголовков колонок матрицы на 90°, а не сокращение содержания,
как предполагалось изначально. Итоговые PNG лежат в [`plos_final/`](plos_final/),
исполняемый патч — в [`plos_final/apply_plos_figure_sizing.py`](plos_final/apply_plos_figure_sizing.py).
Правки в `python_project/src/analysis.py` **не внесены**: патч применяется к копии.

## Содержание

1. [Результат](#результат)
2. [Требования PLOS ONE](#требования-plos-one)
3. [Почему это вообще проблема](#почему-это-вообще-проблема)
4. [Тупики: что не сработало](#тупики-что-не-сработало)
   1. [Тупик 1: снизить dpi](#тупик-1-снизить-dpi)
   2. [Тупик 2: пропорционально сжать и поднять кегли](#тупик-2-пропорционально-сжать-и-поднять-кегли)
   3. [Отменённая гипотеза: резать содержание матрицы](#отменённая-гипотеза-резать-содержание-матрицы)
5. [Рабочее решение](#рабочее-решение)
   1. [Принцип: холст = печатный размер](#принцип-холст--печатный-размер)
   2. [Ключ: поворот заголовков колонок](#ключ-поворот-заголовков-колонок)
   3. [Подводные камни](#подводные-камни)
6. [Правки по функциям](#правки-по-функциям)
   1. [craft_transfer_matrix_figure](#craft_transfer_matrix_figure)
   2. [craft_scenario_accuracy_by_decoder_figure](#craft_scenario_accuracy_by_decoder_figure)
   3. [craft_baseline_vs_cross_subject_figure](#craft_baseline_vs_cross_subject_figure)
7. [Патч целиком](#патч-целиком)
8. [Как воспроизвести](#как-воспроизвести)
9. [Что осталось сделать](#что-осталось-сделать)

## Результат

| Фигура | Было | Стало | Кегли | Статус |
| --- | --- | --- | --- | --- |
| `transfer_matrix` | 13.05 × 8.86 in | **7.07 × 7.18 in** | 7.5–10 pt | в габаритах |
| `scenario_accuracy_by_decoder` | 12.20 × 12.90 in | **7.50 × 8.20 in** | 8–10 pt | в габаритах |
| `baseline_vs_cross_subject` | 11.27 × 5.15 in | **7.30 × 3.67 in** | 8–10 pt | в габаритах |

Содержание сохранено полностью: матрица переноса по-прежнему показывает 11 колонок
source–target, 25 строк (5 источников × 5 декодеров) и оба summary-столбца.

## Требования PLOS ONE

Источник: <https://journals.plos.org/plosone/s/figures>.

| Параметр | Требование | Эквивалент при 300 dpi |
| --- | --- | --- |
| Ширина | 2.63–7.5 in | 789–2250 px |
| Высота | ≤ 8.75 in | ≤ 2625 px |
| Разрешение | 300–600 dpi | — |
| Текст внутри фигуры | 8–12 pt | — |
| Формат | TIFF (LZW) или EPS | — |

Существенно, что ширина и разрешение — **независимые** требования. Нельзя выполнить
одно за счёт другого.

## Почему это вообще проблема

Фигуры изначально верстались без привязки к печатному формату журнала. Кегли в
`analysis.py` заданы абсолютными числами (`fontsize=8.4`, `fontsize=15`), а холст —
под удобство просмотра на экране (12–18 дюймов). При смене холста кегли не
масштабируются автоматически, поэтому любое изменение `figsize` ломает типографику.

Отдельная тонкость: `latex_documents/Makefile` при сборке комплекта подачи
(`make prepare-plos-submission`) только **предупреждал** о нарушении габаритов,
но не падал:

```text
WARNING: transfer_matrix.png is 3915x2658px; verify PLOS figure dimensions.
```

Предупреждение легко пропустить, и оно не мешает собрать формально «готовый» комплект.

## Тупики: что не сработало

### Тупик 1: снизить dpi

Первая идея — сохранять PNG в 150–200 dpi вместо 300. Тогда при неизменном
`figsize` число пикселей падает и файл укладывается в 2250 px.

**Не работает.** PLOS требует не ниже 300 dpi. Понижение разрешения выполняет одно
требование ценой нарушения другого. Кроме того, `figsize` в matplotlib задаётся в
дюймах и уже равен печатному размеру — dpi на него не влияет:

```text
figsize=(7.5, 4) at dpi=300  ->  2250x1200 px, на бумаге 7.5 x 4 in
figsize=(7.5, 4) at dpi=600  ->  4500x2400 px, на бумаге те же 7.5 x 4 in
```

### Тупик 2: пропорционально сжать и поднять кегли

Вторая идея выглядела корректной: уменьшить холст до 7.5 in и умножить все `fontsize`
на обратный коэффициент, чтобы физический размер букв на печати не изменился.
Коэффициенты: ×1.740 для матрицы, ×1.6265 и ×1.5027 для остальных.

**Результат оказался хуже исходного.** Проверено экспериментально:

| Фигура | После «пропорционального» сжатия |
| --- | --- |
| `baseline_vs_cross_subject` | **15.75 × 3.57 in** — стала шире, чем была |
| `scenario_accuracy_by_decoder` | 7.50 × 7.93 in, но легенда обрезана по правому краю |
| `transfer_matrix` | 8.22 × 5.34 in, заголовки слились в нечитаемую кашу |

Причина роста ширины `baseline_vs_cross_subject` неочевидна и стоит отдельного
внимания: легенда рисуется одним горизонтальным рядом из шести элементов
(`ncols=len(handles)`). При кегле ×1.5 ряд стал шире холста, а сохранение идёт с
`bbox_inches="tight"` — тот честно расширил область до содержимого. **Вывод: с
`bbox_inches="tight"` уменьшение `figsize` не гарантирует уменьшения файла.**

У матрицы заголовки колонок наложились друг на друга и превратились в
`StAoirthpnMetihrcolulSIBAoritpnMetihrcoluIlBull A`.

### Отменённая гипотеза: резать содержание матрицы

На основании тупика 2 был сделан вывод, что 11 колонок × 25 строк при читаемом шрифте
в 7.5 дюйма не помещаются физически, и предлагалось сокращать содержание: разбить
матрицу на две фигуры по блокам, убрать два summary-столбца или транспонировать её.

**Гипотеза неверна.** Ширину распирали не данные, а горизонтальные подписи колонок.
Ниже — арифметика, которая это показывает.

## Рабочее решение

### Принцип: холст = печатный размер

Поскольку `figsize` в дюймах и есть итоговый печатный размер, кегли задаются **сразу
в диапазоне PLOS 8–12 pt**, а не пересчитываются из старых значений. Это одновременно
проще и надёжнее: то, что видно в файле, и есть то, что увидит читатель.

### Ключ: поворот заголовков колонок

Расчёт для матрицы при ширине холста 7.5 in:

```text
матрица занимает matrix_width = 0.54 от ширины фигуры
  -> 0.54 x 7.5 = 4.05 in на 11 колонок
  -> 0.368 in на колонку = ~26 pt

содержимое ячейки "69.3" при 8 pt   ~ 18 pt  -> помещается
заголовок колонки "Arithmetic" 8 pt ~ 40 pt  -> НЕ помещается, распирает таблицу
```

То есть ограничителем были подписи, а не числа. Поворот заголовков на 90°
(`rotation=90`) убирает их вклад в ширину полностью, перенося его в высоту — а по
высоте бюджет 8.75 in был свободен. После поворота потребовалось увеличить
вертикальный запас под шапку:

```python
group_header_height = 1.5
header_height = group_header_height + 5.4   # было: 2 * group_header_height
```

### Подводные камни

1. **`bbox_inches="tight"` расширяет файл под вылезающее содержимое.** Легенды,
   не помещающиеся в холст, увеличивают итоговую ширину вместо того, чтобы обрезаться.
   Лечится переносом легенды в несколько рядов (`ncols`), а не подгонкой `figsize`.
2. **Итоговую ширину нельзя вычислить заранее.** Из-за `bbox_inches="tight"`
   соотношение между `figsize` и сохранённым размером зависит от вёрстки. Для матрицы
   пришлось подбирать `figure_width` итеративно: 10.30 → 10.05 → 9.55 → 9.30.
3. **`scenario_accuracy_by_decoder` сохраняется без `tight`** (см. комментарий в
   `figure_specs`), поэтому её нижнее поле обязано вместить подпись оси само —
   `bottom` в `subplots_adjust` пришлось увеличивать вручную.
4. **Левые жёлоба матрицы** заданы в единицах колонок (`source_left`, `decoder_left`).
   При уменьшении ширины колонки длинные подписи `Arithmetic` и `EEGConformer` начали
   касаться рамок; жёлоба расширены с −3.10/−1.85 до −4.15/−2.35.
5. **Заголовки summary-столбцов** (`Baseline vs cross-dataset mean diff`) не помещались
   в колонку и накладывались друг на друга; потребовалась и другая расстановка
   переносов строк, и уменьшение кегля до 7.5 pt, и расширение самих колонок.

## Правки по функциям

### craft_transfer_matrix_figure

| Что | Было | Стало |
| --- | --- | --- |
| `figure_width, figure_height` | `18.0, 8.8` | `9.30, 7.10` |
| `group_header_height` | `1.3` | `1.5` |
| `header_height` | `2 * group_header_height` | `group_header_height + 5.4` |
| `source_left, decoder_left` | `-3.10, -1.85` | `-4.15, -2.35` |
| Заголовки колонок | горизонтальные, 9 pt | **`rotation=90`**, 8 pt |
| Числа в ячейках | 8.4 pt | 8.0 pt |
| Подписи источника и декодера | 9 pt | 8 pt |
| `Target (test)` / `Source (train)` | 11 pt | 10 pt |
| Подпись колорбара | 9 pt | 8 pt |
| `summary_left, summary_width` | `0.62, 0.13` | `0.625, 0.155` |
| Заголовки summary-столбцов | 9 pt, 4 строки | 7.5 pt, иные переносы |
| Значения summary | 11 pt | 10 pt |

### craft_scenario_accuracy_by_decoder_figure

| Что | Было | Стало |
| --- | --- | --- |
| `figsize` | `(12.2, 12.9)` | `(7.5, 8.2)` |
| Легенда | `ncols=6`, 11 pt | `ncols=3`, 8 pt |
| `bbox_to_anchor` легенды | `(0.37, 1.054)` | `(0.30, 1.075)` |
| Подписи строк | 11 / 10 pt | 9 / 8 pt |
| Подпись оси X | 12 pt | 10 pt |
| `chance level` | 10 pt | 8 pt |
| `subplots_adjust` | `left=0.22, top=0.9385, bottom=0.048` | `left=0.30, top=0.930, bottom=0.058` |

### craft_baseline_vs_cross_subject_figure

| Что | Было | Стало |
| --- | --- | --- |
| `figsize` | `(11.4, 5.6)` | `(7.5, 3.9)` |
| Легенда | `ncols=len(handles)` (=6), 11 pt | `ncols=3`, 8 pt |
| Заголовки панелей | 15 pt, `pad=12` | 10 pt, `pad=8` |
| Подписи оси X | 12 pt | 9 pt |
| `labelsize` тиков | 11 | 8 |
| Подпись оси Y | 12 pt | 10 pt |
| `subplots_adjust` | `left=0.055, top=0.865, bottom=0.145` | `left=0.085, top=0.885, bottom=0.235` |

## Патч целиком

Исполняемая версия: [`plos_final/apply_plos_figure_sizing.py`](plos_final/apply_plos_figure_sizing.py).
Скрипт читает текущий `src/analysis.py`, применяет замены и пишет результат в указанный
файл. Каждая замена защищена `assert` на единственность вхождения — если исходник
изменится, патч упадёт, а не применится частично.

```python
import re, pathlib, sys
src=pathlib.Path("/home/svc-jax-dlh/.work/python_project/src/analysis.py").read_text()
t=src

def sub1(old,new):
    global t
    assert t.count(old)==1, f"count={t.count(old)}: {old[:60]}"
    t=t.replace(old,new)

# ---------------- transfer_matrix ----------------
sub1("figure_width, figure_height = 18.0, 8.8",
     "figure_width, figure_height = 9.30, 7.10")
sub1("    group_header_height = 1.3\n    header_height = 2 * group_header_height",
     "    group_header_height = 1.5\n    header_height = group_header_height + 5.4")
sub1("source_left, decoder_left = -3.10, -1.85",
     "source_left, decoder_left = -4.15, -2.35")
# vertical target-column headers: they were the element that destroyed the layout
sub1("""            column + 0.5, label_header_top / 2, label, ha="center", va="center",
            fontsize=9, fontweight="bold", clip_on=False,""",
     """            column + 0.5, label_header_top / 2, label, ha="center", va="center",
            rotation=90, fontsize=8, fontweight="bold", clip_on=False,""")
# fonts into the PLOS 8-12 pt band (canvas is now final print size)
sub1("""                fontsize=8.4,
                fontweight="bold" if not np.isnan(value) else "normal",""",
     """                fontsize=8.0,
                fontweight="bold" if not np.isnan(value) else "normal",""")
sub1("""            label, ha="center", va="center", fontsize=9, fontweight="bold",
            clip_on=False,
        )
        for decoder_index, model_name in enumerate(model_names):""",
     """            label, ha="center", va="center", fontsize=8, fontweight="bold",
            clip_on=False,
        )
        for decoder_index, model_name in enumerate(model_names):""")
sub1("""                -0.08, y_position + decoder_index + 0.5, model_name,
                ha="right", va="center", fontsize=9, fontweight="bold",""",
     """                -0.08, y_position + decoder_index + 0.5, model_name,
                ha="right", va="center", fontsize=8, fontweight="bold",""")
sub1("""        va="center", fontsize=11, fontweight="bold", clip_on=False,
    )
    ax.text(
        source_left - 0.28,""",
     """        va="center", fontsize=10, fontweight="bold", clip_on=False,
    )
    ax.text(
        source_left - 0.28,""")
sub1("""        va="center", rotation=90, fontsize=11, fontweight="bold", clip_on=False,""",
     """        va="center", rotation=90, fontsize=10, fontweight="bold", clip_on=False,""")
sub1("""    colorbar.set_label(f"{ARTICLE_METRIC_LABEL}, %", fontsize=9, fontweight="bold")""",
     """    colorbar.set_label(f"{ARTICLE_METRIC_LABEL}, %", fontsize=8, fontweight="bold")""")

# ---------------- scenario_accuracy_by_decoder ----------------
sub1("figsize=(12.2, 12.9)", "figsize=(7.5, 8.2)")
sub1("""    ax.text(0.5, 1.0, "chance level", color="#62718A", ha="center", va="bottom", fontsize=10,""",
     """    ax.text(0.5, 1.0, "chance level", color="#62718A", ha="center", va="bottom", fontsize=8,""")
sub1("""    ax.set_xlabel(ARTICLE_METRIC_LABEL, fontsize=12, fontweight="bold", color="#24354F")
    ax.set_yticks""",
     """    ax.set_xlabel(ARTICLE_METRIC_LABEL, fontsize=10, fontweight="bold", color="#24354F")
    ax.set_yticks""")
sub1("""                fontsize=11 if label in protocol_label_set else 10)""",
     """                fontsize=9 if label in protocol_label_set else 8)""")
sub1("""    legend = ax.legend(
        ncols=6,
        loc="upper center",
        bbox_to_anchor=(0.37, 1.054),
        frameon=False,
        fontsize=11,
        markerscale=1.15,
        handletextpad=0.35,
        columnspacing=0.75,
    )""",
     """    legend = ax.legend(
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.30, 1.075),
        frameon=False,
        fontsize=8,
        markerscale=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
    )""")
sub1("""    fig.subplots_adjust(left=0.22, right=0.98, top=0.9385, bottom=0.048)""",
     """    fig.subplots_adjust(left=0.30, right=0.985, top=0.930, bottom=0.058)""")

# ---------------- baseline_vs_cross_subject ----------------
sub1("figsize=(11.4, 5.6)", "figsize=(7.5, 3.9)")
sub1("""        ax.set_title(title, fontsize=15, fontweight="bold", color="#172B4D", loc="left", pad=12)""",
     """        ax.set_title(title, fontsize=10, fontweight="bold", color="#172B4D", loc="left", pad=8)""")
sub1("""        ax.set_xticks(x_positions, [label for _, label in protocol_specs], fontsize=12,""",
     """        ax.set_xticks(x_positions, [label for _, label in protocol_specs], fontsize=9,""")
sub1("""        ax.tick_params(axis="both", length=0, labelsize=11)""",
     """        ax.tick_params(axis="both", length=0, labelsize=8)""")
sub1("""    axes[0].set_ylabel(ARTICLE_METRIC_LABEL, fontsize=12, fontweight="bold", color="#24354F")""",
     """    axes[0].set_ylabel(ARTICLE_METRIC_LABEL, fontsize=10, fontweight="bold", color="#24354F")""")
sub1("""        ncols=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=11,""",
     """        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=8,""")
sub1("""    fig.subplots_adjust(left=0.055, right=0.985, top=0.865, bottom=0.145, wspace=0.07)""",
     """    fig.subplots_adjust(left=0.085, right=0.985, top=0.885, bottom=0.235, wspace=0.07)""")

# summary header labels collided: narrower wording, smaller type, wider columns
sub1("""    summary_left, summary_width = 0.62, 0.13""",
     """    summary_left, summary_width = 0.625, 0.155""")
sub1("""        "Baseline\\nvs\\ncross-task\\nmean diff",
        "Baseline\\nvs\\ncross-dataset\\nmean diff",""",
     """        "Baseline vs\\ncross-task\\nmean diff",
        "Baseline vs\\ncross-\\ndataset\\nmean diff",""")
sub1("""            column + 0.5, -header_height / 2, label, ha="center", va="center",
            fontsize=9, fontweight="bold", clip_on=False,""",
     """            column + 0.5, -header_height / 2, label, ha="center", va="center",
            fontsize=7.5, fontweight="bold", clip_on=False,""")
sub1("""                fontsize=11,
                fontweight="bold" if not np.isnan(value) else "normal",
                color="#263341",""",
     """                fontsize=10,
                fontweight="bold" if not np.isnan(value) else "normal",
                color="#263341",""")

pathlib.Path(sys.argv[1]).write_text(t)
print("patched ok")
```

## Как воспроизвести

Патч применяется к **копии** `analysis.py`, которая подкладывается через `PYTHONPATH`,
поэтому рабочий исходник остаётся нетронутым.

```bash
cd /home/svc-jax-dlh/.work/python_project
export WORK=$(mktemp -d)

# 1. собрать пропатченную копию модуля
python3 /home/svc-jax-dlh/.work/cvg/plos_final/apply_plos_figure_sizing.py "$WORK/analysis.py"

# 2. отрисовать фигуры этой копией
PYTHONPATH="$WORK" ANALYSIS_OUTPUT_DIR="$WORK/out" uv run python scripts/run_analysis.py

# 3. проверить габариты
uv run --with pillow python - <<'PY'
from PIL import Image
import glob, os
for f in sorted(glob.glob(os.environ["WORK"] + "/out/figures/source_png/*.png")):
    w, h = Image.open(f).size
    ok = "OK  " if (789 <= w <= 2250 and h <= 2625) else "FAIL"
    print(ok, os.path.basename(f), f"{w/300:.2f}x{h/300:.2f} in")
PY
```

Ожидаемый вывод:

```text
OK   baseline_vs_cross_subject.png 7.30x3.67 in
OK   scenario_accuracy_by_decoder.png 7.50x8.20 in
OK   transfer_matrix.png 7.07x7.18 in
```

Входные данные берутся из `ANALYSIS_INPUT_DIR`; по умолчанию это
`results/scenario_decoder_maybe_new (2026-09-01 % 08-54-09)` (см. `scripts/run_analysis.py`).

## Что осталось сделать

- [ ] Перенести правки в `python_project/src/analysis.py` и перегенерировать фигуры
      прямо в `latex_documents/revision/manuscript_clean/` (`make pull-artifacts`).
- [ ] Пересобрать комплект подачи: `make prepare-plos-submission SOURCE=revision/manuscript_clean OUTPUT=revision/manuscript_clean_plos`.
- [ ] Заменить `WARNING` в `latex_documents/Makefile` на `SystemExit`, чтобы нарушение
      габаритов роняло сборку, а не проходило незамеченным.
- [ ] Завести константу лимита PLOS в `analysis.py`, чтобы новые фигуры рождались уже
      в габаритах, а не подгонялись задним числом.
- [ ] Убрать из ответа на Journal Requirement 1 абзац с признанием, что габариты
      выдержать не удалось (`latex_documents/revision/response_to_reviewers/response_to_reviewers.tex`) —
      требование теперь выполняется, и абзац стал неверным.
- [ ] Косметика: в `scenario_accuracy_by_decoder` заголовок `Cross-task (Dataset B)`
      слегка задевает горизонтальную линейку.
- [ ] Удалить `cvg/plos_try1/` — там сохранена сломанная попытка из тупика 2,
      оставлена только для сравнения.
