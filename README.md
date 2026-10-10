# Воспроизведение статьи «Sparse Attention as Compact Kernel Regression» (Santos et al.)

Статья: https://arxiv.org/abs/2601.22766.

Воспроизводятся два направления: **language modeling (LM)** на BabiStories и
**in-context learning (ICL)** на RegBench. В обоих сравниваются Softmax,
Sparsemax и 1-ReLUmax. Используется авторская реализация Memory Mosaics.

## Структура

```text
reproduction_report.md          Отчёт о воспроизведении
pres/
  slides.typ                   Исходник презентации
  slides.pdf                   Презентация
figures/                       Иллюстрации для отчёта
sparse_kernel_regression/      Код авторов статьи
  Library/                     Модели и обучение LM
  ICLL/                        Модели, обучение и анализ ICL
reproduction.ipynb             Подготовительные эксперименты и замеры в Kaggle
lm_interactive.ipynb           Запуск LM в интерактивной сессии Kaggle
kaggle/
  config.json                  Настройки экспериментов
  run_lm.py                    Запуск LM и общая подготовка окружения
  run_icl_sort.py              Запуск ICL: использовать --task icl
  lm/kernel-metadata.json      Метаданные Kaggle-джобы LM
  icl/kernel-metadata.json     Метаданные Kaggle-джобы ICL
scripts/
  export_lm_artifacts.py        Проверка checkpoints LM экспорт таблиц и графиков
  recover_lm_logs.py            Восстановление результатов LM из вывода ноутбука (ноутбук упал в интерактивной сессии)
  run_icl_probe.py              Подготовка данных, обучение, оценка и сводка ICL
  requirements-icl.txt          Дополнительные зависимости ICL
  plot_icl_results.py           Построение графика ICL для отчёта из сохранённой истории
artifacts/
  lm_recovered/                Восстановленные таблицы и графики LM
  kaggle/icl-full-lite/        Итоговые таблицы, истории обучения и проверки ICL
```

### Language modeling

Основной запуск — `kaggle/run_lm.py`; авторский trainer находится в
`sparse_kernel_regression/Library/train_memory_mosaics.py`. Скрипт скачивает
BabiStories, подготавливает данные и последовательно обучает три метода.
Конфигурация: один слой, 1000 шагов на метод вместо авторских 80000,
warmup 25 и cosine decay 1000, один seed, FP16, две T4.

В сохранённом эксперименте Softmax и Sparsemax дошли до 1000 шагов;
последняя записанная оценка 1-ReLUmax — на шаге 800. Поэтому итоговое
сравнение всех методов проводится на одинаковом шаге 800.
[Результаты](artifacts/lm_recovered/results_summary.md) восстановлены из
сохранённого вывода ноутбука через `scripts/recover_lm_logs.py`;
checkpoints этого запуска не сохранились.

### In-context learning

Основной запуск — `kaggle/run_icl_sort.py --task icl`. Он использует
`scripts/run_icl_probe.py` для запуска авторского кода из `ICLL` в окружении
`seq_icl`. Несмотря на название `probe`, скрипт поддерживает полное обучение.

Все три метода прошли авторские **200 эпох**: 1000 обучающих примеров,
500 validation и 500 test, один seed, FP32, две T4. Обозначение **1k**
относится к числу обучающих примеров, а не к длине последовательности.

Результаты находятся в
`artifacts/kaggle/icl-full-lite/results/icl/full_20261010_143712_771702/in_context_learning/`:
`summary.csv` содержит метрики, а папки методов — `timing.json`,
конфигурации и результаты проверок. Суффикс `lite` означает, что скачаны
лёгкие артефакты без checkpoints, датасета и больших дампов предсказаний;
обучение выполнено полностью.

## Запуск

Команды выполняются из корня репозитория. Для построения графиков локально
достаточно Python с установленным `matplotlib`:

```bash
python3 -m pip install matplotlib
python3 scripts/recover_lm_logs.py lm_interactive.ipynb
python3 scripts/plot_icl_results.py artifacts/kaggle/icl-full-lite/results/icl/full_20261010_143712_771702/in_context_learning
```

Первая команда обработки сохраняет таблицы и графики LM в
`artifacts/lm_recovered/`, вторая — график ICL в
`figures/regbench_validation_loss.png` и `.pdf`. Восстановление LM требует
сохранённых выводов ячеек в `lm_interactive.ipynb`.

### Обучение в Kaggle

Готовые скрипты рассчитаны на окружение Kaggle. Они сами устанавливают Python 3.11 и зависимости, готовят данные,
обучают три метода и сохраняют результаты.

В новом Kaggle Notebook выполните ячейку:

```python
!git clone https://github.com/kostkornilov/hw2.git /kaggle/working/hw2
```

Затем выполните команду нужного направления. **LM**:

```python
!python /kaggle/working/hw2/kaggle/run_lm.py --mode full
```

**ICL**:

```python
!python /kaggle/working/hw2/kaggle/run_icl_sort.py --task icl --mode full
```

Эти команды используют код из опубликованной ветки `main` и настройки
`kaggle/config.json`: 1000 шагов LM или 200 эпох ICL на каждый метод.
`full` означает полный объём нашей конфигурации; для LM он сокращён
относительно статьи.

Результаты и ZIP-архив появятся в `/kaggle/working/results/lm/` или
`/kaggle/working/results/icl/`. Скачайте их из панели файлов Kaggle;
внутри находятся checkpoints, логи, конфигурации, таблицы и графики.
