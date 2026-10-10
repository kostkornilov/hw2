# Запуск воспроизведения

Текущий план: две независимые джобы — RegBench и sort. LM исключён из
длительного запуска. Обе используют `run_icl_sort.py` с `--task icl` или
`--task sort`; упаковка фиксирует задачу, поэтому в Kaggle аргументы не нужны.
Исходный объединённый режим `--task icl_sort` и `run_lm.py` сохранены.
Параметры находятся в `config.json`. Все методы внутри направления имеют один
seed, одинаковый объём обучения и одинаковое расписание LR.

| Режим | LM, шагов/метод | RegBench, эпох/метод | Sort, шагов/метод |
|---|---:|---:|---:|
| smoke | 4 | 3 (по одному batch) | 2 |
| full | 1000 | 200 | 6000 |

В обоих режимах LM использует warmup 25 / decay 1000; sort — 1536 / 6000.
RegBench сохраняет авторское расписание на 200 эпох. Smoke проверяет весь
pipeline, а не качество или сходимость. Финальная оценка sort использует
64/128/256, по 2 примера в smoke и по 1000 в full. RegBench в обоих
режимах использует `icl_eval_examples` примеров на каждый split: 2 validation +
2 test в smoke, 500 + 500 в full. Этот размер используется и во время обучения,
и при финальном экспорте. Кэши для разных размеров сохраняются отдельно.
RegBench smoke использует 32 train примера, лимит строки 16 символов и два
in-context слова длиной 1–3 токена. Это только проверка pipeline; для full
сохраняется авторская генерация. Скорость укороченного smoke не используется
для прогноза полного обучения; прогноз опирается на прежнюю пробу ноутбука.

## Обязательная проверка перед full

Общий pipeline уже проверен скачиванием smoke-архивов. При разделении задач
проверяем локально выбор этапов, упаковку и проверку отдельных архивов из
скачанных smoke-результатов. Обучение, данные и гиперпараметры не меняются.
Текущие команды упаковки двух независимых полных запусков:

```bash
python3 kaggle/run_icl_sort.py --task icl --mode full --package /tmp/hw2-icl-full
python3 kaggle/run_icl_sort.py --task sort --mode full --package /tmp/hw2-sort-full
kaggle kernels push -p /tmp/hw2-icl-full
kaggle kernels push -p /tmp/hw2-sort-full
```

Ниже приведены команды исходной проверки LM и объединённой джобы.

Сначала закоммитить и запушить изменения `kaggle/` и helpers в `scripts/`.
Ноутбук для этого не нужен. Упаковка фиксирует локальный HEAD; Kaggle скачивает
ровно этот commit, включая конфиг. Метаданные указывают приватные джобы и T4 ×2.

```bash
python3 kaggle/run_lm.py --package /tmp/hw2-lm-smoke --mode smoke
python3 kaggle/run_icl_sort.py --package /tmp/hw2-icl-sort-smoke --mode smoke
kaggle kernels push -p /tmp/hw2-lm-smoke --accelerator NvidiaTeslaT4
kaggle kernels push -p /tmp/hw2-icl-sort-smoke --accelerator NvidiaTeslaT4
kaggle kernels status konstantinkornilov/hw2-lm
kaggle kernels status konstantinkornilov/hw2-icl-sort
```

После завершения скачать ZIP до следующего push в эту джобу. Каталог скачивания
должен быть новым. В установленном Kaggle CLI 2.2.4 команда `kernels output`
игнорирует суффикс номера версии и получает последние результаты. При проверке
сверить `repository_commit` и `mode` в `configuration.json` внутри архива с
проверяемым запуском; результаты smoke сохранить до отправки full.

```bash
kaggle kernels output konstantinkornilov/hw2-lm -p artifacts/kaggle/lm-smoke --file-pattern '.*[.]zip$'
kaggle kernels output konstantinkornilov/hw2-icl-sort -p artifacts/kaggle/icl-sort-smoke --file-pattern '.*[.]zip$'
python3 kaggle/run_lm.py --check-output <DOWNLOADED_LM_ZIP>
python3 kaggle/run_icl_sort.py --check-output <DOWNLOADED_ICL_SORT_ZIP>
```

Проверка ZIP читает все файлы и сверяет размеры/SHA256, завершение всех subprocess,
число шагов/эпох и результаты проверок артефактов каждого метода. На GPU до
создания ZIP дополнительно проверяются состояния checkpoint и метрики из
сохранённых предсказаний. Нулевой exact match в smoke не считается ошибкой.

**Full разрешён только после успешного smoke обеих джоб, скачивания и проверки
архивов, сверки оставшейся квоты и отдельного указания пользователя на запуск.**
Упаковка full делается теми же командами с `--mode full` и новыми каталогами.
Не требуется менять конфиг или trainer. Если код изменился после smoke,
проверку затронутого pipeline повторить перед full.

## Артефакты

Код, Python 3.11, зависимости и большие временные данные находятся в `/tmp`.
Результаты — `/kaggle/working/results/<job>/<mode_timestamp>/` и ZIP рядом:
конфигурация/commit, environment, логи, checkpoints, таблицы CSV/MD, графики
PNG/PDF, предсказания и `files_manifest.json`. RegBench сохраняет также кэш
датасета; sort — тестовые данные и неизменённый авторский генератор с manifest.
Обучающие данные sort воспроизводятся из сохранённых seed и параметров.

При ошибке предыдущие результаты сохраняются; статус джобы и архив помечаются
как failed. Длительное обучение автоматически после smoke не начинается.

`--plan --mode full` печатает настройки локально без запуска. Указанные в конфиге
гиперпараметры LM дополнены сокращением обучения и сжатием LR schedule; результаты
нельзя трактовать как точное воспроизведение 80k-step эксперимента статьи.
