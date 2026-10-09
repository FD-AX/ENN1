# HW2 · Cut Cross-Entropy

Воспроизведение вычисления loss и gradients из *Cut Your Losses in Large-Vocabulary Language Models*.
Папка `HW2/` самостоятельна и расположена рядом с `hw1/` в том же репозитории.

- [Notebook с пояснениями, кодом и выводами](cce_reproduction.ipynb).
- [Подробный отчёт: выкладки, результаты, доказательства и текст доклада](REPORT.md).
- [Презентация HTML](presentation.html) и [PDF](presentation.pdf): 12 основных слайдов + 5 приложений.
- [Исходные измерения и profiler-трассы](results/), [статья PDF](sources/paper.pdf).

Весь код воспроизведения находится в notebook. Вспомогательные рабочие `.py` сохранены
локально и исключены через `.gitignore`; notebook их не импортирует и не запускает.
Авторская CCE-библиотека, PyTorch и Triton остаются внешними зависимостями.

## Быстро прочитать работу

В notebook: §2 — формулы, §4 — correctness, §§10–12 — основной Gemma experiment,
§§5–6 — синтетическое масштабирование, §14 — выводы. §§7–9 и §13 содержат diagnostics
и audit. Для исполнения используется обычный порядок Run All сверху вниз.

## Открыть и проверить сохранённые результаты

Команды из корня репозитория:

```powershell
python -m venv HW2/.venv
HW2/.venv/Scripts/python.exe -m pip install -r HW2/requirements-notebook.txt
HW2/.venv/Scripts/python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
HW2/.venv/Scripts/python.exe -m jupyterlab HW2/cce_reproduction.ipynb
```

Выбрать kernel этого окружения и выполнить **Run All**. Если уже есть окружение с PyTorch,
достаточно установить notebook-зависимости. На Linux команды аналогичны, путь к Python —
`HW2/.venv/bin/python`. Численные GPU-серии измерены на Windows; другая платформа не
объявляется идентичной исходной среде.

По умолчанию выполняются пять CPU-проверок формул, построение таблиц и графиков из JSON,
сверка 15 исходных трасс с summaries и проверка medians. CUDA, Triton и скачивание модели
в этом режиме не нужны. Сохранённые outputs в notebook получены выполнением этих ячеек.
Чтение архива явно отделено от новых GPU-измерений.

## Повторить GPU-эксперименты

Для измеренной Windows-среды вместо CPU PyTorch установить CUDA-сборку и Triton:

```powershell
HW2/.venv/Scripts/python.exe -m pip install --upgrade --force-reinstall torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
HW2/.venv/Scripts/python.exe -m pip install -r HW2/requirements-windows.txt
```

В первой code cell notebook включить нужные флаги, затем выполнить Run All:

| Флаг | Действие |
|---|---|
| `RUN_GPU_SMOKE` | 60 correctness checks и маленький benchmark для проверки среды |
| `RUN_SYNTHETIC` | Полная сетка 88 строк, correctness и 100 classifier SGD updates |
| `RUN_DIAGNOSTICS` | Отдельный backward и сравнение режимов |
| `RUN_CACHE` | 54 записи повторного benchmark по двум протоколам |
| `RUN_PROFILER` | 12 CPU/CUDA/memory трасс синтетической серии |
| `DOWNLOAD_REAL` | Закреплённые snapshots Gemma и Alpaca |
| `PREPARE_REAL` | Извлечение 8192 next-token позиций из Gemma |
| `RUN_REAL` | Head benchmark на реальных входах, correctness и 3 трассы |

При первом GPU-запуске notebook клонирует авторскую библиотеку в `vendor/` и закрепляет
commit `3de376c106a1916bc5e1b619f9c77c87a461ee1c`; нужен Git и доступ к GitHub.
При повторных запусках проверяет commit и отсутствие изменений в исходниках kernels.
Измеренная среда: RTX 5070 Ti, Python 3.11.9, PyTorch 2.11.0+cu128,
Triton Windows 3.6.0.post26. Полные сведения находятся в `results/*/environment.json`.

Для Gemma дополнительно установить `requirements-real.txt`. Доступ Hugging Face к модели
должен быть уже предоставлен; токен в notebook не записывается. Веса и подготовленные входы
занимают примерно 6.5 GB сверх окружения и не коммитятся. Если inputs уже подготовлены,
включать только `RUN_REAL`; повторная extraction защищена от перезаписи.

Новые измерения сохраняются в `results/notebook_runs/<UTC timestamp>/`, отдельно от исходных.
Сначала сохранить notebook: provenance новых запусков включает хеш его code cells на диске.
Таблицы с подписью «Архив» продолжают показывать исходную серию; для анализа нового запуска
нужно явно направить соответствующее чтение JSON на его каталог.

## Что подтверждают результаты

Основной результат — **classifier-head forward+backward на извлечённых activations Gemma 2 2B / Alpaca**:
N1024/V256000/D2304, BF16, softcap30. Extra allocated peak **3500.001 → 1132.734 MiB**,
время **62.751 → 31.877 ms** (baseline_author → CCE default; 1.97× по времени head).
Backbone не входит в измерение. [Подробности и full allocated peak](REPORT.md#main-result).

Синтетика служит дополнительной проверкой: N4096/V65536/D512, BF16, forward+backward —
extra peak 3072.001 → 68.533 MiB (−97.8%), время 22.710 → 14.034 ms.
Extra peak = peak allocated − allocated before operation; это не полная VRAM процесса.
Runtime benefit зависит от shape и режима: при N1024/V65536/D2048 CCE медленнее на 12.3%.
Полное обучение LLM и качество модели не воспроизводились.

Неподтвердившиеся 4.228 ms small baseline исключены из выводов о скорости и timing-кривой;
исходный JSON сохранён. Baseline Gemma N8192 заранее пропущен по memory guard — OOM не
наблюдался. Причины расхождений, skipped tiles и аппаратные DRAM/occupancy counters не заявляются.
Все ограничения и основания цифр приведены в отчёте.

Исторические SHA256 в environment относятся к рабочим скриптам первоначальных запусков;
они сохранены без изменений и не выдаются за хеш notebook. Перенос кода проверяется
выполнением notebook из копии публичных файлов без вспомогательных `.py`.

Проверка переноса: 44 code cells выполнены, 5 CPU-тестов и сверка 15 трасс прошли.
Отдельно выполнены 60 новых GPU correctness checks и smoke benchmark из 8 строк.
[Протокол проверки](results/notebook_validation.json) и
[сырые результаты smoke](results/notebook_smoke/) сохранены. Короткие smoke timings
служат проверкой работоспособности кода и не заменяют основные измерения отчёта.

HTML-презентация: стрелки переключают слайды, `N` открывает заметки. Для MathJax/fonts
используются внешние CDN; PDF содержит готовые формулы и доступен офлайн.
