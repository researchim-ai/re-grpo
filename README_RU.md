# Минимальная реализация GRPO

Это форк проекта: https://github.com/open-thought/tiny-grpo  
с целью упрощения использования.

Цель: Рабочая демонстрационная реализация локального RL-обучения llama-3.2-1b с использованием GRPO. Понимание алгоритма и гиперпараметров. Всё выполняется локально на одном узле.

В оригинальном проекте llama-3.2-1b не запускалась на домашних видеокартах уровня хотя бы 3090-4090 с 24ГБ VRAM. А хочется.

Сейчас добавлена квантизация и lora для экономии ресурсов.

Вполне можно расширять функционал и экспериментировать (изначальный проект для того и создан).

### Настройка

1. Создайте окружение conda

```
conda create --name grpo python=3.12 -y
conda activate grpo
```

2. Установите зависимости

```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

3. Поэкспериментируйте с исходным кодом в файле `train.py`

```
python train.py
```

### Эксперименты с исходным кодом

Каждый эксперимент теперь находится в своей собственной директории.

**1. Базовый эксперимент (одна GPU, без Accelerate)**

Для запуска основного скрипта обучения:
```bash
python experiment_base/train.py
```

**2. Эксперимент с Accelerate (одна/несколько GPU)**

Сначала настройте Accelerate, если вы этого еще не сделали:
```bash
accelerate config
```
Следуйте инструкциям. Для одной GPU вы можете выбрать "This machine" и "No distributed training". Для нескольких GPU настройте соответственно.

Затем запустите обучение:
```bash
accelerate launch experiment_accelerate/train_accelerate.py
```

**3. Эксперимент с Accelerate DDP (несколько GPU с Accelerate DDP)**

Убедитесь, что Accelerate настроен для DDP (Distributed Data Parallel).
```bash
accelerate config
```
Выберите "This machine" и укажите количество GPU, которые вы хотите использовать для распределенного обучения.

Затем запустите обучение:
```bash
accelerate launch experiment_accelerate_ddp/train_accelerate_ddp.py
```

**4. Эксперимент с многоходовым калькулятором (Multiturn Calculator Tool Calling)**

Этот скрипт обучает модель для многоходовых взаимодействий с использованием инструмента-калькулятора.
```bash
python experiment_multiturn_tool_calling/train_multiturn_calc_tool_calling.py
```
Скрипт принимает аргументы командной строки. Используйте `--help` для просмотра доступных опций:
```bash
python experiment_multiturn_tool_calling/train_multiturn_calc_tool_calling.py --help
```

**5. Эксперимент с многоходовыми вопросами-ответами и инструментом поиска (Multiturn QA with Search Tool)**

Этот скрипт обучает модель для многоходовой системы вопросов и ответов, которая может использовать инструмент поиска.
```bash
python experiment_multiturn_qa/train_multiturn_qa.py
```
Скрипт также принимает аргументы командной строки. Используйте `--help` для просмотра доступных опций:
```bash
python experiment_multiturn_qa/train_multiturn_qa.py --help
```

**6. Эксперимент Dr. GRPO (GRPO с техниками уменьшения смещений)**

Директория `experiment_drgrpo` содержит реализацию GRPO с модификациями, вдохновленными "Dr. GRPO", для уменьшения определенных смещений (bias).
Ключевые изменения:
- **Удаление смещения по длине (Length Bias Removal)**: Агрегация значений критика была изменена для использования маскированной суммы с постоянным нормализатором (на основе `generate_max_length`) вместо простого маскированного среднего. Это контролируется параметром `critic_type="drgrpo"` в классе `GRPOLoss` и соответствующей конфигурацией в `train.py`.
- **Удаление смещения по сложности (Difficulty Bias Removal)**: Расчет преимущества Монте-Карло (Monte Carlo advantage) скорректирован таким образом, чтобы не производить нормализацию по стандартному отклонению вознаграждений. Это изменение реализовано в функции `group_advantages` в `train.py`.

Эти модификации направлены на улучшение устойчивости и производительности алгоритма GRPO.
Для запуска этого эксперимента:
```bash
python experiment_drgrpo/train.py
```

### Проект вдохновлен

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)

### Дополнительные ресурсы

- [Технический отчёт DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)

### Параметры командной строки для скриптов с `tool_calling` и `qa`

Скрипты `experiment_multiturn_tool_calling/train_multiturn_calc_tool_calling.py` и `experiment_multiturn_qa/train_multiturn_qa.py` принимают следующие аргументы командной строки:

*   `--run-name` (строка, по умолчанию: `None`):
    *   Пользовательское имя для запуска логирования (переопределяет автоматически сгенерированное имя).
*   `--log-dir` (строка, по умолчанию: `"runs"`):
    *   Корневая директория для логов TensorBoard.
*   `--wandb` (флаг):
    *   Использовать Weights & Biases (WandB) для логирования.
*   `--wandb-project` (строка, по умолчанию: `"tiny_grpo_v2"`):
    *   Название проекта в WandB.
*   `--seed` (целое число, по умолчанию: `42`):
    *   Начальное значение для генератора случайных чисел.
*   `--device-index` (целое число, по умолчанию: `0`):
    *   Индекс CUDA-устройства (например, `0` для `cuda:0`).
*   `--model-name` (строка, по умолчанию: `"Qwen/Qwen2.5-3B-Instruct"`):
    *   Имя или путь к модели из Hugging Face Hub.
*   `--checkpoint-path` (строка, по умолчанию: `"./output"`):
    *   Путь для сохранения чекпоинтов.
*   `--checkpoint-interval` (целое число, по умолчанию: `20`):
    *   Сохранять чекпоинт каждые N батчей.
*   `--data-path` (строка):
    *   Путь к файлу с данными для обучения. Для `calc`: по умолчанию `"data/math_tasks.jsonl"`. Для `qa`: по умолчанию `"qa_generated_data/questions.json"`.
*   `--max-prompts` (целое число, по умолчанию: `1024`):
    *   Максимальное количество промптов для загрузки.
*   `--group-size` (целое число, по умолчанию: `4`):
    *   Количество "роллаутов" на одну задачу.
*   `--rollouts-per-step` (целое число, по умолчанию: `4`):
    *   Количество задач в одном батче на этапе генерации "роллаутов".
*   `--train-batch-size` (целое число, по умолчанию: `4`):
    *   Размер батча для этапа обучения.
*   `--lr` (число с плавающей точкой, по умолчанию: `5e-6`):
    *   Скорость обучения.
*   `--temperature` (число с плавающей точкой, по умолчанию: `0.7`):
    *   Температура для семплирования при генерации.
*   `--max-length` (целое число, по умолчанию: `512`):
    *   Максимальная длина последовательности.
*   `--log-completions-interval` (целое число, по умолчанию: `10`):
    *   Логировать примеры сгенерированных последовательностей каждые N батчей.