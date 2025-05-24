# Руководство по логированию в Weights & Biases (wandb)

Данное руководство описывает настройку и использование wandb логирования во всех экспериментах проекта re-grpo.

## Обзор изменений

Все эксперименты теперь включают расширенное логирование в wandb со следующими метриками:

### Основные метрики для всех экспериментов:
- **Роллауты (Rollouts)**:
  - `rollouts/returns_sum` - суммарные вознаграждения 
  - `rollouts/returns_mean` - средние вознаграждения
  - `rollouts/returns_std` - стандартное отклонение вознаграждений
  - `rollouts/returns_max` - максимальные вознаграждения
  - `rollouts/returns_min` - минимальные вознаграждения
  - `rollouts/returns_best_mean` - лучшее среднее значение за все время
  - `rollouts/kl_mean/std/max` - статистики KL дивергенции
  - `rollouts/advantages_mean/std` - статистики преимуществ (advantages)
  - `rollouts/total_rollouts` - общее количество роллаутов
  - `rollouts/batch_step` - номер батча

- **Обучение (Training)**:
  - `training_step/loss` - потери на каждом шаге
  - `training_step/kl` - KL дивергенция на каждом шаге
  - `training_step/grad_norm` - норма градиентов
  - `training_step/learning_rate` - текущий learning rate
  - `training_epoch/loss_mean/std` - средние и стандартные отклонения потерь по эпохе
  - `training_epoch/kl_mean/std` - статистики KL по эпохе
  - `training_epoch/grad_norm_mean/max` - статистики градиентов по эпохе

- **Система (System)**:
  - `system/gpu_memory_allocated_gb` - использованная GPU память в GB
  - `system/gpu_memory_reserved_gb` - зарезервированная GPU память в GB
  - `system/replay_buffer_size` - размер буфера переигровки

- **Чекпоинты (Checkpoints)**:
  - `checkpoint/saved` - индикатор сохранения чекпоинта
  - `checkpoint/step` - шаг на котором сохранен чекпоинт

- **Финальные метрики (Final)**:
  - `final/total_steps` - общее количество шагов обучения
  - `final/total_batches` - общее количество батчей
  - `final/total_rollouts` - общее количество роллаутов
  - `final/best_mean_return` - лучший средний результат

## Специфичные метрики по экспериментам

### DR-GRPO (`experiment_drgrpo`)
Дополнительно логируются:
- `dr_grpo/advantages_variance` - дисперсия преимуществ
- `dr_grpo/advantages_skewness` - асимметрия преимуществ
- `dr_grpo/critic_type` - тип критика (если настроен)
- `training_step/dr_component_*` - компоненты DR-GRPO loss функции

### Accelerate (`experiment_accelerate`)
Дополнительно логируются:
- `accelerate/device` - используемое устройство
- `accelerate/sync_gradients` - синхронизация градиентов

### Accelerate DDP (`experiment_accelerate_ddp`)
Дополнительно логируются:
- `ddp/num_processes` - количество процессов
- `ddp/process_index` - индекс текущего процесса
- `ddp/local_process_index` - локальный индекс процесса
- `ddp/sync_gradients` - синхронизация градиентов

### Multiturn QA и Tool Calling
Используют продвинутый Logger класс с дополнительными возможностями:
- `batch_returns/*` - детальные метрики по батчам
- `examples/completion` - примеры сгенерированных завершений
- `training/invalid_loss_count` - счетчик невалидных потерь

## Запуск экспериментов

### Базовые эксперименты

#### 1. Базовый GRPO (`experiment_base`)
```bash
cd experiment_base
python train.py
```
Проект wandb: `tiny_grpo`

#### 2. DR-GRPO (`experiment_drgrpo`)
```bash
cd experiment_drgrpo
python train.py
```
Проект wandb: `drgrpo_experiment`

#### 3. Accelerate GRPO (`experiment_accelerate`)
```bash
cd experiment_accelerate
python train_accelerate.py
```
Проект wandb: `accelerate_grpo`

#### 4. Accelerate DDP GRPO (`experiment_accelerate_ddp`)
```bash
cd experiment_accelerate_ddp
# Одна GPU
python train_accelerate_ddp.py

# Несколько GPU
accelerate launch --multi_gpu --num_processes=2 train_accelerate_ddp.py
```
Проект wandb: `grpo_accelerate_ddp`

### Продвинутые эксперименты с аргументами

#### 5. Multiturn QA (`experiment_multiturn_qa`)
```bash
cd experiment_multiturn_qa

# С wandb (по умолчанию включен)
python train_multiturn_qa.py --model-name "Qwen/Qwen2.5-3B-Instruct" --max-prompts 512

# Без wandb
python train_multiturn_qa.py --no-wandb

# Кастомная настройка
python train_multiturn_qa.py \
    --wandb-project "my_qa_experiment" \
    --run-name "qa_test_run" \
    --lr 1e-5 \
    --group-size 8 \
    --rollouts-per-step 8 \
    --train-batch-size 8
```
Проект wandb: `tiny_grpo_multiturn_qa`

#### 6. Multiturn Tool Calling (`experiment_multiturn_tool_calling`)
```bash
cd experiment_multiturn_tool_calling

# С wandb (по умолчанию включен)
python train_multiturn_calc_tool_calling.py --model-name "Qwen/Qwen2.5-3B-Instruct"

# Без wandb
python train_multiturn_calc_tool_calling.py --no-wandb

# Кастомная настройка
python train_multiturn_calc_tool_calling.py \
    --wandb-project "my_tool_calling_experiment" \
    --temperature 0.8 \
    --max-length 1024
```
Проект wandb: `tiny_grpo_multiturn_tool_calling`

## Параметры командной строки

### Общие параметры для multiturn экспериментов:
- `--wandb` - использовать wandb (включен по умолчанию)
- `--no-wandb` - отключить wandb
- `--wandb-project` - название проекта в wandb
- `--run-name` - кастомное имя запуска
- `--log-dir` - директория для логов TensorBoard (если wandb отключен)
- `--model-name` - название модели
- `--lr` - learning rate
- `--group-size` - количество роллаутов на задачу
- `--rollouts-per-step` - количество задач в батче роллаутов
- `--train-batch-size` - размер батча для обучения
- `--temperature` - температура генерации
- `--max-length` - максимальная длина последовательности
- `--checkpoint-interval` - интервал сохранения чекпоинтов
- `--seed` - random seed

## Настройка wandb

### Установка и настройка
```bash
# Установка
pip install wandb

# Логин (нужно сделать один раз)
wandb login

# Или установить переменную окружения
export WANDB_API_KEY="your_api_key_here"
```

### Переменные окружения
```bash
# Отключить wandb для всех запусков
export WANDB_MODE="offline"

# Установить проект по умолчанию
export WANDB_PROJECT="my_default_project"

# Установить entity (команду)
export WANDB_ENTITY="my_team"
```

## Мониторинг экспериментов

После запуска экспериментов вы можете отслеживать их прогресс в веб-интерфейсе wandb:

1. Перейдите на https://wandb.ai
2. Найдите ваш проект (например, `tiny_grpo`, `drgrpo_experiment`, и т.д.)
3. Просматривайте графики в реальном времени

### Полезные графики для анализа:
- `rollouts/returns_mean` - основная метрика производительности
- `training_step/loss` - прогресс обучения
- `training_step/kl` - контроль KL дивергенции
- `system/gpu_memory_*` - использование ресурсов

## Сравнение экспериментов

Для сравнения разных экспериментов:
1. Все эксперименты используют единую схему именования метрик
2. Можно сравнивать runs из разных проектов на общих графиках
3. Используйте wandb Sweeps для автоматической оптимизации гиперпараметров

## Отладка

Если возникают проблемы с wandb:
```bash
# Проверить статус
wandb status

# Запустить в offline режиме
export WANDB_MODE="offline"
python train.py

# Синхронизировать offline runs
wandb sync wandb/offline-run-*/
```

## Примеры использования

### Быстрый тест
```bash
# Базовый тест
cd experiment_base && python train.py

# Multiturn тест с ограниченным количеством промптов
cd experiment_multiturn_qa
python train_multiturn_qa.py --max-prompts 100 --rollouts-per-step 2
```

### Продвинутый эксперимент
```bash
cd experiment_multiturn_tool_calling
python train_multiturn_calc_tool_calling.py \
    --wandb-project "advanced_tool_calling" \
    --run-name "qwen_3b_high_temp" \
    --model-name "Qwen/Qwen2.5-3B-Instruct" \
    --temperature 0.9 \
    --lr 3e-6 \
    --group-size 12 \
    --rollouts-per-step 16 \
    --max-prompts 2048
```

Все эксперименты теперь готовы для полноценного логирования и мониторинга через wandb! 