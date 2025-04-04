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

### Проект вдохновлен

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)

### Дополнительные ресурсы

- [Технический отчёт DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)