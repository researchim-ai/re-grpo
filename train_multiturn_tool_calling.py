import os
# так можно выбирать устройство для запуска LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import json
import random
from pathlib import Path
from typing import Any, Iterator, Optional, Callable
from collections.abc import Sequence

import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
from bitsandbytes.optim import AdamW32bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from multiturn_loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


###############################################################################
# БЛОК С ОПРЕДЕЛЕНИЕМ "ИНСТРУМЕНТОВ" (tools) И ФУНКЦИИ ВЫЗОВА
###############################################################################
TOOLS = {}

def register_tool(name: str):
    """
    Декоратор для регистрации инструмента по имени.
    """
    def decorator(func):
        TOOLS[name] = func
        return func
    return decorator

@register_tool("calc")
def calc_tool(expression: str) -> str:
    """
    Простой инструмент для вычисления арифметических выражений.
    Используется eval, в реальном продакшене нужно быть осторожным.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calc error: {e}"

@register_tool("echo")
def echo_tool(message: str) -> str:
    """
    Тривиальный пример: просто возвращает 'ECHO: ' + исходное сообщение.
    """
    return f"ECHO: {message}"

def detect_and_call_tools(generated_text: str) -> str:
    """
    Ищет во входном тексте все фрагменты вида <tool_call:имя_инструмента>...</tool_call>.
    Для каждого вызова:
      - Определяет имя инструмента (tool_name),
      - Передаёт содержимое (tool_input) соответствующей функции,
      - Заменяет вызов в тексте на результат работы инструмента.
    Возвращает текст, в котором вызовы инструментов заменены на результат.
    """
    pattern = r"<tool_call:(\w+)>(.*?)</tool_call>"

    def _replace_tool_call(match: re.Match) -> str:
        tool_name = match.group(1)
        tool_input = match.group(2)
        tool = TOOLS.get(tool_name)
        if tool is None:
            return f"[Tool '{tool_name}' not found]"
        else:
            print(f"tool calling: {tool_name}")
        result = tool(tool_input.strip())
        return result

    updated_text = re.sub(pattern, _replace_tool_call, generated_text, flags=re.DOTALL)
    return updated_text


###############################################################################
# БЛОК С ЗАГРУЗКОЙ МОДЕЛИ
###############################################################################
def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
    use_4bit: bool = True,
    use_lora: bool = True,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        print("Используется 4-битная квантизация")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    else:
        print("Используется 8-битная квантизация")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        quantization_config=quantization_config,
    )

    # Проверяем тип параметров модели
    print("\nПроверка типов параметров модели:")
    total_size = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            param_size = param.numel() * param.element_size() / 1024**2
            total_size += param_size
            print(f"{name}: {param.dtype}, размер: {param_size:.2f} MB")

    print(f"\nОбщий размер модели: {total_size:.2f} MB")

    if use_lora:
        print("\nПрименяется LoRA")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


###############################################################################
# PROMPT (добавляем инструкцию вызывать калькулятор)
###############################################################################
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
<think> reasoning process here </think>
<answer> answer here </answer>.

IMPORTANT: 
If you need to do any arithmetic or calculation, ALWAYS call:
    <tool_call:calc>some_expression</tool_call>
Then wait for the result (it will be injected into your answer), 
and only after that provide the final <answer> to the user.
"""


###############################################################################
# СПИСОК ФУНКЦИЙ НАГРАД (reward_fns) И ИХ СУММАТОР
###############################################################################
def reward_for_tool_usage(generated_text: str, tool_name: str = "calc", reward_amount: float = 0.2) -> float:
    """
    Если в тексте встречается <tool_call:calc>, даём дополнительную награду 
    (чтобы модель была мотивирована использовать инструмент).
    """
    pattern = rf"<tool_call:{tool_name}>(.*?)</tool_call>"
    if re.search(pattern, generated_text, flags=re.DOTALL):
        return reward_amount
    return 0.0

def reward_for_correct_answer(generated_text: str, oracle_answer: str, correct_reward: float = 1.0, partial_reward: float = 0.5, wrong_reward: float = 0.01) -> float:
    """
    Ищем <answer>...</answer> в сгенерированном тексте.
    Сравниваем со строкой oracle_answer:
      - Если совпадает по trim(), даём correct_reward,
      - Если substring, даём partial_reward,
      - Иначе wrong_reward.
    """
    match = re.search(r"<answer>(.*?)</answer>", generated_text, flags=re.DOTALL)
    if match:
        ans = match.group(1).strip()
        if ans == oracle_answer:
            return correct_reward
        elif oracle_answer in ans:
            return partial_reward
        else:
            return wrong_reward
    return 0.0  # Если вообще нет тега <answer> - награда 0

def sum_rewards(text: str, oracle: str, reward_fns: list[Callable[[str, str], float]]) -> float:
    """
    Проходимся по списку функций награды, суммируем результат.
    Каждая функция имеет сигнатуру: fn(generated_text, oracle_answer) -> float
    """
    total = 0.0
    for fn in reward_fns:
        total += fn(text, oracle)
    return total


###############################################################################
# ФУНКЦИЯ ROLLOUT, ОБНОВЛЕННАЯ ДЛЯ ВЫЗОВА НАГРАДОК
###############################################################################
@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    reward_fns: list[Callable[[str, str], float]],
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tool_loops: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Генерирует ответы модели (num_rollouts раз) + проверка вызовов инструментов.
    
    Параметры:
      - reward_fns: список функций награды, каждая принимает (generated_text, oracle_answer).
        Сумма их возвратов = общая награда (returns[i]).

    Возвращает:
      - final_sequence_ids: тензор токенов (B x SeqLen) после применения инструментов,
      - returns: награды (B x 1),
      - action_mask: булева маска (B x SeqLen), какая часть "действие" (для RL),
      - final_texts: список строк финальных ответов.
    """
    model.eval()

    # 1. Формируем prompt
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": task},
    ]
    # Если есть у tokenizer специальный метод для chat, используем:
    if hasattr(tokenizer, "apply_chat_template"):
        chat_prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        chat_prompt = system_prompt + "\nUser: " + task + "\nAssistant:"

    # Подготовка к генерации
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    ).to("cuda")

    # Дублируем prompt num_rollouts раз
    model_inputs["input_ids"] = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(num_rollouts, 1)

    # 2. Генерируем "сырой" ответ
    raw_sequences = model.generate(**model_inputs, generation_config=generation_config)

    # 3. Раскодируем ответы и обрабатываем вызовы инструментов
    completions = tokenizer.batch_decode(
        raw_sequences[:, model_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    final_texts = []
    for text in completions:
        updated_text = text
        for _ in range(max_tool_loops):
            # Вызываем инструменты, если они есть
            new_text = detect_and_call_tools(updated_text)
            if new_text == updated_text:
                # Если вызовов инструментов больше нет, завершаем
                break
            # Иначе обновляем текст
            updated_text = new_text
        final_texts.append(updated_text)

    # 4. Склеиваем prompt + итоговый ответ (после инструментов)
    prompt_input_ids = model_inputs["input_ids"]
    new_inputs = tokenizer(final_texts, return_tensors="pt", padding=True).to("cuda")

    # Собираем единый тензор
    all_sequences = []
    for i in range(num_rollouts):
        # prompt
        prompt_seq = prompt_input_ids[i]
        # финальный ответ
        answer_seq = new_inputs["input_ids"][i]
        # склеиваем
        fused = torch.cat([prompt_seq, answer_seq], dim=0)
        all_sequences.append(fused.unsqueeze(0))

    final_sequence_ids = torch.cat(all_sequences, dim=0)

    # 5. Считаем награды: суммируем результаты функций из reward_fns
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float, device=final_sequence_ids.device)
    for i, completion in enumerate(final_texts):
        total_reward = sum_rewards(completion, oracle_answer, reward_fns)
        returns[i] = total_reward

    # 6. Формируем action_mask (где часть после prompt является "действием")
    action_mask = torch.zeros_like(final_sequence_ids, dtype=torch.bool)
    prompt_len = prompt_input_ids.shape[1]
    for i in range(num_rollouts):
        seq_len = final_sequence_ids[i].shape[0]
        action_mask[i, prompt_len:seq_len] = True

    return final_sequence_ids, returns, action_mask, final_texts


###############################################################################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЧТЕНИЯ ДАННЫХ
###############################################################################
def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


###############################################################################
# ДРУГИЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ RL-ТРЕНИРОВКИ
###############################################################################
def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Нормализация возвратов (returns) -> advantages.
    """
    return (returns - returns.mean()) / (returns.std() + eps)

def sequence_log_probs_from_logits(
    logits: torch.Tensor, output_ids: torch.Tensor
) -> torch.Tensor:
    """
    Вычислить log_softmax + выбрать индекс выходного токена.
    Возвращает размер [B, seq_len-1], если logits = [B, seq_len, vocab],
    output_ids = [B, seq_len], срезая последний логит и первый токен.
    """
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)

def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Считаем log_probs (форма [B, seq_len-1]) по сдвинутой схеме:
    - logits[:, :-1, :] vs sequence_ids[:, 1:].
    """
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]  # [B, seq_len, vocab]
    # убираем последний логит и первый токен
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    # log_probs.shape = [B, seq_len-1, vocab]
    next_tokens = sequence_ids[:, 1:]  # [B, seq_len-1]
    gathered = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)  # [B, seq_len-1]
    return gathered


###############################################################################
# ОСНОВНАЯ ФУНКЦИЯ TRAIN (main)
###############################################################################
def main():
    # ------------------------------
    # ПАРАМЕТРЫ
    # ------------------------------
    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 0
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 8
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    # Параметры квантизации и LoRA
    use_4bit = True
    use_lora = True
    bf16 = True

    group_size = 8
    rollouts_per_step = 8
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # ------------------------------
    # СПИСОК ФУНКЦИЙ НАГРАД
    # ------------------------------
    reward_fns = [
        # 1) награда за использование тула "calc"
        lambda text, oracle: reward_for_tool_usage(text, "calc", 0.2),
        # 2) награда за правильный ответ
        lambda text, oracle: reward_for_correct_answer(text, oracle),
    ]

    # ------------------------------
    # МОДЕЛЬ-РЕФЕРЕНС
    # ------------------------------
    reference_model, _ = load_model(
        model_name,
        device_map=device,
        use_4bit=use_4bit,
        use_lora=False,
        bf16=bf16
    )
    reference_model.eval()

    # ------------------------------
    # МОДЕЛЬ ДЛЯ ОБУЧЕНИЯ
    # ------------------------------
    model, tokenizer = load_model(
        model_name,
        device_map=device,
        use_4bit=use_4bit,
        use_lora=use_lora,
        bf16=bf16
    )
    optimizer = AdamW32bit(model.parameters(), lr=lr, is_paged=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    # ------------------------------
    # ДАТАСЕТ
    # ------------------------------
    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128 and x["num_terms"] <= 3 and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    # ------------------------------
    # GRPO-LOOP
    # ------------------------------
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    # ------------------------------
    # ОСНОВНОЙ ЦИКЛ
    # ------------------------------
    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []
        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                # Вызов нашего rollout c учетом инструментов и списка функций наград
                sequence_ids, returns, action_mask, completions = rollout(
                    model=model,
                    tokenizer=tokenizer,
                    task=q,
                    oracle_answer=a,
                    num_rollouts=group_size,
                    reward_fns=reward_fns,  # <-- передаём список функций наград
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, "
                    f"replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())

                # Вычисляем advantages
                advantages = group_advantages(returns)

                # attention_mask: где токены не паддинг
                attention_mask = (sequence_ids != pad_token_id)

                # Считаем лог-вероятности (форма [B, seq_len-1])
                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                log_probs_ref = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )

                # Обрезаем action_mask до [B, seq_len-1]
                action_mask = action_mask[:, 1:]

                # KL
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                # Сохраняем всё в опыт
                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))

        # ------------------------------
        # ЛОГИ, ОЧИСТКА
        # ------------------------------
        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

        # ------------------------------
        # ОБУЧЕНИЕ НА СОБРАННОМ ОПЫТЕ (replay_buffer)
        # ------------------------------
        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()
            for exp in experience_sampler:
                exp: Experience
                exp = exp.to(device)

                optimizer.zero_grad()
                log_probs = sequences_log_probs(
                    model,
                    sequence_ids=exp.sequences,
                    attention_mask=exp.attention_mask
                )

                loss, kl_val = objective(log_probs=log_probs, experience=exp)
                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={exp.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}: kl={kl_val: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl_val, "grad_norm": grad_norm})

                optimizer.step()

        # ------------------------------
        # ЧЕКПОИНТ
        # ------------------------------
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    # Финальное сохранение
    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
