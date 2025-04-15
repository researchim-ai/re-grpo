# -*- coding: utf-8 -*-
import os
# так можно выбирать устройство для запуска LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional, Dict, Union, Tuple
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)
from bitsandbytes.optim import AdamW32bit, AdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
import argparse
from datetime import datetime # Добавляем импорт datetime

# --- Добавляем константы для цветов ---
COLOR_RESET = "\033[0m"
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
# --- Конец констант для цветов ---


class Logger:
    """
    Универсальный логгер с поддержкой wandb и tensorboard.
    Автоматически создает имя запуска/директории на основе имени скрипта и времени.
    """
    def __init__(self, script_name: str, use_wandb: bool = False, log_root_dir: str = "logs", wandb_project: str = "tiny_grpo", config: Optional[Dict] = None):
        self.use_wandb = use_wandb
        self.writer = None
        self.run_name = f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=self.run_name,
                    config=config if config else {},
                )
                print(f"{COLOR_BLUE}WandB run initialized: {self.run_name}{COLOR_RESET}")
            except Exception as e:
                print(f"{COLOR_RED}Failed to initialize WandB: {e}{COLOR_RESET}")
                self.use_wandb = False # Отключаем wandb если инициализация не удалась

        if not self.use_wandb:
            # Создаем директорию для TensorBoard
            tb_log_dir = Path(log_root_dir) / self.run_name
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_log_dir))
            print(f"{COLOR_BLUE}TensorBoard log directory: {tb_log_dir}{COLOR_RESET}")

    def log(self, metrics: Dict[str, Union[float, str, wandb.Table]], step: Optional[int] = None):
        """Логирует метрики в выбранный бэкенд."""
        if self.use_wandb and wandb.run:
            try:
                 # wandb.log ожидает числовые значения или wandb.* объекты
                 wandb_metrics = {k: v for k, v in metrics.items() if not isinstance(v, str)}
                 # Отдельно логируем таблицы или другие объекты wandb
                 wandb_objects = {k: v for k, v in metrics.items() if isinstance(v, wandb.Table)}

                 if wandb_metrics:
                     wandb.log(wandb_metrics, step=step)
                 if wandb_objects:
                      wandb.log(wandb_objects, step=step) # Логируем объекты отдельно

            except Exception as e:
                 print(f"{COLOR_YELLOW}WandB log warning: {e}{COLOR_RESET}")
        elif self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, global_step=step)
                # Tensorboard не поддерживает таблицы или строки напрямую в add_scalar

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
         """Логирует текстовые данные."""
         if self.use_wandb and wandb.run:
             # WandB может логировать текст через wandb.log с произвольным ключом
             # или можно использовать wandb.Html для форматирования
             try:
                 wandb.log({tag: text}, step=step)
             except Exception as e:
                 print(f"{COLOR_YELLOW}WandB log_text warning: {e}{COLOR_RESET}")
         elif self.writer:
             # TensorBoard логирует текст через add_text
             self.writer.add_text(tag, text, global_step=step)

    def log_table(self, key: str, columns: list, data: list, step: Optional[int] = None):
         """Логирует таблицу (в основном для WandB)."""
         if self.use_wandb and wandb.run:
             try:
                 table = wandb.Table(columns=columns, data=data)
                 self.log({key: table}, step=step)
             except Exception as e:
                 print(f"{COLOR_YELLOW}WandB log_table warning: {e}{COLOR_RESET}")
         else:
             # TensorBoard не имеет прямого аналога таблиц, можно логировать как текст
             table_str = "| " + " | ".join(columns) + " |\n"
             table_str += "|-" + "-|".join(['-' * len(c) for c in columns]) + "-|\n"
             for row in data:
                 table_str += "| " + " | ".join(map(str, row)) + " |\n"
             self.log_text(f"{key}_table", f"<pre>{table_str}</pre>", step=step)


    def close(self):
        """Закрывает логгер."""
        if self.use_wandb and wandb.run:
            wandb.finish()
            print(f"{COLOR_BLUE}WandB run finished.{COLOR_RESET}")
        if self.writer:
            self.writer.close()
            print(f"{COLOR_BLUE}TensorBoard writer closed.{COLOR_RESET}")


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
        # Добавим простую очистку, но основная логика форматирования должна быть на LLM
        expression = expression.strip()
        # Убедимся, что строка не пустая после strip
        if not expression:
            return "Calc error: Empty expression"
        result = eval(expression, {'__builtins__': {}}, {}) # Ограничиваем eval
        return str(result)
    except Exception as e:
        # Возвращаем более информативную ошибку
        return f"Calc error: Cannot evaluate '{expression}'. Details: {e}"

# --- Изменяем detect_and_call_tools ---
def detect_and_call_tools(generated_text: str) -> Optional[Tuple[str, str, str]]:
    """
    Находит *первый* вызов инструмента, выполняет его и возвращает кортеж:
    (tool_name, tool_input, tool_result_str).
    Возвращает None, если инструмент не найден или не вызывался.
    """
    pattern = r"<tool:(\w+)>(.*?)</tool>"
    match = re.search(pattern, generated_text, flags=re.DOTALL)

    if match:
        tool_name = match.group(1)
        tool_input = match.group(2).strip()
        tool_func = TOOLS.get(tool_name)
        tool_result_str: Optional[str] = None

        if tool_func:
            try:
                tool_result_str = tool_func(tool_input)
            except Exception as e:
                tool_result_str = f"Error executing tool '{tool_name}': {e}"
        else:
            tool_result_str = f"[Tool '{tool_name}' not found]"

        # Возвращаем имя, ввод и результат
        if tool_result_str is not None:
            return tool_name, tool_input, tool_result_str
        else:
             # Случай, когда tool_func вернул None, хотя не должен
             return tool_name, tool_input, "[Error: Tool function returned None]"
    else:
        return None # Инструмент не вызывался


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

    print("\nПроверка типов параметров модели:")
    total_size = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            param_size = param.numel() * param.element_size() / 1024**2
            total_size += param_size
            # print(f"{name}: {param.dtype}, размер: {param_size:.2f} MB") # Убрал для краткости логов

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

system_prompt = """You are a helpful assistant that reason, call tools and provide answer. The user provde tasks and you solve it as it described."""
# Первый системный промпт - только для рассуждения и вызова инструмента
first_step_prompt = """- Think about the reasoning process and explain it within <reasoning>...</reasoning> tags
- Call the calculation tool using: <tool:calc>user asked question for calculation</tool>

Here is the format example:

Calculate 2 + 2

<reasoning>I need to add these numbers together</reasoning>
<tool:calc>2 + 2</tool>

Your task:
"""

# Второй системный промпт - только для ответа
second_step_prompt = """A conversation between User and Assistant. Now you need to copy answer from tool to answer tag.

- Your response MUST contain only the answer tag
- After receiving the tool result, provide the final answer within <answer>...</answer> tags

Format Example:

Tool result: 4
<answer>4</answer>

Here is Tool output:
"""

@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    logger: Logger, # Принимаем объект Logger
    global_step: int,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    all_sequences = []
    all_completions_text = []
    all_rewards_dicts = []

    # Метрики, агрегированные по группе роллаутов (для одной задачи)
    group_stats = {
        "total_reward_sum": 0.0,
        "tool_called_count": 0,
        "tool_executed_ok_count": 0,
        "answer_format_ok_count": 0,
        "answer_correct_count": 0,
    }

    for rollout_idx in range(num_rollouts):
        rewards = {
            "step1_tool_call_format": 0.0,
            "step1_tool_execution": 0.0,
            "step2_answer_format": 0.0,
            "step2_answer_content": 0.0,
        }
        rollout_stats = { # Статистика для одного этого роллаута
             "step1_completion": "", "tool_called": False, "tool_input": None,
             "tool_result": None, "step2_completion": "", "final_answer": None,
             "is_correct_answer": False, "error_type": None
        }

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_step_prompt + task},
        ]

        current_messages = chat_messages.copy()
        full_dialog_text_for_log = "" # Текст для логирования примеров
        steps_count = 0
        max_steps = 2
        rollout_tokens = []
        actual_tool_result: Optional[str] = None
        step1_failed = False

        initial_prompt_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        full_dialog_text_for_log += f"**Prompt:**\n```\n{initial_prompt_text}\n```\n"
        prompt_tokens = tokenizer(
            initial_prompt_text, return_tensors="pt", padding=False
        ).input_ids.to("cuda")
        rollout_tokens.append(prompt_tokens[0])

        # --- Шаг 1 ---
        steps_count += 1
        chat_prompt_text_step1 = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs_step1 = tokenizer(
            chat_prompt_text_step1, return_tensors="pt", padding=False
        ).to("cuda")

        generation_config = GenerationConfig(
            do_sample=True, top_p=top_p, temperature=temperature,
            max_new_tokens=128, pad_token_id=tokenizer.eos_token_id,
        )
        sequence_ids_step1 = model.generate(**model_inputs_step1, generation_config=generation_config)
        new_tokens_step1 = sequence_ids_step1[0, model_inputs_step1["input_ids"].shape[1]:]
        rollout_tokens.append(new_tokens_step1)

        completion_step1 = tokenizer.decode(new_tokens_step1, skip_special_tokens=True)
        rollout_stats["step1_completion"] = completion_step1
        full_dialog_text_for_log += f"**Step 1 Completion:**\n```\n{completion_step1}\n```\n"
        current_messages.append({"role": "assistant", "content": completion_step1})

        # Вызов и проверка инструмента
        tool_call_info = detect_and_call_tools(completion_step1)
        if tool_call_info:
            tool_name, tool_input, actual_tool_result = tool_call_info
            rewards["step1_tool_call_format"] += 0.2
            rollout_stats["tool_called"] = True
            group_stats["tool_called_count"] += 1
            rollout_stats["tool_input"] = tool_input
            rollout_stats["tool_result"] = actual_tool_result
            full_dialog_text_for_log += f"**Tool Call:** `{tool_name}({tool_input})` -> `{actual_tool_result}`\n"

            if "error" in actual_tool_result.lower():
                rewards["step1_tool_execution"] -= 1.0
                step1_failed = True
                rollout_stats["error_type"] = "Tool Execution Error"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_RED}Tool Error:{COLOR_RESET} {actual_tool_result}")
            else:
                rewards["step1_tool_execution"] += 0.5
                group_stats["tool_executed_ok_count"] += 1
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_GREEN}Tool OK:{COLOR_RESET} {tool_input} -> {actual_tool_result}")
        else:
            rewards["step1_tool_call_format"] -= 0.5
            step1_failed = True
            rollout_stats["error_type"] = "Tool Format Error"
            full_dialog_text_for_log += "**Tool Call:** Failed (Format Error)\n"
            print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_RED}Tool Call Format Error{COLOR_RESET}")

        # --- Шаг 2 ---
        if not step1_failed and actual_tool_result is not None:
            steps_count += 1
            user_message_step2 = f"{second_step_prompt}\n\nTool result: {actual_tool_result}"
            current_messages.append({"role": "user", "content": user_message_step2})
            full_dialog_text_for_log += f"**Prompt Step 2 (User):**\n```\nTool result: {actual_tool_result}\n```\n"

            chat_prompt_text_step2 = tokenizer.apply_chat_template(
                current_messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs_step2 = tokenizer(
                chat_prompt_text_step2, return_tensors="pt", padding=False
            ).to("cuda")

            sequence_ids_step2 = model.generate(**model_inputs_step2, generation_config=generation_config)
            new_tokens_step2 = sequence_ids_step2[0, model_inputs_step2["input_ids"].shape[1]:]
            rollout_tokens.append(new_tokens_step2)

            completion_step2 = tokenizer.decode(new_tokens_step2, skip_special_tokens=True)
            rollout_stats["step2_completion"] = completion_step2
            full_dialog_text_for_log += f"**Step 2 Completion:**\n```\n{completion_step2}\n```\n"
            current_messages.append({"role": "assistant", "content": completion_step2})

            answer_match = re.match(r"^\s*<answer>(.*?)</answer>\s*$", completion_step2, flags=re.DOTALL)
            if answer_match:
                rewards["step2_answer_format"] += 0.3
                group_stats["answer_format_ok_count"] += 1
                final_answer = answer_match.group(1).strip()
                rollout_stats["final_answer"] = final_answer
                full_dialog_text_for_log += f"**Final Answer:** `{final_answer}`\n"

                # Сравниваем с oracle_answer вместо actual_tool_result
                if final_answer == oracle_answer:
                    rewards["step2_answer_content"] += 1.0
                    rollout_stats["is_correct_answer"] = True
                    group_stats["answer_correct_count"] += 1
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_GREEN}Answer OK:{COLOR_RESET} {final_answer} (matches oracle: {oracle_answer})")
                else:
                    rewards["step2_answer_content"] -= 0.5
                    rollout_stats["error_type"] = "Answer Content Mismatch"
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_YELLOW}Answer Content Mismatch:{COLOR_RESET} Got '{final_answer}', Expected '{oracle_answer}' (Tool result was: {actual_tool_result})")
            else:
                rewards["step2_answer_format"] -= 0.8
                rollout_stats["error_type"] = "Answer Format Error"
                full_dialog_text_for_log += "**Final Answer:** Failed (Format Error)\n"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_RED}Answer Format Error:{COLOR_RESET} {completion_step2[:50]}...") # Показываем начало ошибки
        else:
             full_dialog_text_for_log += "**Step 2:** Skipped\n"
             print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_YELLOW}Skipped{COLOR_RESET}")

        total_reward = sum(rewards.values())
        group_stats["total_reward_sum"] += total_reward

        # Логируем детальные награды для *каждого* роллаута (может быть шумно, но полезно для отладки)
        logger.log({
            f"rollout_rewards/total": total_reward,
            f"rollout_rewards/step1_format": rewards["step1_tool_call_format"],
            f"rollout_rewards/step1_exec": rewards["step1_tool_execution"],
            f"rollout_rewards/step2_format": rewards["step2_answer_format"],
            f"rollout_rewards/step2_content": rewards["step2_answer_content"],
        }, step=global_step)

        if rollout_tokens:
            full_sequence = torch.cat(rollout_tokens)
            all_sequences.append(full_sequence)
        else:
            all_sequences.append(torch.tensor([], dtype=torch.long, device="cuda"))

        all_completions_text.append(full_dialog_text_for_log) # Сохраняем текст с разметкой
        all_rewards_dicts.append(rewards)

    # --- Расчет и логирование агрегированных метрик для группы ---
    avg_group_reward = group_stats["total_reward_sum"] / num_rollouts if num_rollouts > 0 else 0.0
    tool_called_rate = group_stats["tool_called_count"] / num_rollouts if num_rollouts > 0 else 0.0
    tool_exec_ok_rate = group_stats["tool_executed_ok_count"] / group_stats["tool_called_count"] if group_stats["tool_called_count"] > 0 else 0.0
    answer_format_ok_rate = group_stats["answer_format_ok_count"] / num_rollouts if num_rollouts > 0 else 0.0 # Или от числа успешных шагов 1? Пока от всех
    answer_correct_rate = group_stats["answer_correct_count"] / group_stats["answer_format_ok_count"] if group_stats["answer_format_ok_count"] > 0 else 0.0

    logger.log({
        "group_avg/reward": avg_group_reward,
        "group_rates/tool_called": tool_called_rate,
        "group_rates/tool_exec_ok": tool_exec_ok_rate,
        "group_rates/answer_format_ok": answer_format_ok_rate,
        "group_rates/answer_correct": answer_correct_rate,
    }, step=global_step)

    # --- Конец изменений в rollout ---

    # Паддинг и создание маски (остается как было в предыдущей версии)
    if not all_sequences:
        print(f"{COLOR_YELLOW}WARNING: No valid sequences generated in this group.{COLOR_RESET}")
        return torch.empty(0, 0, device="cuda"), \
               torch.empty(0, 1, device="cuda"), \
               torch.empty(0, 0, dtype=torch.bool, device="cuda"), \
               []

    non_empty_sequences = [seq for seq in all_sequences if seq.numel() > 0]
    if not non_empty_sequences:
        print(f"{COLOR_YELLOW}WARNING: All sequences in the group are empty.{COLOR_RESET}")
        return torch.empty(0, 0, device="cuda"), \
               torch.empty(0, 1, device="cuda"), \
               torch.empty(0, 0, dtype=torch.bool, device="cuda"), \
               []

    max_seq_length = max(seq.size(0) for seq in non_empty_sequences)

    padded_sequences = []
    original_lengths = []
    for seq in all_sequences:
        seq_len = seq.size(0)
        original_lengths.append(seq_len)
        padding_length = max_seq_length - seq_len
        if padding_length >= 0:
            padded_seq = torch.cat([seq, torch.full((padding_length,), tokenizer.pad_token_id, device=seq.device)])
        else:
            padded_seq = seq[:max_seq_length]
        padded_sequences.append(padded_seq)

    sequence_ids = torch.stack(padded_sequences)

    action_mask = torch.zeros_like(sequence_ids[:, 1:], dtype=torch.bool)

    len_prompt = rollout_tokens[0].size(0) if rollout_tokens else 0 # Длина первого промпта
    len_comp1 = rollout_tokens[1].size(0) if len(rollout_tokens) > 1 else 0 # Длина ответа 1
    len_comp2 = rollout_tokens[2].size(0) if len(rollout_tokens) > 2 else 0 # Длина ответа 2

    for i, total_len in enumerate(original_lengths):
        start1 = len_prompt
        end1 = start1 + len_comp1
        mask_start1 = max(0, start1 - 1)
        mask_end1 = max(0, end1 - 1)
        # Исправляем условие, чтобы не выходить за пределы маски
        if mask_end1 > mask_start1 and mask_start1 < action_mask.shape[1]:
             actual_end1 = min(mask_end1, action_mask.shape[1]) # Убедимся, что не выходим за границу
             action_mask[i, mask_start1 : actual_end1] = True

        start2 = end1
        end2 = start2 + len_comp2
        mask_start2 = max(0, start2 - 1)
        mask_end2 = max(0, end2 - 1)
        # Исправляем условие
        if mask_end2 > mask_start2 and mask_start2 < action_mask.shape[1]:
             actual_end2 = min(mask_end2, action_mask.shape[1])
             action_mask[i, mask_start2 : actual_end2] = True

        valid_len_mask = total_len - 1
        if valid_len_mask < action_mask.shape[1]:
             action_mask[i, valid_len_mask:] = False
        # Дополнительно обрежем маску по максимальной длине (уже не нужно из-за min выше)
        # action_mask[i, max_seq_length-1:] = False # Можно убрать

    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, rew_dict in enumerate(all_rewards_dicts):
        returns[i] = sum(rew_dict.values())

    # Возвращаем текст completions для возможного логирования примеров
    return sequence_ids, returns.to(sequence_ids.device), action_mask, all_completions_text


def init_rng(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # Дополнительно для GPU, если используется
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Для multi-GPU
        # Не всегда нужно для воспроизводимости, может замедлить
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Нормализует преимущества в группе."""
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    """Вычисляет лог-вероятности для последовательности токенов."""
    log_prob = F.log_softmax(logits, dim=-1)
    # Используем gather для выбора лог-вероятностей нужных токенов
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Получает лог-вероятности для всех токенов в батче последовательностей."""
    # Создаем position_ids для корректной работы attention
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1) # Заменяем 0 на 1 там где маска 0?
    # position_ids = position_ids.clamp(min=0) # Альтернатива: обрезаем отрицательные

    # Пропускаем через модель
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False, # Не используем кэш при обучении
    )
    logits = output["logits"]

    # Вычисляем лог-вероятности для всех токенов кроме последнего
    # так как мы предсказываем следующий токен
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32), # Берем логиты для всех кроме последнего токена
        output_ids=sequence_ids[:, 1:], # Сравниваем с реальными токенами начиная со второго
    )
    return log_probs


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    count = 0
    try:
        # Читаем весь JSON файл, а не построчно
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Загружено {len(data)} вопросов из {file_name}")
        
        for x in data:
            if predicate is None or predicate(x):
                rows.append(x)
                count += 1
                if max_rows is not None and count >= max_rows:
                    break
    except FileNotFoundError as e:
        print(f"{COLOR_RED}Error reading prompts: {e}{COLOR_RESET}")
        return [] # Возвращаем пустой список при ошибке
    except json.JSONDecodeError as e:
        print(f"{COLOR_RED}Error parsing JSON: {e}{COLOR_RESET}")
        return []
    return rows


def parse_args():
    # Определяем имя скрипта для логгера
    script_name = Path(__file__).stem

    parser = argparse.ArgumentParser(description='Train a model with GRPO for multi-turn tool calling.')
    parser.add_argument('--run-name', type=str, default=None, help='Custom run name for logging (overrides auto-generated)')
    parser.add_argument('--log-dir', type=str, default="runs", help='Root directory for logs')
    parser.add_argument('--wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--wandb-project', type=str, default="tiny_grpo_v2", help='WandB project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device-index', type=int, default=0, help='CUDA device index')
    parser.add_argument('--model-name', type=str, default="Qwen/Qwen2.5-3B-Instruct", help='Model name or path')
    parser.add_argument('--checkpoint-path', type=str, default="./output", help='Path to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=20, help='Save checkpoint every N batches')
    parser.add_argument('--data-path', type=str, default="data/math_tasks.jsonl", help='Path to training data')
    parser.add_argument('--max-prompts', type=int, default=1024, help='Maximum number of prompts to load (reduced for faster testing)')
    parser.add_argument('--group-size', type=int, default=4, help='Number of rollouts per task (num_rollouts)')
    parser.add_argument('--rollouts-per-step', type=int, default=4, help='Number of tasks per batch (batch size for rollout phase)')
    parser.add_argument('--train-batch-size', type=int, default=4, help='Batch size for training phase (experience buffer)')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length for generation/padding')
    parser.add_argument('--log-completions-interval', type=int, default=10, help='Log example completions every N batches')

    args = parser.parse_args()
    # Добавляем имя скрипта в аргументы для передачи логгеру
    args.script_name = script_name
    return args

def main():
    args = parse_args()

    seed = args.seed
    device_index = args.device_index
    model_name = args.model_name
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_interval = args.checkpoint_interval
    train_batch_size = args.train_batch_size
    lr = args.lr
    group_size = args.group_size
    rollouts_per_step = args.rollouts_per_step
    max_length = args.max_length
    temperature = args.temperature
    data_path = args.data_path
    max_prompts = args.max_prompts
    log_completions_interval = args.log_completions_interval

    use_4bit = True
    use_lora = True
    bf16 = True

    epochs_per_step = 1 # Можно сделать параметром
    max_norm = 1.0 # Можно сделать параметром
    top_p = 1.0 # Можно сделать параметром
    kl_weight=0.01 # Можно сделать параметром
    clip_eps=0.2 # Можно сделать параметром

    # Собираем конфиг для логгера
    run_config = vars(args) # Преобразуем Namespace в dict

    # --- Инициализация логгера ---
    logger = Logger(
        script_name=args.script_name if args.run_name is None else Path(args.run_name).stem, # Используем имя скрипта или кастомное
        use_wandb=args.wandb,
        log_root_dir=args.log_dir,
        wandb_project=args.wandb_project,
        config=run_config # Передаем конфиг запуска
    )
    # Если указано кастомное имя, используем его
    if args.run_name:
        logger.run_name = args.run_name


    # Создаем директорию для чекпоинтов, если ее нет
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"Using device: {device}")
    init_rng(seed)

    print("Загрузка референсной модели...")
    reference_model, _ = load_model(
        model_name, device_map="auto", use_4bit=use_4bit, use_lora=False, bf16=bf16
    )
    print("Загрузка основной модели...")
    model, tokenizer = load_model(
        model_name, device_map="auto", use_4bit=use_4bit, use_lora=use_lora, bf16=bf16
    )

    optimizer = AdamW32bit(model.parameters(), lr=lr, is_paged=True)

    reference_model.eval()
    # Проверяем, есть ли градиент чекпоинтинг, прежде чем включать
    if hasattr(model, "gradient_checkpointing_enable"):
         model.gradient_checkpointing_enable(
             gradient_checkpointing_kwargs={"use_reentrant": False}
         )
         print("Gradient checkpointing enabled.")
    else:
         print(f"{COLOR_YELLOW}Warning: Model does not support gradient_checkpointing_enable directly.{COLOR_RESET}")

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        data_path,
        predicate=lambda x: len(x.get("question", "")) < 128,
        max_rows=max_prompts,
    )
    print(f"Найдено {len(prompts)} подходящих промптов")
    if not prompts:
         print(f"{COLOR_RED}Ошибка: Не найдено подходящих промптов в {data_path}. Обучение прервано.{COLOR_RESET}")
         logger.close()
         return

    prompt_loader = DataLoader(
        prompts, batch_size=rollouts_per_step, shuffle=True, drop_last=True, pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    global_step = 0 # Счетчик шагов оптимизации
    batch_step = 0 # Счетчик обработанных батчей промптов
    best_batch_mean_return = float('-inf')

    print(f"{COLOR_BLUE}Начало цикла обучения...{COLOR_RESET}")
    for k, prompt_batch in enumerate(prompt_loader):
        batch_step = k + 1
        print(f"\n--- Batch {batch_step} ---")
        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"] # Oracle answers

        batch_total_rewards = []
        batch_completions_for_log = [] # Собираем примеры для логирования

        with torch.no_grad():
            for q_idx, (q, a) in enumerate(zip(questions, answers)):
                print(f"  Задача {q_idx+1}/{len(questions)}: {q[:80]}... (Ожидаемый: {a})") # Обрезаем длинные задачи
                sequence_ids, returns, action_mask, completions = rollout(
                    model, tokenizer, q, a, num_rollouts=group_size,
                    logger=logger, global_step=global_step, max_length=max_length,
                    temperature=temperature, top_p=top_p,
                )

                if sequence_ids.numel() == 0:
                     print(f"  {COLOR_YELLOW}Пропуск задачи {q_idx+1} из-за пустого результата rollout.{COLOR_RESET}")
                     continue

                batch_total_rewards.extend([r.item() for r in returns])
                # Сохраняем первый completion из группы для возможного логирования
                if completions:
                    batch_completions_for_log.append(completions[0])

                if returns.numel() > 0:
                    advantages = group_advantages(returns)
                    attention_mask = sequence_ids != pad_token_id

                    # Вычисляем лог-вероятности до переноса на CPU
                    log_probs = sequences_log_probs(
                        model=model, sequence_ids=sequence_ids, attention_mask=attention_mask,
                    )
                    log_probs_ref = sequences_log_probs(
                        model=reference_model, sequence_ids=sequence_ids, attention_mask=attention_mask,
                    )
                    kl = approx_kl_divergence(
                        log_probs=log_probs, log_probs_ref=log_probs_ref, action_mask=action_mask,
                    )

                    experience = Experience(
                        sequences=sequence_ids, action_log_probs=log_probs, log_probs_ref=log_probs_ref,
                        returns=returns, advantages=advantages, attention_mask=attention_mask,
                        action_mask=action_mask, kl=kl,
                    )
                    replay_buffer.append(experience.to(cpu_device)) # Переносим на CPU для экономии VRAM
                else:
                     print(f"  {COLOR_YELLOW}Пропуск добавления в буфер для задачи {q_idx+1} из-за пустого returns.{COLOR_RESET}")

            # Логирование агрегированных метрик по батчу
            current_batch_mean_return = 0.0
            if batch_total_rewards:
                current_batch_mean_return = sum(batch_total_rewards) / len(batch_total_rewards)
                current_batch_max_return = max(batch_total_rewards)
                best_batch_mean_return = max(best_batch_mean_return, current_batch_mean_return)

                logger.log({
                    "batch_returns/mean": current_batch_mean_return,
                    "batch_returns/max": current_batch_max_return,
                    "batch_returns/best_mean": best_batch_mean_return,
                }, step=global_step) # Используем global_step
                print(f"  Batch {batch_step} Mean Return: {current_batch_mean_return:.4f}")
            else:
                 print(f"  {COLOR_YELLOW}Предупреждение: Батч {batch_step} не содержит валидных роллаутов.{COLOR_RESET}")

            # Логирование примеров completions
            if log_completions_interval > 0 and batch_step % log_completions_interval == 0 and batch_completions_for_log:
                 example_text = f"## Example Completion (Batch {batch_step}, Return: {batch_total_rewards[0]:.2f})\n\n" + batch_completions_for_log[0]
                 logger.log_text("examples/completion", example_text, step=global_step)
                 print(f"  {COLOR_BLUE}Logged example completion.{COLOR_RESET}")


        if len(replay_buffer) == 0:
             print(f"  {COLOR_YELLOW}Предупреждение: Replay buffer пуст после батча {batch_step}. Пропуск шага оптимизации.{COLOR_RESET}")
             continue

        experience_sampler = DataLoader(
            replay_buffer, batch_size=train_batch_size, shuffle=True,
            drop_last=True, collate_fn=join_experience_batch,
        )

        print(f"  Запуск {epochs_per_step} эпох оптимизации (global_step: {global_step})...")
        epoch_metrics = {"loss": [], "kl": [], "grad_norm": []} # Собираем метрики за эпоху
        for step_epoch in range(epochs_per_step):
            model.train()
            for i, exp in enumerate(experience_sampler):
                exp: Experience
                exp = exp.to(device) # Переносим батч на GPU

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"    {COLOR_YELLOW}Warning: Invalid loss ({loss.item()}) at global step {global_step}, skipping step.{COLOR_RESET}")
                    logger.log({
                        "training/loss": float('inf'), "training/kl": float('inf'),
                    }, step=global_step)
                    continue

                loss.backward()
                # clip_grad_norm_ может возвращать тензор или число в зависимости от версии torch
                grad_norm_tensor = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                grad_norm = grad_norm_tensor.item() if torch.is_tensor(grad_norm_tensor) else grad_norm_tensor

                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["kl"].append(kl.item())
                epoch_metrics["grad_norm"].append(grad_norm)

                # Логируем метрики на каждом шаге оптимизатора
                logger.log({
                    "training_step/loss": loss.item(),
                    "training_step/kl": kl.item(),
                    "training_step/grad_norm": grad_norm,
                }, step=global_step)

                optimizer.step()
                global_step += 1 # Увеличиваем global_step только после успешного шага

            # Логируем средние значения по эпохе оптимизации
            avg_epoch_metrics = {f"training_epoch/{key}": (sum(val) / len(val) if val else 0.0)
                                 for key, val in epoch_metrics.items()}
            if avg_epoch_metrics:
                 logger.log(avg_epoch_metrics, step=global_step)
                 print(f"    Epoch {step_epoch+1} Avg Loss: {avg_epoch_metrics['training_epoch/loss']:.4f}, Avg KL: {avg_epoch_metrics['training_epoch/kl']:.4f}")


        # Сохранение чекпоинта
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and batch_step % checkpoint_interval == 0 # Сохраняем каждые N батчей
        ):
            save_path = checkpoint_path / f"step_{global_step}"
            print(f"  {COLOR_BLUE}Сохранение чекпоинта в {save_path}...{COLOR_RESET}")
            try:
                 model.save_pretrained(save_path)
                 tokenizer.save_pretrained(save_path)
                 logger.log({
                     "checkpoint/step": global_step, "checkpoint/batch": batch_step,
                 }, step=global_step) # Логируем и global_step и batch_step
            except Exception as e:
                 print(f"{COLOR_RED}Ошибка при сохранении чекпоинта: {e}{COLOR_RESET}")


    # Финальное сохранение
    if checkpoint_path is not None:
        final_save_path = checkpoint_path / f"final_step_{global_step}"
        print(f"{COLOR_BLUE}Финальное сохранение модели в {final_save_path}...{COLOR_RESET}")
        try:
             model.save_pretrained(final_save_path)
             tokenizer.save_pretrained(final_save_path)
             logger.log({
                 "checkpoint/final_step": global_step, "checkpoint/final_batch": batch_step,
                 "training/completed": True,
             }, step=global_step)
        except Exception as e:
             print(f"{COLOR_RED}Ошибка при финальном сохранении: {e}{COLOR_RESET}")


    logger.close()
    print(f"{COLOR_GREEN}Обучение завершено.{COLOR_RESET}")


if __name__ == "__main__":
    main() # Запускаем main, которая теперь использует parse_args() внутри
