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


class Logger:
    """Универсальный логгер с поддержкой wandb и tensorboard."""
    
    def __init__(self, use_wandb: bool = False, log_dir: str = "runs"):
        self.use_wandb = use_wandb
        if not use_wandb:
            self.writer = SummaryWriter(log_dir)
        
    def log(self, metrics: Dict[str, Union[float, str]], step: Optional[int] = None):
        """Логирует метрики в выбранный бэкенд."""
        if self.use_wandb:
            wandb.log(metrics, step=step)
        else:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
    
    def close(self):
        """Закрывает логгер."""
        if not self.use_wandb:
            self.writer.close()


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

def detect_and_call_tools(generated_text: str) -> Tuple[str, Optional[str]]:
    """
    Находит вызов инструмента, выполняет его и возвращает
    обработанный текст И строку с результатом инструмента.
    Если инструментов несколько, обрабатывает только первый найденный.
    """
    pattern = r"<tool:(\w+)>(.*?)</tool>"
    match = re.search(pattern, generated_text, flags=re.DOTALL)
    tool_result_str: Optional[str] = None
    processed_text = generated_text # По умолчанию текст не меняется

    if match:
        tool_name = match.group(1)
        tool_input = match.group(2).strip()
        tool_func = TOOLS.get(tool_name)

        if tool_func:
            try:
                tool_result_str = tool_func(tool_input)
            except Exception as e:
                tool_result_str = f"Error executing tool '{tool_name}': {e}"
            # Заменяем вызов на результат ТОЛЬКО для processed_text, не для истории
            processed_text = generated_text[:match.start()] + \
                             f"<tool_result>{tool_result_str}</tool_result>" + \
                             generated_text[match.end():]
        else:
            tool_result_str = f"[Tool '{tool_name}' not found]"
            processed_text = generated_text[:match.start()] + \
                             f"<tool_result>{tool_result_str}</tool_result>" + \
                             generated_text[match.end():]

    # Возвращаем обработанный текст (для возможной отладки) и сам результат
    return processed_text, tool_result_str


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
    logger: Logger,
    global_step: int,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    all_sequences = []
    all_completions_text = []
    all_rewards_dicts = []

    for rollout_idx in range(num_rollouts):
        rewards = {
            "tool_call_format": 0.0,
            "tool_execution": 0.0,
            "answer_format": 0.0,
            "answer_content": 0.0,
        }
        rollout_stats = {
             "step1_completion": "",
             "tool_called": False,
             "tool_input": None,
             "tool_result": None,
             "step2_completion": "",
             "final_answer": None,
             "is_correct_answer": False,
        }

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_step_prompt + task},
        ]

        current_messages = chat_messages.copy()
        full_dialog_text = ""
        steps_count = 0
        max_steps = 2
        rollout_tokens = []
        actual_tool_result: Optional[str] = None

        initial_prompt_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        full_dialog_text += initial_prompt_text
        prompt_tokens = tokenizer(
            initial_prompt_text, return_tensors="pt", padding=False
        ).input_ids.to("cuda")

        rollout_tokens.append(prompt_tokens[0])

        steps_count += 1
        step1_failed = False

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
        full_dialog_text += completion_step1

        current_messages.append({"role": "assistant", "content": completion_step1})

        tool_call_match = re.search(r"<tool:calc>(.*?)</tool>\s*$", completion_step1, flags=re.DOTALL)
        if tool_call_match:
            rewards["tool_call_format"] += 0.2
            rollout_stats["tool_called"] = True
            rollout_stats["tool_input"] = tool_call_match.group(1).strip()

            _, actual_tool_result = detect_and_call_tools(completion_step1)
            rollout_stats["tool_result"] = actual_tool_result

            if actual_tool_result is not None:
                if "error" in actual_tool_result.lower():
                    rewards["tool_execution"] -= 1.0
                    step1_failed = True
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | Tool Error: {actual_tool_result}")
                else:
                    rewards["tool_execution"] += 0.5
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | Tool OK: {rollout_stats['tool_input']} -> {actual_tool_result}")
            else:
                rewards["tool_execution"] -= 1.0
                step1_failed = True
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | Tool Call Match but no result?")
        else:
            rewards["tool_call_format"] -= 0.5
            step1_failed = True
            print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | Tool Call Format Error")

        if not step1_failed and actual_tool_result is not None:
            steps_count += 1

            user_message_step2 = f"{second_step_prompt}\n\nTool result: {actual_tool_result}"
            current_messages.append({"role": "user", "content": user_message_step2})
            full_dialog_text += f"\n[User message with tool result]\n"

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
            full_dialog_text += completion_step2

            current_messages.append({"role": "assistant", "content": completion_step2})

            answer_match = re.match(r"^\s*<answer>(.*?)</answer>\s*$", completion_step2, flags=re.DOTALL)
            if answer_match:
                rewards["answer_format"] += 0.3
                final_answer = answer_match.group(1).strip()
                rollout_stats["final_answer"] = final_answer

                if actual_tool_result is not None and final_answer == actual_tool_result:
                    rewards["answer_content"] += 1.0
                    rollout_stats["is_correct_answer"] = True
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | Answer OK: {final_answer}")
                else:
                    rewards["answer_content"] -= 0.5
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | Answer Content Mismatch: Got '{final_answer}', Expected '{actual_tool_result}'")
            else:
                rewards["answer_format"] -= 0.8
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | Answer Format Error: {completion_step2}")
        else:
            print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | Skipped due to Step 1 failure or no tool result")

        total_reward = sum(rewards.values())
        logger.log({
            f"rollout/total_reward": total_reward,
            f"rollout/reward_tool_call_format": rewards["tool_call_format"],
            f"rollout/reward_tool_execution": rewards["tool_execution"],
            f"rollout/reward_answer_format": rewards["answer_format"],
            f"rollout/reward_answer_content": rewards["answer_content"],
            f"rollout/stats_tool_called": float(rollout_stats["tool_called"]),
            f"rollout/stats_tool_executed_ok": float(rewards["tool_execution"] > 0),
            f"rollout/stats_answer_format_ok": float(rewards["answer_format"] > 0),
            f"rollout/stats_answer_correct": float(rollout_stats["is_correct_answer"]),
        }, step=global_step)

        if rollout_tokens:
            full_sequence = torch.cat(rollout_tokens)
            all_sequences.append(full_sequence)
        else:
            all_sequences.append(torch.tensor([], dtype=torch.long, device="cuda"))

        all_completions_text.append(full_dialog_text)
        all_rewards_dicts.append(rewards)

    if not all_sequences:
        print("WARNING: No valid sequences generated in this batch.")
        return torch.empty(0, 0, device="cuda"), \
               torch.empty(0, 1, device="cuda"), \
               torch.empty(0, 0, dtype=torch.bool, device="cuda"), \
               []

    non_empty_sequences = [seq for seq in all_sequences if seq.numel() > 0]
    if not non_empty_sequences:
        print("WARNING: All sequences in the batch are empty.")
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

    prompt_len = prompt_tokens.size(1)

    len_prompt = rollout_tokens[0].size(0)
    len_comp1 = rollout_tokens[1].size(0) if len(rollout_tokens) > 1 else 0
    len_comp2 = rollout_tokens[2].size(0) if len(rollout_tokens) > 2 else 0

    for i, total_len in enumerate(original_lengths):
        start1 = len_prompt
        end1 = start1 + len_comp1
        mask_start1 = max(0, start1 - 1)
        mask_end1 = max(0, end1 - 1)
        if mask_end1 > mask_start1:
            action_mask[i, mask_start1 : mask_end1] = True

        start2 = end1
        end2 = start2 + len_comp2
        mask_start2 = max(0, start2 - 1)
        mask_end2 = max(0, end2 - 1)
        if mask_end2 > mask_start2:
            action_mask[i, mask_start2 : mask_end2] = True

        valid_len_mask = total_len - 1
        if valid_len_mask < action_mask.shape[1]:
            action_mask[i, valid_len_mask:] = False

    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, rew_dict in enumerate(all_rewards_dicts):
        returns[i] = sum(rew_dict.values())

    return sequence_ids, returns.to(sequence_ids.device), action_mask, all_completions_text


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: AutoModelForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with GRPO')
    parser.add_argument('--wandb', action='store_true', help='Use WandB for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device-index', type=int, default=0, help='CUDA device index')
    parser.add_argument('--model-name', type=str, default="unsloth/Llama-3.2-1B-Instruct",
                      help='Model name or path')
    parser.add_argument('--checkpoint-path', type=str, default="./output",
                      help='Path to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=20,
                      help='Save checkpoint every N steps')
    parser.add_argument('--data-path', type=str, default="data/math_tasks.jsonl", help='Path to training data')
    parser.add_argument('--max-prompts', type=int, default=64 * 1024, help='Maximum number of prompts to load')
    parser.add_argument('--group-size', type=int, default=8, help='Number of rollouts per task (num_rollouts)')
    parser.add_argument('--rollouts-per-step', type=int, default=8, help='Number of tasks per batch (batch size for rollout phase)')
    parser.add_argument('--train-batch-size', type=int, default=8, help='Batch size for training phase')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length for generation/padding (reduced)')
    return parser.parse_args()

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

    use_4bit = True
    use_lora = True
    bf16 = True

    epochs_per_step = 1
    max_norm = 1.0

    top_p = 1.0

    logger = Logger(use_wandb=args.wandb)

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    if args.wandb:
        wandb.init(
            project="tiny_grpo",
            config={
                "model_name": model_name, "lr": lr, "use_4bit": use_4bit, "use_lora": use_lora,
                "group_size": group_size, "temperature": temperature,
                "train_batch_size": train_batch_size, "rollouts_per_step": rollouts_per_step,
            }
        )

    print("Загрузка референсной модели...")
    reference_model, _ = load_model(
        model_name, device_map=device, use_4bit=use_4bit, use_lora=False, bf16=bf16
    )
    print("Загрузка основной модели...")
    model, tokenizer = load_model(
        model_name, device_map=device, use_4bit=use_4bit, use_lora=use_lora, bf16=bf16
    )

    optimizer = AdamW32bit(model.parameters(), lr=lr, is_paged=True)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        data_path,
        predicate=lambda x: len(x["question"]) < 128 and x["num_terms"] <= 3 and x["num_digits"] <= 3,
        max_rows=max_prompts,
    )
    print(f"Найдено {len(prompts)} подходящих промптов")
    prompt_loader = DataLoader(
        prompts, batch_size=rollouts_per_step, shuffle=True, drop_last=True, pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=0.2, kl_weight=0.01)

    global_step = 0
    best_return = float('-inf')

    print("Начало цикла обучения...")
    for k, prompt_batch in enumerate(prompt_loader):
        print(f"\n--- Шаг {k+1} ---")
        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        batch_total_rewards = []

        with torch.no_grad():
            for q_idx, (q, a) in enumerate(zip(questions, answers)):
                print(f"  Задача {q_idx+1}/{len(questions)}: {q} (Ожидаемый: {a})")
                sequence_ids, returns, action_mask, completions = rollout(
                    model, tokenizer, q, a, num_rollouts=group_size,
                    logger=logger, global_step=global_step, max_length=max_length,
                    temperature=temperature, top_p=top_p,
                )

                if sequence_ids.numel() == 0:
                    print(f"  Пропуск задачи {q_idx+1} из-за пустого результата rollout.")
                    continue

                batch_total_rewards.extend([r.item() for r in returns])

                if returns.numel() > 0:
                    group_mean_return = returns.mean().item()
                    logger.log({
                        f"rollout_group/mean_return": group_mean_return,
                    }, step=global_step)

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

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
                replay_buffer.append(experience.to(cpu_device))

            if batch_total_rewards:
                mean_return = sum(batch_total_rewards) / len(batch_total_rewards)
                max_return = max(batch_total_rewards)
                best_return = max(best_return, max_return)

                logger.log({
                    "returns/batch_mean": mean_return,
                    "returns/batch_max": max_return,
                    "returns/best_ever": best_return,
                }, step=global_step)
            else:
                print("  Предупреждение: Батч не содержит валидных роллаутов.")

        if len(replay_buffer) == 0:
            print(f"  Предупреждение: Replay buffer пуст после шага {k+1}. Пропуск шага оптимизации.")
            continue

        experience_sampler = DataLoader(
            replay_buffer, batch_size=train_batch_size, shuffle=True,
            drop_last=True, collate_fn=join_experience_batch,
        )

        print(f"  Запуск {epochs_per_step} эпох оптимизации...")
        for step_epoch in range(epochs_per_step):
            model.train()
            epoch_losses = []
            epoch_kls = []

            for i, exp in enumerate(experience_sampler):
                exp: Experience
                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"    Предупреждение: Некорнечное значение loss ({loss.item()}) на шаге {global_step}, пропуск.")
                    logger.log({
                        "training/loss": float('inf'), "training/kl": float('inf'),
                    }, step=global_step)
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)

                epoch_losses.append(loss.item())
                epoch_kls.append(kl.item())

                logger.log({
                    "training/loss": loss.item(),
                    "training/kl": kl.item(),
                    "training/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                }, step=global_step)

                optimizer.step()
                global_step += 1

            if epoch_losses:
                logger.log({
                    f"training/epoch_{step_epoch}_mean_loss": sum(epoch_losses) / len(epoch_losses),
                    f"training/epoch_{step_epoch}_mean_kl": sum(epoch_kls) / len(epoch_kls),
                }, step=global_step)

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            save_path = checkpoint_path / f"step_{global_step}"
            print(f"  Сохранение чекпоинта в {save_path}...")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.log({
                "checkpoint/step": global_step, "checkpoint/path": str(save_path),
            }, step=global_step)

    if checkpoint_path is not None:
        save_path = checkpoint_path / f"final_step_{global_step}"
        print(f"Финальное сохранение модели в {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.log({
            "checkpoint/step": global_step, "checkpoint/path": str(save_path),
            "training/completed": True,
        }, step=global_step)

    logger.close()
    print("Обучение завершено.")


if __name__ == "__main__":
    main()
