import os
# так можно выбирать устройство для запуска LLM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
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
    BitsAndBytesConfig,
)
from bitsandbytes.optim import AdamW32bit, AdamW8bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loss import approx_kl_divergence, GRPOLoss
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

def detect_and_call_tools(generated_text: str) -> str:
    """
    Находит и выполняет все вызовы инструментов в тексте.
    """
    pattern = r"<tool:(\w+)>(.*?)</tool>"

    def _replace_tool_call(match: re.Match) -> str:
        tool_name = match.group(1)
        tool_input = match.group(2)
        tool = TOOLS.get(tool_name)
        if tool is None:
            return f"[Tool '{tool_name}' not found]"
        result = tool(tool_input.strip())
        return f"<tool_result>{result}</tool_result>"

    updated_text = re.sub(pattern, _replace_tool_call, generated_text, flags=re.DOTALL)
    return updated_text


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
        # trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        # torch_dtype=None if use_4bit else (torch.bfloat16 if bf16 else "auto"),
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


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it in two steps.

Step 1:
- Think about the reasoning process and explain it within <reasoning>...</reasoning> tags
- Call the calculation tool using: <tool:calc>expression</tool>
- The first response MUST end with a tool call

Step 2:
- After receiving the tool result, provide the final answer within <answer>...</answer> tags
- The second response MUST contain only the answer tag

Example:
User: Calculate 2 + 2
Assistant: <reasoning>I need to add these numbers together</reasoning>
<tool:calc>2 + 2</tool>
System: Tool result: 4
Assistant: <answer>4</answer>
"""

@torch.no_grad()
def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format initial prompt with full conversation structure
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    
    all_sequences = []
    all_completions = []
    
    for _ in range(num_rollouts):
        current_messages = chat_messages.copy()
        full_response = ""
        steps_count = 0
        max_steps = 2  # Ограничиваем до двух шагов
        
        # Токенизируем начальный промпт один раз
        initial_prompt = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tokenizer(
            initial_prompt,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).input_ids.to("cuda")
        
        # Сохраняем все токены для этого rollout
        rollout_tokens = [prompt_tokens[0]]  # Начинаем с промпта
        
        while steps_count < max_steps:
            steps_count += 1
            
            # Форматируем текущий диалог
            chat_prompt = tokenizer.apply_chat_template(
                current_messages, tokenize=False, add_generation_prompt=True
            )
            
            # Проверяем длину контекста
            if len(tokenizer.encode(chat_prompt)) > max_length // 2:
                break
            
            # Токенизируем и генерируем
            model_inputs = tokenizer(
                chat_prompt,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            ).to("cuda")
            
            generation_config = GenerationConfig(
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=128,  # Ограничиваем длину каждой генерации
                pad_token_id=tokenizer.eos_token_id,
            )
            
            sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
            
            # Получаем только новые токены (без промпта)
            new_tokens = sequence_ids[0, model_inputs["input_ids"].shape[1]:]
            rollout_tokens.append(new_tokens)
            
            # Декодируем для обработки
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Подробное логирование в wandb
            log_data = {
                f"rollout_{_}/step_{steps_count}/raw_completion": completion,
                f"rollout_{_}/metadata/task": task,
                f"rollout_{_}/metadata/expected_answer": oracle_answer,
            }

            # Проверяем формат ответа
            if steps_count == 1:
                # Первый шаг должен заканчиваться вызовом инструмента
                if not re.search(r"<tool:calc>.*?</tool>$", completion, flags=re.DOTALL):
                    print(f"WARNING: First step does not end with tool call: {completion}")
                    # Пропускаем этот rollout
                    break
            else:
                # Второй шаг должен содержать только ответ
                if not re.match(r"^\s*<answer>.*?</answer>\s*$", completion, flags=re.DOTALL):
                    print(f"WARNING: Second step should contain only answer tag: {completion}")
                    # Пропускаем этот rollout
                    break

            # Обрабатываем вызовы инструментов
            processed_completion = detect_and_call_tools(completion)

            # Анализируем структуру ответа для wandb
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", completion, flags=re.DOTALL)
            if reasoning_match:
                log_data[f"rollout_{_}/step_{steps_count}/reasoning"] = reasoning_match.group(1).strip()

            # Если был вызов инструмента
            if processed_completion != completion:
                tool_calls = list(re.finditer(r"<tool:(\w+)>(.*?)</tool>", completion, flags=re.DOTALL))
                tool_results = list(re.finditer(r"<tool_result>(.*?)</tool_result>", processed_completion))
                
                for idx, (call, result) in enumerate(zip(tool_calls, tool_results)):
                    tool_name = call.group(1)
                    tool_input = call.group(2).strip()
                    tool_result = result.group(1)
                    
                    log_data.update({
                        f"rollout_{_}/step_{steps_count}/tool_{idx}/name": tool_name,
                        f"rollout_{_}/step_{steps_count}/tool_{idx}/input": tool_input,
                        f"rollout_{_}/step_{steps_count}/tool_{idx}/result": tool_result,
                    })
                    
                    # Выводим на экран информацию о задаче и вызове инструмента
                    if steps_count == 1 and idx == 0:  # Только для первого шага и первого вызова
                        print(f"\nЗадача: {task}")
                        print(f"Ожидаемый ответ: {oracle_answer}")
                        print("-" * 40)
                    print(f"Rollout {_+1}/{num_rollouts} | Step {steps_count} | {tool_name}({tool_input}) = {tool_result}")

            answer_match = re.search(r"<answer>(.*?)</answer>", processed_completion, flags=re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
                log_data[f"rollout_{_}/step_{steps_count}/final_answer"] = answer
                print(f"Rollout {_+1}/{num_rollouts} | Answer: {answer}")
                # Добавляем информацию о правильности ответа
                is_correct = answer == oracle_answer
                is_partially_correct = oracle_answer in answer if not is_correct else False
                print(f"{'✓' if is_correct else ('≈' if is_partially_correct else '✗')} (expected: {oracle_answer})")
                print("-" * 40)

            log_data[f"rollout_{_}/step_{steps_count}/processed_completion"] = processed_completion
            wandb.log(log_data)

            full_response += processed_completion
            
            # Добавляем ответ ассистента в диалог
            current_messages.append({"role": "assistant", "content": processed_completion})
            
            # Если есть результат инструмента, добавляем его как сообщение от системы
            if "<tool_result>" in processed_completion:
                tool_result = re.search(r"<tool_result>(.*?)</tool_result>", processed_completion)
                if tool_result:
                    result_msg = f"Tool result: {tool_result.group(1)}"
                    current_messages.append({"role": "system", "content": result_msg})
                    
                    # Токенизируем результат инструмента
                    tool_result_tokens = tokenizer(
                        result_msg,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).input_ids[0].to("cuda")
                    rollout_tokens.append(tool_result_tokens)
        
        # Собираем все токены в одну последовательность
        full_sequence = torch.cat(rollout_tokens)
        all_sequences.append(full_sequence)
        all_completions.append(full_response)
    
    # Паддинг последовательностей до одинаковой длины
    max_seq_length = max(seq.size(0) for seq in all_sequences)
    padded_sequences = []
    for seq in all_sequences:
        padding_length = max_seq_length - seq.size(0)
        padded_seq = torch.cat([seq, torch.full((padding_length,), tokenizer.pad_token_id, device=seq.device)])
        padded_sequences.append(padded_seq)
    
    sequence_ids = torch.stack(padded_sequences)
    
    # Создаем маску действий
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    for i, seq in enumerate(all_sequences):
        # Маскируем только токены, сгенерированные моделью (не промпт и не результаты инструментов)
        prompt_length = len(rollout_tokens[0])
        for j in range(1, len(rollout_tokens), 2):  # Пропускаем результаты инструментов
            start_idx = sum(len(t) for t in rollout_tokens[:j])
            end_idx = start_idx + len(rollout_tokens[j])
            action_mask[i, start_idx:end_idx] = True
    action_mask[sequence_ids == tokenizer.pad_token_id] = False
    action_mask = action_mask[:, 1:]  # сдвигаем на 1, так как нам нужны следующие токены
    
    # Определяем награды
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(all_completions):
        # Награда за использование инструмента
        if "<tool:calc>" in completion and "<tool_result>" in completion:
            returns[i] += 0.2
            
        # Награда за правильный ответ
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            if answer == oracle_answer:
                returns[i] += 1.0
            elif oracle_answer in answer:
                returns[i] += 0.5
            else:
                returns[i] += 0.01

    return sequence_ids, returns.to(sequence_ids.device), action_mask, all_completions


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


def main():
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

    # Добавляем параметры для квантизации и LoRA
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

    reference_model, _ = load_model(
        model_name, 
        device_map=device,
        use_4bit=use_4bit,
        use_lora=False,
        bf16=bf16
    )
    model, tokenizer = load_model(
        model_name, 
        device_map=device,
        use_4bit=use_4bit,
        use_lora=use_lora,
        bf16=bf16
    )
    
    # Обновляем оптимизатор для работы только с обучаемыми параметрами
    optimizer = AdamW32bit(model.parameters(), lr=lr, is_paged=True)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
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

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                print(
                    f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

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
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

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

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

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
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl, "grad_norm": grad_norm})

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
