import os
# так можно выбирать устройство для запуска LLM
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

# Dr. GRPO Modification: Определяем тип критика и максимальную длину генерации глобально или в main
# Эти значения будут использоваться для инициализации GRPOLoss и в group_advantages
CRITIC_TYPE = "drgrpo"  # Возможные значения: "grpo", "drgrpo"
# generate_max_length должен соответствовать максимальной длине генерации токенов
# В функции rollout используется max_length, давайте его и используем или определим здесь
GENERATE_MAX_LENGTH = 1024 

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
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
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
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
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
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


# Dr. GRPO Modification 2: Remove difficulty bias by just computing the MC advantage without dividing by std for drgrpo.
def group_advantages(returns: torch.Tensor, eps: float = 1e-8, critic_type: str = "grpo") -> torch.Tensor:
    """Normalizes advantages by mean and std if critic_type is 'grpo'. For 'drgrpo', only mean subtraction is performed."""
    advantages = returns - returns.mean()
    if critic_type == "grpo": # Стандартный GRPO нормализует по std
        advantages = advantages / (returns.std() + eps)
    # Для drgrpo нормализация по std не нужна, только вычитание среднего
    return advantages


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
    wandb_project = "drgrpo_experiment"  # Включаем wandb по умолчанию
    device_index = 0
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    # Добавляем параметры для квантизации и LoRA
    use_4bit = True
    use_lora = True
    bf16 = True

    group_size = 12
    rollouts_per_step = 32
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
    # Dr. GRPO Modification: Передаем CRITIC_TYPE и GENERATE_MAX_LENGTH
    grpo_loss_fn = GRPOLoss(
        clip_eps=clip_eps, 
        kl_weight=kl_weight,
        critic_type=CRITIC_TYPE, 
        generate_max_length=GENERATE_MAX_LENGTH 
    )

    # Расширенная конфигурация для wandb
    config = {
        "model_name": model_name,
        "learning_rate": lr,
        "batch_size": train_batch_size,
        "group_size": group_size,
        "rollouts_per_step": rollouts_per_step,
        "epochs_per_step": epochs_per_step,
        "kl_weight": kl_weight,
        "clip_eps": clip_eps,
        "temperature": temperature,
        "top_p": top_p,
        "max_length": max_length,
        "use_4bit": use_4bit,
        "use_lora": use_lora,
        "critic_type": CRITIC_TYPE,
        "generate_max_length": GENERATE_MAX_LENGTH,
        "algorithm": "DR-GRPO"
    }

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, config=config)
        # Добавляем отладочное логирование сразу после инициализации
        print(f"✓ WandB инициализирован для проекта: {wandb_project}")
        print(f"✓ Конфиг: {config}")
        # Логируем начальные метрики для подтверждения работы
        wandb.log({
            "debug/initialization": 1.0,
            "debug/algorithm": "DR-GRPO",
            "debug/critic_type": CRITIC_TYPE,
        }, step=0)
        print("✓ Отладочные метрики отправлены в WandB")

    global_step = 0
    best_mean_return = float('-inf')
    total_rollouts = 0

    for k, prompt_batch in enumerate(prompt_loader):
        print(f"\n🔥 DR-GRPO BATCH {k+1} - НАЧИНАЕМ ЛОГИРОВАНИЕ РАСШИРЕННЫХ МЕТРИК")
        rollout_returns = []
        batch_advantages = []
        batch_kl_divergences = []
        batch_dr_advantages = []  # Для отслеживания DR-GRPO специфичных метрик
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
                total_rollouts += len(returns)

                # Dr. GRPO Modification: Передаем CRITIC_TYPE в group_advantages
                advantages = group_advantages(
                    returns=returns,
                    eps=1e-8,
                    critic_type=CRITIC_TYPE 
                )
                batch_advantages.append(advantages.cpu())
                batch_dr_advantages.append(advantages.cpu())  # Дополнительное отслеживание для DR-GRPO

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
                batch_kl_divergences.append(kl.cpu())

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
        
        # Вычисляем агрегированные метрики для роллаутов
        all_returns = torch.cat(rollout_returns)  # Объединяем все returns
        episode_return_sum = all_returns.sum()
        episode_return_mean = all_returns.mean()
        episode_return_std = all_returns.std()
        episode_return_max = all_returns.max()
        episode_return_min = all_returns.min()
        
        # Обновляем лучший результат
        if episode_return_mean > best_mean_return:
            best_mean_return = episode_return_mean
        
        # KL дивергенция статистики
        if batch_kl_divergences:
            # Сплющиваем и объединяем тензоры разных размеров
            all_kl_values = torch.cat([kl.flatten() for kl in batch_kl_divergences])
            batch_kl_mean = all_kl_values.mean()
            batch_kl_std = all_kl_values.std()
            batch_kl_max = all_kl_values.max()
        else:
            batch_kl_mean = batch_kl_std = batch_kl_max = 0.0

        # Преимущества статистики
        if batch_advantages:
            # Сплющиваем и объединяем тензоры разных размеров
            all_advantages_values = torch.cat([adv.flatten() for adv in batch_advantages])
            advantages_mean = all_advantages_values.mean()
            advantages_std = all_advantages_values.std()
            advantages_max = all_advantages_values.max()
            advantages_min = all_advantages_values.min()
        else:
            advantages_mean = advantages_std = advantages_max = advantages_min = 0.0

        # DR-GRPO специфичные метрики
        if batch_dr_advantages:
            # Сплющиваем и объединяем тензоры для DR-GRPO метрик
            all_dr_advantages = torch.cat([adv.flatten() for adv in batch_dr_advantages])
            dr_advantages_variance = all_dr_advantages.var()
            dr_advantages_skewness = all_dr_advantages.std() / (all_dr_advantages.mean() + 1e-8)
        else:
            dr_advantages_variance = dr_advantages_skewness = 0.0

        print(f"returns of step {k}: {episode_return_sum:.4f}")
        
        # Расширенное логирование роллаутов с DR-GRPO спецификой
        print(f"📊 ЛОГИРУЕМ DR-GRPO МЕТРИКИ: advantages_variance={dr_advantages_variance:.4f}, critic_type={CRITIC_TYPE}")
        wandb.log({
            # Основные метрики возвратов
            "rollouts/returns_sum": episode_return_sum,
            "rollouts/returns_mean": episode_return_mean,
            "rollouts/returns_std": episode_return_std,
            "rollouts/returns_max": episode_return_max,
            "rollouts/returns_min": episode_return_min,
            "rollouts/returns_best_mean": best_mean_return,
            
            # KL дивергенция
            "rollouts/kl_mean": batch_kl_mean,
            "rollouts/kl_std": batch_kl_std,
            "rollouts/kl_max": batch_kl_max,
            
            # Преимущества
            "rollouts/advantages_mean": advantages_mean,
            "rollouts/advantages_std": advantages_std,
            "rollouts/advantages_max": advantages_max,
            "rollouts/advantages_min": advantages_min,
            
            # DR-GRPO специфичные метрики
            "dr_grpo/advantages_variance": dr_advantages_variance,
            "dr_grpo/advantages_skewness": dr_advantages_skewness,
            "dr_grpo/critic_type": CRITIC_TYPE,
            
            # Общие статистики
            "rollouts/total_rollouts": total_rollouts,
            "rollouts/batch_step": k + 1,
        }, step=global_step)

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        epoch_losses = []
        epoch_kls = []
        epoch_grad_norms = []
        epoch_dr_loss_components = []  # Для отслеживания компонентов DR-GRPO loss
        
        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = grpo_loss_fn(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={exp.advantages}")
                    # Логируем невалидные потери
                    wandb.log({
                        "training/invalid_loss_count": 1,
                        "training/loss": float('inf'),
                        "training/kl": float('inf'),
                    }, step=global_step)
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                
                # Собираем метрики эпохи
                epoch_losses.append(loss.item())
                epoch_kls.append(kl.item())
                epoch_grad_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                
                # Можем добавить специфичные DR-GRPO компоненты
                if hasattr(grpo_loss_fn, 'last_loss_components'):
                    epoch_dr_loss_components.append(grpo_loss_fn.last_loss_components)
                
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                
                # Логирование на каждом шаге оптимизации
                step_log = {
                    "training_step/loss": loss.item(),
                    "training_step/kl": kl.item(),
                    "training_step/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "training_step/learning_rate": optimizer.param_groups[0]['lr'],
                }
                
                # Добавляем DR-GRPO специфичные метрики если доступны
                if hasattr(grpo_loss_fn, 'last_loss_components') and grpo_loss_fn.last_loss_components:
                    step_log.update({f"training_step/dr_component_{k}": v 
                                   for k, v in grpo_loss_fn.last_loss_components.items()})
                
                wandb.log(step_log, step=global_step)

                optimizer.step()
                global_step += 1

        # Логирование средних значений по эпохе
        if epoch_losses:
            epoch_log = {
                "training_epoch/loss_mean": sum(epoch_losses) / len(epoch_losses),
                "training_epoch/loss_std": torch.tensor(epoch_losses).std().item(),
                "training_epoch/kl_mean": sum(epoch_kls) / len(epoch_kls),
                "training_epoch/kl_std": torch.tensor(epoch_kls).std().item(),
                "training_epoch/grad_norm_mean": sum(epoch_grad_norms) / len(epoch_grad_norms),
                "training_epoch/grad_norm_max": max(epoch_grad_norms),
                "training_epoch/steps_per_epoch": len(epoch_losses),
            }
            
            # Добавляем DR-GRPO epoch метрики если доступны
            if epoch_dr_loss_components and epoch_dr_loss_components[0]:
                for component_name in epoch_dr_loss_components[0].keys():
                    component_values = [comp[component_name] for comp in epoch_dr_loss_components if component_name in comp]
                    if component_values:
                        epoch_log[f"training_epoch/dr_{component_name}_mean"] = sum(component_values) / len(component_values)
            
            wandb.log(epoch_log, step=global_step)

        # Логирование информации о модели и памяти
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            wandb.log({
                "system/gpu_memory_allocated_gb": gpu_memory_allocated,
                "system/gpu_memory_reserved_gb": gpu_memory_reserved,
                "system/replay_buffer_size": len(replay_buffer),
            }, step=global_step)

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")
            wandb.log({
                "checkpoint/saved": 1,
                "checkpoint/step": k + 1
            }, step=global_step)

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")
        wandb.log({
            "checkpoint/final_saved": 1,
            "training/completed": True
        }, step=global_step)

    # Финальное логирование
    wandb.log({
        "final/total_steps": global_step,
        "final/total_batches": k + 1,
        "final/total_rollouts": total_rollouts,
        "final/best_mean_return": best_mean_return,
        "final/algorithm": "DR-GRPO"
    })


if __name__ == "__main__":
    main()
