import os
from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from bitsandbytes.optim import AdamW32bit
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    device_map="auto",
    use_4bit=True,
    use_lora=True,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation="flash_attention_2",
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    if use_lora:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer


system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags."""


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)



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
    accelerator = Accelerator(cpu=True)
    seed = 42
    wandb_project = "accelerate_grpo"
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    init_rng(seed)

    reference_model, _ = load_model(model_name, use_lora=False)
    model, tokenizer = load_model(model_name, use_lora=True)

    optimizer = AdamW32bit(model.parameters(), lr=lr, is_paged=True)

    reference_model.eval()

    model, optimizer = accelerator.prepare(model, optimizer)

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128 and x["num_terms"] <= 3 and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    prompt_loader = DataLoader(prompts, batch_size=rollouts_per_step, shuffle=True, drop_last=True)

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

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
        "accelerator_device": str(accelerator.device),
        "mixed_precision": str(accelerator.mixed_precision),
        "algorithm": "GRPO-Accelerate"
    }

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, config=config)

    global_step = 0
    best_mean_return = float('-inf')
    total_rollouts = 0

    for k, prompt_batch in enumerate(prompt_loader):
        replay_buffer.clear()
        rollout_returns = []
        batch_advantages = []
        batch_kl_divergences = []

        with torch.no_grad():
            for q, a in zip(prompt_batch["question"], prompt_batch["answer"]):
                sequence_ids, returns, action_mask, completions = rollout(
                    model, tokenizer, q, a, group_size, max_length, temperature, top_p
                )

                rollout_returns.append(returns.cpu())
                total_rollouts += len(returns)
                
                advantages = group_advantages(returns)
                batch_advantages.append(advantages.cpu())
                attention_mask = sequence_ids != tokenizer.eos_token_id

                log_probs = sequences_log_probs(model, sequence_ids, attention_mask)
                log_probs_ref = sequences_log_probs(reference_model, sequence_ids, attention_mask)

                kl = approx_kl_divergence(log_probs, log_probs_ref, action_mask)
                batch_kl_divergences.append(kl.cpu())

                replay_buffer.append(
                    Experience(
                        sequences=sequence_ids,
                        action_log_probs=log_probs,
                        log_probs_ref=log_probs_ref,
                        returns=returns,
                        advantages=advantages,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        kl=kl,
                    ).to("cpu")
                )

        all_returns = torch.cat(rollout_returns)
        episode_return_sum = all_returns.sum()
        episode_return_mean = all_returns.mean()
        episode_return_std = all_returns.std()
        episode_return_max = all_returns.max()
        episode_return_min = all_returns.min()
        
        if episode_return_mean > best_mean_return:
            best_mean_return = episode_return_mean
        
        if batch_kl_divergences:
            all_kl_values = torch.cat([kl.flatten() for kl in batch_kl_divergences])
            batch_kl_mean = all_kl_values.mean()
            batch_kl_std = all_kl_values.std()
            batch_kl_max = all_kl_values.max()
        else:
            batch_kl_mean = batch_kl_std = batch_kl_max = 0.0

        if batch_advantages:
            all_advantages_values = torch.cat([adv.flatten() for adv in batch_advantages])
            advantages_mean = all_advantages_values.mean()
            advantages_std = all_advantages_values.std()
        else:
            advantages_mean = advantages_std = 0.0

        wandb.log({
            "rollouts/returns_sum": episode_return_sum,
            "rollouts/returns_mean": episode_return_mean,
            "rollouts/returns_std": episode_return_std,
            "rollouts/returns_max": episode_return_max,
            "rollouts/returns_min": episode_return_min,
            "rollouts/returns_best_mean": best_mean_return,
            
            "rollouts/kl_mean": batch_kl_mean,
            "rollouts/kl_std": batch_kl_std,
            "rollouts/kl_max": batch_kl_max,
            
            "rollouts/advantages_mean": advantages_mean,
            "rollouts/advantages_std": advantages_std,
            
            "rollouts/total_rollouts": total_rollouts,
            "rollouts/batch_step": k + 1,
            "accelerate/device": str(accelerator.device),
        }, step=global_step)

        experience_sampler = DataLoader(
            replay_buffer, batch_size=train_batch_size, shuffle=True, collate_fn=join_experience_batch
        )

        epoch_losses = []
        epoch_kls = []
        epoch_grad_norms = []

        model.train()
        for exp in experience_sampler:
            exp = exp.to(accelerator.device)

            with accelerator.accumulate(model):
                optimizer.zero_grad()

                log_probs = sequences_log_probs(model, exp.sequences, exp.attention_mask)
                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    wandb.log({
                        "training/invalid_loss_count": 1,
                        "training/loss": float('inf'),
                        "training/kl": float('inf'),
                    }, step=global_step)
                    continue

                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm)
                else:
                    grad_norm = 0.0
                
                epoch_losses.append(loss.item())
                epoch_kls.append(kl.item())
                epoch_grad_norms.append(grad_norm if isinstance(grad_norm, float) else grad_norm.item())
                
                wandb.log({
                    "training_step/loss": loss.item(),
                    "training_step/kl": kl.item(),
                    "training_step/grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                    "training_step/learning_rate": optimizer.param_groups[0]['lr'],
                    "accelerate/sync_gradients": accelerator.sync_gradients,
                }, step=global_step)

                optimizer.step()
                global_step += 1

        if epoch_losses:
            wandb.log({
                "training_epoch/loss_mean": sum(epoch_losses) / len(epoch_losses),
                "training_epoch/loss_std": torch.tensor(epoch_losses).std().item(),
                "training_epoch/kl_mean": sum(epoch_kls) / len(epoch_kls),
                "training_epoch/kl_std": torch.tensor(epoch_kls).std().item(),
                "training_epoch/grad_norm_mean": sum(epoch_grad_norms) / len(epoch_grad_norms),
                "training_epoch/grad_norm_max": max(epoch_grad_norms),
                "training_epoch/steps_per_epoch": len(epoch_losses),
            }, step=global_step)

        if (checkpoint_path is not None and checkpoint_interval is not None and (k + 1) % checkpoint_interval == 0):
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_path / f"step_{k}")
            wandb.log({
                "checkpoint/saved": 1,
                "checkpoint/step": k + 1
            }, step=global_step)

    if checkpoint_path is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_path / f"step_{k}")
        wandb.log({
            "checkpoint/final_saved": 1,
            "training/completed": True
        }, step=global_step)

    wandb.log({
        "final/total_steps": global_step,
        "final/total_batches": k + 1,
        "final/total_rollouts": total_rollouts,
        "final/best_mean_return": best_mean_return,
        "final/algorithm": "GRPO-Accelerate"
    })


if __name__ == "__main__":
    main()
