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
    wandb_project = None
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

    wandb.init(mode="disabled" if wandb_project is None else wandb_project)

    for k, prompt_batch in enumerate(prompt_loader):
        replay_buffer.clear()
        rollout_returns = []

        with torch.no_grad():
            for q, a in zip(prompt_batch["question"], prompt_batch["answer"]):
                sequence_ids, returns, action_mask, completions = rollout(
                    model, tokenizer, q, a, group_size, max_length, temperature, top_p
                )

                rollout_returns.append(returns.cpu())
                advantages = group_advantages(returns)
                attention_mask = sequence_ids != tokenizer.eos_token_id

                log_probs = sequences_log_probs(model, sequence_ids, attention_mask)
                log_probs_ref = sequences_log_probs(reference_model, sequence_ids, attention_mask)

                kl = approx_kl_divergence(log_probs, log_probs_ref, action_mask)

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

        episode_return_sum = torch.stack(rollout_returns).sum()
        wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer, batch_size=train_batch_size, shuffle=True, collate_fn=join_experience_batch
        )

        model.train()
        for _ in range(epochs_per_step):
            for exp in experience_sampler:
                optimizer.zero_grad()
                exp = exp.to(accelerator.device)
                log_probs = sequences_log_probs(model, exp.sequences, exp.attention_mask)
                loss, kl = objective(log_probs=log_probs, experience=exp)
                accelerator.backward(loss)
                clip_grad_norm_(model.parameters(), max_norm=max_norm)
                optimizer.step()

        if checkpoint_path and checkpoint_interval and (k + 1) % checkpoint_interval == 0:
            accelerator.unwrap_model(model).save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path:
        accelerator.unwrap_model(model).save_pretrained(checkpoint_path / f"final")


if __name__ == "__main__":
    main()
