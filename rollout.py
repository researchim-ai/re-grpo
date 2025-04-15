import torch
import re
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from search_module import search

def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    logger,
    global_step: int,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    all_sequences = []
    all_completions_text = []
    all_rewards_dicts = []

    # Метрики для группы роллаутов
    group_stats = {
        "total_reward_sum": 0.0,
        "search_called_count": 0,
        "search_executed_ok_count": 0,
        "answer_format_ok_count": 0,
        "answer_correct_count": 0,
    }

    for rollout_idx in range(num_rollouts):
        rewards = {
            "step1_search_call_format": 0.0,
            "step1_search_execution": 0.0,
            "step2_answer_format": 0.0,
            "step2_answer_content": 0.0,
        }
        rollout_stats = {
            "step1_completion": "", "search_called": False, "search_input": None,
            "search_result": None, "step2_completion": "", "final_answer": None,
            "is_correct_answer": False, "error_type": None
        }

        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_step_prompt + task},
        ]

        current_messages = chat_messages.copy()
        full_dialog_text_for_log = ""
        steps_count = 0
        max_steps = 2
        rollout_tokens = []
        actual_search_result: Optional[str] = None
        step1_failed = False

        # Шаг 1: Поиск информации
        initial_prompt_text = tokenizer.apply_chat_template(
            current_messages, tokenize=False, add_generation_prompt=True
        )
        full_dialog_text_for_log += f"**Prompt:**\n```\n{initial_prompt_text}\n```\n"
        prompt_tokens = tokenizer(
            initial_prompt_text, return_tensors="pt", padding=False
        ).input_ids.to("cuda")
        rollout_tokens.append(prompt_tokens[0])

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

        # Проверка вызова поиска
        search_pattern = r"<tool:search>(.*?)</tool>"
        search_match = re.search(search_pattern, completion_step1, flags=re.DOTALL)
        if search_match:
            search_query = search_match.group(1).strip()
            rewards["step1_search_call_format"] += 0.2
            rollout_stats["search_called"] = True
            group_stats["search_called_count"] += 1
            rollout_stats["search_input"] = search_query

            try:
                actual_search_result = search(search_query, return_type=str, results=2)
                rewards["step1_search_execution"] += 0.5
                group_stats["search_executed_ok_count"] += 1
                rollout_stats["search_result"] = actual_search_result
                full_dialog_text_for_log += f"**Search Results:**\n```\n{actual_search_result}\n```\n"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_GREEN}Search OK:{COLOR_RESET} {search_query}")
            except Exception as e:
                rewards["step1_search_execution"] -= 1.0
                step1_failed = True
                rollout_stats["error_type"] = "Search Execution Error"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_RED}Search Error:{COLOR_RESET} {e}")
        else:
            rewards["step1_search_call_format"] -= 0.5
            step1_failed = True
            rollout_stats["error_type"] = "Search Format Error"
            full_dialog_text_for_log += "**Search Call:** Failed (Format Error)\n"
            print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {COLOR_RED}Search Call Format Error{COLOR_RESET}")

        # Шаг 2: Формирование ответа
        if not step1_failed and actual_search_result is not None:
            steps_count += 1
            user_message_step2 = f"{second_step_prompt}\n\nSearch results: {actual_search_result}"
            current_messages.append({"role": "user", "content": user_message_step2})
            full_dialog_text_for_log += f"**Prompt Step 2 (User):**\n```\nSearch results: {actual_search_result}\n```\n"

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

                if final_answer == oracle_answer:
                    rewards["step2_answer_content"] += 1.0
                    rollout_stats["is_correct_answer"] = True
                    group_stats["answer_correct_count"] += 1
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_GREEN}Answer OK:{COLOR_RESET} {final_answer} (matches oracle: {oracle_answer})")
                else:
                    rewards["step2_answer_content"] -= 0.5
                    rollout_stats["error_type"] = "Answer Content Mismatch"
                    print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_YELLOW}Answer Content Mismatch:{COLOR_RESET} Got '{final_answer}', Expected '{oracle_answer}'")
            else:
                rewards["step2_answer_format"] -= 0.8
                rollout_stats["error_type"] = "Answer Format Error"
                full_dialog_text_for_log += "**Final Answer:** Failed (Format Error)\n"
                print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_RED}Answer Format Error:{COLOR_RESET} {completion_step2[:50]}...")
        else:
            full_dialog_text_for_log += "**Step 2:** Skipped\n"
            print(f"Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {COLOR_YELLOW}Skipped{COLOR_RESET}")

        total_reward = sum(rewards.values())
        group_stats["total_reward_sum"] += total_reward

        # Логирование метрик роллаута
        logger.log({
            f"rollout_rewards/total": total_reward,
            f"rollout_rewards/step1_format": rewards["step1_search_call_format"],
            f"rollout_rewards/step1_exec": rewards["step1_search_execution"],
            f"rollout_rewards/step2_format": rewards["step2_answer_format"],
            f"rollout_rewards/step2_content": rewards["step2_answer_content"],
        }, step=global_step)

        if rollout_tokens:
            full_sequence = torch.cat(rollout_tokens)
            all_sequences.append(full_sequence)
        else:
            all_sequences.append(torch.tensor([], dtype=torch.long, device="cuda"))

        all_completions_text.append(full_dialog_text_for_log)
        all_rewards_dicts.append(rewards)

    # Расчет и логирование агрегированных метрик
    avg_group_reward = group_stats["total_reward_sum"] / num_rollouts if num_rollouts > 0 else 0.0
    search_called_rate = group_stats["search_called_count"] / num_rollouts if num_rollouts > 0 else 0.0
    search_exec_ok_rate = group_stats["search_executed_ok_count"] / group_stats["search_called_count"] if group_stats["search_called_count"] > 0 else 0.0
    answer_format_ok_rate = group_stats["answer_format_ok_count"] / num_rollouts if num_rollouts > 0 else 0.0
    answer_correct_rate = group_stats["answer_correct_count"] / group_stats["answer_format_ok_count"] if group_stats["answer_format_ok_count"] > 0 else 0.0

    logger.log({
        "group_avg/reward": avg_group_reward,
        "group_rates/search_called": search_called_rate,
        "group_rates/search_exec_ok": search_exec_ok_rate,
        "group_rates/answer_format_ok": answer_format_ok_rate,
        "group_rates/answer_correct": answer_correct_rate,
    }, step=global_step)

    # Паддинг и создание маски
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

    len_prompt = rollout_tokens[0].size(0) if rollout_tokens else 0
    len_comp1 = rollout_tokens[1].size(0) if len(rollout_tokens) > 1 else 0
    len_comp2 = rollout_tokens[2].size(0) if len(rollout_tokens) > 2 else 0

    for i, total_len in enumerate(original_lengths):
        start1 = len_prompt
        end1 = start1 + len_comp1
        mask_start1 = max(0, start1 - 1)
        mask_end1 = max(0, end1 - 1)
        if mask_end1 > mask_start1 and mask_start1 < action_mask.shape[1]:
            actual_end1 = min(mask_end1, action_mask.shape[1])
            action_mask[i, mask_start1 : actual_end1] = True

        start2 = end1
        end2 = start2 + len_comp2
        mask_start2 = max(0, start2 - 1)
        mask_end2 = max(0, end2 - 1)
        if mask_end2 > mask_start2 and mask_start2 < action_mask.shape[1]:
            actual_end2 = min(mask_end2, action_mask.shape[1])
            action_mask[i, mask_start2 : actual_end2] = True

        valid_len_mask = total_len - 1
        if valid_len_mask < action_mask.shape[1]:
            action_mask[i, valid_len_mask:] = False

    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, rew_dict in enumerate(all_rewards_dicts):
        returns[i] = sum(rew_dict.values())

    return sequence_ids, returns.to(sequence_ids.device), action_mask, all_completions_text 