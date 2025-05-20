"""
Модуль для выполнения "прогонов" (rollouts) модели в задаче QA с поиском.

Содержит функцию `rollout`, которая моделирует двухэтапный процесс:
1. Модель генерирует поисковый запрос на основе вопроса пользователя.
2. Модель генерирует ответ на основе результатов поиска.

Примечание: Этот файл содержит функцию `rollout`, которая очень похожа на
одноименную функцию в `train_multiturn_qa.py`. Однако, эта версия напрямую
использует `search_module.search` вместо общей системы инструментов, как в
`train_multiturn_qa.py`. Возможно, это более ранняя или специализированная версия.
Необходимы также определения `system_prompt`, `first_step_prompt`, `second_step_prompt`
и цветовых констант, которые здесь не представлены, но предполагаются из контекста
(например, из `train_multiturn_qa.py`).
"""
import torch
import re
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, GenerationConfig
from search_module import search

# Предполагается, что эти переменные определены где-то глобально или передаются иначе.
# Для примера, можно раскомментировать и задать их.
# system_prompt = "Your system prompt here"
# first_step_prompt = "Your first step prompt here"
# second_step_prompt = "Your second step prompt here"
# COLOR_GREEN = "\033[92m"
# COLOR_RED = "\033[91m"
# COLOR_YELLOW = "\033[93m"
# COLOR_RESET = "\033[0m"

def rollout(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str, # Вопрос пользователя
    oracle_answer: str, # Эталонный ответ
    num_rollouts: int, # Количество прогонов для задачи
    logger: object, # Объект логгера (например, из train_multiturn_qa.Logger)
    global_step: int, # Глобальный шаг для логирования
    # Добавляем промпты и цвета как аргументы, чтобы избежать зависимости от глобальных переменных
    system_prompt: str,
    first_step_prompt: str,
    second_step_prompt: str,
    color_green: str,
    color_red: str,
    color_yellow: str,
    color_reset: str,
    max_length: int = 1024, # Макс. длина генерации
    temperature: float = 0.7, # Температура для семплирования
    top_p: float = 1.0, # Top-p для семплирования
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Выполняет "прогоны" (rollouts) модели для задачи QA с использованием поиска.

    Процесс для каждого прогона:
    1.  **Шаг 1 (Поиск):**
        - Модель получает промпт для генерации вызова поискового инструмента
          в формате `<tool:search>ЗАПРОС</tool>`.
        - Если вызов корректен, выполняется поиск с помощью `search_module.search`.
        - Фиксируются награды за формат вызова и успешность выполнения поиска.
    2.  **Шаг 2 (Ответ):**
        - Если поиск успешен, модель получает результаты поиска и промпт для генерации
          ответа в формате `<answer>ОТВЕТ</answer>`.
        - Фиксируются награды за формат ответа и его корректность по сравнению с `oracle_answer`.

    Собирает и логирует детальную статистику по каждому прогону и по группе прогонов.
    Возвращает сгенерированные последовательности, награды и маски для последующего обучения.

    Args:
        model: Языковая модель для генерации.
        tokenizer: Токенизатор для модели.
        task: Входной вопрос пользователя.
        oracle_answer: Эталонный (правильный) ответ.
        num_rollouts: Количество прогонов для выполнения.
        logger: Экземпляр логгера для записи метрик.
        global_step: Текущий глобальный шаг обучения (для логирования).
        system_prompt (str): Системный промпт для модели.
        first_step_prompt (str): Промпт для первого шага (генерация поиска).
        second_step_prompt (str): Промпт для второго шага (генерация ответа).
        color_green (str): Код цвета для успешных сообщений в консоли.
        color_red (str): Код цвета для сообщений об ошибках в консоли.
        color_yellow (str): Код цвета для предупреждающих сообщений в консоли.
        color_reset (str): Код для сброса цвета в консоли.
        max_length: Максимальная длина генерируемых последовательностей.
        temperature: Температура для семплирования при генерации.
        top_p: Параметр top-p (nucleus sampling) для генерации.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
            - `sequence_ids` (torch.Tensor): Паддированные тензоры всех сгенерированных токенов (включая промпты).
            - `returns` (torch.Tensor): Тензор суммарных наград для каждого прогона.
            - `action_mask` (torch.Tensor): Маска, указывающая на сгенерированные моделью токены (действия).
            - `all_completions_text` (list[str]): Список полных текстовых диалогов (промпты + генерация + результаты поиска) для каждого прогона.
    """

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
                print(f"{color_green}Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {color_reset}Search OK:{color_reset} {search_query}")
            except Exception as e:
                rewards["step1_search_execution"] -= 1.0
                step1_failed = True
                rollout_stats["error_type"] = "Search Execution Error"
                print(f"{color_red}Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {color_reset}Search Error:{color_reset} {e}")
        else:
            rewards["step1_search_call_format"] -= 0.5
            step1_failed = True
            rollout_stats["error_type"] = "Search Format Error"
            full_dialog_text_for_log += "**Search Call:** Failed (Format Error)\n"
            print(f"{color_red}Rollout {rollout_idx+1}/{num_rollouts} | Step 1 | {color_reset}Search Call Format Error{color_reset}")

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
                    print(f"{color_green}Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {color_reset}Answer OK:{color_reset} {final_answer} (matches oracle: {oracle_answer})")
                else:
                    rewards["step2_answer_content"] -= 0.5
                    rollout_stats["error_type"] = "Answer Content Mismatch"
                    print(f"{color_yellow}Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {color_reset}Answer Content Mismatch:{color_reset} Got '{final_answer}', Expected '{oracle_answer}'")
            else:
                rewards["step2_answer_format"] -= 0.8
                rollout_stats["error_type"] = "Answer Format Error"
                full_dialog_text_for_log += "**Final Answer:** Failed (Format Error)\n"
                print(f"{color_red}Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {color_reset}Answer Format Error:{color_reset} {completion_step2[:50]}...")
        else:
            full_dialog_text_for_log += "**Step 2:** Skipped\n"
            print(f"{color_yellow}Rollout {rollout_idx+1}/{num_rollouts} | Step 2 | {color_reset}Skipped{color_reset}")

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
        print(f"{color_yellow}WARNING: No valid sequences generated in this group.{color_reset}")
        return torch.empty(0, 0, device="cuda"), \
               torch.empty(0, 1, device="cuda"), \
               torch.empty(0, 0, dtype=torch.bool, device="cuda"), \
               []

    non_empty_sequences = [seq for seq in all_sequences if seq.numel() > 0]
    if not non_empty_sequences:
        print(f"{color_yellow}WARNING: All sequences in the group are empty.{color_reset}")
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