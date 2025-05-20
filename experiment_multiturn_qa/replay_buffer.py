"""
Реализация буфера воспроизведения (Replay Buffer) и связанных структур данных.

Буфер воспроизведения используется для хранения и выборки "опыта" (траекторий),
полученного в результате взаимодействия агента (модели) со средой.
Это стандартный компонент во многих алгоритмах обучения с подкреплением.
"""
from dataclasses import dataclass, fields
from typing import Optional, Self

import torch
import torch.nn.functional as F


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    """
    Дополняет (pads) список тензоров (последовательностей) нулями до максимальной длины в списке.

    Args:
        sequences (list[torch.Tensor]): Список одномерных тензоров (последовательностей).
        side (str): Сторона для дополнения нулями, "left" или "right". По умолчанию "left".

    Returns:
        torch.Tensor: Двумерный тензор, где каждая строка - это дополненная нулями последовательность.

    Raises:
        AssertionError: Если `side` не "left" или "right".
    """
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


@dataclass
class Experience:
    """
    Датакласс для хранения одного элемента "опыта" (траектории) агента.

    Содержит информацию, необходимую для обучения, такую как:
    - сгенерированные последовательности токенов,
    - логарифмы вероятностей действий текущей и референсной политик,
    - полученные награды (returns),
    - вычисленные преимущества (advantages),
    - маски внимания и маски действий.

    Атрибуты:
        sequences (torch.Tensor): Тензор с ID токенов последовательности (промпт + генерация).
        action_log_probs (torch.Tensor): Логарифмы вероятностей действий, предпринятых текущей политикой.
        log_probs_ref (torch.Tensor): Логарифмы вероятностей тех же действий, но согласно референсной (старой) политике.
        returns (Optional[torch.Tensor]): Оцененные награды (returns) для последовательности/шагов.
        advantages (Optional[torch.Tensor]): Оцененные преимущества для последовательности/шагов.
        attention_mask (Optional[torch.Tensor]): Маска внимания для `sequences`.
        action_mask (torch.Tensor): Маска, указывающая, какие токены в `sequences` являются действиями (т.е. были сгенерированы, а не являются частью промпта).
        kl (Optional[torch.Tensor]): KL-дивергенция между текущей и референсной политиками (может быть вычислена позже).
    """
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    kl: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> Self:
        """
        Перемещает все тензорные поля экземпляра `Experience` на указанное устройство.

        Args:
            device (torch.device): Целевое устройство (например, `torch.device("cuda")` или `torch.device("cpu")`).

        Returns:
            Experience: Новый экземпляр `Experience` с тензорами на указанном устройстве.
        """
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


def split_experience_batch(experience: Experience) -> list[Experience]:
    """
    Разделяет один батчированный экземпляр `Experience` на список отдельных экземпляров `Experience`.

    Предполагается, что входной `experience` содержит тензоры, где первое измерение - это размер батча.

    Args:
        experience (Experience): Батчированный экземпляр `Experience`.

    Returns:
        list[Experience]: Список отдельных (небатчированных) экземпляров `Experience`.
    """
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: list[Experience]) -> Experience:
    """
    Объединяет список отдельных экземпляров `Experience` в один батчированный экземпляр `Experience`.

    Тензоры из списка `items` будут объединены вдоль нового первого измерения (батча).
    Для тензоров разной длины (например, `sequences`) будет применено дополнение нулями
    с помощью `zero_pad_sequences` (слева).

    Args:
        items (list[Experience]): Список отдельных экземпляров `Experience`.

    Returns:
        Experience: Один батчированный экземпляр `Experience`.
    """
    batch_data = {}
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = zero_pad_sequences(vals, "left")
        else:
            data = None
        batch_data[key] = data
    return Experience(**batch_data)


class ReplayBuffer:
    """
    Простой буфер воспроизведения (Replay Buffer).

    Хранит коллекцию экземпляров `Experience`. Позволяет добавлять новые "опыты",
    очищать буфер и получать доступ к элементам. Может иметь опциональный лимит
    на количество хранимых элементов (старые удаляются при превышении лимита).

    Атрибуты:
        limit (int): Максимальное количество элементов `Experience`, хранимых в буфере.
                     Если 0, лимит отсутствует.
        items (list[Experience]): Список, хранящий экземпляры `Experience`.
    """
    def __init__(self, limit: int = 0) -> None:
        """
        Инициализирует ReplayBuffer.

        Args:
            limit (int): Максимальное количество элементов `Experience` для хранения.
                         Если 0 (по умолчанию), буфер не имеет ограничения по размеру.
        """
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        """
        Добавляет новый "опыт" (или батч "опытов") в буфер.

        Если `experience` является батчированным, он сначала разделяется на отдельные
        экземпляры с помощью `split_experience_batch`.
        Если установлен лимит `self.limit` и он превышен после добавления, самые старые
        элементы удаляются из начала буфера.

        Args:
            experience (Experience): Экземпляр `Experience` для добавления (может быть батчированным).
        """
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        """Очищает все элементы из буфера воспроизведения."""
        self.items.clear()

    def __len__(self) -> int:
        """Возвращает текущее количество элементов в буфере."""
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        """Позволяет получить доступ к элементу буфера по индексу."""
        return self.items[idx]
