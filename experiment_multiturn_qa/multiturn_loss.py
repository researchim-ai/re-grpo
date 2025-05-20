"""
Определение функции потерь GRPO для многоэтапного взаимодействия (multiturn).

Этот модуль предоставляет реализацию функции потерь GRPO (Generalized Reinforcement
Learning from Human Preferences and Outcomes), адаптированную или предназначенную
для сценариев с многоэтапным взаимодействием.

Примечание: Этот файл очень похож на `qa_loss.py`. Содержит аналогичные функции
`approx_kl_divergence`, `masked_mean` и класс `GRPOLoss`.
Следует рассмотреть возможность их объединения или устранения избыточности,
если их назначение и реализация по сути совпадают.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Вычисляет Монте-Карло аппроксимацию KL-дивергенции (оценщик k3).
    Формула: E[exp(log_probs_ref - log_probs) - (log_probs_ref - log_probs) - 1].

    Предполагается, что `log_probs`, `log_probs_ref`, и `action_mask` имеют форму [B, seq_len-1].

    Args:
        log_probs (torch.Tensor): Логарифмы вероятностей от текущей (новой) политики.
        log_probs_ref (torch.Tensor): Логарифмы вероятностей от референсной (старой) политики.
        action_mask (Optional[torch.Tensor]): Маска, указывающая, какие элементы (действия)
                                             следует учитывать при вычислении KL.

    Returns:
        torch.Tensor: Тензор со значениями аппроксимированной KL-дивергенции для каждого элемента.

    Примечание: В отличие от версии в `qa_loss.py`, здесь отсутствует явный клиппинг `log_ratio` внутри.
    """
    # log_ratio = log_probs_ref - log_probs
    log_ratio = log_probs_ref.float() - log_probs.float()

    # Умножаем на маску, чтобы учитывать только часть токенов.
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # k3-оценка KL: exp(ratio) - ratio - 1
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Вычисляет среднее значение тензора `tensor` с учетом бинарной маски `mask`.
    Элементы, для которых соответствующее значение в `mask` равно 0 (или False),
    не учитываются при вычислении среднего.

    Args:
        tensor (torch.Tensor): Входной тензор.
        mask (Optional[torch.Tensor]): Булев или числовой (0/1) тензор маски.
                                     Должен быть совместим по форме с `tensor` для поэлементного умножения.
        dim (Optional[int]): Измерение или измерения, вдоль которых вычисляется среднее.
                           Если None, вычисляется среднее по всем элементам.

    Returns:
        torch.Tensor: Тензор со средними значениями (скаляр, если `dim`=None и `tensor` не батчированный).
    """
    if mask is None:
        return tensor.mean(dim=dim)
    # Складываем только по тем позициям, где mask=1 (или True),
    # и делим на общее число таких элементов (добавляем epsilon для избежания деления на 0).
    return (tensor * mask.type_as(tensor)).sum(dim=dim) / (mask.type_as(tensor).sum(dim=dim) + 1e-10)


class GRPOLoss(nn.Module):
    """
    Реализация функции потерь GRPO (аналог PPO) для обучения модели-актора.

    Эта версия предназначена или адаптирована для многоэтапных сценариев.
    Логика расчета потерь идентична PPO: комбинация "clipped surrogate objective"
    и штрафа за KL-дивергенцию по отношению к референсной политике.

    Атрибуты:
        clip_eps (float): Коэффициент клиппинга для отношения вероятностей (например, 0.2).
                          Ограничивает изменение политики на одном шаге обновления.
        kl_weight (float): Весовой коэффициент для члена KL-дивергенции в общей функции потерь.
                           Регулирует степень "удержания" новой политики около референсной.

    Примечание: Класс очень похож на `GRPOLoss` в `qa_loss.py`.
    """

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        """
        Инициализирует GRPOLoss.

        Args:
            clip_eps (float): Коэффициент клиппинга (например, 0.2).
            kl_weight (float): Вес для KL-дивергенции (например, 0.01).
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет потери GRPO на основе предсказаний модели и собранного опыта.

        Args:
            log_probs (torch.Tensor): Логарифмы вероятностей действий, рассчитанные
                                      текущей (обучаемой) политикой. Форма: [B, seq_len-1].
            experience (Experience): Объект, содержащий данные из буфера воспроизведения:
                - `action_log_probs` (torch.Tensor): Логарифмы вероятностей действий, сохраненные
                                                   при сборе опыта (старая политика). Форма: [B, seq_len-1].
                - `log_probs_ref` (torch.Tensor): Логарифмы вероятностей действий от референсной
                                                (например, αρχ-модели) политики. Форма: [B, seq_len-1].
                - `advantages` (torch.Tensor): Оценки преимущества (advantage) для каждого действия.
                                             Форма: [B, seq_len-1] или [B, 1] (будетกระจาย).
                - `action_mask` (torch.Tensor): Маска, указывающая, какие токены являются действиями.
                                              Форма: [B, seq_len-1].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - `loss_mean`: Среднее значение функции потерь по батчу (скаляр).
                - `kl_mean`: Среднее значение KL-дивергенции по батчу (скаляр, для логирования).
        """
        old_log_probs = experience.action_log_probs   # [B, seq_len-1]
        log_probs_ref = experience.log_probs_ref      # [B, seq_len-1]
        action_mask = experience.action_mask          # [B, seq_len-1]
        advantages = experience.advantages            # [B, seq_len-1] или [B, 1]

        # 1) KL-дивергенция
        kl_tensor = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )  # => [B, seq_len-1]

        # 2) Расчёт ratio
        ratio = (log_probs - old_log_probs).exp()  # [B, seq_len-1]

        # 3) Классическая PPO-формула
        surr1 = ratio * advantages  # [B, seq_len-1]
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

        # Суммарный лосс (тем меньше, тем лучше). 
        # Отрицательный, т.к. хотим максимизировать surr (=> минимизируем -surr).
        # Также добавляем регуляризацию по KL.
        loss_tensor = -torch.min(surr1, surr2) + self.kl_weight * kl_tensor  # [B, seq_len-1]

        # 4) Усреднение с учётом маски
        loss_mean = masked_mean(loss_tensor, action_mask, dim=-1).mean()
        kl_mean = masked_mean(kl_tensor, action_mask, dim=-1).mean()

        return loss_mean, kl_mean
