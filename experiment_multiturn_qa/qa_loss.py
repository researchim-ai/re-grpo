"""
Определение функции потерь GRPO (Generalized Reinforcement Learning from Human Preferences and Outcomes)
и связанных вспомогательных функций для обучения модели в задаче QA.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Вычисляет Монте-Карло аппроксимацию KL-дивергенции.
    Используется оценщик k3 (см. http://joschu.net/blog/kl-approx.html).
    Применяется клиппинг к `log_ratio` для повышения стабильности обучения.

    Args:
        log_probs (torch.Tensor): Логарифмы вероятностей от текущей (новой) политики.
        log_probs_ref (torch.Tensor): Логарифмы вероятностей от референсной (старой) политики.
        action_mask (Optional[torch.Tensor]): Маска, указывающая, какие элементы следует учитывать
                                             при вычислении KL (обычно маска для действий).

    Returns:
        torch.Tensor: Тензор со значениями аппроксимированной KL-дивергенции.
    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    
    # Добавляем клиппинг для числовой стабильности
    log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
    
    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Вычисляет среднее значение тензора с учетом маски.
    Элементы, соответствующие False в маске, не учитываются в среднем.

    Args:
        tensor (torch.Tensor): Входной тензор.
        mask (Optional[torch.Tensor]): Булев тензор маски той же формы, что и `tensor`,
                                     или совместимый для broadcasting.
        dim (Optional[int]): Измерение или измерения, вдоль которых вычисляется среднее.
                           Если None, вычисляется среднее по всем элементам.

    Returns:
        torch.Tensor: Тензор со средними значениями.
    """
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask.type_as(tensor)).sum(dim=dim) / mask.sum(dim=dim)


class GRPOLoss(nn.Module):
    """
    Реализация функции потерь GRPO (Generalized Reinforcement Learning from Human Preferences and Outcomes).

    Эта функция потерь сочетает в себе суррогатную функцию потерь, похожую на PPO (Proximal Policy Optimization),
    с регуляризацией на основе KL-дивергенции по отношению к референсной политике.

    Атрибуты:
        clip_eps (float): Коэффициент клиппинга для суррогатной функции потерь (обычно 0.1-0.3).
        kl_weight (float): Вес для компоненты KL-дивергенции в общей функции потерь.
    """

    def __init__(self, clip_eps: float = 0.2, kl_weight: float = 0.01) -> None:
        """
        Инициализирует GRPOLoss.

        Args:
            clip_eps (float): Коэффициент клиппинга для отношения вероятностей.
            kl_weight (float): Вес для штрафа за KL-дивергенцию.
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
        Вычисляет потери GRPO.

        Args:
            log_probs (torch.Tensor): Логарифмы вероятностей действий, предсказанные текущей моделью (новой политикой).
            experience (Experience): Объект `Experience`, содержащий:
                - `action_log_probs`: Логарифмы вероятностей действий из старой политики (на момент сбора опыта).
                - `log_probs_ref`: Логарифмы вероятностей действий из референсной политики.
                - `action_mask`: Маска действий.
                - `advantages`: Преимущества (advantages), рассчитанные для собранного опыта.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - `loss`: Скалярное значение функции потерь GRPO.
                - `kl_mean`: Скалярное значение средней KL-дивергенции (для логирования).
        """
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )
        
        # Клиппинг kl для стабильности (дополнительная защита)
        kl = torch.clamp(kl, max=100.0)

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()

# Для обратной совместимости с существующим кодом
__call__ = GRPOLoss.forward 