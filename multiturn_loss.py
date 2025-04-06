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
    Monte-Carlo approximation of KL divergence (k3 estimator).
    Считаем: E[ exp(log_probs_ref - log_probs) - (log_probs_ref - log_probs) - 1 ].
    
    log_probs, log_probs_ref, action_mask имеют форму [B, seq_len-1].
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
    dim: int = None,
) -> torch.Tensor:
    """
    Считает среднее значение тензора (tensor), учитывая маску (mask),
    где mask=1 означает «учитывать элемент», а 0 — «пропустить».
    """
    if mask is None:
        return tensor.mean(dim=dim)
    # Складываем только по тем позициям, где mask=1,
    # и делим на общее число «единичных» элементов.
    return (tensor * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-10)


class GRPOLoss(nn.Module):
    """
    Лосс GRPO (аналог PPO) для обучения модели-актора.

    Параметры:
      clip_eps: «радиус» клиппинга для ratio (например, 0.2).
      kl_weight: коэффициент при KL-дивергенции (регуляризация).
    """

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Аргументы:
          log_probs:     [B, seq_len-1] — новые лог-вероятности актёра.
          experience:    содержит:
            - action_log_probs (старые лог-вероятности) [B, seq_len-1]
            - log_probs_ref (лог-вероятности реф. модели) [B, seq_len-1]
            - returns (не всегда используется тут)
            - advantages [B, seq_len-1] или [B, 1]
            - action_mask [B, seq_len-1]
            - ...
        
        Возвращает (loss, kl), где:
          - loss: скаляр (Mean over batch),
          - kl:   среднее (Mean over batch) KL.

        Принцип:
          - ratio = exp(new_log_probs - old_log_probs).
          - основной штраф PPO = -min(ratio*adv, clip(ratio)*adv).
          - + kl_weight * approx_kl_divergence(...)
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
