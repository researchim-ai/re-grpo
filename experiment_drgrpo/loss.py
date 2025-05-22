from typing import Optional
import torch
import torch.nn as nn

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


def masked_sum(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
    constant_normalizer: Optional[float] = None,
) -> torch.Tensor:
    if mask is None:
        # Если маски нет, неясно, как должен работать constant_normalizer.
        # Для совместимости с masked_mean вернем просто сумму или среднее в зависимости от normalizer.
        if constant_normalizer is not None and constant_normalizer > 0:
            return tensor.sum(axis=dim) / constant_normalizer
        else:
            return tensor.sum(axis=dim) # Или можно вызвать ошибку, если normalizer не предоставлен
    
    summed_tensor = (tensor * mask).sum(axis=dim)
    if constant_normalizer is not None and constant_normalizer > 0:
        return summed_tensor / constant_normalizer
    else:
        # Если normalizer не предоставлен, или он некорректен, возможно, стоит вернуть просто сумму
        # или вызвать ошибку, чтобы указать на неправильное использование.
        # В данном случае, для соответствия логике Dr.GRPO, где normalizer должен быть,
        # вернем просто сумму, если normalizer невалиден, но это может потребовать уточнения.
        return summed_tensor


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float, critic_type: str = "grpo", generate_max_length: Optional[int] = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.critic_type = critic_type
        self.generate_max_length = generate_max_length
        if self.critic_type == "drgrpo" and self.generate_max_length is None:
            raise ValueError("generate_max_length must be provided when critic_type is 'drgrpo'")

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        if self.critic_type == "drgrpo":
            loss = masked_sum(loss, action_mask, dim=-1, constant_normalizer=self.generate_max_length).mean()
        else:
            loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()
