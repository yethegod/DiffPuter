from __future__ import annotations

import numpy as np
import torch


SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
S_CHURN = 1.0
S_MIN = 0.0
S_MAX = float("inf")
S_NOISE = 1.0


class EDMLoss1D:
    def __init__(self, p_mean: float = -1.2, p_std: float = 1.2, sigma_data: float = 1.0) -> None:
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def __call__(self, net: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        noise = torch.randn_like(data) * sigma.view(-1, 1, 1)
        denoised = net(data + noise, sigma)
        loss = weight.view(-1, 1, 1) * torch.square(denoised - data)
        return loss


def sample_step(
    net: torch.nn.Module,
    num_steps: int,
    step_idx: int,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    x_next: torch.Tensor,
) -> torch.Tensor:
    x_cur = x_next
    gamma = min(S_CHURN / num_steps, np.sqrt(2.0) - 1.0) if S_MIN <= float(t_cur) <= S_MAX else 0.0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + torch.sqrt(t_hat**2 - t_cur**2) * S_NOISE * torch.randn_like(x_cur)

    denoised = net(x_hat, t_hat).float()
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if step_idx < num_steps - 1:
        denoised = net(x_next, t_next).float()
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def impute_sequence_batch(
    net: torch.nn.Module,
    init_batch: torch.Tensor,
    observed_batch: torch.Tensor,
    missing_mask: torch.Tensor,
    num_steps: int = 50,
    inner_resamples: int = 20,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if inner_resamples < 1:
        raise ValueError("inner_resamples must be >= 1")

    device = init_batch.device
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)

    sigma_min = max(SIGMA_MIN, float(net.sigma_min))
    sigma_max = min(SIGMA_MAX, float(net.sigma_max))
    t_steps = (
        sigma_max ** (1.0 / RHO)
        + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / RHO) - sigma_max ** (1.0 / RHO))
    ) ** RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    init_batch = init_batch.float()
    observed_batch = observed_batch.float()
    missing_mask = missing_mask.bool()

    x_t = init_batch + torch.randn_like(init_batch) * t_steps[0]

    with torch.no_grad():
        for step_idx, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            if step_idx < num_steps - 1:
                for resample_idx in range(inner_resamples):
                    known_noise = torch.randn_like(x_t) * t_next
                    x_known_t_prev = observed_batch + known_noise
                    x_unknown_t_prev = sample_step(net, num_steps, step_idx, t_cur, t_next, x_t)
                    x_t_prev = torch.where(missing_mask, x_unknown_t_prev, x_known_t_prev)

                    if resample_idx == inner_resamples - 1:
                        x_t = x_t_prev
                    else:
                        refresh_noise = torch.randn_like(x_t) * torch.sqrt(t_cur**2 - t_next**2)
                        x_t = x_t_prev + refresh_noise
            else:
                known_noise = torch.randn_like(x_t) * t_next
                x_known_t_prev = observed_batch + known_noise
                x_unknown_t_prev = sample_step(net, num_steps, step_idx, t_cur, t_next, x_t)
                x_t = torch.where(missing_mask, x_unknown_t_prev, x_known_t_prev)

    return torch.where(missing_mask, x_t, init_batch)
