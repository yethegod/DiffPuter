from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeSeriesSplit:
    full: np.ndarray
    missing_mask: np.ndarray
    observed: np.ndarray
    linear_init: np.ndarray


def make_time_grid(seq_len: int) -> np.ndarray:
    return np.arange(seq_len, dtype=np.float32)


def rbf_kernel_covariance(
    time_grid: np.ndarray,
    signal_variance: float,
    length_scale: float,
    noise_variance: float,
) -> np.ndarray:
    deltas = time_grid[:, None] - time_grid[None, :]
    covariance = signal_variance * np.exp(-(deltas**2) / (2.0 * (length_scale**2)))
    covariance = covariance.astype(np.float32)
    covariance += np.eye(time_grid.shape[0], dtype=np.float32) * (noise_variance + 1e-6)
    return covariance


def sample_gp_sequences(
    num_series: int,
    time_grid: np.ndarray,
    signal_variance: float,
    length_scale: float,
    noise_variance: float,
    seed: int,
) -> np.ndarray:
    if num_series <= 0:
        raise ValueError("num_series must be positive")

    covariance = rbf_kernel_covariance(
        time_grid=time_grid,
        signal_variance=signal_variance,
        length_scale=length_scale,
        noise_variance=noise_variance,
    )

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        mean=np.zeros(time_grid.shape[0], dtype=np.float32),
        cov=covariance,
        size=num_series,
    )
    return samples[:, None, :].astype(np.float32)


def generate_random_missing_mask(
    num_series: int,
    seq_len: int,
    missing_rate: float,
    seed: int,
    max_retries: int = 128,
) -> np.ndarray:
    if not 0.0 < missing_rate < 1.0:
        raise ValueError("missing_rate must be in (0, 1)")
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2 to allow both observed and missing points")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    rng = np.random.default_rng(seed)
    mask = rng.random((num_series, seq_len)) < missing_rate
    valid = mask.any(axis=1) & (~mask).any(axis=1)
    retries = 0

    while not np.all(valid) and retries < max_retries:
        invalid_count = int((~valid).sum())
        mask[~valid] = rng.random((invalid_count, seq_len)) < missing_rate
        valid = mask.any(axis=1) & (~mask).any(axis=1)
        retries += 1

    if not np.all(valid):
        raise RuntimeError(
            "failed to generate valid missing masks within max_retries; "
            "try a larger seq_len or a less extreme missing_rate"
        )

    return mask[:, None, :]


def apply_missing_mask(sequences: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    return np.where(missing_mask, 0.0, sequences).astype(np.float32)


def linear_interpolate_fill(observed_sequences: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    filled = observed_sequences.copy().astype(np.float32)
    seq_len = filled.shape[-1]
    target_idx = np.arange(seq_len, dtype=np.float32)

    for sample_idx in range(filled.shape[0]):
        series = filled[sample_idx, 0]
        missing = missing_mask[sample_idx, 0]
        observed = ~missing
        observed_idx = np.flatnonzero(observed)

        if observed_idx.size == 0:
            raise ValueError("linear interpolation requires at least one observed point")

        observed_vals = series[observed]
        interp = np.interp(target_idx, observed_idx.astype(np.float32), observed_vals).astype(np.float32)
        filled[sample_idx, 0] = interp
        filled[sample_idx, 0, observed] = observed_vals

    return filled


def build_split(full_sequences: np.ndarray, missing_rate: float, seed: int) -> TimeSeriesSplit:
    missing_mask = generate_random_missing_mask(
        num_series=full_sequences.shape[0],
        seq_len=full_sequences.shape[-1],
        missing_rate=missing_rate,
        seed=seed,
    )
    observed = apply_missing_mask(full_sequences, missing_mask)
    linear_init = linear_interpolate_fill(observed, missing_mask)
    return TimeSeriesSplit(
        full=full_sequences.astype(np.float32),
        missing_mask=missing_mask.astype(bool),
        observed=observed.astype(np.float32),
        linear_init=linear_init.astype(np.float32),
    )


def masked_mae_rmse(prediction: np.ndarray, target: np.ndarray, missing_mask: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.shape != missing_mask.shape:
        raise ValueError("prediction, target, and missing_mask must share the same shape")

    diff = prediction[missing_mask] - target[missing_mask]
    if diff.size == 0:
        raise ValueError("masked metrics require at least one missing position")

    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.square(diff).mean()))
    return {"mae": mae, "rmse": rmse}
