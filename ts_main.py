from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ts_data import TimeSeriesSplit, build_split, make_time_grid, masked_mae_rmse, sample_gp_sequences
from ts_diffusion import EDMLoss1D, impute_sequence_batch
from ts_model import EDMPrecond1D, TimeSeriesUNet1D


@dataclass
class TSExperimentConfig:
    seq_len: int = 256
    n_train: int = 10000
    n_val: int = 1000
    n_test: int = 1000
    missing_rate: float = 0.3
    signal_variance: float = 1.0
    length_scale: float = 8.0
    noise_variance: float = 0.05
    em_iters: int = 5
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-4
    num_steps: int = 50
    num_trials: int = 10
    early_stopping_patience: int = 20
    seed: int = 42
    sigma_data: float = 1.0
    base_channels: int = 64
    channel_mults: tuple[int, ...] = (1, 2, 4)
    time_embed_dim: int = 256
    inner_resamples: int = 20
    ckpt_root: str = "ts_ckpt"
    result_root: str = "ts_results"
    device: str | None = None
    verbose: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def _downsample_factor(channel_mults: tuple[int, ...]) -> int:
    return 2 ** (len(channel_mults) - 1)


def validate_config(config: TSExperimentConfig) -> None:
    if config.seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if config.seq_len % _downsample_factor(config.channel_mults) != 0:
        raise ValueError("seq_len must be divisible by the U-Net downsample factor")
    if not 0.0 < config.missing_rate < 1.0:
        raise ValueError("missing_rate must be in (0, 1)")

    for field_name in ("n_train", "n_val", "n_test", "em_iters", "epochs", "batch_size", "num_steps", "num_trials"):
        if getattr(config, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive")

    if config.num_steps < 2:
        raise ValueError("num_steps must be >= 2")
    if config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if config.inner_resamples < 1:
        raise ValueError("inner_resamples must be >= 1")
    if config.length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    if config.signal_variance <= 0.0:
        raise ValueError("signal_variance must be positive")
    if config.noise_variance < 0.0:
        raise ValueError("noise_variance must be non-negative")
    if config.lr <= 0.0:
        raise ValueError("lr must be positive")


def _config_to_json(config: TSExperimentConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["channel_mults"] = list(config.channel_mults)
    return payload


def _make_run_name(config: TSExperimentConfig) -> str:
    mr = int(round(config.missing_rate * 100))
    return (
        f"seq{config.seq_len}_mr{mr}_seed{config.seed}_"
        f"em{config.em_iters}_steps{config.num_steps}_trials{config.num_trials}"
    )


def _build_splits(config: TSExperimentConfig) -> dict[str, TimeSeriesSplit]:
    time_grid = make_time_grid(config.seq_len)
    full_train = sample_gp_sequences(
        num_series=config.n_train,
        time_grid=time_grid,
        signal_variance=config.signal_variance,
        length_scale=config.length_scale,
        noise_variance=config.noise_variance,
        seed=config.seed,
    )
    full_val = sample_gp_sequences(
        num_series=config.n_val,
        time_grid=time_grid,
        signal_variance=config.signal_variance,
        length_scale=config.length_scale,
        noise_variance=config.noise_variance,
        seed=config.seed + 1,
    )
    full_test = sample_gp_sequences(
        num_series=config.n_test,
        time_grid=time_grid,
        signal_variance=config.signal_variance,
        length_scale=config.length_scale,
        noise_variance=config.noise_variance,
        seed=config.seed + 2,
    )

    return {
        "train": build_split(full_train, config.missing_rate, config.seed + 101),
        "val": build_split(full_val, config.missing_rate, config.seed + 102),
        "test": build_split(full_test, config.missing_rate, config.seed + 103),
    }


def _select_device(config: TSExperimentConfig) -> str:
    if config.device is not None:
        return config.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_loader(array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(array)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate_edm_loss(
    net: torch.nn.Module,
    loss_fn: EDMLoss1D,
    data: np.ndarray,
    batch_size: int,
    device: str,
) -> float:
    loader = _make_loader(data, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_count = 0
    net.eval()

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device).float()
            loss = loss_fn(net, batch).mean()
            total_loss += float(loss.item()) * batch.shape[0]
            total_count += batch.shape[0]

    return total_loss / max(total_count, 1)


def _train_one_iteration(
    config: TSExperimentConfig,
    train_input: np.ndarray,
    val_input: np.ndarray,
    iteration_dir: Path,
    device: str,
    iteration_idx: int,
) -> tuple[EDMPrecond1D, dict[str, float]]:
    denoise_fn = TimeSeriesUNet1D(
        in_channels=1,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        time_embed_dim=config.time_embed_dim,
    ).to(device)
    net = EDMPrecond1D(denoise_fn=denoise_fn, sigma_data=config.sigma_data).to(device)
    loss_fn = EDMLoss1D(sigma_data=config.sigma_data)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    best_path = iteration_dir / "model.pt"
    train_loader = _make_loader(train_input, batch_size=config.batch_size, shuffle=True)

    last_train_loss = float("inf")
    epochs_ran = 0
    epoch_iterator = tqdm(
        range(config.epochs),
        desc=f"[ts] iter {iteration_idx} train",
        disable=not config.verbose,
        leave=False,
    )
    for epoch in epoch_iterator:
        net.train()
        total_loss = 0.0
        total_count = 0

        for (batch,) in train_loader:
            batch = batch.to(device).float()
            loss = loss_fn(net, batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * batch.shape[0]
            total_count += batch.shape[0]

        last_train_loss = total_loss / max(total_count, 1)
        val_loss = _evaluate_edm_loss(net, loss_fn, val_input, batch_size=config.batch_size, device=device)
        epochs_ran = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = last_train_loss
            epochs_without_improvement = 0
            torch.save({"state_dict": net.state_dict()}, best_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

        epoch_iterator.set_postfix(
            train_loss=f"{last_train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            best_val=f"{best_val_loss:.4f}",
        )

    state = torch.load(best_path, map_location=device)
    net.load_state_dict(state["state_dict"])

    stats = {
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "epochs_ran": float(epochs_ran),
    }
    return net, stats


def _impute_split(
    net: torch.nn.Module,
    split: TimeSeriesSplit,
    init_sequences: np.ndarray,
    config: TSExperimentConfig,
    device: str,
    split_name: str,
    iteration_idx: int,
) -> np.ndarray:
    outputs = []
    net.eval()

    batch_iterator = tqdm(
        range(0, init_sequences.shape[0], config.batch_size),
        desc=f"[ts] iter {iteration_idx} impute {split_name}",
        disable=not config.verbose,
        leave=False,
    )
    for start_idx in batch_iterator:
        end_idx = start_idx + config.batch_size
        init_batch = torch.from_numpy(init_sequences[start_idx:end_idx]).to(device).float()
        observed_batch = torch.from_numpy(split.observed[start_idx:end_idx]).to(device).float()
        missing_batch = torch.from_numpy(split.missing_mask[start_idx:end_idx]).to(device)

        trial_outputs = []
        for _ in range(config.num_trials):
            imputed = impute_sequence_batch(
                net=net,
                init_batch=init_batch,
                observed_batch=observed_batch,
                missing_mask=missing_batch,
                num_steps=config.num_steps,
                inner_resamples=config.inner_resamples,
            )
            trial_outputs.append(imputed.cpu())

        mean_imputation = torch.stack(trial_outputs, dim=0).mean(dim=0)
        outputs.append(mean_imputation.numpy().astype(np.float32))

    return np.concatenate(outputs, axis=0).astype(np.float32)


def _combine_with_observed(split: TimeSeriesSplit, imputed: np.ndarray) -> np.ndarray:
    return np.where(split.missing_mask, imputed, split.linear_init).astype(np.float32)


def _baseline_metrics(splits: dict[str, TimeSeriesSplit]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = {"zero_fill": {}, "linear_interp": {}}
    for split_name, split in splits.items():
        metrics["zero_fill"][split_name] = masked_mae_rmse(split.observed, split.full, split.missing_mask)
        metrics["linear_interp"][split_name] = masked_mae_rmse(split.linear_init, split.full, split.missing_mask)
    return metrics


def run_experiment(config: TSExperimentConfig) -> dict[str, Any]:
    validate_config(config)
    device = _select_device(config)
    run_name = _make_run_name(config)
    ckpt_dir = Path(config.ckpt_root) / run_name
    result_dir = Path(config.result_root) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if config.verbose:
        print(f"[ts] device={device} run={run_name}")

    splits = _build_splits(config)
    baseline_metrics = _baseline_metrics(splits)
    train_input = splits["train"].linear_init.copy()

    result_payload: dict[str, Any] = {
        "config": _config_to_json(config),
        "device": device,
        "run_name": run_name,
        "baselines": baseline_metrics,
        "iterations": [],
        "selection_metric": "val_rmse",
    }

    best_iteration = -1
    best_val_rmse = float("inf")

    outer_iterator = tqdm(
        range(config.em_iters),
        desc="[ts] outer iterations",
        disable=not config.verbose,
    )
    for iteration in outer_iterator:
        iteration_dir = ckpt_dir / f"iter_{iteration}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            print(f"[ts] training outer iteration {iteration}")

        net, train_stats = _train_one_iteration(
            config=config,
            train_input=train_input,
            val_input=splits["val"].linear_init,
            iteration_dir=iteration_dir,
            device=device,
            iteration_idx=iteration,
        )

        train_imputed = _impute_split(
            net, splits["train"], train_input, config, device, split_name="train", iteration_idx=iteration
        )
        val_imputed = _impute_split(
            net, splits["val"], splits["val"].linear_init, config, device, split_name="val", iteration_idx=iteration
        )
        test_imputed = _impute_split(
            net, splits["test"], splits["test"].linear_init, config, device, split_name="test", iteration_idx=iteration
        )

        iteration_metrics = {
            "train": masked_mae_rmse(train_imputed, splits["train"].full, splits["train"].missing_mask),
            "val": masked_mae_rmse(val_imputed, splits["val"].full, splits["val"].missing_mask),
            "test": masked_mae_rmse(test_imputed, splits["test"].full, splits["test"].missing_mask),
        }

        result_payload["iterations"].append(
            {
                "iteration": iteration,
                "checkpoint": str(iteration_dir / "model.pt"),
                "train_stats": train_stats,
                "metrics": iteration_metrics,
            }
        )

        if iteration_metrics["val"]["rmse"] < best_val_rmse:
            best_val_rmse = iteration_metrics["val"]["rmse"]
            best_iteration = iteration

        train_input = _combine_with_observed(splits["train"], train_imputed)
        outer_iterator.set_postfix(
            best_iter=best_iteration,
            best_val_rmse=f"{best_val_rmse:.4f}",
        )

    result_payload["best_iteration"] = best_iteration
    result_payload["best_val_rmse"] = best_val_rmse
    result_payload["final_test"] = result_payload["iterations"][best_iteration]["metrics"]["test"]

    metrics_path = result_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    if config.verbose:
        print(f"[ts] metrics saved to {metrics_path}")

    return result_payload


def parse_args() -> TSExperimentConfig:
    parser = argparse.ArgumentParser(description="Phase-1 time-series Gaussian/RBF imputation prototype")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--signal-variance", type=float, default=1.0)
    parser.add_argument("--length-scale", type=float, default=8.0)
    parser.add_argument("--noise-variance", type=float, default=0.05)
    parser.add_argument("--em-iters", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TSExperimentConfig(
        seq_len=args.seq_len,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        missing_rate=args.missing_rate,
        signal_variance=args.signal_variance,
        length_scale=args.length_scale,
        noise_variance=args.noise_variance,
        em_iters=args.em_iters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_steps=args.num_steps,
        num_trials=args.num_trials,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    run_experiment(config)


if __name__ == "__main__":
    main()
