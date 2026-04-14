from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ts_data import build_split, generate_random_missing_mask, linear_interpolate_fill, make_time_grid, sample_gp_sequences
from ts_diffusion import impute_sequence_batch
from ts_main import TSExperimentConfig, run_experiment, validate_config
from ts_model import EDMPrecond1D, TimeSeriesUNet1D


class TimeSeriesPrototypeTests(unittest.TestCase):
    def test_gp_generation_is_reproducible(self) -> None:
        time_grid = make_time_grid(32)
        sample_a = sample_gp_sequences(4, time_grid, 1.0, 0.15, 0.05, seed=7)
        sample_b = sample_gp_sequences(4, time_grid, 1.0, 0.15, 0.05, seed=7)
        self.assertEqual(sample_a.shape, (4, 1, 32))
        self.assertTrue(np.allclose(sample_a, sample_b))

    def test_random_missing_mask_keeps_observed_and_missing_points(self) -> None:
        full = np.arange(5 * 16, dtype=np.float32).reshape(5, 1, 16)
        split = build_split(full, missing_rate=0.3, seed=9)
        mask = split.missing_mask[:, 0, :]
        self.assertTrue(np.all(mask.any(axis=1)))
        self.assertTrue(np.all((~mask).any(axis=1)))

    def test_random_missing_mask_raises_after_retry_budget_exhausted(self) -> None:
        with self.assertRaises(RuntimeError):
            generate_random_missing_mask(num_series=16, seq_len=2, missing_rate=0.999, seed=3, max_retries=0)

    def test_linear_interpolation_preserves_observed_values_and_removes_nans(self) -> None:
        observed = np.array([[[1.0, 0.0, 0.0, 4.0, 0.0]]], dtype=np.float32)
        missing_mask = np.array([[[False, True, True, False, True]]])
        filled = linear_interpolate_fill(observed, missing_mask)
        self.assertFalse(np.isnan(filled).any())
        self.assertEqual(float(filled[0, 0, 0]), 1.0)
        self.assertEqual(float(filled[0, 0, 3]), 4.0)

    def test_unet_forward_preserves_shape(self) -> None:
        model = TimeSeriesUNet1D(in_channels=1, base_channels=16, channel_mults=(1, 2, 4), time_embed_dim=64)
        x = torch.randn(3, 1, 32)
        sigma = torch.randn(3)
        output = model(x, sigma)
        self.assertEqual(output.shape, x.shape)

    def test_sampler_preserves_observed_positions(self) -> None:
        denoise = TimeSeriesUNet1D(in_channels=1, base_channels=16, channel_mults=(1, 2, 4), time_embed_dim=64)
        net = EDMPrecond1D(denoise, sigma_data=1.0)
        init_batch = torch.randn(2, 1, 32)
        observed_batch = init_batch.clone()
        missing_mask = torch.zeros_like(init_batch, dtype=torch.bool)
        missing_mask[:, :, 4:8] = True
        observed_batch = torch.where(missing_mask, torch.zeros_like(observed_batch), observed_batch)

        imputed = impute_sequence_batch(
            net=net,
            init_batch=init_batch,
            observed_batch=observed_batch,
            missing_mask=missing_mask,
            num_steps=4,
            inner_resamples=2,
        )
        observed_positions = ~missing_mask
        self.assertTrue(torch.allclose(imputed[observed_positions], init_batch[observed_positions]))

    def test_validate_config_rejects_invalid_sampling_controls(self) -> None:
        with self.assertRaises(ValueError):
            validate_config(TSExperimentConfig(num_steps=1))
        with self.assertRaises(ValueError):
            validate_config(TSExperimentConfig(inner_resamples=0))

    def test_tiny_end_to_end_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_root = Path(tmp_dir) / "ckpt"
            result_root = Path(tmp_dir) / "results"
            config = TSExperimentConfig(
                seq_len=32,
                n_train=32,
                n_val=8,
                n_test=8,
                em_iters=1,
                epochs=2,
                batch_size=8,
                num_steps=4,
                num_trials=2,
                early_stopping_patience=1,
                base_channels=16,
                time_embed_dim=64,
                ckpt_root=str(ckpt_root),
                result_root=str(result_root),
                verbose=False,
            )

            payload = run_experiment(config)
            metrics_path = result_root / payload["run_name"] / "metrics.json"
            self.assertTrue(metrics_path.exists())
            self.assertIn("zero_fill", payload["baselines"])
            self.assertIn("linear_interp", payload["baselines"])
            self.assertEqual(len(payload["iterations"]), 1)
            self.assertIn("train", payload["iterations"][0]["metrics"])
            with metrics_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["best_iteration"], 0)


if __name__ == "__main__":
    unittest.main()
