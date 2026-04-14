# DiffPuter
Official Implementation of DiffPuter: Empowering Diffusion Models for Missing Data Imputation, at ICLR 2025


## Installing Dependencies
To run experiments of all the baselines, we have to create three different environments.


```
conda create -n diffputer python=3.12       
conda activate diffputer
pip install -r requirements/diffputer.txt
```

## Preparing Datasets
Run the following command to prepare all the datasets, splits and masks.

```
python download_and_process.py
```

## Reproducing the results


To run DiffPuter on a single dataset under a single mask, use the following command
```
conda activate diffputer
python main.py --dataname [NAME_OF_DATASET] --split_idx [MASK_IDX] 
```

[NAME_OF_DATASET]: california magic bean gesture letter adult default shoppers news
[MASK_IDX]: 0 1 2 3 4 5 6 7 8 9

## Phase 1 Time-Series Prototype
This repository also includes an isolated phase-1 time-series imputation prototype based on univariate Gaussian-process data, a compact 1D U-Net backbone, and a DiffPuter-style EM outer loop.

Run the default experiment with:

```bash
python ts_main.py
```

Override key settings from the CLI, for example:

```bash
python ts_main.py --seq-len 256 --n-train 2000 --n-val 200 --n-test 200 --missing-rate 0.4 --em-iters 3 --epochs 50 --num-steps 25 --num-trials 5 --seed 7
```

The time-series prototype writes checkpoints under `ts_ckpt/` and metrics under `ts_results/`.
