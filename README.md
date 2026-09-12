# CFM-OT — Conditional Flow Matching with Optimal Transport

![License](https://img.shields.io/github/license/hashirama21/CFM-OT)
![Repo Size](https://img.shields.io/github/repo-size/hashirama21/CFM-OT)
[![arXiv](https://img.shields.io/badge/method-arXiv%202503.00266-b31b1b.svg)](https://arxiv.org/abs/2503.00266)

**CFM-OT** generates medical images with **conditional flow matching** along an
**optimal-transport** probability path. A velocity field is trained to transport
Gaussian noise to the data distribution and is integrated with an ODE solver at
sampling time — producing high-quality samples in **few steps**, across **2D/3D**
and **unconditional / class / mask (image-to-image)** setups.

<p align="center">
  <img src="./images/framework.png" width="950">
</p>

> CFM-OT builds on MOTFM (Yazdani et al., MICCAI 2025 — see [Citation](#citation)).
> This repository refactors it around **Hydra** configuration, adds a hardware-agnostic
> **HPC** training profile, and ships a BraTS **synT1CE** application (predicting
> contrast-enhanced T1CE from non-contrast MRI), see [`synt1ce_cfm_ot.ipynb`](./synt1ce_cfm_ot.ipynb).

---

## Method

- **Path** — `AffineProbPath` with a `CondOTScheduler` (conditional optimal-transport
  interpolation between noise `x₀ ∼ N(0, I)` and data `x₁`).
- **Objective** — the network predicts the velocity `v(x_t, t)`; the loss is
  `MSE(v_pred, dx_t)` at a random time `t ∈ [0, 1]` (`trainer.py:_compute_loss`).
- **Backbone** — MONAI `DiffusionModelUNet`, optionally paired with a `ControlNet`
  for **mask / image conditioning**; **class conditioning** is injected via
  cross-attention context (`utils/utils_fm.py:MergedModel`).
- **Sampling** — `flow_matching.solver.ODESolver` integrates the velocity field
  from `t=0` to `t=1` (`method` = `euler` / `midpoint`, `time_points` steps).

The four conditioning modes are config groups (`conf/conditioning/`):
`unconditional`, `class`, `mask`, `mask_class`.

---

## Requirements

- Python: **3.9+**
- Core stack (version floors in `pyproject.toml`; the platform's existing
  `torch`/`numpy` are reused rather than reinstalled):
  - `torch>=2.2`
  - `flow_matching>=1.0.10`
  - `pytorch-lightning>=2.2`
  - `numpy>=1.26`
  - `monai_generative>=0.2.3`
  - `hydra-core` + `omegaconf` (configuration)

Install (editable is recommended):
```bash
pip install -e .
```

> Run the CLI from a repo checkout (or an editable install): Hydra loads the `conf/`
> tree relative to the scripts, so the `cfmot-train` / `cfmot-infer` entry points
> resolve the configs from the working tree.

---

## Data

Two dataset loaders are selected via `data_args.loader`.

### `pickle` — a single `.pkl` (default)

```python
{
  "train": [  # and "valid", "test"
    {
      "image": "Tensor[C, H, W, ...] (float32)",   # generation target
      "mask":  "Tensor[C, H, W, ...]",              # optional (mask/image conditioning)
      "class": "int",                                # optional (class conditioning)
      "metadata": "dict (optional)"
    },
    ...
  ]
}
```
Set `data_args.pickle_path` and the split keys (`split_train`, `split_val`).

### `lazy_pt` — one `.pt` per sample, read on demand

Avoids holding a giant pickle in RAM (DDP-safe, flat memory). Each `.pt` is a dict
with an **image** key (target) and a **cond** key (conditioning image):

```python
# <tensors_dir>/<pid>_z<z>.pt
{ "y": Tensor[1, H, W],   # target        -> "images"   (image_key, default "y")
  "x": Tensor[C, H, W] }  # conditioning  -> "masks"    (cond_key,  default "x")
```

Indexed by two CSVs:
- `slice_index_csv` — columns `file, pid` (+ optional `z`, `has_tumour`, …); `has_tumour`
  is used as the class label when present.
- `splits_csv` — columns `patient_id` (or the first column) and `split`.

Extra `lazy_pt` features: `modality_dropout` (zero 1–2 input channels per train sample,
for robustness to missing modalities) and deterministic patient-level subsampling via
`fraction` / `fraction_seed` (use the same values for training and evaluation so
predictions stay aligned).

---

## Configuration (Hydra)

Configuration is managed with **[Hydra](https://hydra.cc) + OmegaConf**. The config
tree lives in `conf/` and is composed from independent groups:

```
conf/
  config.yaml            # defaults list (which group option to load)
  model/                 # unet2d | unet3d                                  -> model_args
  conditioning/          # unconditional | class | mask | mask_class        -> model_args overlay
  data/                  # pickle | lazy_pt                                  -> data_args
  train/                 # default | hpc                                     -> train_args
  solver/                # default                                          -> solver_args
  infer/                 # default                                          -> infer_args
  experiment/            # ready-made compositions
```

The typed schema is defined in `utils/config_schema.py` (registered with Hydra's
`ConfigStore`), so unknown keys and wrong types are caught at composition time.

Pick a ready-made experiment or compose groups, and override any field on the CLI
using dotted paths:

```bash
# Inspect the fully composed config WITHOUT running anything:
python trainer.py --cfg job experiment=brats_synt1ce

# Compose groups explicitly:
python trainer.py model@model_args=unet3d conditioning@model_args=unconditional data@data_args=pickle

# Override individual fields:
python trainer.py experiment=camus_mask_class train_args.lr=2e-4 train_args.num_epochs=300
```

Ready-made experiments (`conf/experiment/`):

| Experiment | Setup |
| --- | --- |
| `camus_mask_class` | 2D CAMUS, mask + class conditioning, `pickle` loader |
| `mri3d_uncond` | 3D brain MRI, unconditional, `pickle` loader |
| `brats_synt1ce` | 2D BraTS synT1CE, ControlNet 3-channel input + class, `lazy_pt` loader, `hpc` profile |

### Hardware / HPC

The `train/hpc` profile is hardware-agnostic and adapts automatically:
- `accelerator=auto`, `devices=auto` use every visible GPU (respects `CUDA_VISIBLE_DEVICES`).
- `strategy=null` → single device, or safe multi-GPU **DDP** with `find_unused_parameters`
  (configurable via `train_args.ddp_find_unused_parameters`).
- `precision=null` → **bf16** on Ampere+/Hopper (A100/H100/H200), **fp16** on T4,
  **fp32** on CPU. TF32 matmul is enabled via `matmul_precision=high`.
- `use_compile` (torch.compile) and `use_ema` are on; `cudnn_benchmark` autotunes convs.
- **OOM-safe memory**: `gpu_mem_fraction=0.95` caps per-process VRAM (always keeps
  ≥5% free), and `auto_batch_size` grows the per-GPU `batch_size` (up to the
  `batch_size` ceiling) to the largest that fits that budget — measured with a real
  train step. The same config trains without OOM on a 15 GB T4 or a 141 GB H200.

Example (any node, uses all allocated GPUs):
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python trainer.py experiment=brats_synt1ce data_args.tensors_dir=/path/to/tensors
```

> Flash attention (`model_args.use_flash_attention`) is left **off** for portability
> (it requires CUDA + a compatible kernel and fails on e.g. T4). Enable it on
> A100/H100/H200 with `model_args.use_flash_attention=true`.

---

## Training

```bash
python trainer.py experiment=camus_mask_class
# or, after `pip install -e .`:
cfmot-train experiment=camus_mask_class
```

Prepare your data first (a single `.pkl` for `pickle`, or the `.pt` tensors + CSVs for
`lazy_pt`). Checkpoints and TensorBoard logs are written under
`train_args.checkpoint_dir/<run_name>`; training resumes automatically from the last
checkpoint found there.

---

## Inference

`inferer.py` generates synthetic samples from a trained checkpoint and saves them as a `.pkl`.

```bash
python inferer.py experiment=camus_mask_class \
    infer_args.model_path=checkpoints/camus_mask_class \
    infer_args.num_samples=200
# or:
cfmot-infer experiment=camus_mask_class infer_args.num_samples=200
```

### `infer_args`

- **`model_path`** (optional): checkpoint `.ckpt` file or directory. If omitted, resolves from `train_args.checkpoint_dir/<run_name>`.
- **`num_samples`** (optional): number of samples to save (default: all validation samples).
- **`num_inference_steps`** (optional): solver time points during sampling (default: `solver_args.time_points`).
- **`output_path`** (optional): explicit output `.pkl` path.
- **`overwrite`** (bool): overwrite an existing `output_path`.
- **`output_norm`** (default `per_sample_minmax`): one of `clip_0_1`, `per_sample_minmax`, `global_minmax`, `none`.
- **`allow_config_mismatch`** (bool): allow loading a checkpoint whose saved critical model fields differ from the current config.
- **`seed`** (optional): RNG seed (defaults to `train_args.seed`).

Checkpoints saved from a `torch.compile`'d model (keys prefixed with `_orig_mod.`) and
EMA weights are handled automatically.

### Checkpoint & output resolution

- If `model_path` is omitted → `train_args.checkpoint_dir/<run_name>`. If provided →
  `<model_path>`, then `<model_path>/<run_name>`, then `<model_path>/latest`.
- In a directory, `last.ckpt` is preferred, else the most recently modified `*.ckpt`.
- Default output name: `samples_<run_name>_<checkpoint_name>_steps<time_points>.pkl` in the
  checkpoint directory; a timestamp suffix is added if it exists and `overwrite` is false.
- Samples are produced from the validation split and stored under `data_args.split_train`
  (and `data_args.split_val` if different).

### CPU-only note

For inference on CPU, set `model_args.use_flash_attention=false` (flash attention requires CUDA).

---

## synT1CE notebook

[`synt1ce_cfm_ot.ipynb`](./synt1ce_cfm_ot.ipynb) is an end-to-end application on BraTS-MEN:
it clones/installs this repo, trains `experiment=brats_synt1ce` (input `x = [T1n, T2w, T2f]`,
target `y = T1CE`), then runs modality-dropout evaluation, full test-set inference (NFE=100),
3D TIFF export, and the metrics suite (SSIM/PSNR, EVarΔ, Wasserstein-1, multimodality).

On managed notebooks, run the environment-fix cells first (they pin a matching
`torchvision`, reinstall `Pillow`, and keep the platform NumPy) and **restart the runtime**
when prompted.

---

## 3D Evaluation

`evaluation_3d/evaluate_3d.py` computes 3D metrics between two datasets — **MMD**,
**MS-SSIM**, and **3D-FID** (R3D-18 features + MONAI `FIDMetric`):

```bash
python evaluation_3d/evaluate_3d.py \
    --generated_path /path/to/generated.pkl \
    --reference_path /path/to/reference.pkl \
    --generated_split train --reference_split valid \
    --num_samples 200
```
Use `--skip_fid` when torchvision video weights are unavailable.

---

## Repository layout

```
trainer.py             # Hydra entry point: Lightning DataModule + Module + training loop
inferer.py             # Hydra entry point: sampling from a checkpoint -> .pkl
utils/
  config_schema.py     # typed Hydra/OmegaConf schema (ConfigStore)
  utils_fm.py          # model (UNet + optional ControlNet), solver, sampling helpers
  general_utils.py     # pickle loading, normalization, image saving
  data_lazy.py         # LazyPtDataset + split resolver (lazy_pt loader)
  callbacks.py         # EMACallback
conf/                  # Hydra config groups (see above)
evaluation_3d/         # standalone 3D metrics
synt1ce_cfm_ot.ipynb   # BraTS synT1CE application
```

---

## Citation

CFM-OT builds on MOTFM. If you use this work, please cite:

```BibTeX
@inproceedings{yazdani2025flow,
  title={Flow matching for medical image synthesis: Bridging the gap between speed and quality},
  author={Yazdani, Milad and Medghalchi, Yasamin and Ashrafian, Pooria and Hacihaliloglu, Ilker and Shahriari, Dena},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={216--226},
  year={2025},
  organization={Springer}
}
```

---

**Enjoy working with CFM-OT!** Issues and pull requests are welcome.
