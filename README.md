# MOTFM (Medical Optimal Transport Flow Matching)

[![arXiv](https://img.shields.io/badge/arXiv-2503.00266-b31b1b.svg)](https://arxiv.org/abs/2503.00266)
![License](https://img.shields.io/github/license/milad1378yz/MOTFM)
![Stars](https://img.shields.io/github/stars/milad1378yz/MOTFM?style=social)
![Repo Size](https://img.shields.io/github/repo-size/milad1378yz/MOTFM)
![MICCAI 2025](https://img.shields.io/badge/MICCAI-2025%20Accepted-4ea94b)


**MOTFM** (Medical Optimal Transport Flow Matching) accelerates medical image generation while preserving, and often improving, quality, across **2D/3D** and **class/mask-conditional** setups.

### [Paper](https://www.arxiv.org/abs/2503.00266)
### [Checkpoints](https://drive.google.com/drive/folders/1iwqLcqXdoJ8w60FVDbi4KXBfvK1Oje6h?usp=sharing)
### [Synthetic Data](https://drive.google.com/drive/folders/1iwqLcqXdoJ8w60FVDbi4KXBfvK1Oje6h?usp=sharing)

<br>

<p align="center">
  <img src="./images/framework.png" width="950">
</p>

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

To install from `pyproject.toml`, run:
```bash
pip install -e .
```

> Run the CLI from a repo checkout (or an editable install as above): Hydra loads
> the `conf/` tree relative to the scripts, so the `motfm-train` / `motfm-infer`
> entry points resolve the configs from the working tree.


---

## Data Preparation

**Important Note**:  
- Your training data **must** be stored in a single `.pkl` file, which itself must follow the structure below.  

Within that `.pkl` file, your data dictionary should look like:
```python
{
  "train": [  # List of training samples
    {
      "image": "Tensor[Channels, Height, Width, ...] (float32, normalized)",
      "mask":  "Tensor[1, Height, Width, ...] (int32)",
      "class": "Scalar integer (int32)",
      "metadata": "Structured data (dict or other format)"
    },
    ...
  ],

  "valid": [  # List of validation samples
    {
      "image": "Tensor[Channels, Height, Width, ...] (float32, normalized)",
      "mask":  "Tensor[1, Height, Width, ...] (int32)",
      "class": "Scalar integer (int32)",
      "metadata": "Structured data (dict or other format)"
    },
    ...
  ],

  "test": [  # List of test samples
    {
      "image": "Tensor[Channels, Height, Width, ...] (float32, normalized)",
      "mask":  "Tensor[1, Height, Width, ...] (int32)",
      "class": "Scalar integer (int32)",
      "metadata": "Structured data (dict or other format)"
    },
    ...
  ]
}
```

Make sure your dataset adheres to the described data structure, saved in a single `.pkl` file, before running the training or inference pipelines.

---

## Configuration (Hydra)

Configuration is managed with **[Hydra](https://hydra.cc) + OmegaConf**. The config
tree lives in `conf/` and is composed from independent groups:

```
conf/
  config.yaml            # defaults list (which group option to load)
  model/                 # unet2d | unet3d          -> model_args
  conditioning/          # unconditional | class | mask | mask_class (overlays model_args)
  data/                  # pickle | lazy_pt          -> data_args
  train/                 # default | hpc            -> train_args
  solver/                # default                   -> solver_args
  infer/                 # default                   -> infer_args
  experiment/            # ready-made compositions (camus_mask_class, mri3d_uncond, brats_synt1ce)
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

Available experiments:

| Experiment | Setup |
| --- | --- |
| `camus_mask_class` | 2D CAMUS, mask+class conditioning, `pickle` loader (reproduces the historical `default`) |
| `mri3d_uncond` | 3D brain MRI, unconditional, `pickle` loader |
| `brats_synt1ce` | 2D BraTS synT1CE, ControlNet 3-channel input + class, `lazy_pt` loader, `hpc` profile |

Two dataset loaders are supported via `data_args.loader`:
- **`pickle`** — the single `.pkl` format described above.
- **`lazy_pt`** — reads one `.pt` per sample on demand (no giant pickle in RAM),
  indexed by `slice_index_csv` + `splits_csv`; supports `modality_dropout` and
  deterministic patient-level subsampling via `data_args.fraction` (< 1.0 for quick
  runs; use the same `fraction`/`fraction_seed` for aligned evaluation).

### Hardware / HPC

The `train/hpc` profile is hardware-agnostic and adapts automatically:
- `accelerator=auto`, `devices=auto` use every visible GPU (respects `CUDA_VISIBLE_DEVICES`).
- `strategy=null` → single device, or safe multi-GPU **DDP** with
  `find_unused_parameters` (configurable via `train_args.ddp_find_unused_parameters`).
- `precision=null` → **bf16** on Ampere+/Hopper (A100/H100/H200), **fp16** on T4,
  **fp32** on CPU. TF32 matmul is enabled via `matmul_precision=high`.
- `use_compile` (torch.compile) and `use_ema` are on; `cudnn_benchmark` autotunes convs.

Example (any node, uses all allocated GPUs):
```bash
python trainer.py experiment=brats_synt1ce data_args.tensors_dir=/path/to/tensors
```

> Flash attention (`model_args.use_flash_attention`) is left **off** for portability
> (it requires CUDA + a compatible kernel and fails on e.g. T4). Enable it on
> A100/H100/H200 with `model_args.use_flash_attention=true`.

---

## Training

To train the model, run:
```bash
python trainer.py experiment=camus_mask_class
```
or (after installation):
```bash
motfm-train experiment=camus_mask_class
```

**Note**: Make sure you have prepared your dataset (a single `.pkl` file for the
`pickle` loader, or the `.pt` tensors + CSVs for the `lazy_pt` loader) before
starting training. Checkpoints and TensorBoard logs are written under
`train_args.checkpoint_dir/<run_name>`.

---

## Inference

Use `inferer.py` to generate synthetic samples from a trained checkpoint and save them as a `.pkl`.

### Quick start

Reuse the same experiment/config used for training, plus the `infer_args` group:
```bash
python inferer.py experiment=camus_mask_class \
    infer_args.model_path=mask_class_conditioning_checkpoints/camus_mask_class \
    infer_args.num_samples=200
```
or (after installation):
```bash
motfm-infer experiment=camus_mask_class infer_args.num_samples=200
```

### `infer_args`

- **`model_path`** (optional): Checkpoint `.ckpt` file or directory. If omitted, resolves from `train_args.checkpoint_dir/<run_name>`.
- **`num_samples`** (optional): Number of samples to save. If omitted, saves all validation samples.
- **`num_inference_steps`** (optional): Number of solver time points used during sampling. If omitted, uses `solver_args.time_points`.
- **`output_path`** (optional): Explicit output `.pkl` path.
- **`overwrite`** (bool): Overwrite an existing file at `output_path`.
- **`output_norm`** (default: `per_sample_minmax`): One of `clip_0_1`, `per_sample_minmax`, `global_minmax`, `none`.
- **`allow_config_mismatch`** (bool): Allow loading a checkpoint whose saved critical model fields differ from the current config.
- **`seed`** (optional): RNG seed for reproducible inference. Defaults to `train_args.seed`.

Checkpoints saved from a `torch.compile`'d model (keys prefixed with `_orig_mod.`)
are handled automatically.

### Checkpoint resolution behavior

If `infer_args.model_path` is omitted, inferer searches:
- `train_args.checkpoint_dir/<run_name>`

If `infer_args.model_path` is provided, inferer checks (in order):
- `<model_path>`
- `<model_path>/<run_name>`
- `<model_path>/latest`

If a directory is selected, checkpoint preference is:
- `last.ckpt` (if present)
- otherwise, the most recently modified `*.ckpt`

### Output behavior

- If `infer_args.output_path` is omitted, output is saved in the resolved checkpoint directory as:
  - `samples_<run_name>_<checkpoint_name>_steps<time_points>.pkl`
- If output file exists and `infer_args.overwrite` is false, a timestamp suffix is appended automatically.
- Generated samples are produced from the validation split and saved under:
  - `data_args.split_train`
  - and also `data_args.split_val` if that key is different.

### CPU-only note

If you run inference on CPU, set `model_args.use_flash_attention=false`.
Flash attention requires CUDA and will raise an error otherwise.

### Kaggle / Colab environment notes

On managed notebook environments you may hit dependency ABI issues unrelated to
MOTFM itself. Fix them *before* importing the code:
- **`torchvision::nms` mismatch** — reinstall the `torchvision` matching your
  `torch` with `pip install --no-deps --force-reinstall torchvision==<x>`, then
  restart the runtime.
- **`PIL._typing._Ink` ImportError** — `pip install --force-reinstall Pillow==10.4.0`, then restart.
- **NumPy 2.x already present (Kaggle)** — do not let extra installs downgrade it;
  pin it (`pip install nibabel numpy==<current>`).

---

## 3D Evaluation

A dedicated script is available in `evaluation_3d/` to compute 3D metrics between two datasets:

- **MMD**
- **MS-SSIM**
- **3D-FID** (R3D-18 features + MONAI FIDMetric)

```bash
python evaluation_3d/evaluate_3d.py \
    --generated_path /path/to/generated.pkl \
    --reference_path /path/to/reference.pkl \
    --generated_split train \
    --reference_split valid \
    --num_samples 200
```

Use `--skip_fid` to skip 3D-FID when torchvision video weights are unavailable.

---


## News
- **`2025-04-09`** | Code released.
- **`2025-03-29`** | The paper became available on arXiv.
- **`2025-05-27`** | The paper was accepted to MICCAI 2025.
---

## Citation

If you find this code or our work useful in your research, please cite:

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

## Checkpoint Data Dimensions

Released checkpoints:

| Checkpoint family | Expected data |
| --- | --- |
| `mask_class_conditioning_checkpoints` | 2D CAMUS: `image: [1, 384, 384]`, `mask: [1, 384, 384]`, `class` |
| `unconditional_checkpoints_3d_mri` | `image: [1, 96, 96, 96]` |

---

**Enjoy working with MOTFM!** Feel free to open an issue or pull request if you have any questions or suggestions.
