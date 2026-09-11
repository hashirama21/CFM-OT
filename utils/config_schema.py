"""Structured configuration schema for MOTFM (Hydra + OmegaConf).

The root keys (``model_args``/``data_args``/``train_args``/``solver_args``/
``infer_args``) are kept identical to the historical YAML layout so that the
existing runtime code (``build_model``, checkpoint validation, inference) keeps
working with minimal changes. Dataclasses give Hydra type-checking and defaults;
the concrete values live in the ``conf/`` config groups.

Fields typed as ``Any`` (e.g. ``image_norm``, ``devices``, ``limit_val_batches``)
are intentionally untyped because they may hold either a scalar or a
list/dict (for example ``image_norm`` can be a plain string or a normalization
sub-dict).
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class ModelArgs:
    # Architecture
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    num_res_blocks: List[int] = field(default_factory=lambda: [2, 2, 2, 2, 2])
    num_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 224, 256])
    attention_levels: List[bool] = field(
        default_factory=lambda: [False, False, False, True, True]
    )
    norm_num_groups: int = 32
    resblock_updown: bool = True
    num_head_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 224, 256])
    transformer_num_layers: int = 4
    use_flash_attention: bool = True
    dropout_cattn: Optional[float] = None
    max_timestep: int = 1000

    # Conditioning (populated/overridden by the `conditioning` config group)
    with_conditioning: bool = False
    mask_conditioning: bool = False
    cross_attention_dim: Optional[int] = None
    conditioning_embedding_num_channels: Optional[List[int]] = None
    # Number of channels of the ControlNet conditioning image (1 for a binary
    # segmentation mask, 3 for the synT1CE multi-modality input).
    conditioning_embedding_in_channels: Optional[int] = None


@dataclass
class DataArgs:
    # "pickle" -> historical single-.pkl loader; "lazy_pt" -> per-sample .pt reader.
    loader: str = "pickle"

    # Split keys (dict keys in the pickle, or `split` values in splits.csv).
    split_train: str = "train"
    split_val: str = "valid"
    split_test: str = "test"

    # --- pickle loader ---
    pickle_path: Optional[str] = None
    image_norm: Any = "minmax_0_1"
    mask_norm: Any = "minmax_0_1"
    norm_scope: str = "global"
    clip_percentiles: Optional[List[float]] = None
    norm_eps: float = 1e-6
    class_values: Optional[List[Any]] = None

    # --- lazy_pt loader ---
    tensors_dir: Optional[str] = None
    slice_index_csv: Optional[str] = None
    splits_csv: Optional[str] = None
    # Keys inside each .pt record: image (target) and conditioning (input).
    image_key: str = "y"
    cond_key: str = "x"
    # Probability of zeroing 1-2 input channels per TRAIN sample (0 disables).
    modality_dropout: float = 0.0
    # Deterministic patient-level subsampling (1.0 = full split). Applies to the
    # lazy loader; keep the same value/seed everywhere for aligned evaluation.
    fraction: float = 1.0
    fraction_seed: int = 42


@dataclass
class TrainArgs:
    num_epochs: int = 200
    batch_size: int = 1
    lr: float = 1e-4
    checkpoint_dir: str = "checkpoints"

    # Reproducibility / runtime
    seed: Optional[int] = None
    deterministic: bool = False
    device: str = "cuda"
    accelerator: str = "auto"
    devices: Any = "auto"
    strategy: Optional[str] = None  # None/"auto" -> resolved per device count
    # For multi-GPU DDP: guard against sub-modules that receive no gradient.
    ddp_find_unused_parameters: bool = True
    precision: Optional[str] = None  # None -> auto-detect bf16/fp16/32 per GPU
    matmul_precision: Optional[str] = None  # e.g. "high" to enable TF32 (Ampere+)
    cudnn_benchmark: bool = True  # autotune convolutions for fixed-shape inputs

    # Dataloader
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    drop_last: bool = False
    class_balanced_sampling: bool = False
    class_balance_power: float = 1.0

    # Optimizer / scheduler
    optimizer: str = "adam"  # "adam" | "adamw"
    weight_decay: float = 0.0
    scheduler: str = "none"  # "none" | "cosine"
    warmup_steps: Optional[int] = None
    min_lr_ratio: float = 0.05

    # GPU memory safety (any hardware: T4 -> H200)
    gpu_mem_fraction: float = 0.95  # cap per-process VRAM -> keeps >= 5% free
    auto_batch_size: bool = False   # grow batch_size to the largest that fits the cap
    min_batch_size: int = 1         # floor for the auto batch-size search

    # Trainer knobs
    gradient_accumulation_steps: int = 8
    grad_clip_norm: float = 0.0
    val_freq: int = 5
    num_val_samples: int = 16
    limit_val_batches: Any = 1.0
    log_every_n_steps: int = 50
    num_sanity_val_steps: int = 0
    ckpt_path: Optional[str] = None
    print_every: int = 1

    # Optimizations
    use_compile: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999


@dataclass
class SolverArgs:
    method: str = "euler"
    step_size: float = 0.1
    time_points: int = 10


@dataclass
class InferArgs:
    num_samples: Optional[int] = None
    model_path: Optional[str] = None
    num_inference_steps: Optional[int] = None
    output_path: Optional[str] = None
    overwrite: bool = False
    output_norm: str = "per_sample_minmax"  # clip_0_1|per_sample_minmax|global_minmax|none
    allow_config_mismatch: bool = False
    seed: Optional[int] = None


@dataclass
class Config:
    # Used for the TensorBoard run name and the checkpoint sub-directory.
    run_name: str = "default"
    model_args: ModelArgs = field(default_factory=ModelArgs)
    data_args: DataArgs = field(default_factory=DataArgs)
    train_args: TrainArgs = field(default_factory=TrainArgs)
    solver_args: SolverArgs = field(default_factory=SolverArgs)
    infer_args: InferArgs = field(default_factory=InferArgs)


def register_configs() -> None:
    """Register the base schema so Hydra validates types across config groups."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
