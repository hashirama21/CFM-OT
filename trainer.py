import contextlib
import math
import os
from typing import Optional, Union

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from utils.callbacks import EMACallback
from utils.config_schema import register_configs
from utils.data_lazy import LazyPtDataset, resolve_split_files
from utils.general_utils import create_dataloader, load_and_prepare_data
from utils.motfm_logging import get_logger
from utils.utils_fm import build_model, validate_and_save_samples

logger = get_logger(__name__)

register_configs()


def _make_class_balanced_sampler(
    class_idxs: torch.Tensor, num_classes: int, power: float = 1.0
) -> torch.utils.data.WeightedRandomSampler:
    """Inverse-frequency weighted sampler with optional tempering via ``power``."""
    counts = torch.bincount(class_idxs, minlength=num_classes).to(dtype=torch.float32)
    class_weights = counts.clamp_min(1.0).pow(-power)
    sample_weights = class_weights[class_idxs].to(dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


def _peek_sample_shapes(datamodule):
    """Return (image_shape, mask_shape|None, num_classes|None) from one sample."""
    ds = getattr(datamodule, "_lazy_train_ds", None)
    if ds is not None and len(ds) > 0:
        s = ds[0]
        img = tuple(s["images"].shape)
        mask = tuple(s["masks"].shape) if "masks" in s else None
        ncls = int(s["classes"].shape[0]) if "classes" in s else None
        return img, mask, ncls
    td = getattr(datamodule, "train_data", None)
    if td is not None and td.get("images") is not None and td["images"].ndim > 1 and td["images"].shape[1] > 0:
        img = tuple(td["images"].shape[1:])
        mask = tuple(td["masks"].shape[1:]) if td.get("masks") is not None else None
        cl = td.get("classes")
        ncls = int(cl.shape[1]) if cl is not None and getattr(cl, "ndim", 1) == 2 else None
        return img, mask, ncls
    return None


def _autotune_batch_size(config, datamodule, device, fraction, ceiling, min_bs=1):
    """Largest power-of-two per-GPU batch (<= ceiling) whose real train step
    (forward+backward+optimizer.step) stays within ``fraction`` of total VRAM.

    Searches upward from ``min_bs`` so it never starts with an OOM-sized batch;
    keeps a margin below the hard cap set by set_per_process_memory_fraction.
    """
    import gc

    shapes = _peek_sample_shapes(datamodule)
    if shapes is None:
        return ceiling
    img_shape, mask_shape, n_classes = shapes
    total = torch.cuda.get_device_properties(device).total_memory
    budget = max(0.05, fraction - 0.05) * total  # stay under the hard cap
    model_args = config["model_args"]
    path = AffineProbPath(scheduler=CondOTScheduler())

    def _peak_for(bs: int) -> int:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        m = build_model(model_args, device=device)
        m.train()
        opt = optim.AdamW(m.parameters(), lr=1e-4)
        imgs = torch.randn((bs,) + img_shape, device=device)
        masks = torch.randn((bs,) + mask_shape, device=device) if mask_shape else None
        cond = None
        if n_classes:
            cond = torch.zeros(bs, n_classes, device=device)
            cond[:, 0] = 1.0
        t = torch.rand(bs, device=device)
        x0 = torch.randn_like(imgs)
        info = path.sample(t=t, x_0=x0, x_1=imgs)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = F.mse_loss(m(x=info.x_t, t=info.t, masks=masks, cond=cond), info.dx_t)
        loss.backward()
        opt.step()
        peak = int(torch.cuda.max_memory_allocated(device))
        del m, opt, imgs, masks, cond, t, x0, info, loss
        gc.collect()
        torch.cuda.empty_cache()
        return peak

    best, bs = 0, max(1, int(min_bs))
    while bs <= int(ceiling):
        try:
            peak = _peak_for(bs)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            break
        if peak <= budget:
            best, bs = bs, bs * 2
        else:
            break
    return best if best >= 1 else max(1, int(min_bs))


class FlowMatchingDataModule(pl.LightningDataModule):
    """Lightning ``DataModule`` wrapping the existing data helpers."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.train_data: Optional[dict] = None
        self.val_data: Optional[dict] = None
        # Lazy datasets (only populated when data_args.loader == "lazy_pt").
        self._lazy_train_ds: Optional[LazyPtDataset] = None
        self._lazy_val_ds: Optional[LazyPtDataset] = None
        model_config = self.config.get("model_args", {})
        self.mask_conditioning = bool(model_config.get("mask_conditioning", False))
        self.class_conditioning = bool(model_config.get("with_conditioning", False))
        self.loader = str(self.config.get("data_args", {}).get("loader", "pickle"))

    def setup(self, stage: Optional[str] = None) -> None:
        if self.loader == "lazy_pt":
            self._setup_lazy(stage)
            return

        data_config = self.config["data_args"]
        model_config = self.config.get("model_args", {})
        logger.info(
            f"Setting up data module for stage='{stage}' with pickle='{data_config['pickle_path']}'."
        )

        spatial_dims = model_config.get("spatial_dims", None)
        if spatial_dims is not None:
            spatial_dims = int(spatial_dims)

        # Normalization knobs (optional in config; defaults preserve existing behavior).
        image_norm = data_config.get("image_norm", "minmax_0_1")
        mask_norm = data_config.get("mask_norm", "minmax_0_1")
        norm_scope = data_config.get("norm_scope", "global")
        clip_percentiles = data_config.get("clip_percentiles", None)
        if clip_percentiles is not None:
            clip_percentiles = (float(clip_percentiles[0]), float(clip_percentiles[1]))
        norm_eps = float(data_config.get("norm_eps", 1e-6))

        # Class mapping: prefer explicit ordering if provided.
        class_values = data_config.get("class_values", None)
        class_to_idx = {c: i for i, c in enumerate(class_values)} if class_values else None

        class_conditioning = bool(model_config.get("with_conditioning", False))
        expected_num_classes = None
        if class_conditioning:
            if model_config.get("cross_attention_dim", None) is None:
                raise ValueError(
                    "`model_args.with_conditioning` is True but `model_args.cross_attention_dim` is missing."
                )
            expected_num_classes = int(model_config["cross_attention_dim"])
            if class_values and expected_num_classes != len(class_values):
                raise ValueError(
                    f"`model_args.cross_attention_dim`={expected_num_classes} does not match "
                    f"`data_args.class_values` length ({len(class_values)})."
                )

        def _load(split: str) -> dict:
            return load_and_prepare_data(
                pickle_path=data_config["pickle_path"],
                split=split,
                convert_classes_to_onehot=self.class_conditioning,
                spatial_dims=spatial_dims,
                image_norm=image_norm,
                mask_norm=mask_norm,
                norm_scope=norm_scope,
                clip_percentiles=clip_percentiles,
                norm_eps=norm_eps,
                class_to_idx=class_to_idx,
                num_classes=expected_num_classes,
                class_mapping_split=data_config.get("split_train", "train"),
            )

        def _assert_required_keys(data: dict, *, split_name: str) -> None:
            if self.mask_conditioning and "masks" not in data:
                raise ValueError(
                    f"`model_args.mask_conditioning` is True but split '{split_name}' has no masks."
                )
            if self.class_conditioning and "classes" not in data:
                raise ValueError(
                    f"`model_args.with_conditioning` is True but split '{split_name}' has no classes."
                )

        if stage in (None, "fit"):
            self.train_data = _load(data_config["split_train"])
            self.val_data = _load(data_config["split_val"])
            _assert_required_keys(self.train_data, split_name=data_config["split_train"])
            _assert_required_keys(self.val_data, split_name=data_config["split_val"])
            logger.info(
                "Loaded train/val splits: "
                f"train={int(self.train_data['images'].shape[0])}, "
                f"val={int(self.val_data['images'].shape[0])}."
            )
        elif stage == "validate":
            self.val_data = _load(data_config["split_val"])
            _assert_required_keys(self.val_data, split_name=data_config["split_val"])
            logger.info(f"Loaded validation split: val={int(self.val_data['images'].shape[0])}.")

    def _make_lazy_dataset(self, split_key: str, default: str):
        """Build (LazyPtDataset, meta-dict) for a split.

        The meta-dict exposes empty tensors just so that ``.shape[0]`` and
        ``class_map`` are available (the contract expected by ``trainer.py`` and
        ``inferer.py``); the actual samples are read on demand.
        """
        dc = self.config["data_args"]
        split = dc.get(split_key, default)
        files, classes = resolve_split_files(
            dc["slice_index_csv"], dc["splits_csv"], split,
            fraction=float(dc.get("fraction", 1.0)),
            fraction_seed=int(dc.get("fraction_seed", 42)),
        )
        num_classes = int(self.config.get("model_args", {}).get("cross_attention_dim", 2) or 2)
        ds = LazyPtDataset(
            files,
            classes,
            dc["tensors_dir"],
            mask_conditioning=self.mask_conditioning,
            class_conditioning=self.class_conditioning,
            num_classes=num_classes,
            image_key=dc.get("image_key", "y"),
            cond_key=dc.get("cond_key", "x"),
            norm_scope=dc.get("norm_scope", "sample"),
            eps=float(dc.get("norm_eps", 1e-6)),
            modality_dropout=float(dc.get("modality_dropout", 0.0)),
            split=split,
        )
        meta: dict = {
            "images": torch.empty((len(ds), 0)),
            "class_map": {i: i for i in range(num_classes)},
        }
        if self.mask_conditioning:
            meta["masks"] = torch.empty((len(ds), 0))
        if self.class_conditioning:
            meta["classes"] = torch.empty((len(ds), 0))
        return ds, meta

    def _setup_lazy(self, stage: Optional[str] = None) -> None:
        data_config = self.config["data_args"]
        logger.info(
            f"Setting up LAZY data module for stage='{stage}' from "
            f"tensors_dir='{data_config['tensors_dir']}'."
        )
        if stage in (None, "fit"):
            self._lazy_train_ds, self.train_data = self._make_lazy_dataset("split_train", "train")
            self._lazy_val_ds, self.val_data = self._make_lazy_dataset("split_val", "val")
        elif stage == "validate":
            self._lazy_val_ds, self.val_data = self._make_lazy_dataset("split_val", "val")

    def _lazy_dataloader(
        self, dataset: LazyPtDataset, shuffle: bool, sampler=None
    ) -> torch.utils.data.DataLoader:
        tr = self.config["train_args"]
        num_workers = int(tr.get("num_workers", 0))
        kwargs = dict(
            batch_size=tr["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=tr.get("pin_memory", torch.cuda.is_available()),
            persistent_workers=(num_workers > 0),
            drop_last=bool(tr.get("drop_last", False)) and (shuffle or sampler is not None),
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(tr.get("prefetch_factor", 4) or 4)
        return torch.utils.data.DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.loader == "lazy_pt":
            tr = self.config["train_args"]
            sampler = None
            classes = getattr(self._lazy_train_ds, "classes", None)
            if bool(tr.get("class_balanced_sampling", False)) and classes is not None:
                cls = torch.as_tensor(classes, dtype=torch.long)
                sampler = _make_class_balanced_sampler(
                    cls, int(self._lazy_train_ds.num_classes),
                    float(tr.get("class_balance_power", 1.0)),
                )
                # Under multi-GPU DDP Lightning replaces custom samplers unless
                # use_distributed_sampler=False; class balancing then applies only
                # in single-process runs.
                logger.info("Using class-balanced sampling for the lazy train loader.")
            return self._lazy_dataloader(
                self._lazy_train_ds, shuffle=(sampler is None), sampler=sampler
            )

        tr_args = self.config["train_args"]
        sampler = None
        shuffle = True

        if bool(tr_args.get("class_balanced_sampling", False)):
            classes = None if self.train_data is None else self.train_data.get("classes")
            if classes is None:
                logger.warning(
                    "Class-balanced sampling is enabled but no classes were found; "
                    "falling back to shuffle=True."
                )
            else:
                if classes.ndim == 2:
                    class_idxs = classes.argmax(dim=1).to(dtype=torch.long)
                    num_classes = int(classes.shape[1])
                elif classes.ndim == 1:
                    class_idxs = classes.to(dtype=torch.long)
                    num_classes = int(class_idxs.max().item() + 1)
                else:
                    raise ValueError(
                        f"Unexpected classes tensor shape {tuple(classes.shape)}; "
                        "expected [N] indices or [N, K] one-hot."
                    )

                power = float(tr_args.get("class_balance_power", 1.0))
                sampler = _make_class_balanced_sampler(class_idxs, num_classes, power)
                shuffle = False
                logger.info(
                    f"Using class-balanced sampling with {num_classes} classes and power={power:.3f}."
                )

        return create_dataloader(
            Images=self.train_data["images"],
            Masks=self.train_data.get("masks") if self.mask_conditioning else None,
            classes=self.train_data.get("classes") if self.class_conditioning else None,
            batch_size=tr_args["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(tr_args.get("num_workers", 0)),
            pin_memory=tr_args.get("pin_memory", None),
            persistent_workers=tr_args.get("persistent_workers", None),
            drop_last=bool(tr_args.get("drop_last", False)),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.loader == "lazy_pt":
            return self._lazy_dataloader(self._lazy_val_ds, shuffle=False)

        tr_args = self.config["train_args"]
        return create_dataloader(
            Images=self.val_data["images"],
            Masks=self.val_data.get("masks") if self.mask_conditioning else None,
            classes=self.val_data.get("classes") if self.class_conditioning else None,
            batch_size=tr_args["batch_size"],
            shuffle=False,
            num_workers=int(tr_args.get("num_workers", 0)),
            pin_memory=tr_args.get("pin_memory", None),
            persistent_workers=tr_args.get("persistent_workers", None),
            drop_last=False,
        )


class FlowMatchingLightningModule(pl.LightningModule):
    """Lightning ``Module`` for the flow matching model."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = build_model(config["model_args"])
        self.mask_conditioning = config["model_args"]["mask_conditioning"]
        self.class_conditioning = config["model_args"]["with_conditioning"]
        self.path = AffineProbPath(scheduler=CondOTScheduler())

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        im_batch = batch["images"]
        if self.mask_conditioning:
            if "masks" not in batch:
                raise KeyError(
                    "mask_conditioning is enabled but the dataloader batch has no 'masks' key."
                )
            mask_batch = batch["masks"]
        else:
            mask_batch = None

        if self.class_conditioning:
            if "classes" not in batch:
                raise KeyError(
                    "class_conditioning is enabled but the dataloader batch has no 'classes' key."
                )
            class_batch = batch["classes"]
        else:
            class_batch = None

        # Flow-matching target: learn velocity at a random interpolation point between noise and data.
        x_0 = torch.randn_like(im_batch)
        t = torch.rand(im_batch.shape[0], device=im_batch.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=im_batch)

        v_pred = self.model(
            x=sample_info.x_t,
            t=sample_info.t,
            masks=mask_batch,
            cond=class_batch,
        )
        return F.mse_loss(v_pred, sample_info.dx_t)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        ta = self.hparams["train_args"]
        lr = ta["lr"]
        weight_decay = float(ta.get("weight_decay", 0.0))
        optimizer_name = str(ta.get("optimizer", "adam")).lower()

        if optimizer_name == "adamw":
            opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.Adam(self.model.parameters(), lr=lr)

        if str(ta.get("scheduler", "none")).lower() != "cosine":
            return opt

        # Cosine decay with linear warmup, stepped per optimizer step.
        try:
            total = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total = 0
        if total <= 1:
            return opt

        warmup = ta.get("warmup_steps", None)
        if warmup is None:
            warmup = max(1, int(0.03 * total))
        warmup = min(int(warmup), max(1, total - 1))
        floor = float(ta.get("min_lr_ratio", 0.05))

        def _lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_validation_epoch_end(self) -> None:
        """Run sampling/visualization at epoch end similar to utils.validate_and_save_samples."""
        # Avoid duplicate work under DDP.
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        # Pull required configs
        tr = self.hparams.get("train_args", {})
        solver_args = self.hparams.get("solver_args", {})

        # Resolve output directory from logger; fallback to default_root_dir
        log_dir = None
        if getattr(self.trainer, "logger", None) is not None and hasattr(
            self.trainer.logger, "log_dir"
        ):
            log_dir = self.trainer.logger.log_dir
        if not log_dir:
            log_dir = self.trainer.default_root_dir

        # Get a fresh val dataloader
        val_loader = self.trainer.datamodule.val_dataloader()

        # Execute the validation sampling and saving.
        # This hook runs OUTSIDE Lightning's autocast, so under mixed precision the
        # (fp16/bf16) params would clash with fp32 inputs inside the ControlNet.
        # Wrap the sampling in autocast to match the trainer precision.
        logger.info(f"Running validation sample export for epoch {self.current_epoch}.")
        precision = str(getattr(self.trainer, "precision", "32-true"))
        if "16" in precision and self.device.type == "cuda":
            amp_dtype = torch.bfloat16 if "bf16" in precision else torch.float16
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            validate_and_save_samples(
                model=self.model,
                val_loader=val_loader,
                device=self.device,
                checkpoint_dir=log_dir,
                epoch=self.current_epoch,
                solver_config=solver_args,
                max_samples=tr.get("num_val_samples", 16),
                class_map=None,
                mask_conditioning=self.mask_conditioning,
                class_conditioning=self.class_conditioning,
            )


def _resolve_resume_checkpoint(
    explicit_ckpt_path: Optional[str], root_ckpt_dir: str, run_name: str
) -> Optional[str]:
    """Return the checkpoint path to resume from, if any."""
    if explicit_ckpt_path:
        return explicit_ckpt_path

    ckpt_dir = os.path.join(root_ckpt_dir, run_name)
    if not os.path.isdir(ckpt_dir):
        return None

    # Prefer Lightning's rolling checkpoint for exact resume state.
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    # Fallback for older/manual checkpoint naming: pick the newest .ckpt file.
    _candidates = [
        os.path.join(ckpt_dir, fname)
        for fname in os.listdir(ckpt_dir)
        if fname.endswith(".ckpt") and os.path.isfile(os.path.join(ckpt_dir, fname))
    ]
    if not _candidates:
        return None

    return max(_candidates, key=os.path.getmtime)


def _count_devices(
    accelerator: Union[str, int, list, tuple],
    devices: Union[str, int, list, tuple],
) -> int:
    """Best-effort count of the devices Lightning will actually use."""
    accelerator_name = str(accelerator).lower()
    gpu_available = torch.cuda.is_available()
    use_gpu = accelerator_name in {"gpu", "cuda"} or (
        accelerator_name == "auto" and gpu_available
    )
    if not use_gpu:
        return 1  # CPU / single device

    if isinstance(devices, bool):
        return torch.cuda.device_count() if devices else 1
    if isinstance(devices, int):
        return max(1, devices)
    if isinstance(devices, (list, tuple)):
        return max(1, len(devices))
    if isinstance(devices, str):
        d = devices.strip().lower()
        if d == "auto" or d == "-1":
            return max(1, torch.cuda.device_count())
        if d.isdigit():
            return max(1, int(d))
        if "," in d:
            return max(1, len([x for x in d.split(",") if x.strip()]))
    return max(1, torch.cuda.device_count())


def _resolve_strategy(
    accelerator: Union[str, int, list, tuple],
    devices: Union[str, int, list, tuple],
    find_unused_parameters: bool = True,
):
    """
    Resolve the Lightning strategy for the available hardware.

    Use DDP only for true multi-GPU execution; for a single device (or CPU) keep
    "auto" to avoid needless overhead. ``find_unused_parameters`` is enabled by
    default so multi-GPU DDP does not crash when a sub-module (e.g. an unused
    ControlNet branch at some step) receives no gradient.
    """
    if _count_devices(accelerator, devices) > 1:
        return DDPStrategy(find_unused_parameters=find_unused_parameters)
    return "auto"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve to a plain container so no OmegaConf types leak into torch/Lightning.
    config = OmegaConf.to_container(cfg, resolve=True)

    run_name = str(config.get("run_name", "default"))
    tr = config["train_args"]
    root_ckpt_dir = tr["checkpoint_dir"]
    logger.info("Resolved config:\n" + OmegaConf.to_yaml(cfg))
    logger.info(f"Run name: {run_name}")
    logger.info(f"Checkpoint root directory: {root_ckpt_dir}")

    # Optional TF32 for faster matmuls on Ampere+ GPUs.
    matmul_precision = tr.get("matmul_precision", None)
    if matmul_precision:
        try:
            torch.set_float32_matmul_precision(str(matmul_precision))
            logger.info(f"Set float32 matmul precision='{matmul_precision}' (TF32).")
        except Exception as exc:
            logger.warning(f"Could not set matmul precision: {exc}")

    seed = tr.get("seed")
    if seed is not None:
        seed = int(seed)
        pl.seed_everything(seed, workers=True)
        logger.info(f"Using seed={seed} for reproducible training.")

    deterministic = bool(tr.get("deterministic", False))
    # Autotune convolution algorithms for fixed-shape inputs (disabled when the
    # run is deterministic, which requires stable algorithms).
    if bool(tr.get("cudnn_benchmark", True)) and not deterministic:
        torch.backends.cudnn.benchmark = True

    # Data and model modules
    datamodule = FlowMatchingDataModule(config)
    model = FlowMatchingLightningModule(config)

    # GPU memory safety: cap per-process VRAM (keeps >= (1-fraction) free) and,
    # if enabled, auto-tune the per-GPU batch size so a real train step fits the
    # cap on the current hardware (T4 -> small batch, A100/H100/H200 -> large).
    if torch.cuda.is_available():
        frac = float(tr.get("gpu_mem_fraction", 0.95) or 0.95)
        dev = torch.cuda.current_device()
        try:
            torch.cuda.set_per_process_memory_fraction(frac, dev)
            logger.info(
                f"GPU memory cap: {frac:.0%} of device {dev} "
                f"(>= {(1.0 - frac) * 100:.0f}% VRAM kept free)."
            )
        except Exception as exc:
            logger.warning(f"Could not set GPU memory fraction: {exc}")

        if bool(tr.get("auto_batch_size", False)):
            try:
                datamodule.setup("fit")  # needed to peek sample shapes
                tuned = _autotune_batch_size(
                    config, datamodule, dev, frac,
                    ceiling=int(tr["batch_size"]),
                    min_bs=int(tr.get("min_batch_size", 1)),
                )
                if tuned and tuned != int(tr["batch_size"]):
                    logger.info(
                        f"Auto batch size: {tr['batch_size']} -> {tuned} per GPU "
                        f"(fits within {frac:.0%} of VRAM)."
                    )
                    tr["batch_size"] = tuned
            except Exception as exc:
                logger.warning(
                    f"Auto batch-size tuning skipped ({exc}); keeping batch_size={tr['batch_size']}."
                )

    # Optional model compilation.
    if bool(tr.get("use_compile", False)):
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled.")
        except Exception as exc:
            logger.warning(f"torch.compile unavailable ({exc}); continuing without.")

    # Logging and callbacks
    tb_logger = TensorBoardLogger(save_dir=root_ckpt_dir, name=run_name)
    ckpt_every = max(1, int(tr.get("val_freq", 5)))
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(root_ckpt_dir, run_name),
        filename="epoch{epoch:03d}-valloss{val/loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=ckpt_every,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    cbs = [ckpt_cb, lr_cb]
    if bool(tr.get("use_ema", False)):
        cbs.append(EMACallback(decay=float(tr.get("ema_decay", 0.999))))
        logger.info(f"EMA enabled (decay={tr.get('ema_decay', 0.999)}).")

    # Precision auto-detection per GPU: bf16 only on Ampere+/Hopper (compute
    # capability >= 8.0, e.g. A100/H100/H200); fp16 on older GPUs (e.g. T4, which
    # has no native bf16); fp32 on CPU.
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(torch.cuda.current_device())[0]
        bf16_ok = major >= 8 and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        default_precision = "bf16-mixed" if bf16_ok else "16-mixed"
    else:
        default_precision = "32-true"
    precision = tr.get("precision") or default_precision

    resume_ckpt = _resolve_resume_checkpoint(tr.get("ckpt_path"), root_ckpt_dir, run_name)
    if resume_ckpt:
        logger.info(f"Resuming training from checkpoint: {resume_ckpt}")
    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    accelerator = tr.get("accelerator", "auto")
    devices = tr.get("devices", "auto")
    find_unused = bool(tr.get("ddp_find_unused_parameters", True))
    # "auto"/None -> resolve DDP (with find_unused) for true multi-GPU, else single.
    # "ddp" -> DDP with the configured find_unused flag. Any other value (e.g.
    # "dp", "ddp_spawn", "fsdp") is passed straight through to Lightning.
    strategy_cfg = tr.get("strategy", None)
    if strategy_cfg in (None, "auto"):
        strategy = _resolve_strategy(accelerator, devices, find_unused_parameters=find_unused)
    elif str(strategy_cfg).lower() == "ddp":
        strategy = (
            DDPStrategy(find_unused_parameters=find_unused)
            if _count_devices(accelerator, devices) > 1
            else "auto"
        )
    else:
        strategy = strategy_cfg
    logger.info(
        f"Trainer runtime: accelerator={accelerator}, devices={devices}, "
        f"strategy={strategy}, precision={precision}, deterministic={deterministic}."
    )

    trainer = pl.Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=tr["num_epochs"],
        precision=precision,
        accumulate_grad_batches=tr.get("gradient_accumulation_steps", 8),
        gradient_clip_val=tr.get("grad_clip_norm", 0.0) or None,
        check_val_every_n_epoch=ckpt_every,
        enable_progress_bar=True,
        logger=tb_logger,
        callbacks=cbs,
        # Distributed/accelerator knobs
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        deterministic=deterministic,
        log_every_n_steps=tr.get("log_every_n_steps", 50),
        num_sanity_val_steps=tr.get("num_sanity_val_steps", 0),
        limit_val_batches=tr.get("limit_val_batches", 1.0),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
