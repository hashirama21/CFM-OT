"""Training callbacks for CFM-OT."""

import pytorch_lightning as pl
import torch

from .cfmot_logging import get_logger

logger = get_logger(__name__)


class EMACallback(pl.Callback):
    """Exponential moving average of model weights.

    The EMA weights are made ACTIVE during validation and checkpointing (they are
    restored to the raw weights at the start of the next training epoch), so the
    saved ``.ckpt`` contains the EMA weights directly — downstream inference loads
    them without any special handling. The shadow is also stored under
    ``ema_shadow`` in the checkpoint for exact resume. Compatible with DDP (the
    same ops run on every rank).
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = float(decay)
        self.shadow: dict = {}
        self.backup: dict = {}

    def on_fit_start(self, trainer, pl_module) -> None:
        if not self.shadow:
            self.shadow = {
                n: p.detach().clone().float()
                for n, p in pl_module.named_parameters()
                if p.requires_grad
            }

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        d = self.decay
        for n, p in pl_module.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # Restore raw weights if we swapped to EMA for the previous validation.
        if self.backup:
            for n, p in pl_module.named_parameters():
                if n in self.backup:
                    p.data.copy_(self.backup[n])
            self.backup = {}

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if not self.shadow:
            return
        self.backup = {
            n: p.detach().clone()
            for n, p in pl_module.named_parameters()
            if n in self.shadow
        }
        for n, p in pl_module.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.dtype))
        # NB: weights are NOT restored here -> checkpoint is saved with EMA weights.

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint["ema_shadow"] = {k: v.cpu() for k, v in self.shadow.items()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if "ema_shadow" in checkpoint:
            self.shadow = {k: v.clone().float() for k, v in checkpoint["ema_shadow"].items()}
