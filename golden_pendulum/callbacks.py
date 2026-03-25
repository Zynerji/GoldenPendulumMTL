"""Framework integration callbacks for Golden Pendulum MTL.

Provides plug-and-play integration with:
- PyTorch Lightning (LightningModule callback)
- Hugging Face Transformers (Trainer callback)
- Standalone weight logging
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    raise ImportError("golden_pendulum requires PyTorch. Install with: pip install torch")


class WeightLogger:
    """Logs Golden Pendulum task weights to file, console, or TensorBoard.

    Args:
        log_file: Optional path to write JSONL weight logs.
        log_every: Log every N steps. Default 100.
        tensorboard_writer: Optional ``torch.utils.tensorboard.SummaryWriter``.

    Example::

        from golden_pendulum import GoldenPendulumMTL, WeightLogger

        logger = WeightLogger(log_file="weights.jsonl", log_every=50)
        balancer = GoldenPendulumMTL(n_tasks=4)

        for step, batch in enumerate(loader):
            weights = balancer.backward(losses, model)
            logger.log(step, weights)
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        log_every: int = 100,
        tensorboard_writer: Any = None,
    ):
        self.log_file = Path(log_file) if log_file else None
        self.log_every = log_every
        self.tb_writer = tensorboard_writer
        self._file_handle = None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, weights: Dict[str, float]) -> None:
        """Record task weights for the given step."""
        if step % self.log_every != 0:
            return

        # Console
        parts = [f"{k}={v:.4f}" for k, v in weights.items()]
        balance = min(weights.values()) / max(weights.values()) if weights else 0
        logger.info("Step %d | weights: %s | balance: %.3f", step, ", ".join(parts), balance)

        # JSONL file
        if self.log_file:
            record = {"step": step, "weights": weights, "balance_ratio": round(balance, 4)}
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")

        # TensorBoard
        if self.tb_writer is not None:
            for name, w in weights.items():
                self.tb_writer.add_scalar(f"golden_pendulum/weight_{name}", w, step)
            self.tb_writer.add_scalar("golden_pendulum/balance_ratio", balance, step)


class GoldenPendulumCallback:
    """PyTorch Lightning callback for Golden Pendulum MTL.

    Replaces the standard backward pass with golden-ratio gradient balancing.
    Attach to your LightningModule by overriding ``training_step`` to return
    a dict of task losses.

    Args:
        lam: Golden-ratio regularization strength. Default 0.5.
        n_iter: QP solver iterations. Default 25.
        pcgrad: Enable PCGrad. Default True.
        log_every: Log weight stats every N steps. Default 100.

    Example with PyTorch Lightning::

        import pytorch_lightning as pl
        from golden_pendulum import GoldenPendulumCallback

        class MyModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.automatic_optimization = False  # Required
                self.golden = GoldenPendulumCallback(lam=0.5)
                # ... define model ...

            def training_step(self, batch, batch_idx):
                losses = {"seg": seg_loss, "depth": depth_loss, "normal": normal_loss}
                opt = self.optimizers()
                opt.zero_grad()
                weights = self.golden.on_train_batch(losses, self, batch_idx)
                opt.step()
                self.log_dict({f"w_{k}": v for k, v in weights.items()})
                return sum(losses.values()).detach()
    """

    def __init__(
        self,
        lam: float = 0.5,
        n_iter: int = 25,
        pcgrad: bool = True,
        log_every: int = 100,
    ):
        from golden_pendulum.core import GoldenPendulumMTL

        self.balancer = GoldenPendulumMTL(
            lam=lam, n_iter=n_iter, pcgrad=pcgrad
        )
        self.log_every = log_every

    def on_train_batch(
        self,
        losses: Dict[str, torch.Tensor],
        model: torch.nn.Module,
        batch_idx: int,
    ) -> Dict[str, float]:
        """Call this instead of loss.backward() in your training_step."""
        weights = self.balancer.backward(losses, model)

        if batch_idx % self.log_every == 0:
            balance = self.balancer.weight_balance_ratio
            logger.info(
                "Step %d | Golden Pendulum balance: %.3f | weights: %s",
                batch_idx,
                balance,
                {k: round(v, 4) for k, v in weights.items()},
            )

        return weights
