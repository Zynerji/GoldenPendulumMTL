"""Adaptive lambda tuning for Golden Pendulum MTL.

Auto-tunes the golden-ratio regularization strength (lambda) based on
real-time gradient conflict severity. When conflicts are high, lambda
increases to enforce anti-resonant weights more strongly. When gradients
are aligned, lambda decreases to allow more Pareto-optimal movement.

This replaces manual lambda tuning — the #1 hyperparameter question.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from golden_pendulum.core import (
    _collect_task_gradients,
    _pcgrad_resolve,
    _solve_golden_qp,
    golden_ratio_weights,
)


class AdaptiveLambda:
    """Automatically tune lambda based on gradient conflict severity.

    Monitors the conflict ratio (fraction of task pairs with negative cosine
    similarity) and adjusts lambda using an exponential moving average:
    - High conflict (>0.5): lambda increases toward lam_max
    - Low conflict (<0.2): lambda decreases toward lam_min
    - Moderate conflict: lambda stays near current value

    Also supports loss-ratio-aware scaling: when loss magnitudes diverge more,
    lambda is boosted proportionally to prevent the QP from being dominated by
    the magnitude-driven term.

    Args:
        lam_init: Starting lambda. Default 0.5.
        lam_min: Minimum lambda floor. Default 0.05.
        lam_max: Maximum lambda ceiling. Default 2.0.
        ema_decay: Smoothing factor for conflict EMA. Default 0.95.
        conflict_target: Target conflict ratio to maintain. Default 0.3.
        sensitivity: How aggressively lambda adapts. Default 1.0.
        loss_ratio_aware: Scale lambda by log(max_loss/min_loss). Default True.

    Example::

        from golden_pendulum.pro import AdaptiveLambda

        adaptive = AdaptiveLambda(lam_init=0.5)

        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            losses = compute_losses(model, batch)
            weights = adaptive.backward(losses, model)
            optimizer.step()

            if step % 100 == 0:
                print(f"lambda={adaptive.current_lam:.3f}, "
                      f"conflict={adaptive.conflict_ratio:.3f}")
    """

    def __init__(
        self,
        lam_init: float = 0.5,
        lam_min: float = 0.05,
        lam_max: float = 2.0,
        ema_decay: float = 0.95,
        conflict_target: float = 0.3,
        sensitivity: float = 1.0,
        loss_ratio_aware: bool = True,
        n_iter: int = 25,
        min_weight_fraction: float = 0.02,
        pcgrad: bool = True,
    ):
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.ema_decay = ema_decay
        self.conflict_target = conflict_target
        self.sensitivity = sensitivity
        self.loss_ratio_aware = loss_ratio_aware
        self.n_iter = n_iter
        self.min_weight_fraction = min_weight_fraction
        self.pcgrad = pcgrad

        self._lam = lam_init
        self._conflict_ema = conflict_target  # Initialize at target
        self._loss_ratio_ema = 1.0
        self._step = 0
        self._history: List[Dict] = []

    @property
    def current_lam(self) -> float:
        """Current effective lambda value."""
        return self._lam

    @property
    def conflict_ratio(self) -> float:
        """EMA of fraction of conflicting task pairs."""
        return self._conflict_ema

    @property
    def loss_ratio(self) -> float:
        """EMA of max_loss / min_loss magnitude ratio."""
        return float(self._loss_ratio_ema)

    def _compute_conflict_ratio(self, GTG: Tensor) -> float:
        """Fraction of off-diagonal entries in GTG that are negative."""
        n = GTG.shape[0]
        if n < 2:
            return 0.0
        n_pairs = n * (n - 1) // 2
        n_conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                if GTG[i, j] < 0:
                    n_conflicts += 1
        return float(n_conflicts / n_pairs)

    def _update_lambda(self, conflict_ratio: float, loss_ratio: float) -> None:
        """Adapt lambda based on conflict severity and loss disparity."""
        # Update EMAs
        self._conflict_ema = (
            self.ema_decay * self._conflict_ema + (1 - self.ema_decay) * conflict_ratio
        )
        self._loss_ratio_ema = (
            self.ema_decay * self._loss_ratio_ema + (1 - self.ema_decay) * loss_ratio
        )

        # Conflict-driven adjustment
        conflict_delta = self._conflict_ema - self.conflict_target
        # Positive delta = more conflict than target = increase lambda
        adjustment = self.sensitivity * conflict_delta * 0.1
        self._lam = self._lam * math.exp(adjustment)

        # Loss-ratio boost: when losses are more disparate, strengthen regularization
        if self.loss_ratio_aware and self._loss_ratio_ema > 1.0:
            ratio_boost = 1.0 + 0.1 * math.log(max(self._loss_ratio_ema, 1.0))
            self._lam *= ratio_boost

        # Clamp to bounds
        self._lam = max(self.lam_min, min(self.lam_max, self._lam))

    def backward(
        self,
        losses: Dict[str, Tensor],
        model: nn.Module,
        shared_params: Optional[List[nn.Parameter]] = None,
    ) -> Dict[str, float]:
        """Run Golden Pendulum backward with adaptive lambda.

        Args:
            losses: Dict of task name -> scalar loss.
            model: Model to compute gradients for.
            shared_params: Optional explicit shared parameter list.

        Returns:
            Dict of task name -> weight.
        """
        task_names = list(losses.keys())
        n_tasks = len(task_names)

        if n_tasks <= 1:
            total = sum(losses.values())
            total.backward()
            return {k: 1.0 for k in task_names}

        if shared_params is None:
            shared_params = [p for p in model.parameters() if p.requires_grad]

        device = shared_params[0].device

        # Compute loss magnitude ratio
        loss_vals = [v.detach().item() for v in losses.values()]
        loss_min = max(min(loss_vals), 1e-10)
        loss_max = max(loss_vals)
        loss_ratio = loss_max / loss_min

        # Collect and normalize gradients
        G = _collect_task_gradients(losses, shared_params)
        g_norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G_hat = G / g_norms

        # Scale-free Gram matrix
        GTG = G_hat @ G_hat.T

        # Measure conflict
        conflict_ratio = self._compute_conflict_ratio(GTG)

        # Update lambda
        self._update_lambda(conflict_ratio, loss_ratio)

        # Solve QP with current adaptive lambda
        alpha_golden = golden_ratio_weights(n_tasks, device=device, dtype=GTG.dtype)
        alpha = _solve_golden_qp(
            GTG, alpha_golden, lam=self._lam, n_iter=self.n_iter,
            min_weight_fraction=self.min_weight_fraction,
        )

        # PCGrad
        if self.pcgrad:
            G_resolved = _pcgrad_resolve(G_hat, alpha)
        else:
            G_resolved = G_hat

        # Final gradient
        final_grad = (alpha.unsqueeze(1) * G_resolved).sum(dim=0)

        # Write back
        offset = 0
        for p in shared_params:
            p.grad = None
        for p in shared_params:
            numel = p.numel()
            p.grad = final_grad[offset : offset + numel].reshape(p.shape).clone()
            offset += numel

        weights = {name: alpha[i].item() for i, name in enumerate(task_names)}

        # Record history
        self._history.append({
            "step": self._step,
            "lam": round(self._lam, 6),
            "conflict_ratio": round(conflict_ratio, 4),
            "loss_ratio": round(loss_ratio, 2),
            "weights": weights,
        })
        self._step += 1

        return weights

    def get_history(self, last_n: int = 0) -> List[Dict]:
        """Return adaptive tuning history.

        Args:
            last_n: Return last N entries. 0 = all.
        """
        if last_n > 0:
            return self._history[-last_n:]
        return list(self._history)

    def reset(self) -> None:
        """Reset adaptive state."""
        self._conflict_ema = self.conflict_target
        self._loss_ratio_ema = 1.0
        self._step = 0
        self._history.clear()
