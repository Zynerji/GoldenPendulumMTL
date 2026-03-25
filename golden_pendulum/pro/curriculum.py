"""Curriculum scheduling for multi-phase training with Golden Pendulum.

Manages the Phase A/B/C/D curriculum from the paper:
- Phase A (Ranking): Backbone unfrozen, ranking-focused tasks
- Phase B (Risk): Backbone frozen, risk/sizing tasks
- Phase C (Meta): Backbone frozen, meta/calibration tasks
- Phase D (Discovery): Backbone frozen, discovery/latent tasks

Each phase can have its own task set, lambda, learning rate, and backbone
freeze state. The scheduler handles transitions automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import torch.nn as nn
from torch import Tensor

from golden_pendulum.core import GoldenPendulumMTL

logger = logging.getLogger(__name__)


@dataclass
class Phase:
    """A training phase in the curriculum.

    Attributes:
        name: Phase identifier (e.g., "A_ranking", "B_risk").
        tasks: Set of task names active in this phase.
        steps: Number of training steps for this phase.
        freeze_backbone: Whether to freeze the backbone during this phase.
        lam: Golden-ratio regularization strength for this phase.
        lr: Learning rate for this phase (applied via scheduler callback).
        pcgrad: Whether to enable PCGrad in this phase.
        warmup_steps: Linear warmup steps at phase start.
        on_enter: Optional callback called when phase begins. Receives (model, phase).
        on_exit: Optional callback called when phase ends. Receives (model, phase).
    """
    name: str
    tasks: Set[str]
    steps: int
    freeze_backbone: bool = False
    lam: float = 0.5
    lr: float = 1e-4
    pcgrad: bool = True
    warmup_steps: int = 0
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None


class CurriculumScheduler:
    """Multi-phase curriculum scheduler with Golden Pendulum integration.

    Manages phase transitions, backbone freezing, and per-phase Golden
    Pendulum configuration.

    Args:
        phases: Ordered list of training phases.
        backbone_params: Optional callable that returns backbone parameters
            (used for freeze/unfreeze). If None, freezing is skipped.

    Example::

        from golden_pendulum.pro import CurriculumScheduler, Phase

        curriculum = CurriculumScheduler(
            phases=[
                Phase("A_ranking",
                    tasks={"horizon_returns", "trade_quality", "short_rank", "embedding"},
                    steps=15000, freeze_backbone=False, lam=0.5, lr=5e-5),
                Phase("B_risk",
                    tasks={"vol_forecast", "mae", "kelly", "risk_score", "horizon_mag"},
                    steps=10000, freeze_backbone=True, lam=0.3, lr=1e-4),
                Phase("C_meta",
                    tasks={"regime", "calibration", "confidence", "optimal_exit"},
                    steps=10000, freeze_backbone=True, lam=0.3, lr=1e-4),
                Phase("D_discovery",
                    tasks={"reconstruction", "prediction", "decorrelation"},
                    steps=5000, freeze_backbone=True, lam=0.8, lr=5e-5),
            ],
            backbone_params=lambda model: model.backbone.parameters(),
        )

        for step in range(curriculum.total_steps):
            phase = curriculum.current_phase
            active_losses = {k: v for k, v in all_losses.items() if k in phase.tasks}
            weights = curriculum.backward(active_losses, model)
            optimizer.step()
            optimizer.zero_grad()
            curriculum.step()
    """

    def __init__(
        self,
        phases: List[Phase],
        backbone_params: Optional[Callable] = None,
    ):
        if not phases:
            raise ValueError("At least one phase is required")

        self.phases = phases
        self._backbone_params_fn = backbone_params
        self._phase_idx = 0
        self._phase_step = 0
        self._global_step = 0
        self._balancers: Dict[str, GoldenPendulumMTL] = {}
        self._entered = False

        # Create per-phase balancers
        for phase in phases:
            self._balancers[phase.name] = GoldenPendulumMTL(
                n_tasks=len(phase.tasks),
                lam=phase.lam,
                pcgrad=phase.pcgrad,
            )

    @property
    def current_phase(self) -> Phase:
        """Return the current active phase."""
        return self.phases[self._phase_idx]

    @property
    def current_phase_name(self) -> str:
        return self.current_phase.name

    @property
    def current_phase_progress(self) -> float:
        """Return progress through current phase as fraction [0, 1]."""
        return self._phase_step / max(self.current_phase.steps, 1)

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def total_steps(self) -> int:
        return sum(p.steps for p in self.phases)

    @property
    def is_complete(self) -> bool:
        return self._phase_idx >= len(self.phases)

    @property
    def current_lr(self) -> float:
        """Return the learning rate for the current phase, with warmup."""
        phase = self.current_phase
        if phase.warmup_steps > 0 and self._phase_step < phase.warmup_steps:
            return phase.lr * (self._phase_step + 1) / phase.warmup_steps
        return phase.lr

    def _enter_phase(self, model: Optional[nn.Module] = None) -> None:
        """Handle phase entry: freeze/unfreeze backbone, call on_enter."""
        phase = self.current_phase
        logger.info(
            "Entering phase '%s': %d tasks, %d steps, backbone=%s, lam=%.2f",
            phase.name, len(phase.tasks), phase.steps,
            "frozen" if phase.freeze_backbone else "unfrozen", phase.lam,
        )

        if model is not None and self._backbone_params_fn is not None:
            for p in self._backbone_params_fn(model):
                p.requires_grad = not phase.freeze_backbone

        if phase.on_enter is not None:
            phase.on_enter(model, phase)

        self._entered = True

    def _exit_phase(self, model: Optional[nn.Module] = None) -> None:
        """Handle phase exit: call on_exit."""
        phase = self.current_phase
        logger.info("Exiting phase '%s' after %d steps", phase.name, self._phase_step)

        if phase.on_exit is not None:
            phase.on_exit(model, phase)

        self._entered = False

    def backward(
        self,
        losses: Dict[str, Tensor],
        model: nn.Module,
        shared_params: Optional[List] = None,
    ) -> Dict[str, float]:
        """Run Golden Pendulum backward for the current phase.

        Only losses matching the current phase's task set are used.
        Extra losses are silently ignored.

        Args:
            losses: Dict of task name -> scalar loss.
            model: Model to compute gradients for.
            shared_params: Optional shared parameter list.

        Returns:
            Dict of active task name -> weight.
        """
        if self.is_complete:
            raise RuntimeError("Curriculum is complete — no more phases")

        if not self._entered:
            self._enter_phase(model)

        phase = self.current_phase
        balancer = self._balancers[phase.name]

        # Filter to active tasks
        active_losses = {k: v for k, v in losses.items() if k in phase.tasks}

        if not active_losses:
            logger.warning(
                "Phase '%s' expects tasks %s but got %s — no active losses",
                phase.name, phase.tasks, list(losses.keys()),
            )
            return {}

        # Allow fewer tasks than declared (some might be unavailable)
        balancer.n_tasks = 0  # Disable count check
        return balancer.backward(active_losses, model, shared_params=shared_params)

    def step(self, model: Optional[nn.Module] = None) -> bool:
        """Advance by one step. Returns True if phase changed.

        Args:
            model: Optional model for backbone freeze/unfreeze on phase transition.

        Returns:
            True if a phase transition occurred.
        """
        self._phase_step += 1
        self._global_step += 1

        if self._phase_step >= self.current_phase.steps:
            self._exit_phase(model)
            self._phase_idx += 1
            self._phase_step = 0

            if not self.is_complete:
                self._enter_phase(model)

            return True
        return False

    def state_dict(self) -> Dict:
        """Serialize scheduler state for checkpointing."""
        return {
            "phase_idx": self._phase_idx,
            "phase_step": self._phase_step,
            "global_step": self._global_step,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore scheduler state from checkpoint."""
        self._phase_idx = state["phase_idx"]
        self._phase_step = state["phase_step"]
        self._global_step = state["global_step"]
        self._entered = False
