"""Real-time gradient conflict diagnostics for Golden Pendulum MTL.

Provides deep visibility into multi-task training dynamics:
- Per-pair conflict detection and severity measurement
- Resonance detection (periodic weight oscillations)
- Gradient magnitude tracking and imbalance alerts
- Convergence monitoring toward golden equilibrium
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from golden_pendulum.core import golden_ratio_weights


@dataclass
class ConflictReport:
    """Snapshot of gradient conflict state at a single step.

    Attributes:
        step: Training step number.
        conflict_ratio: Fraction of task pairs with negative cosine similarity.
        conflict_pairs: List of (task_i, task_j, cosine_sim) for conflicting pairs.
        gradient_norms: Dict of task name -> raw gradient L2 norm.
        norm_ratio: max_norm / min_norm — magnitude imbalance.
        weight_deviation: L2 distance from golden-ratio target weights.
        balance_ratio: w_min / w_max.
        resonance_score: 0-1 score measuring periodic weight oscillation.
    """
    step: int
    conflict_ratio: float
    conflict_pairs: List[Tuple[str, str, float]]
    gradient_norms: Dict[str, float]
    norm_ratio: float
    weight_deviation: float
    balance_ratio: float
    resonance_score: float = 0.0


class DiagnosticsEngine:
    """Monitor gradient conflicts and training dynamics in real-time.

    Attaches to a Golden Pendulum training loop and generates diagnostic
    reports at each step. Useful for debugging task starvation, detecting
    resonance, and validating that the golden equilibrium is maintained.

    Args:
        window_size: Number of steps to keep in the analysis window. Default 500.
        alert_conflict_ratio: Alert when conflict ratio exceeds this. Default 0.6.
        alert_norm_ratio: Alert when gradient norm ratio exceeds this. Default 100.
        alert_balance_floor: Alert when balance drops below this. Default 0.10.

    Example::

        from golden_pendulum.pro import DiagnosticsEngine

        diag = DiagnosticsEngine(window_size=500)

        for step, batch in enumerate(loader):
            losses = compute_losses(model, batch)

            # Analyze BEFORE backward (needs access to compute graph)
            report = diag.analyze(losses, model)

            weights = balancer.backward(losses, model)
            diag.record_weights(step, weights)
            optimizer.step()

            if report.conflict_ratio > 0.6:
                print(f"High conflict at step {step}: {report.conflict_pairs}")

            if step % 500 == 0:
                print(diag.summary())
    """

    def __init__(
        self,
        window_size: int = 500,
        alert_conflict_ratio: float = 0.6,
        alert_norm_ratio: float = 100.0,
        alert_balance_floor: float = 0.10,
    ):
        self.window_size = window_size
        self.alert_conflict_ratio = alert_conflict_ratio
        self.alert_norm_ratio = alert_norm_ratio
        self.alert_balance_floor = alert_balance_floor

        self._weight_history: deque = deque(maxlen=window_size)
        self._reports: deque = deque(maxlen=window_size)
        self._alerts: List[str] = []
        self._step = 0

    def analyze(
        self,
        losses: Dict[str, Tensor],
        model: torch.nn.Module,
        shared_params: Optional[List] = None,
    ) -> ConflictReport:
        """Analyze gradient conflicts without modifying model gradients.

        Computes per-task gradients in a temporary context, measures conflict
        severity, and returns a diagnostic report.

        Args:
            losses: Dict of task name -> scalar loss.
            model: Model to analyze.
            shared_params: Optional shared parameter list.

        Returns:
            ConflictReport with full diagnostics.
        """
        task_names = list(losses.keys())
        n_tasks = len(task_names)

        if shared_params is None:
            shared_params = [p for p in model.parameters() if p.requires_grad]

        # Save existing gradients
        saved_grads = {}
        for i, p in enumerate(shared_params):
            if p.grad is not None:
                saved_grads[i] = p.grad.clone()

        # Collect per-task gradients
        task_grads = []
        for i, (name, loss) in enumerate(losses.items()):
            for p in shared_params:
                if p.grad is not None:
                    p.grad = None
            loss.backward(retain_graph=True)
            parts = []
            for p in shared_params:
                if p.grad is not None:
                    parts.append(p.grad.detach().flatten())
                else:
                    parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            task_grads.append(torch.cat(parts))

        G = torch.stack(task_grads)

        # Restore saved gradients
        for p in shared_params:
            p.grad = None
        for i, p in enumerate(shared_params):
            if i in saved_grads:
                p.grad = saved_grads[i]

        # Compute gradient norms
        norms = G.norm(dim=1)
        grad_norms = {task_names[i]: norms[i].item() for i in range(n_tasks)}
        norm_ratio = (norms.max() / norms.min().clamp(min=1e-10)).item()

        # Normalize and compute cosine similarities
        G_hat = G / norms.unsqueeze(1).clamp(min=1e-8)
        cosine_sim = G_hat @ G_hat.T

        # Find conflicting pairs
        conflict_pairs = []
        n_pairs = 0
        n_conflicts = 0
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                sim = cosine_sim[i, j].item()
                n_pairs += 1
                if sim < 0:
                    n_conflicts += 1
                    conflict_pairs.append((task_names[i], task_names[j], round(sim, 4)))

        conflict_ratio = n_conflicts / max(n_pairs, 1)

        # Weight deviation from golden targets
        weight_deviation = 0.0
        balance_ratio = 0.0
        if self._weight_history:
            last_weights = self._weight_history[-1]
            if last_weights:
                targets = golden_ratio_weights(len(last_weights))
                vals = list(last_weights.values())
                target_vals = targets.tolist()
                weight_deviation = sum((a - b) ** 2 for a, b in zip(vals, target_vals)) ** 0.5
                balance_ratio = min(vals) / max(vals) if max(vals) > 0 else 0.0

        # Resonance detection (periodic weight oscillation)
        resonance_score = self._detect_resonance()

        report = ConflictReport(
            step=self._step,
            conflict_ratio=round(conflict_ratio, 4),
            conflict_pairs=conflict_pairs,
            gradient_norms=grad_norms,
            norm_ratio=round(norm_ratio, 2),
            weight_deviation=round(weight_deviation, 6),
            balance_ratio=round(balance_ratio, 4),
            resonance_score=round(resonance_score, 4),
        )

        # Check alerts
        self._alerts.clear()
        if conflict_ratio > self.alert_conflict_ratio:
            self._alerts.append(
                f"HIGH CONFLICT: {conflict_ratio:.2%} of task pairs conflicting "
                f"(threshold: {self.alert_conflict_ratio:.2%})"
            )
        if norm_ratio > self.alert_norm_ratio:
            self._alerts.append(
                f"GRADIENT IMBALANCE: norm ratio {norm_ratio:.1f}x "
                f"(threshold: {self.alert_norm_ratio:.1f}x). "
                f"Gradient normalization is critical."
            )
        if balance_ratio > 0 and balance_ratio < self.alert_balance_floor:
            self._alerts.append(
                f"LOW BALANCE: w_min/w_max = {balance_ratio:.3f} "
                f"(threshold: {self.alert_balance_floor:.3f}). "
                f"Consider increasing lambda."
            )
        if resonance_score > 0.7:
            self._alerts.append(
                f"RESONANCE DETECTED: score {resonance_score:.3f}. "
                f"Weights are oscillating periodically. Increase lambda."
            )

        self._reports.append(report)
        self._step += 1
        return report

    def record_weights(self, step: int, weights: Dict[str, float]) -> None:
        """Record task weights for resonance detection and tracking."""
        self._weight_history.append(weights)

    def _detect_resonance(self) -> float:
        """Detect periodic weight oscillation using autocorrelation.

        Returns a score from 0 (no resonance) to 1 (strong resonance).
        Resonance indicates harmonic lock-in — the thing Golden Pendulum prevents.
        """
        if len(self._weight_history) < 50:
            return 0.0

        # Use the first task's weight history as a proxy
        history = list(self._weight_history)
        first_task = list(history[0].keys())[0]
        series = [h.get(first_task, 0.0) for h in history[-100:]]

        if len(series) < 50:
            return 0.0

        # Compute autocorrelation at lag 2-20
        mean_s = sum(series) / len(series)
        var_s = sum((s - mean_s) ** 2 for s in series) / len(series)

        if var_s < 1e-10:
            return 0.0  # Constant weights — no resonance

        max_acf = 0.0
        for lag in range(2, min(21, len(series) // 2)):
            acf = sum(
                (series[t] - mean_s) * (series[t - lag] - mean_s)
                for t in range(lag, len(series))
            ) / (len(series) * var_s)
            max_acf = max(max_acf, abs(acf))

        return min(max_acf, 1.0)

    @property
    def alerts(self) -> List[str]:
        """Return current alerts from the last analyze() call."""
        return list(self._alerts)

    def summary(self) -> str:
        """Return a human-readable summary of recent diagnostics."""
        if not self._reports:
            return "No diagnostics recorded yet."

        recent = list(self._reports)[-10:]
        avg_conflict = sum(r.conflict_ratio for r in recent) / len(recent)
        avg_norm_ratio = sum(r.norm_ratio for r in recent) / len(recent)
        avg_balance = sum(r.balance_ratio for r in recent) / len(recent)
        avg_resonance = sum(r.resonance_score for r in recent) / len(recent)

        lines = [
            f"=== Golden Pendulum Diagnostics (last {len(recent)} steps) ===",
            f"  Avg conflict ratio:   {avg_conflict:.3f}",
            f"  Avg norm ratio:       {avg_norm_ratio:.1f}x",
            f"  Avg balance:          {avg_balance:.3f}",
            f"  Avg resonance score:  {avg_resonance:.3f}",
        ]

        if self._alerts:
            lines.append("  ALERTS:")
            for a in self._alerts:
                lines.append(f"    ! {a}")

        return "\n".join(lines)
