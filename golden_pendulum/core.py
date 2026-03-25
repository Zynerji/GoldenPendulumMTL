"""Core implementation of Golden Pendulum MTL.

Algorithm 1 from Knopp (2026):
    1. Compute per-task gradients g_k = nabla_theta L_k
    2. Normalize: g_hat_k = g_k / ||g_k||_2
    3. Compute G_hat^T G_hat (scale-free Gram matrix)
    4. Compute golden-ratio target weights alpha^(phi)
    5. Solve modified QP with golden-ratio L1 regularizer (25 iterations)
    6. Apply PCGrad conflict resolution on normalized gradients
    7. Set final gradient = weighted sum of resolved gradients
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

PHI = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio: 1.6180339887...


def golden_ratio_weights(
    n_tasks: int,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Compute golden-ratio-spaced target weights for K tasks.

    The weight for task k is:  phi^(k-1) / sum_{j=1}^{K} phi^(j-1)

    These weights are maximally incommensurate: no pair has a rational ratio,
    preventing harmonic lock-in between task gradients.

    Args:
        n_tasks: Number of tasks K.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape (n_tasks,) summing to 1.0, with golden-ratio spacing.

    Example::

        >>> golden_ratio_weights(4)
        tensor([0.1056, 0.1708, 0.2764, 0.4472])
    """
    powers = torch.tensor(
        [PHI ** k for k in range(n_tasks)], device=device, dtype=dtype
    )
    return powers / powers.sum()


def _collect_task_gradients(
    losses: Dict[str, Tensor],
    shared_params: List[Tensor],
) -> Tensor:
    """Compute per-task gradients by sequential backward passes.

    Args:
        losses: Dict mapping task name -> scalar loss tensor.
        shared_params: List of shared model parameters.

    Returns:
        Gradient matrix G of shape (n_tasks, n_params).
    """
    task_names = list(losses.keys())
    n_tasks = len(task_names)
    task_grads = []

    for i, (name, loss) in enumerate(losses.items()):
        # Zero existing gradients
        for p in shared_params:
            if p.grad is not None:
                p.grad = None

        # Backward — retain graph for all but last task
        retain = i < n_tasks - 1
        loss.backward(retain_graph=retain)

        # Flatten and collect
        parts = []
        for p in shared_params:
            if p.grad is not None:
                parts.append(p.grad.detach().flatten())
            else:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        task_grads.append(torch.cat(parts))

    return torch.stack(task_grads)  # (n_tasks, n_params)


def _solve_golden_qp(
    GTG: Tensor,
    alpha_golden: Tensor,
    lam: float = 0.5,
    n_iter: int = 25,
    min_weight_fraction: float = 0.02,
) -> Tensor:
    """Solve the golden-ratio regularized QP over the simplex.

    Minimizes:  alpha^T GTG alpha  +  lambda * ||alpha - alpha_golden||_1

    Using projected gradient descent with decaying step size.

    Args:
        GTG: Scale-free Gram matrix, shape (K, K).
        alpha_golden: Golden-ratio target weights, shape (K,).
        lam: Regularization strength for golden-ratio attraction.
        n_iter: Number of QP solver iterations.
        min_weight_fraction: Minimum weight per task = min_weight_fraction / K.

    Returns:
        Optimized task weights on the simplex, shape (K,).
    """
    n_tasks = GTG.shape[0]
    alpha = alpha_golden.clone()
    min_weight = min_weight_fraction / n_tasks

    for t in range(n_iter):
        # QP gradient: d/dalpha (alpha^T GTG alpha) = 2 * GTG @ alpha
        qp_grad = 2.0 * GTG @ alpha

        # Golden-ratio attraction: d/dalpha ||alpha - alpha_golden||_1 = sign(...)
        golden_grad = lam * torch.sign(alpha - alpha_golden)

        # Step with decaying learning rate
        lr = 0.05 / (1.0 + 0.1 * t)
        alpha = alpha - lr * (qp_grad + golden_grad)

        # Project onto simplex with minimum weight floor
        alpha = alpha.clamp(min=min_weight)
        alpha = alpha / alpha.sum()

    return alpha


def _pcgrad_resolve(
    G: Tensor,
    alpha: Tensor,
) -> Tensor:
    """Apply PCGrad-style conflict resolution on normalized gradients.

    For each conflicting pair (i, j) where <g_i, g_j> < 0, project the
    lower-weight gradient onto the normal plane of the higher-weight gradient.

    Args:
        G: Normalized gradient matrix, shape (K, D).
        alpha: Task weights, shape (K,).

    Returns:
        Conflict-resolved gradient matrix, shape (K, D).
    """
    resolved = G.clone()
    n_tasks = G.shape[0]

    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            dot = (resolved[i] * resolved[j]).sum()
            if dot < 0:
                if alpha[i] < alpha[j]:
                    # Project i onto normal plane of j
                    proj = dot / (resolved[j].norm().square() + 1e-8) * resolved[j]
                    resolved[i] = resolved[i] - proj
                else:
                    # Project j onto normal plane of i
                    proj = dot / (resolved[i].norm().square() + 1e-8) * resolved[i]
                    resolved[j] = resolved[j] - proj

    return resolved


def golden_nash_backward(
    losses: Dict[str, Tensor],
    model: nn.Module,
    lam: float = 0.5,
    n_iter: int = 25,
    min_weight_fraction: float = 0.02,
    pcgrad: bool = True,
    shared_params: Optional[List[nn.Parameter]] = None,
) -> Dict[str, float]:
    """Golden Pendulum Nash-MTL backward pass (functional API).

    Computes per-task gradients, solves the golden-ratio regularized QP to find
    anti-resonant task weights, optionally applies PCGrad conflict resolution,
    and writes the combined gradient back to model parameters.

    This is the complete Algorithm 1 from Knopp (2026).

    Args:
        losses: Dict mapping task name -> scalar loss tensor. Must have >= 2 entries.
        model: The model whose .parameters() will receive the combined gradient.
        lam: Golden-ratio regularization strength (lambda). Default 0.5.
            Higher values pull weights closer to golden-ratio targets.
            Lower values allow more deviation toward Pareto-optimal direction.
        n_iter: Number of QP solver iterations. Default 25.
        min_weight_fraction: Minimum weight per task = fraction / K. Default 0.02.
        pcgrad: Whether to apply PCGrad conflict resolution. Default True.
        shared_params: Optional explicit list of shared parameters. If None, uses
            all model parameters with requires_grad=True.

    Returns:
        Dict mapping task name -> final weight (float). Useful for logging.

    Example::

        losses = {"cls": cls_loss, "reg": reg_loss, "rank": rank_loss}
        weights = golden_nash_backward(losses, model, lam=0.5)
        optimizer.step()
        print(f"Task weights: {weights}")
    """
    task_names = list(losses.keys())
    n_tasks = len(task_names)

    # Trivial case: single task
    if n_tasks <= 1:
        total = sum(losses.values())
        total.backward()
        return {k: 1.0 for k in task_names}

    # Determine shared parameters
    if shared_params is None:
        shared_params = [p for p in model.parameters() if p.requires_grad]

    if not shared_params:
        raise ValueError("No parameters with requires_grad=True found in model")

    device = shared_params[0].device

    # Step 1: Collect per-task gradients
    G = _collect_task_gradients(losses, shared_params)

    # Step 2: Normalize each task gradient (remove magnitude disparity)
    g_norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
    G_hat = G / g_norms

    # Step 3: Compute scale-free Gram matrix
    GTG = G_hat @ G_hat.T

    # Step 4: Compute golden-ratio target weights
    alpha_golden = golden_ratio_weights(n_tasks, device=device, dtype=GTG.dtype)

    # Step 5: Solve the golden-ratio regularized QP
    alpha = _solve_golden_qp(GTG, alpha_golden, lam=lam, n_iter=n_iter,
                             min_weight_fraction=min_weight_fraction)

    # Step 6: PCGrad conflict resolution
    if pcgrad:
        G_resolved = _pcgrad_resolve(G_hat, alpha)
    else:
        G_resolved = G_hat

    # Step 7: Compute final gradient (weighted sum)
    final_grad = (alpha.unsqueeze(1) * G_resolved).sum(dim=0)

    # Write back to parameters
    offset = 0
    for p in shared_params:
        if p.grad is not None:
            p.grad = None
    for p in shared_params:
        numel = p.numel()
        p.grad = final_grad[offset : offset + numel].reshape(p.shape).clone()
        offset += numel

    return {name: alpha[i].item() for i, name in enumerate(task_names)}


class GoldenPendulumMTL:
    """Stateful wrapper for Golden Pendulum MTL gradient balancing.

    Tracks task weight history and provides a clean OOP interface.

    Args:
        n_tasks: Expected number of tasks. Used only for validation.
        lam: Golden-ratio regularization strength. Default 0.5.
        n_iter: QP solver iterations. Default 25.
        min_weight_fraction: Minimum weight floor = fraction / K. Default 0.02.
        pcgrad: Enable PCGrad conflict resolution. Default True.

    Example::

        balancer = GoldenPendulumMTL(n_tasks=4, lam=0.5)

        for batch in dataloader:
            losses = compute_losses(model, batch)
            weights = balancer.backward(losses, model)
            optimizer.step()
            optimizer.zero_grad()

        # Inspect weight history
        print(balancer.weight_history[-1])
    """

    def __init__(
        self,
        n_tasks: int = 0,
        lam: float = 0.5,
        n_iter: int = 25,
        min_weight_fraction: float = 0.02,
        pcgrad: bool = True,
    ):
        self.n_tasks = n_tasks
        self.lam = lam
        self.n_iter = n_iter
        self.min_weight_fraction = min_weight_fraction
        self.pcgrad = pcgrad
        self.weight_history: List[Dict[str, float]] = []
        self._step = 0

    def backward(
        self,
        losses: Dict[str, Tensor],
        model: nn.Module,
        shared_params: Optional[List[nn.Parameter]] = None,
    ) -> Dict[str, float]:
        """Run Golden Pendulum backward pass and record weights.

        Args:
            losses: Dict mapping task name -> scalar loss.
            model: Model to compute gradients for.
            shared_params: Optional explicit shared parameter list.

        Returns:
            Dict of task name -> weight.
        """
        if self.n_tasks > 0 and len(losses) != self.n_tasks:
            raise ValueError(
                f"Expected {self.n_tasks} tasks, got {len(losses)}: {list(losses.keys())}"
            )

        weights = golden_nash_backward(
            losses=losses,
            model=model,
            lam=self.lam,
            n_iter=self.n_iter,
            min_weight_fraction=self.min_weight_fraction,
            pcgrad=self.pcgrad,
            shared_params=shared_params,
        )

        self.weight_history.append(weights)
        self._step += 1
        return weights

    @property
    def golden_targets(self) -> Optional[Dict[str, float]]:
        """Return the golden-ratio target weights for the last-seen task set."""
        if not self.weight_history:
            return None
        task_names = list(self.weight_history[-1].keys())
        targets = golden_ratio_weights(len(task_names))
        return {name: targets[i].item() for i, name in enumerate(task_names)}

    @property
    def weight_balance_ratio(self) -> float:
        """Return w_min / w_max for the most recent step. 1.0 = perfectly balanced."""
        if not self.weight_history:
            return 0.0
        vals = list(self.weight_history[-1].values())
        return min(vals) / max(vals) if max(vals) > 0 else 0.0

    def mean_weights(self, last_n: int = 100) -> Dict[str, float]:
        """Return mean task weights over the last N steps."""
        if not self.weight_history:
            return {}
        history = self.weight_history[-last_n:]
        keys = list(history[0].keys())
        return {
            k: sum(h[k] for h in history) / len(history)
            for k in keys
        }

    def reset(self) -> None:
        """Clear weight history."""
        self.weight_history.clear()
        self._step = 0
