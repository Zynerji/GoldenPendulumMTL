"""Dynamic task grouping and K adaptation for Golden Pendulum MTL.

When training with many tasks (K=16+), not all tasks need to participate in
every gradient balancing step. DynamicK automatically groups tasks by
gradient similarity and runs Golden Pendulum within each group, then merges.

This reduces the O(K^2) cost of the Gram matrix computation and allows
tasks to be added/removed dynamically during training (e.g., curriculum phases).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch import Tensor

from golden_pendulum.core import (
    _collect_task_gradients,
    _pcgrad_resolve,
    _solve_golden_qp,
    golden_ratio_weights,
)


@dataclass
class TaskGroup:
    """A group of tasks that are balanced together.

    Attributes:
        name: Group name (e.g., "ranking", "risk", "meta").
        tasks: Set of task names in this group.
        weight: Group-level weight in the final gradient (allocated by golden ratio).
        lam: Per-group lambda override. None = use global lambda.
    """
    name: str
    tasks: Set[str] = field(default_factory=set)
    weight: float = 0.0
    lam: Optional[float] = None


class DynamicK:
    """Dynamic task grouping with automatic or manual group assignment.

    Modes:
    - **Manual groups**: Define task groups explicitly (e.g., Phase A/B/C tasks).
      Golden Pendulum runs within each group, then golden-ratio weights are
      assigned across groups.
    - **Auto-grouping**: Cluster tasks by gradient cosine similarity every N steps.
      Tasks with aligned gradients are grouped together (they don't conflict),
      and Golden Pendulum only runs between groups that conflict.

    Args:
        groups: Optional list of predefined TaskGroups. If None, all tasks
            are in a single group (equivalent to standard Golden Pendulum).
        auto_group: Enable automatic gradient-similarity clustering. Default False.
        regroup_every: Steps between re-clustering (if auto_group=True). Default 500.
        similarity_threshold: Cosine similarity threshold for grouping. Default 0.5.
        lam: Default lambda for all groups. Default 0.5.
        n_iter: QP solver iterations. Default 25.
        pcgrad: Enable PCGrad. Default True.

    Example with manual groups::

        from golden_pendulum.pro import DynamicK, TaskGroup

        dk = DynamicK(groups=[
            TaskGroup("ranking",
                      tasks={"horizon_returns", "short_rank", "trade_quality", "embedding"}),
            TaskGroup("risk", tasks={"vol_forecast", "mae", "kelly", "risk_score"}),
            TaskGroup("meta", tasks={"regime", "calibration", "confidence", "optimal_exit"}),
        ])

        # In training loop:
        weights = dk.backward(all_16_losses, model)

    Example with auto-grouping::

        dk = DynamicK(auto_group=True, regroup_every=500, similarity_threshold=0.5)
        weights = dk.backward(losses, model)
        print(dk.current_groups)  # See discovered groups
    """

    def __init__(
        self,
        groups: Optional[List[TaskGroup]] = None,
        auto_group: bool = False,
        regroup_every: int = 500,
        similarity_threshold: float = 0.5,
        lam: float = 0.5,
        n_iter: int = 25,
        min_weight_fraction: float = 0.02,
        pcgrad: bool = True,
    ):
        self.auto_group = auto_group
        self.regroup_every = regroup_every
        self.similarity_threshold = similarity_threshold
        self.lam = lam
        self.n_iter = n_iter
        self.min_weight_fraction = min_weight_fraction
        self.pcgrad = pcgrad

        self._groups = groups or []
        self._step = 0
        self._weight_history: List[Dict[str, float]] = []

    @property
    def current_groups(self) -> List[TaskGroup]:
        """Return current task groups."""
        return list(self._groups)

    def _auto_cluster(self, G_hat: Tensor, task_names: List[str]) -> List[TaskGroup]:
        """Cluster tasks by gradient cosine similarity using greedy agglomerative."""
        n = G_hat.shape[0]
        cosine_sim = G_hat @ G_hat.T  # Already normalized, so this is cosine sim

        # Greedy agglomerative clustering
        assigned = [False] * n
        groups = []
        group_idx = 0

        for i in range(n):
            if assigned[i]:
                continue
            # Start new group with task i
            group_tasks = {task_names[i]}
            assigned[i] = True

            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                # Check if j is similar to all current group members
                all_similar = True
                for member_name in group_tasks:
                    member_idx = task_names.index(member_name)
                    if cosine_sim[member_idx, j].item() < self.similarity_threshold:
                        all_similar = False
                        break
                if all_similar:
                    group_tasks.add(task_names[j])
                    assigned[j] = True

            groups.append(TaskGroup(
                name=f"auto_group_{group_idx}",
                tasks=group_tasks,
            ))
            group_idx += 1

        return groups

    def _ensure_groups(
        self, task_names: List[str], G_hat: Optional[Tensor] = None,
    ) -> List[TaskGroup]:
        """Ensure groups are set up, auto-clustering if needed."""
        if self.auto_group and G_hat is not None:
            if self._step % self.regroup_every == 0 or not self._groups:
                self._groups = self._auto_cluster(G_hat, task_names)
            return self._groups

        if self._groups:
            return self._groups

        # Default: single group with all tasks
        return [TaskGroup("all", tasks=set(task_names))]

    def backward(
        self,
        losses: Dict[str, Tensor],
        model: nn.Module,
        shared_params: Optional[List[nn.Parameter]] = None,
    ) -> Dict[str, float]:
        """Run hierarchical Golden Pendulum backward.

        1. Collect all task gradients
        2. Assign tasks to groups
        3. Run Golden Pendulum QP within each group
        4. Assign golden-ratio weights across groups
        5. Combine and write final gradient

        Args:
            losses: Dict of task name -> scalar loss.
            model: Model to compute gradients for.
            shared_params: Optional shared parameter list.

        Returns:
            Dict of task name -> final weight.
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

        # Collect and normalize all task gradients
        G = _collect_task_gradients(losses, shared_params)
        g_norms = G.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G_hat = G / g_norms

        # Get task groups
        groups = self._ensure_groups(task_names, G_hat)

        # Filter to groups that have tasks present in current losses
        active_groups = []
        for g in groups:
            present = g.tasks & set(task_names)
            if present:
                active_groups.append((g, present))

        # Handle ungrouped tasks (put them in a catch-all)
        grouped_tasks = set()
        for _, present in active_groups:
            grouped_tasks |= present
        ungrouped = set(task_names) - grouped_tasks
        if ungrouped:
            active_groups.append((TaskGroup("_ungrouped", tasks=ungrouped), ungrouped))

        n_groups = len(active_groups)

        # Step 1: Golden-ratio weights across groups
        group_alpha = golden_ratio_weights(n_groups, device=device, dtype=G_hat.dtype)

        # Step 2: Within each group, run Golden Pendulum QP
        final_weights = {}

        for gi, (group, present_tasks) in enumerate(active_groups):
            group_task_list = sorted(present_tasks)
            group_indices = [task_names.index(t) for t in group_task_list]
            n_group = len(group_indices)

            if n_group == 1:
                # Single task in group gets the full group weight
                final_weights[group_task_list[0]] = group_alpha[gi].item()
                continue

            # Extract sub-Gram matrix for this group
            idx = torch.tensor(group_indices, device=device)
            G_sub = G_hat[idx]
            GTG_sub = G_sub @ G_sub.T

            # Solve intra-group QP
            group_lam = group.lam if group.lam is not None else self.lam
            alpha_golden_sub = golden_ratio_weights(n_group, device=device, dtype=GTG_sub.dtype)
            alpha_sub = _solve_golden_qp(
                GTG_sub, alpha_golden_sub, lam=group_lam, n_iter=self.n_iter,
                min_weight_fraction=self.min_weight_fraction,
            )

            # Assign final weight = group_weight * intra_group_weight
            for ti, task_name in enumerate(group_task_list):
                final_weights[task_name] = group_alpha[gi].item() * alpha_sub[ti].item()

        # Normalize final weights to sum to 1
        total_w = sum(final_weights.values())
        if total_w > 0:
            final_weights = {k: v / total_w for k, v in final_weights.items()}

        # Build final gradient using the hierarchical weights
        alpha_tensor = torch.tensor(
            [final_weights[t] for t in task_names], device=device, dtype=G_hat.dtype
        )

        # PCGrad on full gradient set
        if self.pcgrad:
            G_resolved = _pcgrad_resolve(G_hat, alpha_tensor)
        else:
            G_resolved = G_hat

        final_grad = (alpha_tensor.unsqueeze(1) * G_resolved).sum(dim=0)

        # Write back
        offset = 0
        for p in shared_params:
            p.grad = None
        for p in shared_params:
            numel = p.numel()
            p.grad = final_grad[offset : offset + numel].reshape(p.shape).clone()
            offset += numel

        self._weight_history.append(final_weights)
        self._step += 1
        return final_weights

    def add_task(self, task_name: str, group_name: str) -> None:
        """Add a task to a group dynamically (e.g., when a new phase starts)."""
        for g in self._groups:
            if g.name == group_name:
                g.tasks.add(task_name)
                return
        self._groups.append(TaskGroup(group_name, tasks={task_name}))

    def remove_task(self, task_name: str) -> None:
        """Remove a task from all groups."""
        for g in self._groups:
            g.tasks.discard(task_name)
