"""Benchmark: Golden Pendulum vs Nash-MTL vs Equal Weights vs GradNorm.

Reproduces the paper's core finding: Nash-MTL produces corner solutions
when task losses have disparate magnitudes, while Golden Pendulum
maintains balanced gradient allocation.

Run: python benchmarks/compare_methods.py
"""

import math
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from golden_pendulum.core import golden_nash_backward, golden_ratio_weights


# ── Baseline Methods ──────────────────────────────────────────────────


def equal_weight_backward(losses: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
    """Standard equal-weight backward (the naive baseline)."""
    total = sum(losses.values())
    total.backward()
    n = len(losses)
    return {k: 1.0 / n for k in losses}


def nash_mtl_backward(losses: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, float]:
    """Standard Nash-MTL without golden-ratio regularization.

    This is the method from Navon et al. (2022) that produces corner solutions.
    Equivalent to Golden Pendulum with lam=0 and no PCGrad.
    """
    return golden_nash_backward(losses, model, lam=0.0, pcgrad=False)


def gradnorm_backward(
    losses: Dict[str, torch.Tensor], model: nn.Module, alpha: float = 1.5
) -> Dict[str, float]:
    """Simplified GradNorm (Chen et al., 2018).

    Uses gradient norm ratios to rebalance task weights.
    """
    task_names = list(losses.keys())
    n = len(task_names)
    shared_params = [p for p in model.parameters() if p.requires_grad]

    # Get gradient norms
    norms = []
    for i, (name, loss) in enumerate(losses.items()):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad_parts = []
        for p in shared_params:
            if p.grad is not None:
                grad_parts.append(p.grad.flatten())
            else:
                grad_parts.append(torch.zeros(p.numel(), device=p.device))
        norms.append(torch.cat(grad_parts).norm().item())

    # GradNorm weighting: inverse gradient norm
    mean_norm = sum(norms) / len(norms)
    weights_raw = [(mean_norm / (n + 1e-8)) ** alpha for n in norms]
    total_w = sum(weights_raw)
    weights = {name: w / total_w for name, w in zip(task_names, weights_raw)}

    # Recompute weighted backward
    model.zero_grad(set_to_none=True)
    total_loss = sum(weights[name] * loss for name, loss in losses.items())
    total_loss.backward()

    return weights


# ── Benchmark Model ───────────────────────────────────────────────────


class BenchmarkModel(nn.Module):
    """Multi-head model with intentionally disparate loss scales."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(32, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.head_a = nn.Linear(d_model, 1)   # MSE ~0.01
        self.head_b = nn.Linear(d_model, 1)   # BCE ~0.69
        self.head_c = nn.Linear(d_model, 1)   # MSE ~200 (ranking)
        self.head_d = nn.Linear(d_model, 16)  # InfoNCE ~0.05

    def forward(self, x):
        h = self.backbone(x)
        return self.head_a(h), self.head_b(h), self.head_c(h), self.head_d(h)


def make_losses(model, x, device):
    """Create losses with disparate magnitudes (paper's failure mode)."""
    out_a, out_b, out_c, out_d = model(x)
    y_a = torch.randn(x.shape[0], 1, device=device) * 0.01
    y_b = torch.randint(0, 2, (x.shape[0], 1), device=device).float()
    y_c = torch.randn(x.shape[0], 1, device=device) * 100.0

    embed = F.normalize(out_d, dim=1)
    sim = embed @ embed.T / 0.1
    nce_labels = torch.arange(x.shape[0], device=device)

    return {
        "horizon_returns": F.mse_loss(out_a, y_a),           # ~0.01
        "trade_quality": F.binary_cross_entropy_with_logits(out_b, y_b),  # ~0.69
        "ranking": F.mse_loss(out_c, y_c),                   # ~200
        "embedding": F.cross_entropy(sim, nce_labels),        # ~0.05
    }


def run_benchmark(n_steps: int = 200, seed: int = 42):
    """Run all methods and compare weight distributions."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    methods = {
        "Equal Weights": equal_weight_backward,
        "Nash-MTL": nash_mtl_backward,
        "GradNorm": gradnorm_backward,
        "Golden Pendulum": lambda losses, model: golden_nash_backward(losses, model, lam=0.5),
    }

    results = {name: {"weights": [], "time_ms": 0.0} for name in methods}

    for method_name, backward_fn in methods.items():
        torch.manual_seed(seed)
        model = BenchmarkModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        t0 = time.time()
        for step in range(n_steps):
            optimizer.zero_grad()
            x = torch.randn(32, 32, device=device)
            losses = make_losses(model, x, device)
            weights = backward_fn(losses, model)
            optimizer.step()
            results[method_name]["weights"].append(weights)
        results[method_name]["time_ms"] = (time.time() - t0) * 1000

    # Print comparison
    print("=" * 80)
    print(f"BENCHMARK: Golden Pendulum vs Baselines ({n_steps} steps, {device})")
    print("=" * 80)
    print()

    golden_targets = golden_ratio_weights(4).tolist()
    task_names = list(results["Equal Weights"]["weights"][0].keys())

    print(f"Golden-ratio targets: {', '.join(f'{t:.4f}' for t in golden_targets)}")
    print()

    for method_name, data in results.items():
        # Use last 50% of weights for steady-state analysis
        steady = data["weights"][n_steps // 2:]
        mean_w = {k: sum(w[k] for w in steady) / len(steady) for k in task_names}
        vals = list(mean_w.values())
        balance = min(vals) / max(vals)
        max_w = max(vals)

        print(f"  {method_name:20s} | balance={balance:.3f} | max_weight={max_w:.3f} | "
              f"time={data['time_ms']:.0f}ms")
        print(f"  {'':20s} | weights: {', '.join(f'{k}={v:.4f}' for k, v in mean_w.items())}")

        if max_w > 0.85:
            print(f"  {'':20s} | ** CORNER SOLUTION DETECTED **")
        print()

    print("Paper claims:")
    print("  - Nash-MTL: balance ~0.04, max_weight ~0.89 (corner solution)")
    print("  - Golden Pendulum: balance ~0.24, max_weight ~0.45 (anti-resonant)")


if __name__ == "__main__":
    run_benchmark()
