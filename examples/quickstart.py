"""Quickstart: Golden Pendulum MTL in 30 lines.

Demonstrates the core API on a simple multi-task regression problem.
"""

import torch
import torch.nn as nn

from golden_pendulum import GoldenPendulumMTL

# Simple shared-backbone + multi-head model
class MultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.head_regression = nn.Linear(32, 1)
        self.head_classification = nn.Linear(32, 1)
        self.head_ranking = nn.Linear(32, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_regression(h), self.head_classification(h), self.head_ranking(h)


torch.manual_seed(42)
model = MultiTaskNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
balancer = GoldenPendulumMTL(n_tasks=3, lam=0.5)

# Synthetic data
x = torch.randn(64, 20)
y_reg = torch.randn(64, 1)
y_cls = torch.randint(0, 2, (64, 1)).float()
y_rank = torch.randn(64, 1)

for step in range(200):
    optimizer.zero_grad()

    reg_out, cls_out, rank_out = model(x)
    losses = {
        "regression": nn.functional.mse_loss(reg_out, y_reg),
        "classification": nn.functional.binary_cross_entropy_with_logits(cls_out, y_cls),
        "ranking": 100.0 * nn.functional.mse_loss(rank_out, y_rank),  # 100x larger!
    }

    weights = balancer.backward(losses, model)
    optimizer.step()

    if step % 50 == 0:
        total = sum(v.item() for v in losses.values())
        print(f"Step {step:3d} | loss={total:.4f} | weights={weights} | balance={balancer.weight_balance_ratio:.3f}")

print(f"\nFinal balance ratio: {balancer.weight_balance_ratio:.3f}")
print(f"Mean weights (last 100): {balancer.mean_weights(100)}")
print(f"Golden targets: {balancer.golden_targets}")
