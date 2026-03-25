"""Example: Golden Pendulum Pro features.

Demonstrates:
1. AdaptiveLambda — auto-tune regularization strength
2. DynamicK — hierarchical task grouping
3. CurriculumScheduler — multi-phase training with backbone freeze
4. DiagnosticsEngine — real-time conflict analysis
5. Presets — battle-tested configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from golden_pendulum.pro import (
    AdaptiveLambda,
    CurriculumScheduler,
    DiagnosticsEngine,
    DynamicK,
    Phase,
    TaskGroup,
    get_preset,
    list_presets,
)


class MultiHeadModel(nn.Module):
    def __init__(self, d_in=16, d_model=64, n_heads=8):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU())
        self.heads = nn.ModuleDict({f"task_{i}": nn.Linear(d_model, 1) for i in range(n_heads)})

    def forward(self, x):
        h = self.backbone(x)
        return {name: head(h) for name, head in self.heads.items()}


def demo_adaptive_lambda():
    """Auto-tune lambda based on gradient conflict severity."""
    print("\n" + "=" * 60)
    print("1. AdaptiveLambda — Auto-tuning regularization")
    print("=" * 60)

    torch.manual_seed(42)
    model = MultiHeadModel(n_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    adaptive = AdaptiveLambda(lam_init=0.5, lam_min=0.05, lam_max=2.0)

    for step in range(200):
        optimizer.zero_grad()
        x = torch.randn(16, 16)
        outputs = model(x)

        # Create disparate-magnitude losses
        losses = {
            "task_0": 0.01 * F.mse_loss(outputs["task_0"], torch.randn(16, 1)),
            "task_1": 0.69 * F.mse_loss(outputs["task_1"], torch.randn(16, 1)),
            "task_2": 200.0 * F.mse_loss(outputs["task_2"], torch.randn(16, 1)),
            "task_3": 0.05 * F.mse_loss(outputs["task_3"], torch.randn(16, 1)),
        }

        weights = adaptive.backward(losses, model)
        optimizer.step()

        if step % 50 == 0:
            print(f"  Step {step:3d} | lam={adaptive.current_lam:.3f} | "
                  f"conflict={adaptive.conflict_ratio:.3f} | "
                  f"loss_ratio={adaptive.loss_ratio:.0f}")

    print(f"\n  Final lambda: {adaptive.current_lam:.3f} (started at 0.5)")


def demo_dynamic_k():
    """Hierarchical task grouping with golden-ratio weights."""
    print("\n" + "=" * 60)
    print("2. DynamicK — Hierarchical task grouping")
    print("=" * 60)

    torch.manual_seed(42)
    model = MultiHeadModel(n_heads=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Group tasks by function (like the paper's Phase A/B/C)
    dk = DynamicK(groups=[
        TaskGroup("ranking", tasks={"task_0", "task_1", "task_2"}),
        TaskGroup("risk", tasks={"task_3", "task_4", "task_5"}),
        TaskGroup("meta", tasks={"task_6", "task_7"}),
    ])

    for step in range(100):
        optimizer.zero_grad()
        x = torch.randn(16, 16)
        outputs = model(x)
        losses = {k: F.mse_loss(v, torch.randn_like(v)) for k, v in outputs.items()}
        weights = dk.backward(losses, model)
        optimizer.step()

        if step == 0:
            print(f"  Groups: {[g.name for g in dk.current_groups]}")
            print(f"  Initial weights: {', '.join(f'{k}={v:.4f}' for k, v in sorted(weights.items()))}")

    print(f"  Final weights: {', '.join(f'{k}={v:.4f}' for k, v in sorted(weights.items()))}")


def demo_curriculum():
    """Multi-phase training with backbone freeze control."""
    print("\n" + "=" * 60)
    print("3. CurriculumScheduler — Multi-phase training")
    print("=" * 60)

    torch.manual_seed(42)
    model = MultiHeadModel(n_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    curriculum = CurriculumScheduler(
        phases=[
            Phase("A_ranking", tasks={"task_0", "task_1"}, steps=50,
                  freeze_backbone=False, lam=0.5, lr=5e-5, warmup_steps=5),
            Phase("B_risk", tasks={"task_2", "task_3"}, steps=30,
                  freeze_backbone=True, lam=0.3, lr=1e-4),
        ],
        backbone_params=lambda m: m.backbone.parameters(),
    )

    print(f"  Total steps: {curriculum.total_steps}")

    while not curriculum.is_complete:
        optimizer.zero_grad()
        x = torch.randn(16, 16)
        outputs = model(x)
        all_losses = {k: F.mse_loss(v, torch.randn_like(v)) for k, v in outputs.items()}

        weights = curriculum.backward(all_losses, model)
        optimizer.step()

        step = curriculum.global_step
        if step % 20 == 0 or curriculum.step(model):
            phase = curriculum.current_phase if not curriculum.is_complete else "DONE"
            phase_name = phase.name if hasattr(phase, "name") else phase
            print(f"  Step {step:3d} | phase={phase_name} | "
                  f"lr={curriculum.current_lr if not curriculum.is_complete else 0:.6f} | "
                  f"weights={weights}")
        else:
            pass  # step() already called above in the if condition

    print(f"  Curriculum complete after {curriculum.global_step} steps")


def demo_diagnostics():
    """Real-time gradient conflict analysis."""
    print("\n" + "=" * 60)
    print("4. DiagnosticsEngine — Conflict analysis")
    print("=" * 60)

    torch.manual_seed(42)
    model = MultiHeadModel(n_heads=4)
    diag = DiagnosticsEngine(alert_norm_ratio=10.0)

    x = torch.randn(16, 16)
    outputs = model(x)
    losses = {
        "task_0": 0.01 * F.mse_loss(outputs["task_0"], torch.randn(16, 1)),
        "task_1": F.mse_loss(outputs["task_1"], torch.randn(16, 1)),
        "task_2": 200.0 * F.mse_loss(outputs["task_2"], torch.randn(16, 1)),
        "task_3": 0.05 * F.mse_loss(outputs["task_3"], torch.randn(16, 1)),
    }

    report = diag.analyze(losses, model)

    print(f"  Conflict ratio: {report.conflict_ratio:.3f}")
    print(f"  Norm ratio: {report.norm_ratio:.1f}x")
    print(f"  Conflicting pairs: {report.conflict_pairs}")
    print(f"  Gradient norms: {', '.join(f'{k}={v:.4f}' for k, v in report.gradient_norms.items())}")

    if diag.alerts:
        print(f"  ALERTS:")
        for alert in diag.alerts:
            print(f"    ! {alert}")


def demo_presets():
    """Battle-tested configurations for common use cases."""
    print("\n" + "=" * 60)
    print("5. Presets — Pre-configured settings")
    print("=" * 60)

    print("\n  Available presets:")
    for name, desc in list_presets().items():
        print(f"    {name}: {desc[:80]}...")

    preset = get_preset("finance_4phase")
    print(f"\n  finance_4phase:")
    print(f"    Total steps: {preset.total_steps}")
    for p in preset.phases:
        print(f"    {p.name}: {len(p.tasks)} tasks, {p.steps} steps, "
              f"backbone={'frozen' if p.freeze_backbone else 'unfrozen'}, lam={p.lam}")


if __name__ == "__main__":
    demo_adaptive_lambda()
    demo_dynamic_k()
    demo_curriculum()
    demo_diagnostics()
    demo_presets()
    print("\n" + "=" * 60)
    print("All Pro features demonstrated successfully!")
    print("=" * 60)
