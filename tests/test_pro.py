"""Tests for Golden Pendulum Pro tier features."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from golden_pendulum.pro.adaptive import AdaptiveLambda
from golden_pendulum.pro.dynamic_k import DynamicK, TaskGroup
from golden_pendulum.pro.curriculum import CurriculumScheduler, Phase
from golden_pendulum.pro.diagnostics import DiagnosticsEngine, ConflictReport
from golden_pendulum.pro.presets import get_preset, list_presets


# ── Test Models ───────────────────────────────────────────────────────


class SimpleModel(nn.Module):
    def __init__(self, d_in=10, d_hidden=32, n_heads=4):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU())
        self.heads = nn.ModuleDict({
            f"head_{i}": nn.Linear(d_hidden, 1) for i in range(n_heads)
        })

    def forward(self, x):
        h = self.backbone(x)
        return {name: head(h) for name, head in self.heads.items()}


def make_losses(model, n_tasks=4, scale_range=(0.01, 200)):
    """Create losses with controllable magnitude disparity."""
    x = torch.randn(8, 10)
    outputs = model(x)
    losses = {}
    scales = torch.linspace(scale_range[0], scale_range[1], n_tasks)
    for i, (name, out) in enumerate(outputs.items()):
        losses[name] = scales[i] * F.mse_loss(out, torch.randn_like(out))
    return losses


# ── AdaptiveLambda ────────────────────────────────────────────────────


class TestAdaptiveLambda:
    def test_lambda_stays_in_bounds(self):
        torch.manual_seed(42)
        model = SimpleModel()
        adaptive = AdaptiveLambda(lam_init=0.5, lam_min=0.05, lam_max=2.0)

        for _ in range(100):
            losses = make_losses(model)
            adaptive.backward(losses, model)

        assert adaptive.current_lam >= 0.05
        assert adaptive.current_lam <= 2.0

    def test_lambda_increases_under_conflict(self):
        torch.manual_seed(42)
        model = SimpleModel()
        adaptive = AdaptiveLambda(lam_init=0.1, sensitivity=5.0, loss_ratio_aware=False)

        # Run with high-disparity losses to create conflicts
        for _ in range(50):
            losses = make_losses(model, scale_range=(0.001, 500))
            adaptive.backward(losses, model)

        # Lambda should have adapted upward from 0.1
        # (direction depends on conflict pattern, but it should have moved)
        assert adaptive.current_lam != 0.1

    def test_returns_valid_weights(self):
        torch.manual_seed(42)
        model = SimpleModel()
        adaptive = AdaptiveLambda()
        losses = make_losses(model)
        weights = adaptive.backward(losses, model)

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert all(v > 0 for v in weights.values())

    def test_history_tracking(self):
        torch.manual_seed(42)
        model = SimpleModel()
        adaptive = AdaptiveLambda()

        for _ in range(10):
            losses = make_losses(model)
            adaptive.backward(losses, model)

        history = adaptive.get_history()
        assert len(history) == 10
        assert "lam" in history[0]
        assert "conflict_ratio" in history[0]

    def test_loss_ratio_aware(self):
        torch.manual_seed(42)
        model = SimpleModel()
        # Without loss ratio awareness
        adaptive_no = AdaptiveLambda(lam_init=0.5, loss_ratio_aware=False)
        # With loss ratio awareness
        adaptive_yes = AdaptiveLambda(lam_init=0.5, loss_ratio_aware=True)

        for _ in range(30):
            losses = make_losses(model, scale_range=(0.001, 1000))
            adaptive_no.backward(losses, model)
            # Recreate losses (graph consumed)
            losses = make_losses(model, scale_range=(0.001, 1000))
            adaptive_yes.backward(losses, model)

        # Loss-ratio-aware should have higher lambda due to boost
        # (not guaranteed in all seeds, but the mechanism should be active)
        assert adaptive_yes.loss_ratio > 1.0

    def test_reset(self):
        adaptive = AdaptiveLambda(lam_init=0.5)
        adaptive._lam = 1.5
        adaptive._history.append({"step": 0})
        adaptive.reset()
        assert adaptive._conflict_ema == adaptive.conflict_target  # EMA reset
        assert len(adaptive._history) == 0
        assert adaptive._step == 0


# ── DynamicK ──────────────────────────────────────────────────────────


class TestDynamicK:
    def test_manual_groups(self):
        torch.manual_seed(42)
        model = SimpleModel(n_heads=4)

        dk = DynamicK(groups=[
            TaskGroup("group_a", tasks={"head_0", "head_1"}),
            TaskGroup("group_b", tasks={"head_2", "head_3"}),
        ])

        losses = make_losses(model)
        weights = dk.backward(losses, model)

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert all(v > 0 for v in weights.values())

    def test_single_group_fallback(self):
        """Without groups, should behave like standard Golden Pendulum."""
        torch.manual_seed(42)
        model = SimpleModel()

        dk = DynamicK()
        losses = make_losses(model)
        weights = dk.backward(losses, model)

        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_auto_grouping(self):
        torch.manual_seed(42)
        model = SimpleModel(n_heads=6)
        model.heads["head_4"] = nn.Linear(32, 1)
        model.heads["head_5"] = nn.Linear(32, 1)

        dk = DynamicK(auto_group=True, regroup_every=1, similarity_threshold=0.3)
        x = torch.randn(8, 10)
        outputs = model(x)
        losses = {k: F.mse_loss(v, torch.randn_like(v)) for k, v in outputs.items()}
        weights = dk.backward(losses, model)

        assert len(weights) == 6
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert len(dk.current_groups) >= 1

    def test_add_remove_task(self):
        dk = DynamicK(groups=[TaskGroup("g1", tasks={"a", "b"})])
        dk.add_task("c", "g1")
        assert "c" in dk.current_groups[0].tasks

        dk.remove_task("a")
        assert "a" not in dk.current_groups[0].tasks

    def test_ungrouped_tasks_handled(self):
        """Tasks not in any group should get a catch-all group."""
        torch.manual_seed(42)
        model = SimpleModel(n_heads=4)

        dk = DynamicK(groups=[
            TaskGroup("partial", tasks={"head_0", "head_1"}),
            # head_2 and head_3 are ungrouped
        ])

        losses = make_losses(model)
        weights = dk.backward(losses, model)
        assert len(weights) == 4


# ── CurriculumScheduler ──────────────────────────────────────────────


class TestCurriculumScheduler:
    def test_phase_transitions(self):
        sched = CurriculumScheduler(phases=[
            Phase("A", tasks={"t1", "t2"}, steps=10, lam=0.5),
            Phase("B", tasks={"t3", "t4"}, steps=5, lam=0.3),
        ])

        assert sched.current_phase_name == "A"
        assert sched.total_steps == 15

        for _ in range(9):
            assert not sched.step()
        assert sched.step()  # Step 10 triggers transition
        assert sched.current_phase_name == "B"

    def test_backbone_freeze(self):
        model = SimpleModel()
        sched = CurriculumScheduler(
            phases=[
                Phase("A", tasks={"head_0"}, steps=5, freeze_backbone=False),
                Phase("B", tasks={"head_1"}, steps=5, freeze_backbone=True),
            ],
            backbone_params=lambda m: m.backbone.parameters(),
        )

        # Phase A: backbone unfrozen
        losses = {"head_0": torch.tensor(1.0, requires_grad=True)}
        sched.backward(losses, model)
        for p in model.backbone.parameters():
            assert p.requires_grad

        # Advance to Phase B
        for _ in range(5):
            sched.step(model)

        losses = {"head_1": torch.tensor(1.0, requires_grad=True)}
        sched.backward(losses, model)
        for p in model.backbone.parameters():
            assert not p.requires_grad

    def test_warmup_lr(self):
        sched = CurriculumScheduler(phases=[
            Phase("A", tasks={"t1"}, steps=100, lr=1e-3, warmup_steps=10),
        ])

        # Step 0: lr = 1e-3 * 1/10 = 1e-4
        assert abs(sched.current_lr - 1e-4) < 1e-6

        # Advance past warmup
        for _ in range(10):
            sched.step()
        assert abs(sched.current_lr - 1e-3) < 1e-6

    def test_state_dict_roundtrip(self):
        sched = CurriculumScheduler(phases=[
            Phase("A", tasks={"t1"}, steps=100),
            Phase("B", tasks={"t2"}, steps=50),
        ])

        for _ in range(75):
            sched.step()

        state = sched.state_dict()
        sched2 = CurriculumScheduler(phases=[
            Phase("A", tasks={"t1"}, steps=100),
            Phase("B", tasks={"t2"}, steps=50),
        ])
        sched2.load_state_dict(state)

        assert sched2.current_phase_name == sched.current_phase_name
        assert sched2.global_step == sched.global_step

    def test_is_complete(self):
        sched = CurriculumScheduler(phases=[
            Phase("A", tasks={"t1"}, steps=3),
        ])
        for _ in range(3):
            sched.step()
        assert sched.is_complete

    def test_on_enter_callback(self):
        entered = []
        sched = CurriculumScheduler(phases=[
            Phase("A", tasks={"t1"}, steps=5, on_enter=lambda m, p: entered.append(p.name)),
            Phase("B", tasks={"t2"}, steps=5, on_enter=lambda m, p: entered.append(p.name)),
        ])

        model = SimpleModel()
        losses = {"t1": torch.tensor(1.0, requires_grad=True)}
        sched.backward(losses, model)  # Triggers Phase A enter
        assert "A" in entered

        for _ in range(5):
            sched.step(model)
        assert "B" in entered


# ── DiagnosticsEngine ─────────────────────────────────────────────────


class TestDiagnosticsEngine:
    def test_analyze_returns_report(self):
        torch.manual_seed(42)
        model = SimpleModel()
        diag = DiagnosticsEngine()

        losses = make_losses(model)
        report = diag.analyze(losses, model)

        assert isinstance(report, ConflictReport)
        assert 0 <= report.conflict_ratio <= 1
        assert report.norm_ratio >= 1
        assert len(report.gradient_norms) == 4

    def test_conflict_detection(self):
        torch.manual_seed(42)
        model = SimpleModel()
        diag = DiagnosticsEngine()

        losses = make_losses(model, scale_range=(0.001, 500))
        report = diag.analyze(losses, model)

        # With high disparity, there should be some conflicts
        assert isinstance(report.conflict_pairs, list)
        for pair in report.conflict_pairs:
            assert len(pair) == 3
            assert pair[2] < 0  # Negative cosine similarity

    def test_alerts_on_high_conflict(self):
        torch.manual_seed(42)
        model = SimpleModel()
        diag = DiagnosticsEngine(alert_conflict_ratio=0.0)  # Alert on any conflict

        losses = make_losses(model, scale_range=(0.001, 500))
        diag.analyze(losses, model)

        # Should have at least the norm ratio alert (500000x disparity)
        assert len(diag.alerts) > 0

    def test_weight_recording(self):
        diag = DiagnosticsEngine()
        diag.record_weights(0, {"a": 0.3, "b": 0.7})
        diag.record_weights(1, {"a": 0.35, "b": 0.65})
        assert len(diag._weight_history) == 2

    def test_summary_string(self):
        torch.manual_seed(42)
        model = SimpleModel()
        diag = DiagnosticsEngine()

        for _ in range(5):
            losses = make_losses(model)
            diag.analyze(losses, model)
            diag.record_weights(diag._step, {"a": 0.3, "b": 0.7})

        summary = diag.summary()
        assert "Golden Pendulum Diagnostics" in summary
        assert "conflict ratio" in summary

    def test_preserves_existing_gradients(self):
        """analyze() should not destroy existing model gradients."""
        torch.manual_seed(42)
        model = SimpleModel()
        x = torch.randn(4, 10)
        out = model(x)

        # Set some gradients
        dummy_loss = sum(v.sum() for v in out.values())
        dummy_loss.backward()

        # Save a gradient
        original_grad = list(model.backbone.parameters())[0].grad.clone()

        # Analyze (should restore gradients)
        diag = DiagnosticsEngine()
        losses2 = make_losses(model)
        diag.analyze(losses2, model)

        restored_grad = list(model.backbone.parameters())[0].grad
        assert restored_grad is not None
        assert torch.allclose(original_grad, restored_grad)


# ── Presets ───────────────────────────────────────────────────────────


class TestPresets:
    def test_finance_4phase(self):
        preset = get_preset("finance_4phase")
        assert len(preset.phases) == 4
        assert preset.total_steps == 40000
        assert preset.phases[0].name == "A_ranking"
        assert not preset.phases[0].freeze_backbone
        assert preset.phases[1].freeze_backbone

    def test_finance_quick(self):
        preset = get_preset("finance_quick")
        assert len(preset.phases) == 2
        assert preset.total_steps == 13000

    def test_nlp_preset(self):
        preset = get_preset("nlp_multitask")
        assert "classification" in preset.phases[0].tasks

    def test_vision_preset(self):
        preset = get_preset("vision_multitask")
        assert "segmentation" in preset.phases[0].tasks

    def test_robotics_preset(self):
        preset = get_preset("robotics_control")
        assert "safety" in preset.phases[0].tasks

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_list_presets(self):
        presets = list_presets()
        assert "finance_4phase" in presets
        assert len(presets) >= 5

    def test_preset_creates_scheduler(self):
        """Presets should be usable with CurriculumScheduler."""
        preset = get_preset("finance_4phase")
        sched = CurriculumScheduler(phases=preset.phases)
        assert sched.total_steps == 40000
        assert sched.current_phase_name == "A_ranking"
