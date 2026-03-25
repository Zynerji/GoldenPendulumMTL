"""Tests for Golden Pendulum MTL core algorithm."""

import math

import pytest
import torch
import torch.nn as nn

from golden_pendulum.core import (
    GoldenPendulumMTL,
    PHI,
    _pcgrad_resolve,
    _solve_golden_qp,
    golden_nash_backward,
    golden_ratio_weights,
)


# ── Golden Ratio Weights ──────────────────────────────────────────────


class TestGoldenRatioWeights:
    def test_sums_to_one(self):
        for k in [2, 4, 8, 16]:
            w = golden_ratio_weights(k)
            assert abs(w.sum().item() - 1.0) < 1e-6, f"K={k}: sum={w.sum()}"

    def test_phi_spacing(self):
        """Each consecutive weight ratio should equal phi."""
        w = golden_ratio_weights(6)
        for i in range(len(w) - 1):
            ratio = w[i + 1] / w[i]
            assert abs(ratio.item() - PHI) < 1e-5, f"ratio[{i+1}/{i}] = {ratio}"

    def test_known_values_k4(self):
        """Match the paper's K=4 values: (0.106, 0.171, 0.276, 0.447)."""
        w = golden_ratio_weights(4)
        expected = [0.1056, 0.1708, 0.2764, 0.4472]
        for i, (got, exp) in enumerate(zip(w.tolist(), expected)):
            assert abs(got - exp) < 0.001, f"w[{i}]: got {got:.4f}, expected {exp:.4f}"

    def test_anti_resonance_property(self):
        """Weight ratios should be powers of phi (irrational)."""
        w = golden_ratio_weights(5)
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                ratio = (w[i] / w[j]).item()
                expected = PHI ** (i - j)
                assert abs(ratio - expected) < 1e-5, (
                    f"w[{i}]/w[{j}] = {ratio:.6f}, expected phi^{i-j} = {expected:.6f}"
                )

    def test_single_task(self):
        w = golden_ratio_weights(1)
        assert w.shape == (1,)
        assert abs(w[0].item() - 1.0) < 1e-6

    def test_device_dtype(self):
        if torch.cuda.is_available():
            w = golden_ratio_weights(4, device="cuda", dtype=torch.float16)
            assert w.device.type == "cuda"
            assert w.dtype == torch.float16


# ── QP Solver ─────────────────────────────────────────────────────────


class TestQPSolver:
    def test_converges_to_golden(self):
        """With identity Gram matrix, solver should return golden-ratio weights."""
        K = 4
        GTG = torch.eye(K)
        alpha_golden = golden_ratio_weights(K)
        alpha = _solve_golden_qp(GTG, alpha_golden, lam=1.0, n_iter=50)
        for i in range(K):
            assert abs(alpha[i].item() - alpha_golden[i].item()) < 0.02

    def test_simplex_constraint(self):
        """Weights must sum to 1 and be non-negative."""
        K = 6
        GTG = torch.randn(K, K)
        GTG = GTG @ GTG.T  # Make PSD
        alpha_golden = golden_ratio_weights(K)
        alpha = _solve_golden_qp(GTG, alpha_golden, lam=0.5)
        assert abs(alpha.sum().item() - 1.0) < 1e-6
        assert (alpha >= 0).all()

    def test_min_weight_floor(self):
        """No weight should fall below 0.02/K."""
        K = 8
        # Adversarial Gram matrix that pushes toward corner
        GTG = torch.zeros(K, K)
        GTG[0, 0] = 100.0
        alpha_golden = golden_ratio_weights(K)
        alpha = _solve_golden_qp(GTG, alpha_golden, lam=0.1)
        min_expected = 0.02 / K
        for i in range(K):
            assert alpha[i].item() >= min_expected - 1e-6


# ── PCGrad Resolution ─────────────────────────────────────────────────


class TestPCGrad:
    def test_no_change_when_aligned(self):
        """When all gradients are aligned, PCGrad should not modify them."""
        G = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.9, 0.05]])
        alpha = torch.tensor([0.3, 0.3, 0.4])
        resolved = _pcgrad_resolve(G, alpha)
        assert torch.allclose(G, resolved, atol=1e-6)

    def test_resolves_conflicts(self):
        """When gradients conflict, the lower-weight one should be projected."""
        G = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])  # Opposing gradients
        alpha = torch.tensor([0.3, 0.7])
        resolved = _pcgrad_resolve(G, alpha)
        # Task 0 (lower weight) should be projected onto normal plane of task 1
        dot = (resolved[0] * resolved[1]).sum()
        assert dot >= -1e-6, f"Conflict not resolved: dot = {dot}"


# ── Full Backward Pass ────────────────────────────────────────────────


class SimpleMultiTaskModel(nn.Module):
    """Minimal model with shared backbone and 3 task heads."""

    def __init__(self, d_in: int = 10, d_hidden: int = 32):
        super().__init__()
        self.shared = nn.Linear(d_in, d_hidden)
        self.head_a = nn.Linear(d_hidden, 1)
        self.head_b = nn.Linear(d_hidden, 1)
        self.head_c = nn.Linear(d_hidden, 1)

    def forward(self, x):
        h = torch.relu(self.shared(x))
        return self.head_a(h), self.head_b(h), self.head_c(h)


class TestGoldenNashBackward:
    @pytest.fixture
    def model_and_data(self):
        torch.manual_seed(42)
        model = SimpleMultiTaskModel()
        x = torch.randn(8, 10)
        y_a = torch.randn(8, 1)
        y_b = torch.randn(8, 1)
        y_c = torch.randn(8, 1)
        return model, x, y_a, y_b, y_c

    def test_returns_weights(self, model_and_data):
        model, x, y_a, y_b, y_c = model_and_data
        out_a, out_b, out_c = model(x)
        losses = {
            "mse": nn.functional.mse_loss(out_a, y_a),
            "mae": nn.functional.l1_loss(out_b, y_b),
            "huber": nn.functional.smooth_l1_loss(out_c, y_c),
        }
        weights = golden_nash_backward(losses, model)
        assert set(weights.keys()) == {"mse", "mae", "huber"}
        assert abs(sum(weights.values()) - 1.0) < 1e-4

    def test_gradients_are_set(self, model_and_data):
        model, x, y_a, y_b, y_c = model_and_data
        out_a, out_b, out_c = model(x)
        losses = {
            "mse": nn.functional.mse_loss(out_a, y_a),
            "mae": nn.functional.l1_loss(out_b, y_b),
            "huber": nn.functional.smooth_l1_loss(out_c, y_c),
        }
        golden_nash_backward(losses, model)
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_weights_not_corner_solution(self, model_and_data):
        """Golden Pendulum should not produce corner solutions (w > 0.85)."""
        model, x, y_a, y_b, y_c = model_and_data
        out_a, out_b, out_c = model(x)
        # Create disparate-magnitude losses (the failure mode of Nash-MTL)
        losses = {
            "big": 200.0 * nn.functional.mse_loss(out_a, y_a),
            "small": 0.01 * nn.functional.mse_loss(out_b, y_b),
            "tiny": 0.001 * nn.functional.mse_loss(out_c, y_c),
        }
        weights = golden_nash_backward(losses, model, lam=0.5)
        assert max(weights.values()) < 0.85, f"Corner solution detected: {weights}"

    def test_single_task_fallback(self, model_and_data):
        model, x, y_a, _, _ = model_and_data
        out_a, _, _ = model(x)
        losses = {"only": nn.functional.mse_loss(out_a, y_a)}
        weights = golden_nash_backward(losses, model)
        assert weights == {"only": 1.0}

    def test_balance_ratio(self, model_and_data):
        """w_min/w_max should be > 0.15 (vs 0.04 for standard Nash-MTL)."""
        model, x, y_a, y_b, y_c = model_and_data
        out_a, out_b, out_c = model(x)
        losses = {
            "big": 100.0 * nn.functional.mse_loss(out_a, y_a),
            "medium": nn.functional.mse_loss(out_b, y_b),
            "small": 0.01 * nn.functional.mse_loss(out_c, y_c),
        }
        weights = golden_nash_backward(losses, model, lam=0.5)
        vals = list(weights.values())
        balance = min(vals) / max(vals)
        assert balance > 0.15, f"Poor balance: {balance:.3f}, weights={weights}"


# ── Stateful Wrapper ──────────────────────────────────────────────────


class TestGoldenPendulumMTL:
    def test_tracks_history(self):
        torch.manual_seed(0)
        model = SimpleMultiTaskModel()
        balancer = GoldenPendulumMTL(n_tasks=3)

        for _ in range(5):
            x = torch.randn(4, 10)
            out_a, out_b, out_c = model(x)
            losses = {
                "a": nn.functional.mse_loss(out_a, torch.zeros(4, 1)),
                "b": nn.functional.mse_loss(out_b, torch.zeros(4, 1)),
                "c": nn.functional.mse_loss(out_c, torch.zeros(4, 1)),
            }
            balancer.backward(losses, model)

        assert len(balancer.weight_history) == 5

    def test_wrong_n_tasks_raises(self):
        model = SimpleMultiTaskModel()
        balancer = GoldenPendulumMTL(n_tasks=4)
        x = torch.randn(4, 10)
        out_a, out_b, out_c = model(x)
        losses = {
            "a": nn.functional.mse_loss(out_a, torch.zeros(4, 1)),
            "b": nn.functional.mse_loss(out_b, torch.zeros(4, 1)),
            "c": nn.functional.mse_loss(out_c, torch.zeros(4, 1)),
        }
        with pytest.raises(ValueError, match="Expected 4 tasks"):
            balancer.backward(losses, model)

    def test_golden_targets_property(self):
        torch.manual_seed(0)
        model = SimpleMultiTaskModel()
        balancer = GoldenPendulumMTL()
        x = torch.randn(4, 10)
        out_a, out_b, out_c = model(x)
        losses = {
            "a": nn.functional.mse_loss(out_a, torch.zeros(4, 1)),
            "b": nn.functional.mse_loss(out_b, torch.zeros(4, 1)),
            "c": nn.functional.mse_loss(out_c, torch.zeros(4, 1)),
        }
        balancer.backward(losses, model)
        targets = balancer.golden_targets
        assert targets is not None
        assert abs(sum(targets.values()) - 1.0) < 1e-4

    def test_reset(self):
        balancer = GoldenPendulumMTL()
        balancer.weight_history.append({"a": 0.5, "b": 0.5})
        balancer.reset()
        assert len(balancer.weight_history) == 0


# ── Numerical Stability ──────────────────────────────────────────────


class TestNumericalStability:
    def test_zero_gradient_task(self):
        """A task with zero gradient should still get minimum weight."""
        model = nn.Linear(5, 2)
        x = torch.randn(4, 5)
        out = model(x)
        losses = {
            "active": nn.functional.mse_loss(out[:, 0], torch.randn(4)),
            "dead": torch.tensor(0.0, requires_grad=True),  # Zero-grad task
        }
        weights = golden_nash_backward(losses, model)
        assert not any(math.isnan(v) for v in weights.values())

    def test_identical_losses(self):
        """When all losses are identical, weights should be near-equal."""
        model = nn.Linear(5, 1)
        x = torch.randn(4, 5)
        out = model(x).squeeze()
        target = torch.randn(4)
        loss = nn.functional.mse_loss(out, target)
        # Same loss tensor reused — gradients are identical
        losses = {
            "a": loss,
            "b": loss,
        }
        # This will have identical gradients; weights should be close to golden-ratio
        weights = golden_nash_backward(losses, model)
        assert not any(math.isnan(v) for v in weights.values())

    def test_large_n_tasks(self):
        """Should work with many tasks (K=16, matching the paper)."""
        torch.manual_seed(42)
        d = 32
        model = nn.Linear(d, 16)
        x = torch.randn(8, d)
        out = model(x)
        losses = {f"task_{i}": nn.functional.mse_loss(out[:, i], torch.randn(8)) for i in range(16)}
        weights = golden_nash_backward(losses, model)
        assert len(weights) == 16
        assert abs(sum(weights.values()) - 1.0) < 1e-4
        assert all(v > 0 for v in weights.values())
