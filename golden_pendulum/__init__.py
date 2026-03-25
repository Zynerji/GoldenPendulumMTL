"""Golden Pendulum MTL — Anti-resonance equilibria for multi-task gradient balancing.

Replace Nash-MTL corner solutions with golden-ratio (phi) weights that prevent
harmonic lock-in between competing task gradients.

Usage::

    from golden_pendulum import GoldenPendulumMTL

    balancer = GoldenPendulumMTL(n_tasks=4)

    # In your training loop:
    losses = {"task_a": loss_a, "task_b": loss_b, "task_c": loss_c, "task_d": loss_d}
    balancer.backward(losses, model)
    optimizer.step()

Reference:
    Knopp, C. (2026). "Golden Pendulum Multi-Task Learning: Anti-Resonance Equilibria
    for Gradient Balancing in Multi-Head Transformer Training."
"""

from golden_pendulum.core import (
    GoldenPendulumMTL,
    golden_ratio_weights,
    golden_nash_backward,
)
from golden_pendulum.callbacks import (
    GoldenPendulumCallback,
    WeightLogger,
)

__version__ = "0.2.0"
__author__ = "Christian Knopp"
__email__ = "cknopp@gmail.com"

__all__ = [
    "GoldenPendulumMTL",
    "golden_ratio_weights",
    "golden_nash_backward",
    "GoldenPendulumCallback",
    "WeightLogger",
]

# Pro tier — lazy import to avoid hard dependency
def __getattr__(name):
    if name == "pro":
        from golden_pendulum import pro as _pro
        return _pro
    raise AttributeError(f"module 'golden_pendulum' has no attribute {name!r}")
