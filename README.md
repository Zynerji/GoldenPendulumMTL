# Golden Pendulum MTL

[![PyPI version](https://img.shields.io/pypi/v/golden-pendulum-mtl)](https://pypi.org/project/golden-pendulum-mtl/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/github/actions/workflow/status/Zynerji/GoldenPendulumMTL/ci.yml?label=tests)](https://github.com/Zynerji/GoldenPendulumMTL/actions)

**Anti-resonance equilibria for gradient balancing in multi-task learning.**

Replace Nash-MTL corner solutions with golden-ratio (&phi;) weights that prevent harmonic lock-in between competing task gradients.

> Standard Nash-MTL allocated 89% of gradient bandwidth to one task while starving others at 3.7%. Golden Pendulum achieves w<sub>min</sub>/w<sub>max</sub> = 0.24 while maintaining Pareto optimality.

## The Problem

Multi-task gradient methods (MGDA, Nash-MTL, PCGrad) suffer from **corner solutions**: when task losses have disparate magnitudes, the optimizer converges to simplex vertices where one task dominates.

| Method | w<sub>min</sub>/w<sub>max</sub> | Max Weight | Corner? |
|--------|:---:|:---:|:---:|
| Equal Weights | 1.00 | 0.25 | No (but ignores conflicts) |
| Nash-MTL | 0.04 | 0.89 | **Yes** |
| GradNorm | ~0.10 | ~0.60 | Sometimes |
| **Golden Pendulum** | **0.24** | **0.45** | **No** |

## The Solution

Golden Pendulum MTL derives from the physics of coupled wave oscillators that the stable equilibrium lies at **golden-ratio-spaced points** (&phi; = (1+&radic;5)/2 &asymp; 1.618), not at simplex corners. These weights are maximally incommensurate &mdash; no pair has a rational ratio &mdash; preventing the harmonic resonances that cause lock-in.

**Algorithm (3 lines to integrate):**

```python
from golden_pendulum import GoldenPendulumMTL

balancer = GoldenPendulumMTL(n_tasks=4, lam=0.5)

# In your training loop (replaces loss.backward()):
weights = balancer.backward(losses, model)
optimizer.step()
```

## Installation

```bash
pip install golden-pendulum-mtl
```

Or from source:
```bash
git clone https://github.com/Zynerji/GoldenPendulumMTL.git
cd GoldenPendulumMTL
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
import torch.nn as nn
from golden_pendulum import GoldenPendulumMTL

# Your multi-task model
model = YourMultiHeadModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
balancer = GoldenPendulumMTL(n_tasks=3, lam=0.5)

for batch in dataloader:
    optimizer.zero_grad()

    # Compute per-task losses (can have 300x magnitude disparity!)
    losses = {
        "ranking": ranking_loss,       # ~200
        "classification": cls_loss,    # ~0.69
        "regression": reg_loss,        # ~0.01
    }

    # Golden Pendulum backward (replaces loss.backward())
    weights = balancer.backward(losses, model)
    optimizer.step()

    # Monitor balance (should be >0.20, not 0.04 like Nash-MTL)
    print(f"Balance: {balancer.weight_balance_ratio:.3f}")
```

## How It Works

### Algorithm 1: Golden Pendulum MTL

1. **Compute per-task gradients** g<sub>k</sub> = &nabla;<sub>&theta;</sub> L<sub>k</sub>
2. **Normalize** each gradient: &gcirc;<sub>k</sub> = g<sub>k</sub> / ||g<sub>k</sub>||<sub>2</sub> (removes magnitude disparity)
3. **Compute scale-free Gram matrix**: &Gcirc;<sup>T</sup>&Gcirc;
4. **Solve golden-ratio QP** (25 iterations of projected gradient descent):
   ```
   min  alpha^T G_hat^T G_hat alpha  +  lambda * ||alpha - alpha_golden||_1
   ```
   where alpha<sub>golden</sub> = [&phi;<sup>k-1</sup> / &Sigma;&phi;<sup>j-1</sup>] are golden-ratio target weights
5. **PCGrad conflict resolution** on normalized gradients
6. **Set gradient** = weighted sum of conflict-resolved gradients

### Why Golden Ratio?

The golden ratio &phi; is the **most irrational number**: its continued fraction [1;1,1,1,...] converges slower than any other. This means:
- No pair of &phi;-spaced weights has a rational ratio
- Gradient updates are **quasiperiodic** (not periodic)
- No task can "pump" energy from others through resonance

This is the same reason &phi; appears in phyllotaxis (sunflower seeds), quasicrystals, and the KAM theorem for orbital stability.

### Golden-Ratio Weights for K Tasks

| K | Weights | w<sub>min</sub>/w<sub>max</sub> |
|---|---------|:---:|
| 2 | (0.382, 0.618) | 0.618 |
| 3 | (0.186, 0.302, 0.488) | 0.382 |
| 4 | (0.106, 0.171, 0.276, 0.447) | 0.237 |
| 8 | (0.019, 0.031, ..., 0.277) | 0.069 |
| 16 | (0.001, 0.001, ..., 0.172) | 0.004 |

## Framework Integration

### PyTorch Lightning

```python
from golden_pendulum import GoldenPendulumCallback

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.golden = GoldenPendulumCallback(lam=0.5)

    def training_step(self, batch, batch_idx):
        losses = {"task_a": loss_a, "task_b": loss_b}
        opt = self.optimizers()
        opt.zero_grad()
        weights = self.golden.on_train_batch(losses, self, batch_idx)
        opt.step()
```

### Hugging Face Transformers

```python
from golden_pendulum import GoldenPendulumMTL

balancer = GoldenPendulumMTL(n_tasks=3, lam=0.5)

# In your custom Trainer.training_step:
weights = balancer.backward(losses, model)
```

### Weight Logging

```python
from golden_pendulum import GoldenPendulumMTL, WeightLogger

logger = WeightLogger(log_file="weights.jsonl", log_every=100)
balancer = GoldenPendulumMTL(n_tasks=4)

for step, batch in enumerate(loader):
    weights = balancer.backward(losses, model)
    logger.log(step, weights)
```

## API Reference

### `GoldenPendulumMTL(n_tasks, lam, n_iter, min_weight_fraction, pcgrad)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_tasks` | 0 | Expected number of tasks (0 = any) |
| `lam` | 0.5 | Golden-ratio regularization strength |
| `n_iter` | 25 | QP solver iterations |
| `min_weight_fraction` | 0.02 | Minimum weight = fraction / K |
| `pcgrad` | True | Enable PCGrad conflict resolution |

**Methods:**
- `backward(losses, model)` &rarr; `Dict[str, float]` task weights
- `weight_balance_ratio` &rarr; w<sub>min</sub>/w<sub>max</sub>
- `mean_weights(last_n)` &rarr; mean weights over last N steps
- `golden_targets` &rarr; target golden-ratio weights

### `golden_nash_backward(losses, model, lam, n_iter, min_weight_fraction, pcgrad)`

Functional API &mdash; same algorithm, no state tracking.

### `golden_ratio_weights(n_tasks)`

Returns the &phi;-spaced target weights for K tasks.

## Pro Features

Advanced capabilities for production multi-task training.

### AdaptiveLambda — Auto-tune regularization

No more manual lambda tuning. Adapts based on real-time gradient conflict severity and loss magnitude disparity.

```python
from golden_pendulum.pro import AdaptiveLambda

adaptive = AdaptiveLambda(lam_init=0.5, lam_min=0.05, lam_max=2.0)

for step, batch in enumerate(loader):
    optimizer.zero_grad()
    losses = compute_losses(model, batch)
    weights = adaptive.backward(losses, model)
    optimizer.step()
    # Lambda auto-adjusts: high conflict -> stronger regularization
```

### CurriculumScheduler — Multi-phase training

Manages Phase A/B/C/D training with automatic backbone freeze/unfreeze.

```python
from golden_pendulum.pro import CurriculumScheduler, Phase

curriculum = CurriculumScheduler(
    phases=[
        Phase("A_ranking", tasks={"returns", "rank", "quality", "embed"},
              steps=15000, freeze_backbone=False, lam=0.5, lr=5e-5),
        Phase("B_risk", tasks={"vol", "mae", "kelly", "risk"},
              steps=10000, freeze_backbone=True, lam=0.3, lr=1e-4),
        Phase("C_meta", tasks={"regime", "calibration", "confidence"},
              steps=10000, freeze_backbone=True, lam=0.3, lr=1e-4),
    ],
    backbone_params=lambda model: model.backbone.parameters(),
)

while not curriculum.is_complete:
    weights = curriculum.backward(all_losses, model)
    optimizer.step()
    curriculum.step(model)  # Auto-freezes backbone at phase transitions
```

### DynamicK — Hierarchical task grouping

Group tasks by function, run Golden Pendulum within and across groups. Reduces O(K^2) cost for K=16+ tasks.

```python
from golden_pendulum.pro import DynamicK, TaskGroup

dk = DynamicK(groups=[
    TaskGroup("ranking", tasks={"returns", "rank", "quality"}),
    TaskGroup("risk", tasks={"vol", "mae", "kelly", "risk"}),
    TaskGroup("meta", tasks={"regime", "calibration", "confidence"}),
])
weights = dk.backward(all_16_losses, model)

# Or auto-group by gradient similarity:
dk = DynamicK(auto_group=True, similarity_threshold=0.5)
```

### DiagnosticsEngine — Real-time conflict analysis

Deep visibility into gradient conflicts, resonance detection, and convergence monitoring.

```python
from golden_pendulum.pro import DiagnosticsEngine

diag = DiagnosticsEngine()
report = diag.analyze(losses, model)
print(f"Conflict ratio: {report.conflict_ratio}")
print(f"Norm ratio: {report.norm_ratio}x")
print(f"Conflicting pairs: {report.conflict_pairs}")
if diag.alerts:
    for alert in diag.alerts:
        print(f"ALERT: {alert}")
```

### Presets — Battle-tested configurations

```python
from golden_pendulum.pro import get_preset, list_presets

list_presets()  # finance_4phase, finance_quick, nlp_multitask, vision_multitask, robotics_control

preset = get_preset("finance_4phase")  # Paper's exact configuration
scheduler = CurriculumScheduler(phases=preset.phases)
```

## Benchmarks

```bash
python benchmarks/compare_methods.py
```

Reproduces the paper's core finding on synthetic multi-head models.

## Paper

**Golden Pendulum Multi-Task Learning: Anti-Resonance Equilibria for Gradient Balancing in Multi-Head Transformer Training**

Christian Knopp, 2026

See [`paper/golden_pendulum_mtl.pdf`](paper/golden_pendulum_mtl.pdf) for the full paper.

**Key results** (42.5M-param, 16-head financial transformer):

| Metric | AdamW+EMA | Nash-MTL | Golden Pendulum |
|--------|:---------:|:--------:|:--------------:|
| GOLD metrics | 5 | 1 | **7** |
| FAIL metrics | 2 | 6 | **2** |
| Trading IC | 0.012 | -0.002 | **0.033** |
| Decile Monotonicity | 0.576 | 0.455 | **0.879** |
| Vol Forecast IC | 0.645 | -0.795 | **0.695** |
| Balance (w<sub>min</sub>/w<sub>max</sub>) | 1.00 | 0.04 | **0.24** |

## Citation

```bibtex
@article{knopp2026golden,
  title={Golden Pendulum Multi-Task Learning: Anti-Resonance Equilibria for
         Gradient Balancing in Multi-Head Transformer Training},
  author={Knopp, Christian},
  year={2026}
}
```

## Author

**Christian Knopp** &mdash; [@Conceptual1](https://x.com/Conceptual1) &mdash; [cknopp@gmail.com](mailto:cknopp@gmail.com)

## License

Apache 2.0. See [LICENSE](LICENSE).
