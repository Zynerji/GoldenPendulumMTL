"""Pre-configured Golden Pendulum settings for common use cases.

Provides battle-tested configurations derived from production deployments,
so users don't have to tune hyperparameters from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from golden_pendulum.pro.curriculum import Phase


@dataclass
class FinancePreset:
    """Pre-configured settings for financial transformer training.

    Based on the paper's OrchestratorV7 (42.5M params, 16 heads) results.

    Attributes:
        name: Preset name.
        description: What this preset is optimized for.
        phases: List of training phases.
        total_steps: Total training steps across all phases.
        recommended_lr: Recommended base learning rate.
        recommended_batch_size: Recommended batch size.
    """
    name: str
    description: str
    phases: list = field(default_factory=list)
    total_steps: int = 0
    recommended_lr: float = 1e-4
    recommended_batch_size: int = 64


_PRESETS: Dict[str, FinancePreset] = {}


def _register(preset: FinancePreset) -> FinancePreset:
    _PRESETS[preset.name] = preset
    return preset


# ── Finance: Full 4-Phase Curriculum ──────────────────────────────────

_register(FinancePreset(
    name="finance_4phase",
    description=(
        "Full 4-phase curriculum from the Golden Pendulum paper. "
        "Phase A: ranking (backbone unfrozen, 15K steps). "
        "Phase B: risk heads (frozen, 10K). "
        "Phase C: meta heads (frozen, 10K). "
        "Phase D: discovery (frozen, 5K). "
        "Total: 40K steps. Validated on 42.5M-param, 16-head transformer."
    ),
    phases=[
        Phase(
            name="A_ranking",
            tasks={"horizon_returns", "trade_quality", "short_horizon_rank", "embedding"},
            steps=15000,
            freeze_backbone=False,
            lam=0.5,
            lr=5e-5,
            warmup_steps=500,
        ),
        Phase(
            name="B_risk",
            tasks={"vol_forecast", "mae", "kelly_fraction", "risk_score", "horizon_magnitude"},
            steps=10000,
            freeze_backbone=True,
            lam=0.3,
            lr=1e-4,
            warmup_steps=200,
        ),
        Phase(
            name="C_meta",
            tasks={"regime", "calibration", "confidence", "optimal_exit"},
            steps=10000,
            freeze_backbone=True,
            lam=0.3,
            lr=1e-4,
            warmup_steps=200,
        ),
        Phase(
            name="D_discovery",
            tasks={"reconstruction", "prediction", "decorrelation"},
            steps=5000,
            freeze_backbone=True,
            lam=0.8,
            lr=5e-5,
            warmup_steps=100,
        ),
    ],
    total_steps=40000,
    recommended_lr=5e-5,
    recommended_batch_size=64,
))


# ── Finance: Quick 2-Phase ────────────────────────────────────────────

_register(FinancePreset(
    name="finance_quick",
    description=(
        "Abbreviated 2-phase curriculum for rapid iteration. "
        "Phase A: ranking + risk combined (backbone unfrozen, 8K steps). "
        "Phase B: meta + sizing (frozen, 5K steps). "
        "Total: 13K steps. Good for hyperparameter search."
    ),
    phases=[
        Phase(
            name="A_combined",
            tasks={"horizon_returns", "trade_quality", "short_horizon_rank",
                   "vol_forecast", "mae", "risk_score"},
            steps=8000,
            freeze_backbone=False,
            lam=0.5,
            lr=5e-5,
            warmup_steps=300,
        ),
        Phase(
            name="B_meta",
            tasks={"regime", "calibration", "confidence", "kelly_fraction", "optimal_exit"},
            steps=5000,
            freeze_backbone=True,
            lam=0.4,
            lr=1e-4,
            warmup_steps=100,
        ),
    ],
    total_steps=13000,
    recommended_lr=5e-5,
    recommended_batch_size=64,
))


# ── NLP: Multi-Task BERT/GPT ─────────────────────────────────────────

_register(FinancePreset(
    name="nlp_multitask",
    description=(
        "Multi-task NLP training (classification + NER + similarity). "
        "Single phase, backbone unfrozen, moderate lambda. "
        "Good for BERT/GPT fine-tuning with 2-5 tasks."
    ),
    phases=[
        Phase(
            name="joint",
            tasks={"classification", "ner", "similarity", "qa", "summarization"},
            steps=20000,
            freeze_backbone=False,
            lam=0.5,
            lr=2e-5,
            warmup_steps=1000,
        ),
    ],
    total_steps=20000,
    recommended_lr=2e-5,
    recommended_batch_size=32,
))


# ── Vision: Multi-Task Segmentation ──────────────────────────────────

_register(FinancePreset(
    name="vision_multitask",
    description=(
        "Multi-task vision (segmentation + depth + normals + edges). "
        "Two phases: joint training then head refinement. "
        "Based on NYUv2/Cityscapes multi-task benchmarks."
    ),
    phases=[
        Phase(
            name="joint",
            tasks={"segmentation", "depth", "surface_normals", "edge_detection"},
            steps=30000,
            freeze_backbone=False,
            lam=0.5,
            lr=1e-4,
            warmup_steps=1000,
        ),
        Phase(
            name="refine",
            tasks={"segmentation", "depth", "surface_normals", "edge_detection"},
            steps=10000,
            freeze_backbone=True,
            lam=0.3,
            lr=1e-5,
            warmup_steps=200,
        ),
    ],
    total_steps=40000,
    recommended_lr=1e-4,
    recommended_batch_size=16,
))


# ── Robotics: Multi-Objective Control ─────────────────────────────────

_register(FinancePreset(
    name="robotics_control",
    description=(
        "Multi-objective robotic control (reward + safety + efficiency). "
        "Safety task gets higher lambda to prevent starvation. "
        "Single phase, backbone unfrozen."
    ),
    phases=[
        Phase(
            name="joint",
            tasks={"reward", "safety", "energy_efficiency", "smoothness"},
            steps=50000,
            freeze_backbone=False,
            lam=0.7,
            lr=3e-4,
            warmup_steps=2000,
        ),
    ],
    total_steps=50000,
    recommended_lr=3e-4,
    recommended_batch_size=256,
))


def get_preset(name: str) -> FinancePreset:
    """Get a pre-configured preset by name.

    Available presets:
    - ``"finance_4phase"``: Full 4-phase curriculum (paper's setting)
    - ``"finance_quick"``: Abbreviated 2-phase for rapid iteration
    - ``"nlp_multitask"``: Multi-task NLP fine-tuning
    - ``"vision_multitask"``: Multi-task vision (seg + depth + normals)
    - ``"robotics_control"``: Multi-objective robotic control

    Args:
        name: Preset name.

    Returns:
        FinancePreset with pre-configured phases.

    Raises:
        KeyError: If preset name is not found.

    Example::

        from golden_pendulum.pro import get_preset, CurriculumScheduler

        preset = get_preset("finance_4phase")
        scheduler = CurriculumScheduler(
            phases=preset.phases,
            backbone_params=lambda m: m.backbone.parameters(),
        )
    """
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return _PRESETS[name]


def list_presets() -> Dict[str, str]:
    """Return dict of preset name -> description."""
    return {name: p.description for name, p in _PRESETS.items()}
