"""Golden Pendulum MTL — Pro tier.

Advanced features for production multi-task training:
- AdaptiveLambda: Auto-tune regularization strength based on gradient conflict
- DynamicK: Automatic task grouping and phase management
- CurriculumScheduler: Phase A/B/C/D curriculum with backbone freeze control
- DiagnosticsEngine: Real-time conflict analysis, resonance detection, alerts
- FinancePresets: Pre-configured settings for financial transformer training

Pro features require: pip install golden-pendulum-mtl[pro]
"""

from golden_pendulum.pro.adaptive import AdaptiveLambda
from golden_pendulum.pro.curriculum import CurriculumScheduler, Phase
from golden_pendulum.pro.diagnostics import ConflictReport, DiagnosticsEngine
from golden_pendulum.pro.dynamic_k import DynamicK, TaskGroup
from golden_pendulum.pro.presets import FinancePreset, get_preset, list_presets

__all__ = [
    "AdaptiveLambda",
    "DynamicK",
    "TaskGroup",
    "CurriculumScheduler",
    "Phase",
    "DiagnosticsEngine",
    "ConflictReport",
    "FinancePreset",
    "get_preset",
    "list_presets",
]
