"""397M DeepSeek-V4-inspired cross-project Muon comparison.

The technical report defines disclosed V4 mechanisms. The supplied audit package
is a peer implementation for numerical comparison, not an oracle. This is not an
official or complete DeepSeek-V4 implementation. Historical API names are kept
for compatibility.
"""

from .config import load_config, report_aligned_config, validate_config
from .model import ReportAlignedDeepSeekV4, load_reference_weights
from .optim import HybridOptimizer, WarmupCosineScheduler, build_optimizer

__all__ = [
    "HybridOptimizer",
    "ReportAlignedDeepSeekV4",
    "WarmupCosineScheduler",
    "build_optimizer",
    "load_config",
    "load_reference_weights",
    "report_aligned_config",
    "validate_config",
]
