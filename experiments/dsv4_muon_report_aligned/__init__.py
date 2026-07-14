"""Report-aligned DeepSeek-V4-like Muon training experiment.

This is a TorchForge assembly for numerical comparison with the supplied audit
package. It is not an official or complete DeepSeek-V4 implementation.
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
