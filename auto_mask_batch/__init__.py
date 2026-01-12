"""Public entrypoint for the auto-mask-batch pipeline."""
from .core import AutoMaskBatchConfig, run_auto_mask_batch, VideoSegments

__all__ = ["AutoMaskBatchConfig", "run_auto_mask_batch", "VideoSegments"]
