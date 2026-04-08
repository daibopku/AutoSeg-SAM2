# Auto Mask Batch (importable)

This folder exposes the original `auto-mask-batch.py` workflow as an importable function for reuse in other projects.

## Public API

```python
from auto_mask_batch import AutoMaskBatchConfig, run_auto_mask_batch

config = AutoMaskBatchConfig(
    video_path="/path/to/frames",  # directory of sequential JPG frames
    output_dir="/path/to/output",
    level="default",               # one of: default, small, middle, large
)

video_segments = run_auto_mask_batch(config)
# video_segments is a dict: frame_idx -> {obj_id: bool_mask_np_array}
```

Key config parameters mirror the CLI flags from the original script (`batch_size`, `detect_stride`, `use_other_level`, `postnms`, `pred_iou_thresh`, `box_nms_thresh`, `stability_score_thresh`, checkpoints, etc.).

## Installation / import in a new project

### Pip install from this repo (recommended)

```bash
pip install /path/to/AutoSeg-SAM2  # or: pip install -e /path/to/AutoSeg-SAM2
```

Inside this repository layout, the preprocessing pipeline resolves SAM2 from the bundled official `submodule/segment-anything-2` checkout at runtime, not from the old copied `sam2/` tree. Plain `pip install` from outside this repo does not bundle that checkout by itself; keep the repository structure intact (or copy the `segment-anything-2/` tree alongside `auto_mask_batch/`) if you want the same runtime behavior. By default, SAM1 still loads from `checkpoints/sam1`, while SAM 2.1 loads from `submodule/segment-anything-2/checkpoints/` in the official repo layout.

### Copy-only

If you prefer to copy the folder into another project, copy **both** `auto_mask_batch/` and `segment-anything-2/`, keep the default relative checkpoint paths, and add the parent directory to `PYTHONPATH`.

## Outputs

When `save_outputs=True` (default), masks and `.npy` dumps are written under
`<output_dir>/<level>/mask_each_frame_sam2` during the iterative search and
`<output_dir>/<level>/final-output` after the final propagation. The returned
`video_segments` already contains the per-frame boolean masks.
