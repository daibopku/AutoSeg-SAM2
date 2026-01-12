"""Importable auto-mask-batch pipeline.

This module wraps the original script logic from ``auto-mask-batch.py`` so it can be
imported and called from other projects.  The main entrypoint is
:func:`run_auto_mask_batch`, which returns a dense ``(F, H, W)`` mask volume and
writes a colorized mask mp4 for convenience.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from loguru import logger
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor


# ---------------------------------------------------------------------------
# Helper visualization and mask utilities (lifted from auto-mask-batch.py)
# ---------------------------------------------------------------------------


def show_anns(anns, borders: bool = True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def mask_nms(masks: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.7, score_thr: float = 0.1, inner_thr: float = 0.2, **kwargs) -> torch.Tensor:
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)

    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou

            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def filter_masks(keep: torch.Tensor, masks_result: Iterable[dict]):
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep:
            result_keep.append(m)
    return result_keep


def masks_update(*args, **kwargs):
    masks_new = ()
    for masks_lvl in args:
        seg_pred = torch.from_numpy(np.stack([m["segmentation"] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m["predicted_iou"] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m["stability_score"] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter_masks(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def show_mask(mask, ax, obj_id=None, random_color: bool = False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def save_masks(mask_list: List[np.ndarray], frame_idx: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    if len(mask_list[0].shape) == 3:
        total_width = mask_list[0].shape[2] * len(mask_list)
        max_height = mask_list[0].shape[1]
        final_image = Image.new("RGB", (total_width, max_height))
        for i, img in enumerate(mask_list):
            img_pil = Image.fromarray((img[0] * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img_pil, (i * img_pil.width, 0))
        final_image.save(os.path.join(save_dir, f"mask_{frame_idx:03}.png"))
    else:
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        final_image = Image.new("RGB", (total_width, max_height))
        for i, img in enumerate(mask_list):
            img_pil = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img_pil, (i * img_pil.width, 0))
        final_image.save(os.path.join(save_dir, f"mask_{frame_idx:03}.png"))


def save_masks_npy(mask_list: List[np.ndarray], frame_idx: int, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"mask_{frame_idx:03}.npy"), np.array(mask_list))


def show_points(coords, labels, ax, marker_size: int = 200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)


def make_enlarge_bbox(origin_bbox, max_width, max_height, ratio):
    width = origin_bbox[2]
    height = origin_bbox[3]
    new_box = [max(origin_bbox[0] - width * (ratio - 1) / 2, 0), max(origin_bbox[1] - height * (ratio - 1) / 2, 0)]
    new_box.append(min(width * ratio, max_width - new_box[0]))
    new_box.append(min(height * ratio, max_height - new_box[1]))
    return new_box


def sample_points(masks, enlarge_bbox, positive_num: int = 1, negtive_num: int = 40):
    ex, ey, ewidth, eheight = enlarge_bbox
    positive_count = positive_num
    negtive_count = negtive_num
    output_points = []
    while True:
        x = int(np.random.uniform(ex, ex + ewidth))
        y = int(np.random.uniform(ey, ey + eheight))
        if masks[y][x] is True and positive_count > 0:
            output_points.append((x, y, 1))
            positive_count -= 1
        elif masks[y][x] is False and negtive_count > 0:
            output_points.append((x, y, 0))
            negtive_count -= 1
        if positive_count == 0 and negtive_count == 0:
            break

    return output_points


def sample_points_from_mask(mask):
    true_indices = np.argwhere(mask)
    if true_indices.size == 0:
        raise ValueError("The mask does not contain any True values.")
    random_index = np.random.choice(len(true_indices))
    sample_point = true_indices[random_index]
    return tuple(sample_point)


def search_new_obj(masks_from_prev, mask_list, other_masks_list=None, mask_ratio_thresh=0, ratio=0.5, area_threash=5000):
    new_mask_list = []

    mask_none = ~masks_from_prev[0].copy()[0]
    for prev_mask in masks_from_prev[1:]:
        mask_none &= ~prev_mask[0]

    for mask in mask_list:
        seg = mask["segmentation"]
        if (mask_none & seg).sum() / seg.sum() > ratio and seg.sum() > area_threash:
            new_mask_list.append(mask)

    for mask in new_mask_list:
        mask_none &= ~mask["segmentation"]
    logger.info(len(new_mask_list))
    logger.info("now ratio:", mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]))
    logger.info("expected ratios:", mask_ratio_thresh)
    if other_masks_list is not None:
        for mask in other_masks_list:
            if mask_none.sum() / (mask_none.shape[0] * mask_none.shape[1]) > mask_ratio_thresh:
                seg = mask["segmentation"]
                if (mask_none & seg).sum() / seg.sum() > ratio and seg.sum() > area_threash:
                    new_mask_list.append(mask)
                    mask_none &= ~seg
            else:
                break
    logger.info(len(new_mask_list))

    return new_mask_list


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    return xmin, ymin, width, height


def cal_no_mask_area_ratio(out_mask_list: List[np.ndarray]):
    h = out_mask_list[0].shape[1]
    w = out_mask_list[0].shape[2]
    mask_none = ~out_mask_list[0].copy()
    for prev_mask in out_mask_list[1:]:
        mask_none &= ~prev_mask
    return mask_none.sum() / (h * w)


def extract_video_frames(video_path: str, dst_dir: str) -> Tuple[List[str], float, Tuple[int, int]]:
    """Extract frames from a video file into ``dst_dir`` as JPEGs.

    Returns the list of generated frame filenames (zero-padded), the source fps,
    and (height, width) of the frames.
    """

    os.makedirs(dst_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_names: List[str] = []
    idx = 0
    success, frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError(f"No frames could be read from video: {video_path}")

    height, width = frame.shape[:2]
    while success:
        fname = f"{idx:05}.jpg"
        cv2.imwrite(os.path.join(dst_dir, fname), frame)
        frame_names.append(fname)
        idx += 1
        success, frame = cap.read()

    cap.release()
    return frame_names, float(fps), (height, width)


def masks_to_color_palette(max_id: int) -> np.ndarray:
    """Build an RGB palette (0-1 float) where index maps to object id.

    Index 0 is black (background).
    """

    palette = np.zeros((max_id + 1, 3), dtype=np.float32)
    if max_id == 0:
        return palette

    cmap = plt.get_cmap("tab20")
    for obj_id in range(1, max_id + 1):
        palette[obj_id] = cmap(obj_id % cmap.N)[:3]
    return palette


def mask_frame_to_rgb(mask_frame: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = (palette[mask_frame] * 255).astype(np.uint8)
    return rgb


def ensure_2d_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is a 2D boolean array.

    Accepts shapes (H, W) or (1, H, W). Raises if shape is unexpected.
    """

    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = np.squeeze(mask_arr, axis=0)
    if mask_arr.ndim != 2:
        raise ValueError(f"Mask must be 2D or (1,H,W); got shape {mask_arr.shape}")
    return mask_arr.astype(bool)


class Prompts:
    def __init__(self, bs: int):
        self.batch_size = bs
        self.prompts = {}
        self.obj_list: List[int] = []
        self.key_frame_list: List[int] = []
        self.key_frame_obj_begin_list: List[int] = []

    def add(self, obj_id, frame_id, mask):
        if obj_id not in self.obj_list:
            new_obj = True
            self.prompts[obj_id] = []
            self.obj_list.append(obj_id)
        else:
            new_obj = False
        self.prompts[obj_id].append((frame_id, mask))
        if frame_id not in self.key_frame_list and new_obj:
            self.key_frame_list.append(frame_id)
            self.key_frame_obj_begin_list.append(obj_id)
            logger.info("key_frame_obj_begin_list:", self.key_frame_obj_begin_list)

    def get_obj_num(self):
        return len(self.obj_list)

    def __len__(self):
        if self.obj_list % self.batch_size == 0:  # type: ignore[arg-type]
            return len(self.obj_list) // self.batch_size
        return len(self.obj_list) // self.batch_size + 1

    def __iter__(self):
        self.start_idx = 0
        self.iter_frameindex = 0
        return self

    def __next__(self):
        if self.start_idx < len(self.obj_list):
            if self.iter_frameindex == len(self.key_frame_list) - 1:
                end_idx = min(self.start_idx + self.batch_size, len(self.obj_list))
            else:
                if self.start_idx + self.batch_size < self.key_frame_obj_begin_list[self.iter_frameindex + 1]:
                    end_idx = self.start_idx + self.batch_size
                else:
                    end_idx = self.key_frame_obj_begin_list[self.iter_frameindex + 1]
                    self.iter_frameindex += 1
            batch_keys = self.obj_list[self.start_idx:end_idx]
            batch_prompts = {key: self.prompts[key] for key in batch_keys}
            self.start_idx = end_idx
            return batch_prompts
        raise StopIteration


def get_video_segments(prompts_loader: Prompts, predictor, inference_state, final_output: bool = False):
    video_segments: Dict[int, Dict[int, np.ndarray]] = {}
    for batch_prompts in tqdm(prompts_loader, desc="processing prompts\n"):
        predictor.reset_state(inference_state)
        for obj_id, prompt_list in batch_prompts.items():
            for prompt in prompt_list:
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state, frame_idx=prompt[0], obj_id=obj_id, mask=prompt[1]
                )
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()

        if final_output:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, reverse=True
            ):
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
    return video_segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class AutoMaskBatchConfig:
    video_path: str
    output_dir: str
    level: str = "default"  # one of {"default", "small", "middle", "large"}
    batch_size: int = 20
    detect_stride: int = 10
    use_other_level: bool = True
    postnms: bool = True
    pred_iou_thresh: float = 0.7
    box_nms_thresh: float = 0.7
    stability_score_thresh: float = 0.85
    sam2_checkpoint: str = "./checkpoints/sam2/sam2_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    sam1_checkpoint: str = "checkpoints/sam1/sam_vit_h_4b8939.pth"
    device: str = "cuda"
    log_path: Optional[str] = None
    save_outputs: bool = True
    vis_stride: Optional[int] = None
    mask_ratio_tolerance: float = 0.01


VideoSegments = Dict[int, Dict[int, np.ndarray]]


def run_auto_mask_batch(config: AutoMaskBatchConfig) -> np.ndarray:
    """Run the auto-mask-batch pipeline and return a dense mask volume.

    Args:
        config: Runtime options for the pipeline.

    Returns:
        A numpy array of shape ``(F, H, W)`` where each value is the object id
        (0 is background, objects start from 1).
    """

    device_type = "cuda" if config.device.startswith("cuda") else "cpu"
    vis_stride = config.vis_stride or config.detect_stride

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    if config.save_outputs:
        os.makedirs(os.path.join(config.output_dir, config.level), exist_ok=True)
        log_path = config.log_path or os.path.join(config.output_dir, config.level, f"{config.level}.log")
        logger.add(log_path, rotation="500 MB")
    elif config.log_path:
        logger.add(config.log_path, rotation="500 MB")

    if device_type == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    video_segments: VideoSegments = {}
    mask_ratio_thresh = 0.0

    # Prepare frames from either a directory of JPGs or a video file (mp4, etc.).
    frames_root = config.video_path
    fps: float = 30.0
    frame_names: List[str] = []
    frame_height: int
    frame_width: int

    if os.path.isdir(config.video_path):
        frame_names = [
            p
            for p in os.listdir(config.video_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if not frame_names:
            raise RuntimeError(f"No JPEG frames found in directory: {config.video_path}")
        sample_img = cv2.imread(os.path.join(config.video_path, frame_names[0]))
        if sample_img is None:
            raise RuntimeError(f"Failed to read sample frame: {frame_names[0]}")
        frame_height, frame_width = sample_img.shape[:2]
    else:
        frames_root = os.path.join(config.output_dir, config.level, "_frames_tmp")
        frame_names, fps, (frame_height, frame_width) = extract_video_frames(config.video_path, frames_root)

    num_frames = len(frame_names)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        predictor = build_sam2_video_predictor(config.sam2_config, config.sam2_checkpoint)
        sam = sam_model_registry["vit_h"](checkpoint=config.sam1_checkpoint).to(config.device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=config.pred_iou_thresh,
            box_nms_thresh=config.box_nms_thresh,
            stability_score_thresh=config.stability_score_thresh,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )

        inference_state = predictor.init_state(video_path=frames_root)
        masks_from_prev: List[np.ndarray] = []
        now_frame = 0
        prompts_loader = Prompts(bs=config.batch_size)

        while True:
            logger.info(f"frame: {now_frame}")
            sum_id = prompts_loader.get_obj_num()
            image_path = os.path.join(frames_root, frame_names[now_frame])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
            except IndexError as exc:  # pragma: no cover - hardware dependent
                logger.error(f"mask_generator.generate IndexError at frame {now_frame}: {exc}")
                now_frame += 1
                if now_frame >= len(frame_names):
                    break
                continue

            if config.postnms:
                masks_default, masks_s, masks_m, masks_l = masks_update(
                    masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5
                )
            if config.level == "default":
                masks = [mask for mask in masks_default]
                other_masks = [mask for mask in masks_l] + [mask for mask in masks_m] + [mask for mask in masks_s]
            elif config.level == "small":
                masks = [mask for mask in masks_s]
                other_masks = None
            elif config.level == "middle":
                masks = [mask for mask in masks_m]
                other_masks = [mask for mask in masks_s]
            elif config.level == "large":
                masks = [mask for mask in masks_l]
                other_masks = [mask for mask in masks_s] + [mask for mask in masks_m]
            else:
                raise NotImplementedError(f"Unsupported level: {config.level}")
            if not config.use_other_level:
                other_masks = None

            if now_frame == 0:
                ann_obj_id_list = range(len(masks))
                if config.save_outputs:
                    save_masks([masks[ann_obj_id]["segmentation"] for ann_obj_id in ann_obj_id_list], now_frame, os.path.join(config.output_dir, config.level, "mask_each_frame-sam1"))

                for ann_obj_id in tqdm(ann_obj_id_list):
                    seg = masks[ann_obj_id]["segmentation"]
                    prompts_loader.add(ann_obj_id, 0, seg)
            else:
                if config.save_outputs:
                    save_masks([mask["segmentation"] for mask in masks], now_frame, os.path.join(config.output_dir, config.level, "mask_each_frame-sam1"))
                new_mask_list = search_new_obj(masks_from_prev, masks, other_masks, mask_ratio_thresh)
                logger.info(f"number of new obj: {len(new_mask_list)}")

                for obj_id, mask in enumerate(masks_from_prev):
                    if mask.sum() == 0:
                        continue
                    prompts_loader.add(obj_id, now_frame, mask[0])

                for i in range(len(new_mask_list)):
                    new_mask = new_mask_list[i]["segmentation"]
                    prompts_loader.add(sum_id + i, now_frame, new_mask)

            logger.info(f"obj num: {prompts_loader.get_obj_num()}")

            if now_frame == 0 or ("new_mask_list" in locals() and len(new_mask_list) != 0):
                video_segments = get_video_segments(prompts_loader, predictor, inference_state)

            save_dir = os.path.join(config.output_dir, config.level, "mask_each_frame_sam2")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(os.path.join(save_dir, f"now_frame_{now_frame}"), exist_ok=True)
            max_area_no_mask = (0.0, -1)
            for out_frame_idx in tqdm(range(0, len(frame_names), vis_stride)):
                if out_frame_idx < now_frame:
                    continue
                out_mask_list = []
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    if config.save_outputs:
                        idx_save_dir = os.path.join(save_dir, f"obj_{out_obj_id:02}")
                        os.makedirs(idx_save_dir, exist_ok=True)
                    out_mask_list.append(out_mask)

                if not out_mask_list:
                    continue
                no_mask_ratio = cal_no_mask_area_ratio(out_mask_list)
                if now_frame == out_frame_idx:
                    mask_ratio_thresh = no_mask_ratio
                    logger.info(f"mask_ratio_thresh: {mask_ratio_thresh}")

                if config.save_outputs:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.set_title(f"frame {out_frame_idx}")
                    img_path = os.path.join(frames_root, frame_names[out_frame_idx])
                    ax.imshow(Image.open(img_path))
                    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                        show_mask(out_mask, ax, obj_id=out_obj_id, random_color=False)
                    save_masks(out_mask_list, out_frame_idx, os.path.join(save_dir, f"now_frame_{now_frame}"))
                    save_masks_npy(out_mask_list, out_frame_idx, os.path.join(save_dir, f"now_frame_{now_frame}"))
                    plt.savefig(os.path.join(save_dir, f"frame_{out_frame_idx}.png"))
                    plt.close(fig)

                if no_mask_ratio > mask_ratio_thresh + config.mask_ratio_tolerance and out_frame_idx > now_frame:
                    masks_from_prev = out_mask_list
                    max_area_no_mask = (no_mask_ratio, out_frame_idx)
                    logger.info(max_area_no_mask)
                    break
            if max_area_no_mask[1] == -1:
                break
            logger.info("max_area_no_mask:", max_area_no_mask)
            now_frame = max_area_no_mask[1]

        final_save_dir = os.path.join(config.output_dir, config.level, "final-output")
        video_segments = get_video_segments(prompts_loader, predictor, inference_state, final_output=True)
        if config.save_outputs:
            for out_frame_idx in tqdm(range(0, len(frame_names), 1)):
                out_mask_list = []
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    out_mask_list.append(out_mask)
                if not out_mask_list:
                    continue
                save_masks(out_mask_list, out_frame_idx, final_save_dir)
                save_masks_npy(out_mask_list, out_frame_idx, final_save_dir)

    # Build dense mask volume (F, H, W) with background=0 and object ids starting at 1.
    mask_volume = np.zeros((num_frames, frame_height, frame_width), dtype=np.int32)
    max_obj_id = 0
    for frame_idx, objs in video_segments.items():
        for obj_id, mask in objs.items():
            target_id = obj_id + 1  # shift by 1 so background stays 0
            mask_2d = ensure_2d_mask(mask)
            if mask_2d.shape != (frame_height, frame_width):
                mask_2d = cv2.resize(mask_2d.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            mask_volume[frame_idx][mask_2d] = target_id
            max_obj_id = max(max_obj_id, target_id)

    # Save a colored mask video to output_dir.
    palette = masks_to_color_palette(max_obj_id)
    mask_video_path = os.path.join(config.output_dir, config.level, "masks.mp4")
    os.makedirs(os.path.dirname(mask_video_path), exist_ok=True)
    writer = cv2.VideoWriter(
        mask_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    for frame_idx in range(num_frames):
        color_frame = mask_frame_to_rgb(mask_volume[frame_idx], palette)
        writer.write(cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR))
    writer.release()
    logger.info(f"colored mask video saved to: {mask_video_path}")

    return mask_volume


__all__ = ["AutoMaskBatchConfig", "run_auto_mask_batch", "VideoSegments"]
