#!/usr/bin/env python3
"""Ducret-Autun known-parent localization with vanilla masked-feature cross-correlation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import retrieve as core

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Configuration and records
# =============================================================================

@dataclass(frozen=True)
class Config:
    query_dir: Path = SCRIPT_DIR / "ducret-autun-t2-virtual-sawn-full-similarity"
    source_dir: Path = SCRIPT_DIR / "ducret-autun-t1-orig"
    manifest_path: Path = query_dir / "_metadata" / "virtual_board_manifest.csv"
    model_name: str = "resnet18.a1_in1k"
    feature_index: int = 2
    source_long_side: int = 576
    angles: tuple[int, ...] = (0, 90, 180, 270)
    scales: tuple[float, ...] = (0.30, 0.40)
    source_crop_pad_fraction: float = 0.02
    refinement_scale_start: float = 0.85
    refinement_scale_stop: float = 1.15
    refinement_scale_step: float = 0.005
    localization_source_long_side: int = -1  # -1 = native cropped source resolution.
    log_path: Path | None = None  # Optional exact mirror of console lines.


@dataclass(frozen=True)
class NativeSource:
    feature: torch.Tensor
    mask: torch.Tensor
    meta: dict[str, int | str]
    input_side: int
    support_h: int
    support_w: int

    @property
    def encoded_long_side(self) -> int:
        return max(int(self.meta["resized_w"]), int(self.meta["resized_h"]))


@dataclass(frozen=True)
class NativeQuery:
    feature: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class Anchor:
    query_x: int
    query_y: int
    source_x: int
    source_y: int
    score: float


@dataclass(frozen=True)
class Refinement:
    box_xyxy: tuple[int, int, int, int]
    score: float
    scale_multiplier: float


# =============================================================================
# Ground truth
# =============================================================================

def manifest_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["query_id"]: (
            int(row["crop_x"]),
            int(row["crop_y"]),
            int(row["crop_x1"]),
            int(row["crop_y1"]),
        )
        for row in rows
    }


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return float(intersection / union) if union > 0 else 0.0


# =============================================================================
# Coarse cross-correlation preparation
# =============================================================================

def build_coarse_source(
    record: core.SourceRecord,
    cfg: Config,
    encoder: core.Encoder,
) -> core.Gallery:
    full_image = core.load_rgb(record.image_path)
    full_mask = core.load_mask(record.mask_path)
    cropped_image, cropped_mask, crop_box = core.crop_source(
        full_image, full_mask, cfg.source_crop_pad_fraction
    )
    square_image, square_mask, content_size = core.resize_and_pad(
        cropped_image,
        cropped_mask,
        cfg.source_long_side,
        cfg.source_long_side,
    )
    features, feature_masks = encoder.encode([square_image], [square_mask])
    feature_shape = (int(features.shape[-2]), int(features.shape[-1]))
    support_h, support_w = core.feature_support_size(content_size, cfg.source_long_side, feature_shape)
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    meta = {
        "source_id": record.source_id,
        "orig_w": full_image.size[0],
        "orig_h": full_image.size[1],
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_w": cropped_image.size[0],
        "crop_h": cropped_image.size[1],
        "resized_w": content_size[0],
        "resized_h": content_size[1],
        "reduction": encoder.reduction,
    }
    sizes = torch.tensor([[support_h, support_w]], dtype=torch.long, device=encoder.device)
    return core.Gallery([meta], features.contiguous(), sizes)


# =============================================================================
# Native-resolution refinement preparation
# =============================================================================

def encode_native_source(
    record: core.SourceRecord,
    cfg: Config,
    encoder: core.Encoder,
) -> NativeSource:
    full_image = core.load_rgb(record.image_path)
    full_mask = core.load_mask(record.mask_path)
    cropped_image, cropped_mask, crop_box = core.crop_source(
        full_image, full_mask, cfg.source_crop_pad_fraction
    )
    native_long_side = max(cropped_image.size)
    encoded_long_side = (
        native_long_side
        if cfg.localization_source_long_side == -1
        else min(native_long_side, cfg.localization_source_long_side)
    )
    input_side = max(32, int(encoded_long_side))
    square_image, square_mask, content_size = core.resize_and_pad(
        cropped_image, cropped_mask, input_side, input_side
    )
    features, feature_masks = encoder.encode([square_image], [square_mask])
    feature = features[0].contiguous()
    mask = feature_masks[0].contiguous()
    support_h, support_w = core.feature_support_size(
        content_size,
        input_side,
        (int(feature.shape[-2]), int(feature.shape[-1])),
    )
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    meta = {
        "source_id": record.source_id,
        "orig_w": full_image.size[0],
        "orig_h": full_image.size[1],
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_w": cropped_image.size[0],
        "crop_h": cropped_image.size[1],
        "resized_w": content_size[0],
        "resized_h": content_size[1],
        "reduction": encoder.reduction,
    }
    return NativeSource(feature, mask, meta, input_side, support_h, support_w)


def encode_native_query(
    query: core.QueryRecord,
    angle: int,
    target_long_side: int,
    encoder: core.Encoder,
) -> NativeQuery:
    image = core.load_rgb(query.image_path)
    mask = core.load_mask(query.mask_path)
    rotated_image, rotated_mask = core.rotate_right_angle(image, mask, angle)
    x0, y0, x1, y1 = core.foreground_bbox(rotated_mask)
    rotated_image = rotated_image.crop((x0, y0, x1, y1))
    rotated_mask = rotated_mask[y0:y1, x0:x1]
    input_side = max(48, int(target_long_side))
    square_image, square_mask, content_size = core.resize_and_pad(
        rotated_image, rotated_mask, input_side, input_side
    )
    features, feature_masks = encoder.encode([square_image], [square_mask])
    feature = features[0]
    mask_feature = feature_masks[0]
    support_h, support_w = core.feature_support_size(
        content_size,
        input_side,
        (int(feature.shape[-2]), int(feature.shape[-1])),
    )
    return NativeQuery(
        feature[:, :support_h, :support_w].contiguous(),
        mask_feature[:, :support_h, :support_w].contiguous(),
    )


# =============================================================================
# Anchored refinement
# =============================================================================

def source_box_to_feature_edges(
    box: tuple[int, int, int, int],
    source: NativeSource,
) -> tuple[float, float, float, float]:
    meta = source.meta
    crop_w = float(max(1, int(meta["crop_w"])))
    crop_h = float(max(1, int(meta["crop_h"])))
    resized_w = float(max(1, int(meta["resized_w"])))
    resized_h = float(max(1, int(meta["resized_h"])))
    feat_w = float(source.feature.shape[-1])
    feat_h = float(source.feature.shape[-2])
    x0, y0, x1, y1 = box
    encoded_x0 = (x0 - int(meta["crop_x0"])) * resized_w / crop_w
    encoded_x1 = (x1 - int(meta["crop_x0"])) * resized_w / crop_w
    encoded_y0 = (y0 - int(meta["crop_y0"])) * resized_h / crop_h
    encoded_y1 = (y1 - int(meta["crop_y0"])) * resized_h / crop_h
    return (
        encoded_x0 * feat_w / float(source.input_side),
        encoded_y0 * feat_h / float(source.input_side),
        encoded_x1 * feat_w / float(source.input_side),
        encoded_y1 * feat_h / float(source.input_side),
    )


def select_anchor(
    source: NativeSource,
    query: NativeQuery,
    coarse_box: tuple[int, int, int, int],
) -> Anchor:
    query_h, query_w = int(query.feature.shape[-2]), int(query.feature.shape[-1])
    x0, y0, x1, y1 = source_box_to_feature_edges(coarse_box, source)
    query_y, query_x = torch.where(query.mask[0] > 0.5)
    source_x = torch.round(x0 + (query_x.float() + 0.5) * ((x1 - x0) / query_w) - 0.5).long()
    source_y = torch.round(y0 + (query_y.float() + 0.5) * ((y1 - y0) / query_h) - 0.5).long()

    inside = (
        (source_x >= 0)
        & (source_y >= 0)
        & (source_x < source.support_w)
        & (source_y < source.support_h)
    )
    query_x, query_y = query_x[inside], query_y[inside]
    source_x, source_y = source_x[inside], source_y[inside]
    valid = source.mask[0, source_y, source_x] > 0.5
    query_x, query_y = query_x[valid], query_y[valid]
    source_x, source_y = source_x[valid], source_y[valid]
    if query_x.numel() == 0:
        raise RuntimeError("retrieval placement contains no valid native feature correspondences")

    query_vectors = query.feature[:, query_y, query_x].transpose(0, 1)
    source_vectors = source.feature[:, source_y, source_x].transpose(0, 1)
    scores = (query_vectors * source_vectors).sum(dim=1)
    best = int(torch.argmax(scores).item())
    return Anchor(
        query_x=int(query_x[best].item()),
        query_y=int(query_y[best].item()),
        source_x=int(source_x[best].item()),
        source_y=int(source_y[best].item()),
        score=float(scores[best].item()),
    )


def refinement_scales(cfg: Config) -> list[float]:
    values: list[float] = []
    value = cfg.refinement_scale_start
    while value <= cfg.refinement_scale_stop + cfg.refinement_scale_step * 0.25:
        values.append(round(value, 10))
        value += cfg.refinement_scale_step
    return sorted(set(values + [1.0]))


def resize_kernel(
    feature: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = int(feature.shape[-2]), int(feature.shape[-1])
    resized_h = max(2, int(round(height * scale)))
    resized_w = max(2, int(round(width * scale)))
    resized_feature = F.interpolate(
        feature.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )[0]
    resized_mask = (
        F.interpolate(mask.unsqueeze(0), size=(resized_h, resized_w), mode="nearest")[0]
        > 0.5
    ).float()
    resized_feature = (
        F.normalize(resized_feature * resized_mask, p=2, dim=0, eps=1e-12)
        * resized_mask
    )
    return resized_feature.contiguous(), resized_mask.contiguous()


def refine(
    source: NativeSource,
    query: NativeQuery,
    coarse_box: tuple[int, int, int, int],
    cfg: Config,
) -> Refinement:
    anchor = select_anchor(source, query, coarse_box)
    base_h, base_w = int(query.feature.shape[-2]), int(query.feature.shape[-1])
    best: Refinement | None = None

    for multiplier in refinement_scales(cfg):
        feature, mask = resize_kernel(query.feature, query.mask, multiplier)
        height, width = int(feature.shape[-2]), int(feature.shape[-1])
        scaled_query_x = (anchor.query_x + 0.5) * (width / float(base_w)) - 0.5
        scaled_query_y = (anchor.query_y + 0.5) * (height / float(base_h)) - 0.5
        x = int(round(anchor.source_x - scaled_query_x))
        y = int(round(anchor.source_y - scaled_query_y))
        if x < 0 or y < 0 or x + width > source.support_w or y + height > source.support_h:
            continue

        source_crop = source.feature[:, y:y + height, x:x + width]
        # Both tensors are already zero outside their masks. Refinement therefore
        # uses ordinary masked-feature correlation, normalized only by query area.
        query_area = mask.sum().clamp_min(1.0)
        score = float(((feature * source_crop).sum() / query_area).item())
        candidate = Refinement(
            box_xyxy=core.feature_box_to_source_pixels(source.meta, x, y, width, height),
            score=score,
            scale_multiplier=multiplier,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        raise RuntimeError("no geometrically valid anchored refinement scale")
    return best


# =============================================================================
# Localization run
# =============================================================================

def run(cfg: Config) -> None:
    if cfg.localization_source_long_side != -1 and cfg.localization_source_long_side < 48:
        raise ValueError("localization_source_long_side must be -1 or at least 48")

    log = core.Logger(cfg.log_path)
    encoder = core.Encoder(cfg.model_name, cfg.feature_index)
    queries = core.query_records(cfg.query_dir)
    sources = {record.source_id: record for record in core.source_records(cfg.source_dir)}
    truth = manifest_boxes(cfg.manifest_path)
    scorer = core.FFTScorer(encoder.device, gallery_batch_size=1)

    log.write(
        f"localization: queries={len(queries)} model={cfg.model_name} feature_index={cfg.feature_index} "
        f"stride={encoder.reduction} channels={encoder.channels} device={encoder.device}"
    )
    log.write(
        f"coarse_angles={cfg.angles} coarse_scales={cfg.scales} "
        "score=vanilla_masked_feature_cross_correlation "
        f"refinement_scales={cfg.refinement_scale_start}:"
        f"{cfg.refinement_scale_stop}:{cfg.refinement_scale_step}"
    )

    coarse_ious: list[float] = []
    refined_ious: list[float] = []
    preproc_times: list[float] = []
    score_times: list[float] = []

    for index, query in enumerate(queries, start=1):
        if query.query_id not in truth:
            raise KeyError(f"missing ground-truth box for {query.query_id} in {cfg.manifest_path}")
        source_record = sources[query.expected_source_id]

        coarse_source, t_coarse_source = core.timed(lambda: build_coarse_source(source_record, cfg, encoder))
        bank, t_bank = core.timed(lambda: core.build_query_bank(query, cfg, encoder))
        coarse_hits, t_coarse_score = core.timed(lambda: scorer.rank(coarse_source, bank))
        if not coarse_hits:
            raise RuntimeError(f"no valid coarse localization for {query.query_id}")
        # Recompute the best orientation, scale and placement on the known parent.
        coarse = coarse_hits[0]

        native_source, t_native_source = core.timed(lambda: encode_native_source(source_record, cfg, encoder))
        target_long_side = max(
            48, int(round(native_source.encoded_long_side * coarse.scale))
        )
        native_query, t_native_query = core.timed(
            lambda: encode_native_query(
                query, coarse.angle, target_long_side, encoder
            )
        )
        refined, t_refine_score = core.timed(
            lambda: refine(native_source, native_query, coarse.box_xyxy, cfg)
        )

        preproc_sec = t_coarse_source + t_bank + t_native_source + t_native_query
        score_sec = t_coarse_score + t_refine_score
        gt_box = truth[query.query_id]
        coarse_iou = box_iou(coarse.box_xyxy, gt_box)
        refined_iou = box_iou(refined.box_xyxy, gt_box)
        coarse_ious.append(coarse_iou)
        refined_ious.append(refined_iou)
        preproc_times.append(preproc_sec)
        score_times.append(score_sec)

        log.write(
            f"[{index}/{len(queries)}] {query.query_id} parent={query.expected_source_id} "
            f"kernel=angle:{coarse.angle},scale:{coarse.scale:g} coarse_iou={coarse_iou:.6f} "
            f"refined_iou={refined_iou:.6f} refine_scale={refined.scale_multiplier:g} "
            f"coarse_box={','.join(map(str, coarse.box_xyxy))} "
            f"refined_box={','.join(map(str, refined.box_xyxy))} "
            f"time_preproc={preproc_sec:.6f}s time_score={score_sec:.6f}s"
        )

    log.write(
        f"summary: mean_coarse_iou={np.mean(coarse_ious):.6f} mean_refined_iou={np.mean(refined_ious):.6f} "
        f"iou@0.75={np.mean(np.asarray(refined_ious) >= 0.75):.6f} "
        f"iou@0.90={np.mean(np.asarray(refined_ious) >= 0.90):.6f} "
        f"mean_preproc={np.mean(preproc_times):.6f}s mean_score={np.mean(score_times):.6f}s"
    )


def main() -> None:
    cfg = Config(
        # Edit defaults here when needed, e.g. log_path=SCRIPT_DIR / "localization_console.txt".
    )
    run(cfg)


if __name__ == "__main__":
    main()
