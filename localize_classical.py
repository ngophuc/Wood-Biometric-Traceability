#!/usr/bin/env python3
"""Classical Ducret-Autun known-parent localization with SIFT + NCC.

Self-contained extraction of the classical localization baseline from the E2E
traceability evaluation. It uses the same default dataset folders as localize.py
and writes nothing to disk: all results are printed to the console.

Protocol:
- source logs are fixed in the sawline coordinate frame;
- each source is prepared once with SIFT/FLANN and NCC preprocessing;
- every board query is processed independently;
- SIFT + FLANN + Lowe filtering + RANSAC estimates query geometry;
- the directed query-axis angle is quantized to {0, 90, 180, 270} degrees;
- the estimated scale resizes the board;
- the inverse quantized turn selects exactly one NCC template;
- cv2.TM_CCOEFF_NORMED localizes that template on the known parent source.

Run from the directory containing the Ducret-Autun folders:

    python localize_classical_sift_ncc_v2.py
"""

from __future__ import annotations

import csv
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_RE = re.compile(r"log(\d+)$", re.I)
QUERY_RE = re.compile(r"log(\d+)_board(\d+)$", re.I)


# =============================================================================
# Configuration and records
# =============================================================================

@dataclass(frozen=True)
class Config:
    query_dir: Path = SCRIPT_DIR / "ducret-autun-t2-virtual-sawn-full-similarity"
    source_dir: Path = SCRIPT_DIR / "ducret-autun-t1-orig"
    manifest_path: Path = query_dir / "_metadata" / "virtual_board_manifest.csv"

    sift_ratio: float = 0.80
    min_good_matches: int = 10
    ransac_threshold_px: float = 5.0
    rotation_hypotheses_deg: tuple[int, ...] = (0, 90, 180, 270)


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    source_id: str
    image_path: Path


@dataclass(frozen=True)
class GeometryEstimate:
    raw_query_axis_angle_deg: float
    quantized_query_axis_angle_deg: int
    quarter_turn_residual_deg: float
    scale: float
    good_matches: int


@dataclass(frozen=True)
class SourceGalleryItem:
    source_id: str
    source_shape: tuple[int, int]
    ncc_gray: np.ndarray
    keypoints: Sequence[Any]
    matcher: Any | None
    preprocess_sec: float


# =============================================================================
# Dataset utilities
# =============================================================================

def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_image(directory: Path, stem: str) -> Path:
    matches = [path for path in image_files(directory) if path.stem.lower() == stem.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one image for {stem!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def source_sort_key(source_id: str) -> int:
    match = SOURCE_RE.fullmatch(source_id)
    if match is None:
        raise ValueError(f"invalid source id: {source_id}")
    return int(match.group(1))


def query_sort_key(query_id: str) -> tuple[int, int]:
    match = QUERY_RE.fullmatch(query_id)
    if match is None:
        raise ValueError(f"invalid query id: {query_id}")
    return int(match.group(1)), int(match.group(2))


def source_records(directory: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for image_path in image_files(directory / "seg"):
        match = SOURCE_RE.fullmatch(image_path.stem)
        if match is None:
            continue
        source_id = f"log{int(match.group(1))}"
        records.append(
            SourceRecord(
                source_id=source_id,
                image_path=image_path,
                mask_path=find_image(directory / "masks", image_path.stem),
            )
        )
    return sorted(records, key=lambda record: source_sort_key(record.source_id))


def query_records(directory: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for image_path in image_files(directory / "seg"):
        match = QUERY_RE.fullmatch(image_path.stem)
        if match is None:
            continue
        source_id = f"log{int(match.group(1))}"
        query_id = f"{source_id}_board{int(match.group(2))}"
        records.append(QueryRecord(query_id, source_id, image_path))
    return sorted(records, key=lambda record: query_sort_key(record.query_id))


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            ImageOps.exif_transpose(image).convert("L"), dtype=np.uint8
        ) > 0


def manifest_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing localization manifest: {path}")
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


# =============================================================================
# Released-method preprocessing and SIFT geometry
# =============================================================================

def preprocess_sift(image_bgr: np.ndarray) -> np.ndarray:
    """BGR grayscale followed by OpenCV default CLAHE."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE().apply(gray)


def preprocess_ncc(image_bgr: np.ndarray) -> np.ndarray:
    """5x5 Gaussian blur, BGR grayscale, then OpenCV default CLAHE."""
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE().apply(gray)


def convex_quadrilateral(points: np.ndarray) -> bool:
    """Diagonal-side convexity test used by the extracted E2E baseline."""
    point = np.asarray(points, dtype=np.float64).reshape(4, 2)
    x1, y1 = point[0]
    x2, y2 = point[1]
    x3, y3 = point[2]
    x4, y4 = point[3]
    z1 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    z2 = (x4 - x1) * (y3 - y1) - (x3 - x1) * (y4 - y1)
    z3 = (x1 - x2) * (y4 - y2) - (x4 - x2) * (y1 - y2)
    z4 = (x3 - x2) * (y4 - y2) - (x4 - x2) * (y3 - y2)
    return bool(z1 * z2 < 0.0 and z3 * z4 < 0.0)


def circular_difference_deg(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def quantize_quarter_turn(angle_deg: float, cfg: Config) -> tuple[int, float]:
    normalized = float(angle_deg) % 360.0
    quantized = int(
        min(
            cfg.rotation_hypotheses_deg,
            key=lambda candidate: circular_difference_deg(normalized, candidate),
        )
    )
    return quantized, circular_difference_deg(normalized, quantized)


def directed_axis_angle(transformed: np.ndarray) -> float | None:
    """Directed query x-axis angle in source-image coordinates."""
    top_edge = transformed[3] - transformed[0]
    bottom_edge = transformed[2] - transformed[1]
    top_norm = float(np.linalg.norm(top_edge))
    bottom_norm = float(np.linalg.norm(bottom_edge))
    if top_norm <= 1e-12 or bottom_norm <= 1e-12:
        return None
    direction = top_edge / top_norm + bottom_edge / bottom_norm
    if float(np.linalg.norm(direction)) <= 1e-12:
        return None
    return float(math.degrees(math.atan2(direction[1], direction[0])) % 360.0)


def estimate_geometry(
    board_shape: tuple[int, int],
    source_shape: tuple[int, int],
    board_keypoints: Sequence[Any],
    board_descriptors: np.ndarray | None,
    source_keypoints: Sequence[Any],
    source_matcher: Any | None,
    cfg: Config,
) -> GeometryEstimate | None:
    """SIFT/FLANN/Lowe/RANSAC geometry for one independent board."""
    if board_descriptors is None or source_matcher is None:
        return None

    matches = source_matcher.knnMatch(board_descriptors, k=2)
    good = [
        first
        for pair in matches
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < cfg.sift_ratio * second.distance
    ]
    if len(good) <= cfg.min_good_matches:
        return None

    board_points = np.float32(
        [board_keypoints[match.queryIdx].pt for match in good]
    ).reshape(-1, 1, 2)
    source_points = np.float32(
        [source_keypoints[match.trainIdx].pt for match in good]
    ).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(
        board_points,
        source_points,
        cv2.RANSAC,
        cfg.ransac_threshold_px,
    )
    if homography is None:
        return None

    height, width = int(board_shape[0]), int(board_shape[1])
    corners = np.float32(
        [[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]
    ).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, homography).reshape(4, 2)

    # Intentionally preserved from the E2E classical baseline: the transformed
    # corner boundary test compares the global min/max against source width.
    source_width = int(source_shape[1])
    if (
        transformed.min() <= 0.0
        or transformed.max() >= source_width
        or not convex_quadrilateral(transformed)
    ):
        return None

    raw_angle = directed_axis_angle(transformed)
    if raw_angle is None:
        return None
    quantized_angle, residual = quantize_quarter_turn(raw_angle, cfg)

    height_1 = np.linalg.norm(transformed[0] - transformed[1])
    height_2 = np.linalg.norm(transformed[2] - transformed[3])
    width_1 = np.linalg.norm(transformed[0] - transformed[3])
    width_2 = np.linalg.norm(transformed[1] - transformed[2])
    scale = float(
        np.mean(
            [
                (height_1 + height_2) / (2.0 * height),
                (width_1 + width_2) / (2.0 * width),
            ]
        )
    )
    if not math.isfinite(scale) or scale <= 0.0:
        return None

    return GeometryEstimate(
        raw_query_axis_angle_deg=raw_angle,
        quantized_query_axis_angle_deg=quantized_angle,
        quarter_turn_residual_deg=residual,
        scale=scale,
        good_matches=len(good),
    )


# =============================================================================
# NCC template preparation and localization
# =============================================================================

def alignment_template_orientation(quantized_query_axis_angle_deg: int) -> int:
    """Quarter turn mapping the query axes back into the fixed source frame."""
    return int((-int(quantized_query_axis_angle_deg)) % 360)


def prepare_ncc_template(
    board_bgr: np.ndarray,
    estimate: GeometryEstimate,
) -> tuple[int, np.ndarray]:
    """Resize by SIFT scale and construct the one selected NCC template."""
    target_width = max(2, int(round(board_bgr.shape[1] * estimate.scale)))
    target_height = max(2, int(round(board_bgr.shape[0] * estimate.scale)))
    scaled = cv2.resize(
        board_bgr,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR,
    )

    orientation_deg = alignment_template_orientation(
        estimate.quantized_query_axis_angle_deg
    )
    if orientation_deg == 0:
        oriented = scaled
    elif orientation_deg == 180:
        # Preserved exactly from the extracted baseline: np.flip with no axis.
        oriented = np.flip(scaled).copy()
    elif orientation_deg == 90:
        oriented = np.rot90(scaled).copy()
    elif orientation_deg == 270:
        oriented = np.flip(np.rot90(scaled)).copy()
    else:
        raise ValueError(f"unsupported quarter-turn orientation: {orientation_deg}")

    return orientation_deg, preprocess_ncc(oriented)


def ncc_search(
    source_gray: np.ndarray,
    orientation_deg: int,
    template_gray: np.ndarray,
) -> tuple[float, np.ndarray | None, int | None]:
    """Search one SIFT-selected template on the fixed known-parent source."""
    source_height, source_width = source_gray.shape[:2]
    template_height, template_width = template_gray.shape[:2]
    if template_height > source_height or template_width > source_width:
        return float("nan"), None, None

    response = cv2.matchTemplate(
        source_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED,
    )
    _, maximum, _, location = cv2.minMaxLoc(response)
    x, y = location
    polygon = np.asarray(
        [
            [x, y],
            [x, y + template_height],
            [x + template_width, y + template_height],
            [x + template_width, y],
        ],
        dtype=np.float64,
    )
    return float(maximum), polygon, int(orientation_deg)


def localization_iou(
    predicted_polygon: np.ndarray,
    ground_truth_box: tuple[int, int, int, int],
    source_shape: tuple[int, int],
) -> float:
    """Binary-mask polygon IoU against the manifest rectangle."""
    height, width = source_shape
    predicted = np.zeros((height, width), dtype=np.uint8)
    truth = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(predicted, [np.rint(predicted_polygon).astype(np.int32)], 1)
    x0, y0, x1, y1 = ground_truth_box
    truth[y0:y1, x0:x1] = 1
    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    return float(intersection / union) if union else 0.0


# =============================================================================
# Source gallery and per-query localization
# =============================================================================

def prepare_source_gallery_item(
    record: SourceRecord,
    sift: Any,
) -> SourceGalleryItem:
    start = time.perf_counter()

    source_rgb = np.asarray(load_rgb(record.image_path), dtype=np.uint8)
    source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    source_mask = load_mask(record.mask_path)
    source_bgr[~source_mask] = 255

    source_sift_gray = preprocess_sift(source_bgr)
    keypoints, descriptors = sift.detectAndCompute(source_sift_gray, None)

    matcher = None
    if descriptors is not None:
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50),
        )
        matcher.add([descriptors])
        matcher.train()

    source_ncc_gray = preprocess_ncc(source_bgr)
    return SourceGalleryItem(
        source_id=record.source_id,
        source_shape=source_bgr.shape[:2],
        ncc_gray=source_ncc_gray,
        keypoints=keypoints,
        matcher=matcher,
        preprocess_sec=time.perf_counter() - start,
    )


def localize_one_query(
    query: QueryRecord,
    source: SourceGalleryItem,
    ground_truth_box: tuple[int, int, int, int],
    sift: Any,
    cfg: Config,
) -> dict[str, Any]:
    # Virtual board crops are already at native crop scale: no acquisition-specific
    # 4x board scaling or compensating source scaling is introduced here.
    query_preprocess_start = time.perf_counter()
    query_rgb = np.asarray(load_rgb(query.image_path), dtype=np.uint8)
    query_bgr = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2BGR)
    query_sift_gray = preprocess_sift(query_bgr)
    board_keypoints, board_descriptors = sift.detectAndCompute(query_sift_gray, None)
    query_feature_preprocess_sec = time.perf_counter() - query_preprocess_start

    geometry_score_start = time.perf_counter()
    estimate = estimate_geometry(
        query_sift_gray.shape,
        source.source_shape,
        board_keypoints,
        board_descriptors,
        source.keypoints,
        source.matcher,
        cfg,
    )
    geometry_score_sec = time.perf_counter() - geometry_score_start

    ncc_input_preprocess_sec = 0.0
    ncc_score_sec = 0.0
    ncc_score = float("nan")
    polygon: np.ndarray | None = None
    orientation_deg: int | None = None

    if estimate is not None:
        ncc_preprocess_start = time.perf_counter()
        orientation_deg, template = prepare_ncc_template(query_bgr, estimate)
        ncc_input_preprocess_sec = time.perf_counter() - ncc_preprocess_start

        ncc_score_start = time.perf_counter()
        ncc_score, polygon, orientation_deg = ncc_search(
            source.ncc_gray,
            orientation_deg,
            template,
        )
        ncc_score_sec = time.perf_counter() - ncc_score_start

    preprocess_sec = query_feature_preprocess_sec + ncc_input_preprocess_sec
    score_sec = geometry_score_sec + ncc_score_sec
    iou = (
        0.0
        if polygon is None
        else localization_iou(polygon, ground_truth_box, source.source_shape)
    )

    return {
        "query_id": query.query_id,
        "source_id": query.source_id,
        "valid": polygon is not None,
        "sift_valid": estimate is not None,
        "source_keypoints": len(source.keypoints),
        "query_keypoints": len(board_keypoints),
        "good_matches": 0 if estimate is None else estimate.good_matches,
        "raw_angle": float("nan") if estimate is None else estimate.raw_query_axis_angle_deg,
        "quantized_angle": None if estimate is None else estimate.quantized_query_axis_angle_deg,
        "angle_residual": float("nan") if estimate is None else estimate.quarter_turn_residual_deg,
        "estimated_scale": float("nan") if estimate is None else estimate.scale,
        "template_orientation": orientation_deg,
        "ncc_score": ncc_score,
        "polygon": polygon,
        "iou": iou,
        "preprocess_sec": preprocess_sec,
        "score_sec": score_sec,
        "total_sec": preprocess_sec + score_sec,
    }


# =============================================================================
# Console-only run
# =============================================================================

def run(cfg: Config) -> None:
    queries = query_records(cfg.query_dir)
    truth = manifest_boxes(cfg.manifest_path)
    all_sources = {record.source_id: record for record in source_records(cfg.source_dir)}

    missing_truth = [query.query_id for query in queries if query.query_id not in truth]
    if missing_truth:
        raise KeyError(f"manifest is missing ground truth for {missing_truth[0]}")

    requested_sources = sorted(
        {query.source_id for query in queries}, key=source_sort_key
    )
    missing_sources = [source_id for source_id in requested_sources if source_id not in all_sources]
    if missing_sources:
        raise KeyError(f"missing source image/mask for {missing_sources[0]}")

    print(
        f"classical localization: queries={len(queries)} parents={len(requested_sources)} "
        f"device=CPU method=SIFT+FLANN+RANSAC+NCC"
    )
    print(
        f"sift_ratio={cfg.sift_ratio:g} min_good_matches=>{cfg.min_good_matches} "
        f"ransac={cfg.ransac_threshold_px:g}px rotations={cfg.rotation_hypotheses_deg}"
    )

    sift = cv2.SIFT_create()
    source_gallery = {
        source_id: prepare_source_gallery_item(all_sources[source_id], sift)
        for source_id in tqdm(requested_sources, desc="Classical SIFT/NCC source gallery")
    }

    source_preparation_times = np.asarray(
        [source_gallery[source_id].preprocess_sec for source_id in requested_sources],
        dtype=np.float64,
    )
    print(
        f"source gallery: prepared={len(requested_sources)} "
        f"preproc_total={source_preparation_times.sum():.6f}s "
        f"preproc_mean={source_preparation_times.mean():.6f}s "
        f"(excluded from per-query timing)"
    )

    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        row = localize_one_query(
            query=query,
            source=source_gallery[query.source_id],
            ground_truth_box=truth[query.query_id],
            sift=sift,
            cfg=cfg,
        )
        rows.append(row)

        polygon = row["polygon"]
        polygon_text = (
            "-"
            if polygon is None
            else ";".join(f"{x:.1f},{y:.1f}" for x, y in polygon)
        )
        quantized = "-" if row["quantized_angle"] is None else str(row["quantized_angle"])
        template_orientation = (
            "-" if row["template_orientation"] is None else str(row["template_orientation"])
        )
        print(
            f"[{index}/{len(queries)}] {query.query_id} parent={query.source_id} "
            f"valid={row['valid']} sift_valid={row['sift_valid']} "
            f"kp={row['query_keypoints']} good={row['good_matches']} "
            f"angle_raw={row['raw_angle']:.3f} angle_q={quantized} "
            f"angle_resid={row['angle_residual']:.3f} scale={row['estimated_scale']:.6f} "
            f"template={template_orientation} ncc={row['ncc_score']:.6f} "
            f"iou={row['iou']:.6f} polygon={polygon_text} "
            f"time_preproc={row['preprocess_sec']:.6f}s "
            f"time_score={row['score_sec']:.6f}s"
        )

    ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
    valid = np.asarray([bool(row["valid"]) for row in rows], dtype=bool)
    preprocess_times = np.asarray(
        [float(row["preprocess_sec"]) for row in rows], dtype=np.float64
    )
    score_times = np.asarray([float(row["score_sec"]) for row in rows], dtype=np.float64)
    total_times = preprocess_times + score_times

    print(
        f"summary: valid={int(valid.sum())}/{len(rows)} failures={int((~valid).sum())} "
        f"mean_iou={ious.mean():.6f} median_iou={np.median(ious):.6f} "
        f"iou@0.50={np.mean(ious >= 0.50):.6f} iou@0.75={np.mean(ious >= 0.75):.6f} "
        f"mean_preproc={preprocess_times.mean():.6f}s "
        f"mean_score={score_times.mean():.6f}s mean_total={total_times.mean():.6f}s"
    )


def main() -> None:
    run(Config())


if __name__ == "__main__":
    main()
