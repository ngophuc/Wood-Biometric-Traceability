#!/usr/bin/env python3
"""Generate the deterministic paired Ducret-Autun virtual board cuts used in the manuscript."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_RE = re.compile(r"log(\d+)$", re.I)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    fresh_logs_dir: Path = SCRIPT_DIR / "ducret-autun-t1-orig"
    aligned_aged_logs_dir: Path = SCRIPT_DIR / "ducret-autun-t2-cut-align-full-similarity"
    aged_boards_dir: Path = SCRIPT_DIR / "ducret-autun-t2-virtual-sawn-full-similarity"
    fresh_boards_dir: Path = SCRIPT_DIR / "ducret-autun-t1-virtual-sawn"
    seed: int = 20260711
    boards_per_log: int = 10
    long_side_fraction: tuple[float, float] = (0.30, 0.40)
    aspect_ratio: tuple[float, float] = (2.0, 4.0)
    rotations: tuple[int, ...] = (0, 90, 180, 270)
    jpeg_quality: int = 95
    overwrite: bool = False


# =============================================================================
# Image and mask utilities
# =============================================================================

def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_image(directory: Path, stem: str) -> Path:
    matches = [p for p in image_files(directory) if p.stem.lower() == stem.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one image for {stem!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("L"), dtype=np.uint8) > 0


def save_jpeg(array: np.ndarray, path: Path, quality: int) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path, quality=quality, subsampling=0)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L").save(path)


# =============================================================================
# Virtual sawing geometry
# =============================================================================

def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("empty foreground mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def integral_image(mask: np.ndarray) -> np.ndarray:
    integral = mask.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


def valid_positions(
    integral: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask_h, mask_w = integral.shape[0] - 1, integral.shape[1] - 1
    if width > mask_w or height > mask_h:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    sums = (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )
    return np.where(sums == width * height)


def sample_dimensions(
    log_max_dim: int,
    rng: np.random.Generator,
    cfg: Config,
) -> tuple[int, int, float]:
    long_side = max(2, int(round(rng.uniform(*cfg.long_side_fraction) * log_max_dim)))
    target_aspect = float(rng.uniform(*cfg.aspect_ratio))
    short_min = max(1, int(math.ceil(long_side / cfg.aspect_ratio[1] - 1e-12)))
    short_max = max(1, int(math.floor(long_side / cfg.aspect_ratio[0] + 1e-12)))
    short_side = min(max(int(round(long_side / target_aspect)), short_min), short_max)
    # Randomize whether the sampled long side is horizontal or vertical.
    if int(rng.integers(2)) == 0:
        return long_side, short_side, long_side / float(short_side)
    return short_side, long_side, long_side / float(short_side)


def rotate_right_angle(array: np.ndarray, degrees: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(array, k=(int(degrees) // 90) % 4))


# =============================================================================
# Output
# =============================================================================

def prepare_output(directory: Path, overwrite: bool) -> None:
    if directory.exists() and not overwrite:
        raise FileExistsError(f"{directory} already exists; set overwrite=True in main() to regenerate")
    if directory.exists():
        import shutil

        shutil.rmtree(directory)
    (directory / "seg").mkdir(parents=True)
    (directory / "masks").mkdir(parents=True)
    (directory / "_metadata").mkdir(parents=True)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Dataset generation
# =============================================================================

def generate(cfg: Config) -> None:
    prepare_output(cfg.aged_boards_dir, cfg.overwrite)
    prepare_output(cfg.fresh_boards_dir, cfg.overwrite)

    rng = np.random.default_rng(cfg.seed)
    rows: list[dict[str, object]] = []
    aged_masks = image_files(cfg.aligned_aged_logs_dir / "masks")

    for mask_path in tqdm(aged_masks, desc="Ducret-Autun virtual cuts"):
        if SOURCE_RE.fullmatch(mask_path.stem) is None:
            continue
        source_id = mask_path.stem.lower()
        aged = load_rgb(find_image(cfg.aligned_aged_logs_dir / "seg", mask_path.stem))
        fresh = load_rgb(find_image(cfg.fresh_logs_dir / "seg", mask_path.stem))
        mask = load_mask(mask_path)
        if aged.shape != fresh.shape or aged.shape[:2] != mask.shape:
            raise ValueError(f"aligned aged/fresh geometry mismatch for {source_id}")

        x0, y0, x1, y1 = foreground_bbox(mask)
        foreground = mask[y0:y1, x0:x1]
        integral = integral_image(foreground)
        log_max_dim = max(x1 - x0, y1 - y0)
        position_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

        for board_index in range(1, cfg.boards_per_log + 1):
            while True:
                width, height, aspect = sample_dimensions(log_max_dim, rng, cfg)
                key = (width, height)
                if key not in position_cache:
                    position_cache[key] = valid_positions(integral, width, height)
                ys, xs = position_cache[key]
                if xs.size:
                    pick = int(rng.integers(xs.size))
                    crop_x = x0 + int(xs[pick])
                    crop_y = y0 + int(ys[pick])
                    rotation = int(rng.choice(np.asarray(cfg.rotations, dtype=np.int32)))
                    break

            query_id = f"{source_id}_board{board_index}"
            aged_crop = aged[crop_y:crop_y + height, crop_x:crop_x + width].copy()
            fresh_crop = fresh[crop_y:crop_y + height, crop_x:crop_x + width].copy()
            query_mask = np.ones((height, width), dtype=bool)

            aged_crop = rotate_right_angle(aged_crop, rotation)
            query_mask = rotate_right_angle(query_mask, rotation)

            save_jpeg(aged_crop, cfg.aged_boards_dir / "seg" / f"{query_id}.jpg", cfg.jpeg_quality)
            save_mask(query_mask, cfg.aged_boards_dir / "masks" / f"{query_id}.png")
            save_jpeg(fresh_crop, cfg.fresh_boards_dir / "seg" / f"{query_id}.jpg", cfg.jpeg_quality)
            save_mask(
                np.ones(fresh_crop.shape[:2], dtype=bool),
                cfg.fresh_boards_dir / "masks" / f"{query_id}.png",
            )

            rows.append(
                {
                    "query_id": query_id,
                    "source_id": source_id,
                    "board_index": board_index,
                    "crop_x": crop_x,
                    "crop_y": crop_y,
                    "crop_w": width,
                    "crop_h": height,
                    "crop_x1": crop_x + width,
                    "crop_y1": crop_y + height,
                    "aspect": f"{aspect:.8f}",
                    "query_rotation_deg": rotation,
                    "expected_alignment_angle_deg": int((-rotation) % 360),
                    "source_width": fresh.shape[1],
                    "source_height": fresh.shape[0],
                }
            )

    if not rows:
        raise RuntimeError("no Ducret-Autun logs were found")

    rows.sort(
        key=lambda row: (
            int(re.search(r"\d+", str(row["source_id"])).group()),
            int(row["board_index"]),
        )
    )
    for directory in (cfg.aged_boards_dir, cfg.fresh_boards_dir):
        write_manifest(directory / "_metadata" / "virtual_board_manifest.csv", rows)

    print(f"generated {len(rows)} paired boards from {len(aged_masks)} Ducret-Autun logs")
    print(f"aged queries:  {cfg.aged_boards_dir}")
    print(f"fresh pairs:   {cfg.fresh_boards_dir}")


def main() -> None:
    cfg = Config(
        # Edit defaults here when needed, e.g. overwrite=True.
    )
    generate(cfg)


if __name__ == "__main__":
    main()
