#!/usr/bin/env python3
"""Default Ducret-Autun board-to-log retrieval by vanilla FFT cross-correlation of masked features."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_RE = re.compile(r"log(\d+)$", re.I)
QUERY_RE = re.compile(r"log(\d+)_board(\d+)$", re.I)
T = TypeVar("T")


# =============================================================================
# Configuration and records
# =============================================================================

@dataclass(frozen=True)
class Config:
    query_dir: Path = SCRIPT_DIR / "ducret-autun-t2-virtual-sawn-full-similarity"
    source_dir: Path = SCRIPT_DIR / "ducret-autun-t1-orig"
    model_name: str = "resnet18.a1_in1k"
    feature_index: int = 2
    source_long_side: int = 576
    angles: tuple[int, ...] = (0, 90, 180, 270)
    scales: tuple[float, ...] = (0.30, 0.40)
    source_crop_pad_fraction: float = 0.02
    source_encode_batch_size: int = 8
    gallery_batch_size: int = 256
    log_path: Path | None = None  # Optional exact mirror of console lines.


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    expected_source_id: str
    image_path: Path
    mask_path: Path


@dataclass
class Gallery:
    meta: list[dict[str, int | str]]
    features: torch.Tensor
    sizes: torch.Tensor


@dataclass
class QueryBank:
    features: torch.Tensor
    areas: torch.Tensor
    sizes: torch.Tensor
    meta: list[dict[str, float | int]]


@dataclass(frozen=True)
class Hit:
    source_index: int
    source_id: str
    score: float
    kernel_index: int
    angle: int
    scale: float
    x_feat: int
    y_feat: int
    support_h: int
    support_w: int
    box_xyxy: tuple[int, int, int, int]


# =============================================================================
# Console logging
# =============================================================================

class Logger:
    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def write(self, line: str = "") -> None:
        print(line)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


# =============================================================================
# Dataset utilities
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


def source_records(directory: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for image_path in image_files(directory / "seg"):
        match = SOURCE_RE.fullmatch(image_path.stem)
        if match:
            source_id = f"log{int(match.group(1))}"
            records.append(
                SourceRecord(
                    source_id,
                    image_path,
                    find_image(directory / "masks", image_path.stem),
                )
            )
    return sorted(records, key=lambda row: int(SOURCE_RE.fullmatch(row.source_id).group(1)))


def query_records(directory: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for image_path in image_files(directory / "seg"):
        match = QUERY_RE.fullmatch(image_path.stem)
        if match:
            source_id = f"log{int(match.group(1))}"
            query_id = f"{source_id}_board{int(match.group(2))}"
            records.append(
                QueryRecord(
                    query_id,
                    source_id,
                    image_path,
                    find_image(directory / "masks", image_path.stem),
                )
            )
    return sorted(
        records,
        key=lambda row: tuple(
            int(value) for value in QUERY_RE.fullmatch(row.query_id).groups()
        ),
    )


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("L"), dtype=np.uint8) > 0


def foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("empty foreground mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    scale = float(long_side) / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def crop_source(
    image: Image.Image,
    mask: np.ndarray,
    pad_fraction: float,
) -> tuple[Image.Image, np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = foreground_bbox(mask)
    pad = int(round(max(x1 - x0, y1 - y0) * pad_fraction))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(image.size[0], x1 + pad), min(image.size[1], y1 + pad)
    return image.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def resize_and_pad(
    image: Image.Image,
    mask: np.ndarray,
    content_long_side: int,
    square_side: int,
) -> tuple[Image.Image, np.ndarray, tuple[int, int]]:
    size = fit_size(image.size[0], image.size[1], content_long_side)
    resized_image = image.resize(size, Image.Resampling.BICUBIC)
    resized_mask = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 0
    square_image = Image.new("RGB", (square_side, square_side), (0, 0, 0))
    square_image.paste(resized_image, (0, 0))
    square_mask = np.zeros((square_side, square_side), dtype=bool)
    square_mask[: size[1], : size[0]] = resized_mask
    return square_image, square_mask, size


def rotate_right_angle(
    image: Image.Image,
    mask: np.ndarray,
    angle: int,
) -> tuple[Image.Image, np.ndarray]:
    k = (int(angle) // 90) % 4
    rotated_image = np.ascontiguousarray(np.rot90(np.asarray(image), k=k))
    rotated_mask = np.ascontiguousarray(np.rot90(mask, k=k))
    return Image.fromarray(rotated_image), rotated_mask


def feature_support_size(
    content_size: tuple[int, int],
    input_side: int,
    feature_shape: tuple[int, int],
) -> tuple[int, int]:
    content_w, content_h = content_size
    feature_h, feature_w = feature_shape
    support_h = max(1, min(feature_h, int(round(content_h * feature_h / float(input_side)))))
    support_w = max(1, min(feature_w, int(round(content_w * feature_w / float(input_side)))))
    return support_h, support_w


# =============================================================================
# Timing and feature conditioning
# =============================================================================

def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call: Callable[[], T]) -> tuple[T, float]:
    synchronize()
    start = time.perf_counter()
    value = call()
    synchronize()
    return value, time.perf_counter() - start


def condition_features(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    valid = masks.float()
    count = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (features * valid).sum(dim=(2, 3), keepdim=True) / count
    centered = features - mean
    variance = ((centered * valid).square()).sum(dim=(2, 3), keepdim=True) / count
    standardized = centered / torch.sqrt(variance.clamp_min(1e-5))
    return (F.normalize(standardized * valid, p=2, dim=1, eps=1e-12) * valid).contiguous()


# =============================================================================
# CNN feature encoding
# =============================================================================

class Encoder:
    def __init__(self, model_name: str, feature_index: int):
        import timm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(feature_index,),
        ).to(self.device).eval()
        config = timm.data.resolve_model_data_config(self.model)
        self.mean = torch.tensor(
            config.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32
        ).view(3, 1, 1)
        self.std = torch.tensor(
            config.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32
        ).view(3, 1, 1)
        info = self.model.feature_info.get_dicts()[0]
        self.reduction = int(info["reduction"])
        self.channels = int(info["num_chs"])

    def tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std

    @torch.inference_mode()
    def encode(
        self,
        images: list[Image.Image],
        masks: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = torch.stack([self.tensor(image) for image in images]).to(self.device)
        features = self.model(batch)[0].float()
        mask_tensor = (
            torch.from_numpy(np.stack(masks).astype(np.float32))
            .unsqueeze(1)
            .to(self.device)
        )
        feature_masks = (
            F.interpolate(mask_tensor, size=features.shape[-2:], mode="nearest") > 0.5
        ).float()
        return condition_features(features, feature_masks), feature_masks.contiguous()


# =============================================================================
# Gallery and query preparation
# =============================================================================

def build_gallery(cfg: Config, encoder: Encoder) -> Gallery:
    records = source_records(cfg.source_dir)
    meta: list[dict[str, int | str]] = []
    feature_rows: list[torch.Tensor] = []
    size_rows: list[tuple[int, int]] = []

    for start in tqdm(range(0, len(records), cfg.source_encode_batch_size), desc="source gallery"):
        batch = records[start:start + cfg.source_encode_batch_size]
        prepared: list[dict[str, object]] = []
        for record in batch:
            full_image = load_rgb(record.image_path)
            full_mask = load_mask(record.mask_path)
            cropped_image, cropped_mask, crop_box = crop_source(
                full_image, full_mask, cfg.source_crop_pad_fraction
            )
            square_image, square_mask, content_size = resize_and_pad(
                cropped_image,
                cropped_mask,
                cfg.source_long_side,
                cfg.source_long_side,
            )
            prepared.append(
                {
                    "record": record,
                    "full_size": full_image.size,
                    "crop_size": cropped_image.size,
                    "crop_box": crop_box,
                    "content_size": content_size,
                    "image": square_image,
                    "mask": square_mask,
                }
            )

        features, feature_masks = encoder.encode(
            [item["image"] for item in prepared],
            [item["mask"] for item in prepared],
        )
        feature_shape = (int(features.shape[-2]), int(features.shape[-1]))
        for i, item in enumerate(prepared):
            support_h, support_w = feature_support_size(
                item["content_size"], cfg.source_long_side, feature_shape
            )
            crop_x0, crop_y0, crop_x1, crop_y1 = item["crop_box"]
            full_w, full_h = item["full_size"]
            crop_w, crop_h = item["crop_size"]
            resized_w, resized_h = item["content_size"]
            record = item["record"]
            meta.append({
                "source_id": record.source_id,
                "orig_w": full_w,
                "orig_h": full_h,
                "crop_x0": crop_x0,
                "crop_y0": crop_y0,
                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
                "crop_w": crop_w,
                "crop_h": crop_h,
                "resized_w": resized_w,
                "resized_h": resized_h,
                "reduction": encoder.reduction,
            })
            feature_rows.append(features[i].contiguous())
            size_rows.append((support_h, support_w))

    return Gallery(
        meta=meta,
        features=torch.stack(feature_rows).contiguous(),
        sizes=torch.tensor(size_rows, dtype=torch.long, device=encoder.device),
    )


def build_query_bank(query: QueryRecord, cfg: Config, encoder: Encoder) -> QueryBank:
    image = load_rgb(query.image_path)
    mask = load_mask(query.mask_path)
    max_target_side = max(max(48, int(round(cfg.source_long_side * scale))) for scale in cfg.scales)
    candidates: list[dict[str, object]] = []

    for scale in cfg.scales:
        target_side = max(48, int(round(cfg.source_long_side * scale)))
        resized_size = fit_size(image.size[0], image.size[1], target_side)
        resized_image = image.resize(resized_size, Image.Resampling.BICUBIC)
        resized_mask = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
                resized_size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
        for angle in cfg.angles:
            rotated_image, rotated_mask = rotate_right_angle(resized_image, resized_mask, angle)
            rx0, ry0, rx1, ry1 = foreground_bbox(rotated_mask)
            rotated_image = rotated_image.crop((rx0, ry0, rx1, ry1))
            rotated_mask = rotated_mask[ry0:ry1, rx0:rx1]
            square_image = Image.new("RGB", (max_target_side, max_target_side), (0, 0, 0))
            square_image.paste(rotated_image, (0, 0))
            square_mask = np.zeros((max_target_side, max_target_side), dtype=bool)
            square_mask[: rotated_image.size[1], : rotated_image.size[0]] = rotated_mask
            candidates.append(
                {
                    "image": square_image,
                    "mask": square_mask,
                    "content_size": rotated_image.size,
                    "angle": angle,
                    "scale": scale,
                }
            )

    features, feature_masks = encoder.encode(
        [item["image"] for item in candidates],
        [item["mask"] for item in candidates],
    )
    feature_shape = (int(features.shape[-2]), int(features.shape[-1]))
    sizes: list[tuple[int, int]] = []
    meta: list[dict[str, float | int]] = []
    for item in candidates:
        support_h, support_w = feature_support_size(item["content_size"], max_target_side, feature_shape)
        sizes.append((support_h, support_w))
        meta.append({"angle": int(item["angle"]), "scale": float(item["scale"])})

    areas = feature_masks.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    return QueryBank(
        features,
        areas.contiguous(),
        torch.tensor(sizes, dtype=torch.long, device=encoder.device),
        meta,
    )


def feature_box_to_source_pixels(
    meta: dict[str, int | str],
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    scale_x = float(meta["crop_w"]) / float(max(1, int(meta["resized_w"])))
    scale_y = float(meta["crop_h"]) / float(max(1, int(meta["resized_h"])))
    reduction = int(meta["reduction"])
    x0 = int(meta["crop_x0"]) + int(round(x * reduction * scale_x))
    y0 = int(meta["crop_y0"]) + int(round(y * reduction * scale_y))
    x1 = int(meta["crop_x0"]) + int(round((x + width) * reduction * scale_x))
    y1 = int(meta["crop_y0"]) + int(round((y + height) * reduction * scale_y))
    return max(0, x0), max(0, y0), min(int(meta["orig_w"]), x1), min(int(meta["orig_h"]), y1)


# =============================================================================
# Vanilla FFT cross-correlation of masked features
# =============================================================================

class FFTScorer:
    """Dense FFT cross-correlation with geometric in-bounds validity only.

    Source and query masks have already been applied during feature conditioning,
    so invalid feature cells are zero. Scoring does not correlate masks, compute
    overlap ratios, or apply an overlap threshold.
    """

    def __init__(
        self,
        device: torch.device,
        gallery_batch_size: int,
    ):
        self.device = device
        self.gallery_batch_size = gallery_batch_size

    def rank(self, gallery: Gallery, bank: QueryBank) -> list[Hit]:
        reduced: list[tuple[torch.Tensor, ...]] = []
        batch_size = min(max(1, self.gallery_batch_size), len(gallery.meta))
        for start in range(0, len(gallery.meta), batch_size):
            stop = min(len(gallery.meta), start + batch_size)
            result = self._reduce_batch(
                gallery.features[start:stop],
                gallery.sizes[start:stop],
                bank,
                start,
            )
            if result is not None:
                reduced.append(result)
        if not reduced:
            return []

        source_indices = torch.cat([x[0] for x in reduced])
        kernel_indices = torch.cat([x[1] for x in reduced])
        peak_x = torch.cat([x[2] for x in reduced])
        peak_y = torch.cat([x[3] for x in reduced])
        scores = torch.cat([x[4] for x in reduced])
        support_sizes = torch.cat([x[5] for x in reduced])

        order = torch.argsort(scores, descending=True)
        source_indices, kernel_indices = source_indices[order], kernel_indices[order]
        peak_x, peak_y, scores = peak_x[order], peak_y[order], scores[order]
        support_sizes = support_sizes[order]

        source_cpu = source_indices.cpu().numpy()
        kernel_cpu = kernel_indices.cpu().numpy()
        x_cpu, y_cpu = peak_x.cpu().numpy(), peak_y.cpu().numpy()
        score_cpu = scores.float().cpu().numpy()
        size_cpu = support_sizes.cpu().numpy()

        hits: list[Hit] = []
        for i in range(len(source_cpu)):
            source_index = int(source_cpu[i])
            kernel_index = int(kernel_cpu[i])
            support_h, support_w = int(size_cpu[i, 0]), int(size_cpu[i, 1])
            x, y = int(x_cpu[i]), int(y_cpu[i])
            meta = gallery.meta[source_index]
            kernel_meta = bank.meta[kernel_index]
            hits.append(Hit(
                source_index=source_index,
                source_id=str(meta["source_id"]),
                score=float(score_cpu[i]),
                kernel_index=kernel_index,
                angle=int(kernel_meta["angle"]),
                scale=float(kernel_meta["scale"]),
                x_feat=x,
                y_feat=y,
                support_h=support_h,
                support_w=support_w,
                box_xyxy=feature_box_to_source_pixels(meta, x, y, support_w, support_h),
            ))
        return hits

    def _reduce_batch(
        self,
        features: torch.Tensor,
        source_sizes: torch.Tensor,
        bank: QueryBank,
        source_offset: int,
    ) -> tuple[torch.Tensor, ...] | None:
        out_h, out_w = self._output_size(source_sizes, bank.sizes)
        if out_h < 1 or out_w < 1:
            return None

        response = self._correlate(features, bank.features, (out_h, out_w))
        response = response / bank.areas.view(1, -1, 1, 1)

        valid = self._valid_positions(source_sizes, bank.sizes, out_h, out_w)
        response = response.masked_fill(~valid, float("-inf"))

        flat = response.flatten(2)
        kernel_peaks, flat_index = flat.max(dim=2)
        peak_y = torch.div(flat_index, out_w, rounding_mode="floor")
        peak_x = flat_index - peak_y * out_w

        source_scores, best_kernel = kernel_peaks.max(dim=1)
        local_source = torch.where(torch.isfinite(source_scores))[0]
        if local_source.numel() == 0:
            return None
        selected_kernel = best_kernel[local_source]
        return (
            local_source + source_offset,
            selected_kernel,
            peak_x[local_source, selected_kernel],
            peak_y[local_source, selected_kernel],
            source_scores[local_source],
            bank.sizes[selected_kernel],
        )

    def _output_size(
        self,
        source_sizes: torch.Tensor,
        kernel_sizes: torch.Tensor,
    ) -> tuple[int, int]:
        heights = source_sizes[:, 0].view(-1, 1) - kernel_sizes[:, 0].view(1, -1) + 1
        widths = source_sizes[:, 1].view(-1, 1) - kernel_sizes[:, 1].view(1, -1) + 1
        fits = (heights > 0) & (widths > 0)
        out_h = int(torch.where(fits, heights, torch.zeros_like(heights)).max().item())
        out_w = int(torch.where(fits, widths, torch.zeros_like(widths)).max().item())
        return out_h, out_w

    def _correlate(
        self,
        source: torch.Tensor,
        kernel: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        fft_h = int(source.shape[-2] + kernel.shape[-2] - 1)
        fft_w = int(source.shape[-1] + kernel.shape[-1] - 1)
        source_fft = torch.fft.rfft2(source.float(), s=(fft_h, fft_w), dim=(-2, -1))
        kernel_fft = torch.fft.rfft2(kernel.float(), s=(fft_h, fft_w), dim=(-2, -1))
        response_fft = torch.einsum(
            "nchw,kchw->nkhw", source_fft, torch.conj(kernel_fft)
        )
        response = torch.fft.irfft2(response_fft, s=(fft_h, fft_w), dim=(-2, -1))
        return response[..., : output_size[0], : output_size[1]].contiguous()

    def _valid_positions(
        self,
        source_sizes: torch.Tensor,
        kernel_sizes: torch.Tensor,
        out_h: int,
        out_w: int,
    ) -> torch.Tensor:
        source_h = source_sizes[:, 0].view(-1, 1, 1, 1)
        source_w = source_sizes[:, 1].view(-1, 1, 1, 1)
        kernel_h = kernel_sizes[:, 0].view(1, -1, 1, 1)
        kernel_w = kernel_sizes[:, 1].view(1, -1, 1, 1)
        y = torch.arange(out_h, device=self.device).view(1, 1, out_h, 1)
        x = torch.arange(out_w, device=self.device).view(1, 1, 1, out_w)
        return (y < source_h - kernel_h + 1) & (x < source_w - kernel_w + 1)


# =============================================================================
# Retrieval run
# =============================================================================

def rank_of(hits: list[Hit], source_id: str) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if hit.source_id == source_id:
            return rank
    return None


def run(cfg: Config) -> None:
    log = Logger(cfg.log_path)
    encoder = Encoder(cfg.model_name, cfg.feature_index)
    gallery, gallery_preproc_sec = timed(lambda: build_gallery(cfg, encoder))
    queries = query_records(cfg.query_dir)
    scorer = FFTScorer(encoder.device, cfg.gallery_batch_size)

    log.write(
        f"gallery: sources={len(gallery.meta)} preproc={gallery_preproc_sec:.3f}s "
        f"model={cfg.model_name} feature_index={cfg.feature_index} "
        f"stride={encoder.reduction} channels={encoder.channels} "
        f"device={encoder.device}"
    )
    log.write(
        f"queries={len(queries)} angles={cfg.angles} scales={cfg.scales} "
        "score=vanilla_masked_feature_cross_correlation"
    )

    ranks: list[int] = []
    preproc_times: list[float] = []
    score_times: list[float] = []
    for index, query in enumerate(queries, start=1):
        bank, preproc_sec = timed(lambda q=query: build_query_bank(q, cfg, encoder))
        hits, score_sec = timed(lambda: scorer.rank(gallery, bank))
        rank = rank_of(hits, query.expected_source_id)
        ranks.append(rank if rank is not None else 10**9)
        preproc_times.append(preproc_sec)
        score_times.append(score_sec)

        if hits:
            best = hits[0]
            log.write(
                f"[{index}/{len(queries)}] {query.query_id} "
                f"match={best.source_id} expected={query.expected_source_id} "
                f"rank={rank if rank is not None else '-'} "
                f"correct={best.source_id == query.expected_source_id} "
                f"sim={best.score:.6f} angle={best.angle} scale={best.scale:g} "
                f"box={','.join(map(str, best.box_xyxy))} "
                f"time_preproc={preproc_sec:.6f}s time_score={score_sec:.6f}s"
            )
        else:
            log.write(
                f"[{index}/{len(queries)}] {query.query_id} no_hit expected={query.expected_source_id} "
                f"rank=- correct=False time_preproc={preproc_sec:.6f}s time_score={score_sec:.6f}s"
            )

    rank_array = np.asarray(ranks, dtype=np.int64)
    log.write(
        f"summary: top1={np.mean(rank_array <= 1):.6f} recall@5={np.mean(rank_array <= 5):.6f} "
        f"recall@10={np.mean(rank_array <= 10):.6f} mean_preproc={np.mean(preproc_times):.6f}s "
        f"mean_score={np.mean(score_times):.6f}s"
    )


def main() -> None:
    cfg = Config(
        # Edit defaults here when needed, e.g. log_path=SCRIPT_DIR / "retrieval_console.txt".
    )
    run(cfg)


if __name__ == "__main__":
    main()
