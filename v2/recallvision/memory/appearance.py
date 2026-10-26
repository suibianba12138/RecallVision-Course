"""Small deterministic HSV appearance descriptors for P2A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class AppearanceExtractionError(RuntimeError):
    """Raised when a valid crop/descriptor cannot be produced."""


@dataclass(frozen=True)
class HSVHistogramConfig:
    """Fixed descriptor settings; V is intentionally ignored."""

    h_bins: int = 32
    s_bins: int = 32


def extract_hsv_histogram(
    frame: Any,
    bbox: tuple[float, float, float, float],
    *,
    config: HSVHistogramConfig = HSVHistogramConfig(),
) -> tuple[float, ...]:
    """Extract an L1-normalized H/S histogram from a BGR object crop."""

    try:
        import cv2
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise AppearanceExtractionError("OpenCV and NumPy are required for HSV extraction") from error
    if config.h_bins <= 0 or config.s_bins <= 0:
        raise ValueError("histogram bins must be positive")
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise AppearanceExtractionError("frame is not a valid image array")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, int(x1)))
    top = max(0, min(height, int(y1)))
    right = max(0, min(width, int(x2 + 0.999999)))
    bottom = max(0, min(height, int(y2 + 0.999999)))
    if right <= left or bottom <= top:
        raise AppearanceExtractionError(f"empty crop for bbox={bbox}")
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise AppearanceExtractionError(f"empty crop for bbox={bbox}")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1], None, [config.h_bins, config.s_bins], [0, 180, 0, 256]
    ).astype(np.float64).reshape(-1)
    total = float(histogram.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise AppearanceExtractionError(f"invalid histogram for bbox={bbox}")
    normalized = histogram / total
    if not np.all(np.isfinite(normalized)):
        raise AppearanceExtractionError(f"non-finite histogram for bbox={bbox}")
    return tuple(float(value) for value in normalized)


def histogram_intersection_similarity(
    first: tuple[float, ...], second: tuple[float, ...]
) -> float:
    """Return the deterministic intersection of two L1-normalized histograms."""

    if not first or len(first) != len(second):
        raise ValueError("histograms must be non-empty and have equal length")
    similarity = sum(min(left, right) for left, right in zip(first, second))
    return max(0.0, min(1.0, float(similarity)))
