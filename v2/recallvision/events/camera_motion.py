"""Sparse, model-free camera-motion estimates for event evidence gating.

Only the candidate support frames are decoded. ORB correspondences and a
RANSAC partial-affine fit estimate the common image transform; object boxes are
excluded from feature selection so target motion does not define the camera
estimate. OpenCV is imported lazily and is never downloaded by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class VideoCameraMotionEstimator:
    """Estimate a normalized image transform between two evidence windows."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        self._cv2: Any = None
        self._np: Any = None
        self._capture: Any = None
        self._frame_size: tuple[int, int] | None = None
        self._processed_size: tuple[int, int] | None = None
        self._orb: Any = None
        self._features: dict[int, tuple[list[Any], Any]] = {}
        self._available_reason: str | None = None
        try:
            import cv2
            import numpy as np
        except ImportError:
            self._available_reason = "OPENCV_UNAVAILABLE"
            return
        if not self.video_path.is_file():
            self._available_reason = "VIDEO_FILE_UNAVAILABLE"
            return
        self._cv2, self._np = cv2, np
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            self._available_reason = "VIDEO_DECODE_UNAVAILABLE"
            capture.release()
            return
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            self._available_reason = "VIDEO_METADATA_UNAVAILABLE"
            capture.release()
            return
        self._capture = capture
        self._frame_size = (width, height)
        scale = min(1.0, 640.0 / width)
        self._processed_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        self._orb = cv2.ORB_create(
            nfeatures=1800,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=12,
            fastThreshold=12,
        )

    def estimate(
        self,
        before_frame: int,
        after_frame: int,
        *,
        before_bboxes: Sequence[Sequence[float]] = (),
        after_bboxes: Sequence[Sequence[float]] = (),
    ) -> dict[str, Any]:
        if self._capture is None:
            return {"status": "UNVERIFIED", "reason": self._available_reason or "VIDEO_MOTION_UNAVAILABLE"}
        if before_frame == after_frame:
            return {"status": "UNVERIFIED", "reason": "IDENTICAL_SUPPORT_FRAMES"}
        left_keypoints, left_descriptors = self._frame_features(before_frame)
        right_keypoints, right_descriptors = self._frame_features(after_frame)
        if left_descriptors is None or right_descriptors is None:
            return {"status": "UNVERIFIED", "reason": "INSUFFICIENT_IMAGE_FEATURES"}

        left_indices = self._indices_outside_boxes(left_keypoints, before_bboxes)
        right_indices = self._indices_outside_boxes(right_keypoints, after_bboxes)
        if len(left_indices) < 20 or len(right_indices) < 20:
            return {"status": "UNVERIFIED", "reason": "INSUFFICIENT_BACKGROUND_FEATURES"}

        cv2, np = self._cv2, self._np
        left_descriptors = left_descriptors[left_indices]
        right_descriptors = right_descriptors[right_indices]
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_descriptors, right_descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(good) < 20:
            return {
                "status": "UNVERIFIED",
                "reason": "INSUFFICIENT_FEATURE_MATCHES",
                "good_match_count": len(good),
            }
        source = np.float32([left_keypoints[left_indices[item.queryIdx]].pt for item in good]).reshape(-1, 1, 2)
        target = np.float32([right_keypoints[right_indices[item.trainIdx]].pt for item in good]).reshape(-1, 1, 2)
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=3000,
            confidence=0.99,
        )
        if affine is None or inlier_mask is None:
            return {
                "status": "UNVERIFIED",
                "reason": "AFFINE_ESTIMATION_FAILED",
                "good_match_count": len(good),
            }
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / len(good)
        if inlier_count < 15 or inlier_ratio < 0.25:
            return {
                "status": "UNVERIFIED",
                "reason": "LOW_GEOMETRIC_CONSENSUS",
                "good_match_count": len(good),
                "inlier_count": inlier_count,
                "inlier_ratio": inlier_ratio,
            }

        proc_width, proc_height = self._processed_size
        normalized_transform = [
            [float(affine[0, 0]), float(affine[0, 1] * proc_height / proc_width), float(affine[0, 2] / proc_width)],
            [float(affine[1, 0] * proc_width / proc_height), float(affine[1, 1]), float(affine[1, 2] / proc_height)],
        ]
        return {
            "status": "ESTIMATED",
            "method": "ORB_RANSAC_PARTIAL_AFFINE",
            "before_frame": before_frame,
            "after_frame": after_frame,
            "transform_normalized": normalized_transform,
            "good_match_count": len(good),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "excluded_target_boxes": {"before": len(before_bboxes), "after": len(after_bboxes)},
        }

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _frame_features(self, frame_index: int) -> tuple[list[Any], Any]:
        if frame_index not in self._features:
            cv2, np = self._cv2, self._np
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = self._capture.read()
            if not ok:
                self._features[frame_index] = ([], None)
            else:
                width, height = self._processed_size
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
                self._features[frame_index] = self._orb.detectAndCompute(gray, None)
        return self._features[frame_index]

    def _indices_outside_boxes(
        self,
        keypoints: Sequence[Any],
        bboxes: Sequence[Sequence[float]],
    ) -> list[int]:
        if not bboxes:
            return list(range(len(keypoints)))
        frame_width, frame_height = self._frame_size
        proc_width, proc_height = self._processed_size
        sx, sy = proc_width / frame_width, proc_height / frame_height
        excluded: list[tuple[float, float, float, float]] = []
        for bbox in bboxes:
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            pad_x, pad_y = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
            excluded.append(((x1 - pad_x) * sx, (y1 - pad_y) * sy, (x2 + pad_x) * sx, (y2 + pad_y) * sy))
        return [
            index
            for index, keypoint in enumerate(keypoints)
            if not any(x1 <= keypoint.pt[0] <= x2 and y1 <= keypoint.pt[1] <= y2 for x1, y1, x2, y2 in excluded)
        ]
