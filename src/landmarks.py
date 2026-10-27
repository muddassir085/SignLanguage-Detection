"""Shared MediaPipe feature extraction used for files and webcam frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

HAND_LANDMARKS = 21
HAND_SHAPE_SIZE = HAND_LANDMARKS * 3
FEATURE_SIZE = 141


@dataclass(frozen=True)
class FeatureExtraction:
    """A feature vector plus the underlying MediaPipe result for optional drawing."""

    features: np.ndarray
    result: Any


def feature_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(
            f"{side}_shape_{landmark}_{axis}"
            for landmark in range(HAND_LANDMARKS)
            for axis in ("x", "y", "z")
        )
        names += [f"{side}_x", f"{side}_y", f"{side}_present"]
    names += ["face_x", "face_y", "face_present"]
    for side in ("left", "right"):
        names += [f"{side}_face_dx", f"{side}_face_dy", f"{side}_face_distance"]
    return names


def create_holistic(static_image_mode: bool = True):
    """Create the single MediaPipe detector configuration used throughout the project."""
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=static_image_mode,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _points(landmarks: Any | None) -> np.ndarray | None:
    if landmarks is None:
        return None
    return np.asarray([[point.x, point.y, point.z] for point in landmarks.landmark], dtype=np.float32)


def _hand_features(landmarks: Any | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return translation/scale-normalised hand shape, image centre, and presence."""
    points = _points(landmarks)
    if points is None or points.shape != (HAND_LANDMARKS, 3):
        return np.zeros(HAND_SHAPE_SIZE, dtype=np.float32), np.zeros(2, dtype=np.float32), 0.0

    wrist = points[0]
    # Landmark 9 (middle-finger MCP) supplies a stable, rotation-agnostic scale.
    scale = float(np.linalg.norm(points[9] - wrist))
    if scale < 1e-6:
        scale = float(np.linalg.norm(np.ptp(points[:, :2], axis=0)))
    scale = max(scale, 1e-6)
    shape = ((points - wrist) / scale).reshape(-1).astype(np.float32)
    centre = points[:, :2].mean(axis=0).astype(np.float32)
    return shape, centre, 1.0


def _face_features(landmarks: Any | None) -> tuple[np.ndarray, float]:
    points = _points(landmarks)
    if points is None or not len(points):
        return np.zeros(2, dtype=np.float32), 0.0
    # Median is robust to an occasional face-mesh point at the contour edge.
    return np.median(points[:, :2], axis=0).astype(np.float32), 1.0


def extract_features_from_result(result: Any) -> np.ndarray:
    """Build the 141-value, zero-filled feature vector from one Holistic result."""
    left_shape, left_centre, left_present = _hand_features(result.left_hand_landmarks)
    right_shape, right_centre, right_present = _hand_features(result.right_hand_landmarks)
    face_centre, face_present = _face_features(result.face_landmarks)

    parts = [
        left_shape,
        left_centre,
        np.asarray([left_present], dtype=np.float32),
        right_shape,
        right_centre,
        np.asarray([right_present], dtype=np.float32),
        face_centre,
        np.asarray([face_present], dtype=np.float32),
    ]
    for centre, present in ((left_centre, left_present), (right_centre, right_present)):
        if present and face_present:
            delta = centre - face_centre
            spatial = np.asarray([delta[0], delta[1], np.linalg.norm(delta)], dtype=np.float32)
        else:
            spatial = np.zeros(3, dtype=np.float32)
        parts.append(spatial)

    vector = np.concatenate(parts).astype(np.float32)
    if vector.shape != (FEATURE_SIZE,):
        raise RuntimeError(f"Unexpected feature size {vector.shape}; expected {(FEATURE_SIZE,)}")
    return vector


def extract_features_from_rgb(rgb_image: np.ndarray, holistic: Any) -> FeatureExtraction:
    """Run Holistic and apply the exact common feature calculation."""
    rgb_image.flags.writeable = False
    result = holistic.process(rgb_image)
    rgb_image.flags.writeable = True
    return FeatureExtraction(extract_features_from_result(result), result)

