"""Turn MediaPipe hand landmarks into a pose-invariant feature vector."""

from __future__ import annotations

import numpy as np

LANDMARKS_PER_HAND = 21
COORDS = 3
HANDS = 2
FEATURE_SIZE = LANDMARKS_PER_HAND * COORDS * HANDS  # 126


def _points(hand) -> np.ndarray:
    pts = hand.landmark if hasattr(hand, "landmark") else hand
    return np.array([[lm.x, lm.y, lm.z] for lm in pts], dtype=np.float32)


def _normalize_hand(xyz: np.ndarray) -> np.ndarray:
    """xyz shape (21, 3). Center on wrist and scale by palm size."""
    wrist = xyz[0].copy()
    centered = xyz - wrist
    scale = np.linalg.norm(centered[9])  # middle-finger MCP
    if scale < 1e-6:
        scale = np.max(np.linalg.norm(centered, axis=1)) or 1.0
    return (centered / scale).reshape(-1)


def _hand_label(hand_info, index: int) -> str:
    if not hand_info:
        return "Right"
    item = hand_info[index]
    if hasattr(item, "classification"):
        return item.classification[0].label
    if isinstance(item, (list, tuple)):
        cat = item[0]
        return getattr(cat, "category_name", None) or getattr(cat, "display_name", "Right")
    return getattr(item, "category_name", None) or "Right"


def landmarks_to_features(multi_hand_landmarks, handedness) -> np.ndarray | None:
    """Build a 126-d vector: left hand then right hand. Missing hands are zeros."""
    if not multi_hand_landmarks:
        return None

    left = np.zeros(LANDMARKS_PER_HAND * COORDS, dtype=np.float32)
    right = np.zeros(LANDMARKS_PER_HAND * COORDS, dtype=np.float32)

    for i, landmarks in enumerate(multi_hand_landmarks):
        vec = _normalize_hand(_points(landmarks))
        label = _hand_label(handedness, i)
        if str(label).lower().startswith("left"):
            left = vec
        else:
            right = vec

    return np.concatenate([left, right]).astype(np.float32)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
