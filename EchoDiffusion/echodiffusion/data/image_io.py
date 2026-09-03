"""Image helpers, isolated so the audio-only path never imports cv2.

Training with ``data.use_image: false`` should run in an environment with no
OpenCV at all, so every import here is deferred into the function bodies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_image_message(msg: dict, path: str | Path, quality: int = 92) -> None:
    """Write a decoded ``Image`` / ``CompressedImage`` message to disk as JPEG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if "format" in msg and isinstance(msg.get("data"), (bytes, bytearray)):
        # CompressedImage: already JPEG/PNG bytes, so re-encoding would only
        # lose quality.  Keep the original container.
        fmt = str(msg["format"]).lower()
        suffix = ".png" if "png" in fmt else ".jpg"
        path.with_suffix(suffix).write_bytes(msg["data"])
        return

    import cv2
    img = np.asarray(msg["data"])
    if img.ndim == 3 and img.shape[2] >= 3:
        img = img[..., ::-1]                      # RGB -> BGR for cv2
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def load_image(path: str | Path, size: tuple[int, int] | None = None
               ) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1], optionally resized to (H, W)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image {path}")
    img = img[..., ::-1]                          # BGR -> RGB
    if size is not None:
        h, w = size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img, dtype=np.float32) / 255.0


#: ImageNet statistics -- DINOv2 was trained with these.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) in [0, 1] -> (3, H, W) normalised, channel-first."""
    return np.ascontiguousarray(
        ((img - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
