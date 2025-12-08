from functools import lru_cache
from multiprocessing.pool import ThreadPool
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import numba as nb

from PIL import Image

from ultralytics.utils import NUM_THREADS, LOGGER, TQDM, colorstr
from .augment import Compose, Format, LetterBox, v8_transforms
from .dataset import YOLODataset

# Marker for full image slice (used in validation to include letterboxed full image)
FULL_IMAGE_SLICE_IDX = -1


# ==================== Numba-accelerated functions ====================

@nb.jit(nopython=True, fastmath=True, cache=True)
def _calculate_grid_coords(
    img_h: int, img_w: int,
    crop_h: int, crop_w: int,
    overlap_ratio: float
) -> np.ndarray:
    """
    Calculate SAHI grid coordinates for an image.
    
    Returns array of shape (N, 4) with [x1, y1, x2, y2] for each crop.
    Last row/column is aligned to image edge to ensure full coverage.
    """
    if img_h <= crop_h and img_w <= crop_w:
        return np.array([[0, 0, img_w, img_h]], dtype=np.int64)
    
    overlap_h = int(overlap_ratio * crop_h)
    overlap_w = int(overlap_ratio * crop_w)
    step_h = max(1, crop_h - overlap_h)
    step_w = max(1, crop_w - overlap_w)

    n_h = 1 if img_h <= crop_h else (img_h - crop_h + step_h - 1) // step_h + 1
    n_w = 1 if img_w <= crop_w else (img_w - crop_w + step_w - 1) // step_w + 1

    coords = np.empty((n_h * n_w, 4), dtype=np.int64)
    
    idx = 0
    for i in range(n_h):
        for j in range(n_w):
            # Align last row/column to image edge
            y1 = img_h - crop_h if (i == n_h - 1 and img_h > crop_h) else i * step_h
            x1 = img_w - crop_w if (j == n_w - 1 and img_w > crop_w) else j * step_w
            
            coords[idx, 0] = x1
            coords[idx, 1] = y1
            coords[idx, 2] = min(x1 + crop_w, img_w)
            coords[idx, 3] = min(y1 + crop_h, img_h)
            idx += 1
    
    return coords


@nb.jit(nopython=True, fastmath=True, cache=True)
def _count_grid_slices(
    img_h: int, img_w: int,
    crop_h: int, crop_w: int,
    overlap_ratio: float
) -> int:
    """Count number of grid slices without computing coordinates."""
    if img_h <= crop_h and img_w <= crop_w:
        return 1

    overlap_h = int(overlap_ratio * crop_h)
    overlap_w = int(overlap_ratio * crop_w)
    step_h = max(1, crop_h - overlap_h)
    step_w = max(1, crop_w - overlap_w)

    n_h = max(1, (img_h - crop_h + step_h - 1) // step_h + 1)
    n_w = max(1, (img_w - crop_w + step_w - 1) // step_w + 1)

    return n_h * n_w


@nb.jit(nopython=True, fastmath=True, cache=True)
def _xywh_to_xyxy(bboxes: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Convert normalized xywh to absolute xyxy coordinates."""
    n = len(bboxes)
    if n == 0:
        return np.empty((0, 4), dtype=np.float64)

    xyxy = np.empty((n, 4), dtype=np.float64)
    for i in range(n):
        cx, cy = bboxes[i, 0] * img_w, bboxes[i, 1] * img_h
        w, h = bboxes[i, 2] * img_w, bboxes[i, 3] * img_h
        xyxy[i, 0] = cx - w / 2
        xyxy[i, 1] = cy - h / 2
        xyxy[i, 2] = cx + w / 2
        xyxy[i, 3] = cy + h / 2
    return xyxy


@nb.jit(nopython=True, fastmath=True, cache=True)
def _filter_bboxes(
    boxes_xyxy: np.ndarray,
    cls: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    min_coverage: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter and transform bboxes to crop coordinates.
    
    Only keeps boxes with at least min_coverage fraction inside the crop.
    Returns normalized xywh coordinates relative to crop.
    """
    crop_w, crop_h = x2 - x1, y2 - y1
    n = len(boxes_xyxy)

    if n == 0:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)

    new_bboxes = np.empty((n, 4), dtype=np.float64)
    new_cls = np.empty(n, dtype=np.float64)
    count = 0

    for i in range(n):
        box = boxes_xyxy[i]

        # Compute intersection
        inter_x1 = max(box[0], x1)
        inter_y1 = max(box[1], y1)
        inter_x2 = min(box[2], x2)
        inter_y2 = min(box[3], y2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            continue

        # Check coverage threshold
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        if inter_area / (box_area + 1e-6) < min_coverage:
            continue

        # Transform to crop coordinates (clip to crop bounds)
        new_x1 = max(box[0] - x1, 0)
        new_y1 = max(box[1] - y1, 0)
        new_x2 = min(box[2] - x1, crop_w)
        new_y2 = min(box[3] - y1, crop_h)

        # Convert to normalized xywh
        new_bboxes[count, 0] = (new_x1 + new_x2) / 2 / crop_w
        new_bboxes[count, 1] = (new_y1 + new_y2) / 2 / crop_h
        new_bboxes[count, 2] = (new_x2 - new_x1) / crop_w
        new_bboxes[count, 3] = (new_y2 - new_y1) / crop_h
        new_cls[count] = cls[i]
        count += 1

    return new_bboxes[:count], new_cls[:count]


# ==================== LRU-cached wrappers ====================

@lru_cache(maxsize=4096)
def cached_grid_coords(img_h: int, img_w: int, crop_h: int, crop_w: int, overlap_ratio: float) -> Tuple[Tuple[int, ...], ...]:
    """Cached grid coordinates (hashable tuple output for LRU cache)."""
    coords = _calculate_grid_coords(img_h, img_w, crop_h, crop_w, overlap_ratio)
    return tuple(tuple(int(c) for c in coord) for coord in coords)


@lru_cache(maxsize=4096)
def cached_grid_count(img_h: int, img_w: int, crop_h: int, crop_w: int, overlap_ratio: float) -> int:
    """Cached grid slice count."""
    return _count_grid_slices(img_h, img_w, crop_h, crop_w, overlap_ratio)


# ==================== SAHIDataset ====================

class SAHIDataset(YOLODataset):
    """
    SAHI-enabled YOLO dataset for small object detection.
    
    Splits large images into overlapping crops for training/validation.
    During validation, also includes letterboxed full image for large object detection.
    
    Args:
        img_path: Path to images directory
        cut_strategy: "grid" for deterministic slicing, "random" for object-centered crops
        crop_size: Size of each crop (square)
        overlap_ratio: Overlap between adjacent crops (0-1)
        sampling_rate: Fraction of grid crops to sample (for random strategy)
        min_object_coverage: Minimum fraction of object area in crop to keep it
        object_crop_prob: Probability to center crop on object (for random strategy)
        buffer_size: Per-worker image cache size
    """

    def __init__(
        self,
        img_path: str,
        cut_strategy: str = "grid",
        crop_size: int = 640,
        overlap_ratio: float = 0.2,
        sampling_rate: float = 1.0,
        min_object_coverage: float = 0.3,
        object_crop_prob: float = 0.7,
        buffer_size: int = 8,
        *args,
        **kwargs,
    ):
        # Normalize strategy name
        self.cut_strategy = "random" if cut_strategy in ("random_crop", "random") else cut_strategy
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.sampling_rate = sampling_rate
        self.min_object_coverage = min_object_coverage
        self.object_crop_prob = object_crop_prob
        self.use_slicing = (self.cut_strategy == "grid")
        self._epoch = 0
        self._buffer_size = buffer_size
        
        self.image_shapes: Dict[int, Tuple[int, int]] = {}

        super().__init__(img_path=img_path, *args, **kwargs)

        self.sahi = True
        self._cache_image_shapes()
        self.slice_indices = self._precompute_slices()
        
        # Sort by image index for better cache locality
        if self.use_slicing:
            self.slice_indices.sort(key=lambda x: x[0])

        # Per-worker image cache (initialized lazily)
        self._worker_image_cache: Optional[Dict[int, np.ndarray]] = None
        self._worker_id: Optional[int] = None

        self._log_config()

    def _cache_image_shapes(self):
        """Pre-cache all image shapes using PIL (fast, doesn't load pixels)."""
        self.image_shapes.clear()
        
        def get_shape(path):
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    return (h, w)
            except Exception:
                return None
        
        with ThreadPool(NUM_THREADS) as pool:
            shapes = pool.map(get_shape, self.im_files)
        
        for idx, shape in enumerate(shapes):
            if shape is not None:
                self.image_shapes[idx] = shape
            else:
                self.image_shapes[idx] = (self.imgsz, self.imgsz)
                LOGGER.warning(f"Could not read shape for {self.im_files[idx]}")
        
        LOGGER.info(f"{colorstr('SAHIDataset')}: Cached {len(self.image_shapes)} image shapes")

    def _get_shape(self, idx: int) -> Tuple[int, int]:
        """Get cached image shape (h, w)."""
        return self.image_shapes.get(idx, (self.imgsz, self.imgsz))

    def __len__(self) -> int:
        return len(self.slice_indices)

    def get_image_and_label(self, index: int) -> Dict[str, Any]:
        """Get crop image and transformed labels for a slice index."""
        img_idx, slice_idx, coords = self.slice_indices[index]
        x1, y1, x2, y2 = coords

        im = self._get_cached_image(img_idx)
        actual_h, actual_w = im.shape[:2]
        is_full_image = (slice_idx == FULL_IMAGE_SLICE_IDX)
        
        if is_full_image:
            # Letterbox full image to crop_size
            crop_im, ratio, (pad_w, pad_h) = self._letterbox_image(im)
            coords = (0, 0, actual_w, actual_h)
            resized_shape = crop_im.shape[:2]
            ratio_pad = (ratio, (pad_w, pad_h))
        else:
            # Extract crop (with bounds checking)
            x1 = min(x1, max(0, actual_w - self.crop_size))
            y1 = min(y1, max(0, actual_h - self.crop_size))
            x2 = min(x1 + self.crop_size, actual_w)
            y2 = min(y1 + self.crop_size, actual_h)
            coords = (x1, y1, x2, y2)

            crop_im = im[y1:y2, x1:x2].copy()
            resized_shape = crop_im.shape[:2]
            ratio_pad = ((1.0, 1.0), (0, 0))

        # Transform labels
        labels = deepcopy(self.labels[img_idx])
        if is_full_image:
            crop_labels = self._transform_labels_for_full_image(labels, ratio_pad, actual_h, actual_w)
        else:
            crop_labels = self._transform_labels_to_crop(labels, coords, actual_h, actual_w)

        labels.update({
            "img": crop_im,
            "ori_shape": (actual_h, actual_w),
            "resized_shape": resized_shape,
            "ratio_pad": ratio_pad,
            "bboxes": crop_labels["bboxes"],
            "cls": crop_labels["cls"],
            "original_img_idx": img_idx,
            "slice_idx": slice_idx,
            "slice_coords": coords,
            "is_full_image": is_full_image,
        })

        return self.update_labels_info(labels)

    def _letterbox_image(self, im: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Resize image with letterbox padding to crop_size."""
        import cv2
        
        h, w = im.shape[:2]
        target = self.crop_size
        scale = min(target / h, target / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        pad_h, pad_w = target - new_h, target - new_w
        top, left = pad_h // 2, pad_w // 2
        
        padded = np.full((target, target, 3), 114, dtype=np.uint8)
        padded[top:top + new_h, left:left + new_w] = resized
        
        return padded, (scale, scale), (left, top)

    def _transform_labels_for_full_image(
        self,
        labels: Dict[str, Any],
        ratio_pad: Tuple[Tuple[float, float], Tuple[int, int]],
        img_h: int,
        img_w: int,
    ) -> Dict[str, np.ndarray]:
        """Transform labels for letterboxed full image."""
        bboxes = labels.get("bboxes", np.zeros((0, 4), dtype=np.float32))
        cls = labels.get("cls", np.zeros((0, 1), dtype=np.float32))

        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32) if bboxes is not None and len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        if not isinstance(cls, np.ndarray):
            cls = np.array(cls, dtype=np.float32) if cls is not None and len(cls) > 0 else np.zeros((0, 1), dtype=np.float32)

        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(-1, 4) if len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        if cls.ndim == 1:
            cls = cls.reshape(-1, 1) if len(cls) > 0 else np.zeros((0, 1), dtype=np.float32)

        if len(bboxes) == 0:
            return {"bboxes": np.zeros((0, 4), dtype=np.float32), "cls": np.zeros((0, 1), dtype=np.float32)}

        ratio, (pad_w, pad_h) = ratio_pad
        scale = ratio[0]
        target = self.crop_size

        # Transform: original normalized -> letterbox normalized
        new_bboxes = np.zeros_like(bboxes)
        new_bboxes[:, 0] = (bboxes[:, 0] * img_w * scale + pad_w) / target
        new_bboxes[:, 1] = (bboxes[:, 1] * img_h * scale + pad_h) / target
        new_bboxes[:, 2] = bboxes[:, 2] * img_w * scale / target
        new_bboxes[:, 3] = bboxes[:, 3] * img_h * scale / target

        return {"bboxes": new_bboxes.astype(np.float32), "cls": cls.astype(np.float32)}

    def _transform_labels_to_crop(
        self,
        labels: Dict[str, Any],
        coords: Tuple[int, int, int, int],
        img_h: int,
        img_w: int,
    ) -> Dict[str, np.ndarray]:
        """Transform labels from full image to crop coordinates."""
        x1, y1, x2, y2 = coords

        bboxes = labels.get("bboxes", np.array([]))
        cls = labels.get("cls", np.array([]))

        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes) if bboxes is not None and len(bboxes) > 0 else np.zeros((0, 4))
        if not isinstance(cls, np.ndarray):
            cls = np.array(cls) if cls is not None and len(cls) > 0 else np.zeros((0,))

        n_cls_cols = cls.shape[1] if cls.ndim > 1 else 1

        if len(bboxes) == 0:
            return {"bboxes": np.zeros((0, 4), dtype=np.float32), "cls": np.zeros((0, n_cls_cols), dtype=np.float32)}

        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(-1, 4)

        cls_flat = cls.flatten() if cls.ndim > 1 else cls

        # Filter and transform bboxes
        boxes_xyxy = _xywh_to_xyxy(bboxes.astype(np.float64), img_h, img_w)
        new_bboxes, new_cls = _filter_bboxes(boxes_xyxy, cls_flat.astype(np.float64), x1, y1, x2, y2, self.min_object_coverage)

        new_bboxes = new_bboxes.astype(np.float32)
        if len(new_bboxes) == 0:
            return {"bboxes": np.zeros((0, 4), dtype=np.float32), "cls": np.zeros((0, n_cls_cols), dtype=np.float32)}
        
        # Reshape cls to match original structure
        cls_values = new_cls.astype(np.float32).flatten()
        if n_cls_cols > 1:
            new_cls = np.zeros((len(cls_values), n_cls_cols), dtype=np.float32)
            new_cls[:, 0] = cls_values
        else:
            new_cls = cls_values.reshape(-1, 1)
        
        return {"bboxes": new_bboxes, "cls": new_cls}

    def _get_cached_image(self, img_idx: int) -> np.ndarray:
        """Get image with per-worker LRU caching."""
        import torch
        
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info else -1
        
        # Reset cache if worker changed
        if self._worker_image_cache is None or self._worker_id != current_worker_id:
            self._worker_image_cache = {}
            self._worker_id = current_worker_id
        
        if img_idx in self._worker_image_cache:
            return self._worker_image_cache[img_idx]
        
        im, _, _ = self.load_image(img_idx)
        if im is None:
            raise FileNotFoundError(f"Image not found: {self.im_files[img_idx]}")
        
        # Simple FIFO eviction
        if len(self._worker_image_cache) >= self._buffer_size:
            oldest_key = next(iter(self._worker_image_cache))
            del self._worker_image_cache[oldest_key]
        
        self._worker_image_cache[img_idx] = im
        return im

    def _precompute_slices(self) -> List[Tuple[int, int, Tuple[int, int, int, int]]]:
        """Precompute all slice indices for the dataset."""
        crop = self.crop_size
        overlap = self.overlap_ratio
        include_full_image = not self.augment and self.use_slicing

        def process_grid(idx: int):
            img_h, img_w = self._get_shape(idx)
            coords = cached_grid_coords(img_h, img_w, crop, crop, overlap)
            return idx, list(coords)

        def process_random(idx: int):
            img_h, img_w = self._get_shape(idx)
            n_grid = cached_grid_count(img_h, img_w, crop, crop, overlap)
            n_sampled = max(1, round(n_grid * self.sampling_rate))
            bboxes = self._get_bboxes_for_idx(idx)
            seed = 42 + self._epoch * 100000 + idx
            coords = _generate_random_coords(img_h, img_w, bboxes, n_sampled, crop, self.object_crop_prob, seed)
            return idx, coords

        process_fn = process_grid if self.use_slicing else process_random
        desc = colorstr("SAHI grid slices" if self.use_slicing else "SAHI random crops")

        slice_indices = []
        max_crops_per_image = {}

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(process_fn, range(self.ni))
            for idx, coords in TQDM(results, total=self.ni, desc=desc):
                n_crops = len(coords)
                for s_idx, c in enumerate(coords):
                    slice_indices.append((idx, s_idx, tuple(c)))
                
                # Add full image for validation
                if include_full_image:
                    img_h, img_w = self._get_shape(idx)
                    slice_indices.append((idx, FULL_IMAGE_SLICE_IDX, (0, 0, img_w, img_h)))
                    n_crops += 1
                
                max_crops_per_image[idx] = n_crops

        self.max_crops_per_image = max_crops_per_image
        if include_full_image:
            LOGGER.info(f"{colorstr('SAHIDataset')}: Added {self.ni} full images for validation")
        
        return slice_indices

    def _get_bboxes_for_idx(self, idx: int) -> np.ndarray:
        """Get bboxes for an image index."""
        bboxes = self.labels[idx].get("bboxes", np.array([]))
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes) if bboxes is not None and len(bboxes) > 0 else np.zeros((0, 4))
        if bboxes.ndim == 1 and len(bboxes) > 0:
            bboxes = bboxes.reshape(-1, 4)
        elif len(bboxes) == 0:
            bboxes = np.zeros((0, 4))
        return bboxes.astype(np.float64)

    def build_transforms(self, hyp: Optional[Any] = None) -> Compose:
        """Build augmentation transforms."""
        if self.augment:
            transforms = v8_transforms(dataset=self, imgsz=self.crop_size, hyp=hyp, stretch=False)
        else:
            transforms = Compose([LetterBox(new_shape=(self.crop_size, self.crop_size), scaleup=False)])

        transforms.append(Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            return_obb=self.use_obb,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio if hyp else 4,
            mask_overlap=hyp.overlap_mask if hyp else True,
            bgr=hyp.bgr if hyp and self.augment else 0.0,
        ))
        return transforms

    def on_epoch_end(self):
        """Regenerate random crops for next epoch."""
        self._epoch += 1
        
        if not self.use_slicing:
            self.slice_indices = self._precompute_slices()
            if self.use_slicing:
                self.slice_indices.sort(key=lambda x: x[0])
            LOGGER.info(f"{colorstr('SAHIDataset')}: Regenerated {len(self.slice_indices)} random crops for epoch {self._epoch}")

    def _log_config(self):
        """Log dataset configuration."""
        mode = "train" if self.augment else "val"
        lines = [
            f"\n{colorstr('SAHIDataset')} ({mode}):",
            f"  strategy: {self.cut_strategy}",
            f"  crop_size: {self.crop_size}",
            f"  images: {self.ni}",
            f"  slices: {len(self.slice_indices)}",
            f"  slices/image: {len(self.slice_indices) / max(1, self.ni):.1f}",
        ]
        if self.use_slicing:
            lines.append(f"  overlap_ratio: {self.overlap_ratio}")
        else:
            lines.extend([f"  sampling_rate: {self.sampling_rate}", f"  object_crop_prob: {self.object_crop_prob}"])
        LOGGER.info("\n".join(lines))


def _generate_random_coords(
    img_h: int,
    img_w: int,
    bboxes: np.ndarray,
    num_crops: int,
    crop_size: int,
    object_crop_prob: float,
    seed: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate random crop coordinates, biased towards objects.
    
    Args:
        object_crop_prob: Probability to center crop on a random object
        seed: Random seed for reproducibility
    """
    rng = random.Random(seed)
    max_x, max_y = max(0, img_w - crop_size), max(0, img_h - crop_size)

    boxes_xyxy = _xywh_to_xyxy(bboxes.astype(np.float64), img_h, img_w) if len(bboxes) > 0 else np.array([])
    has_boxes = len(boxes_xyxy) > 0
    coords = []

    for _ in range(num_crops):
        if rng.random() < object_crop_prob and has_boxes:
            # Center on random object with jitter
            box = boxes_xyxy[rng.randint(0, len(boxes_xyxy) - 1)]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            jitter_x = rng.uniform(-crop_size * 0.3, crop_size * 0.3)
            jitter_y = rng.uniform(-crop_size * 0.3, crop_size * 0.3)
            x1 = int(cx - crop_size / 2 + jitter_x)
            y1 = int(cy - crop_size / 2 + jitter_y)
        else:
            # Random position
            x1 = rng.randint(0, max_x) if max_x > 0 else 0
            y1 = rng.randint(0, max_y) if max_y > 0 else 0

        # Clamp to image bounds
        x1 = max(0, min(x1, max_x))
        y1 = max(0, min(y1, max_y))
        coords.append((x1, y1, min(x1 + crop_size, img_w), min(y1 + crop_size, img_h)))

    return coords
