from functools import lru_cache
from multiprocessing.pool import ThreadPool
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numba as nb

from PIL import Image

from ultralytics.utils import NUM_THREADS, LOGGER, TQDM, colorstr
from .augment import Compose, Format, LetterBox, v8_transforms
from .dataset import YOLODataset

# Marker for full image slice (used in validation to include letterboxed full image)
FULL_IMAGE_SLICE_IDX = -1

@nb.jit(nopython=True, fastmath=True, cache=True)
def _polygon_area(polygon: np.ndarray) -> float:
    """Calculate polygon area using shoelace formula (numba-accelerated)."""
    n = len(polygon)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i, 0] * polygon[j, 1]
        area -= polygon[j, 0] * polygon[i, 1]
    return abs(area) * 0.5


def _clip_polygon_to_rect(
    polygon: np.ndarray,
    x1: float, y1: float, x2: float, y2: float
) -> Optional[np.ndarray]:
    """
    Clip polygon to rectangle using Sutherland-Hodgman algorithm.
    
    Args:
        polygon: [N, 2] array of (x, y) points (absolute coordinates)
        x1, y1, x2, y2: Rectangle bounds
        
    Returns:
        Clipped polygon or None if completely outside
    """
    def inside(p, edge):
        if edge == 'left':
            return p[0] >= x1
        elif edge == 'right':
            return p[0] <= x2
        elif edge == 'top':
            return p[1] >= y1
        else:  # bottom
            return p[1] <= y2
    
    def intersection(p1, p2, edge):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        if edge == 'left':
            t = (x1 - p1[0]) / (dx + 1e-10)
            return np.array([x1, p1[1] + t * dy])
        elif edge == 'right':
            t = (x2 - p1[0]) / (dx + 1e-10)
            return np.array([x2, p1[1] + t * dy])
        elif edge == 'top':
            t = (y1 - p1[1]) / (dy + 1e-10)
            return np.array([p1[0] + t * dx, y1])
        else:  # bottom
            t = (y2 - p1[1]) / (dy + 1e-10)
            return np.array([p1[0] + t * dx, y2])
    
    output = list(polygon)
    
    for edge in ['left', 'right', 'top', 'bottom']:
        if len(output) == 0:
            return None
        
        input_poly = output
        output = []
        
        for i in range(len(input_poly)):
            current = input_poly[i]
            next_pt = input_poly[(i + 1) % len(input_poly)]
            
            if inside(current, edge):
                if inside(next_pt, edge):
                    output.append(next_pt)
                else:
                    output.append(intersection(current, next_pt, edge))
            elif inside(next_pt, edge):
                output.append(intersection(current, next_pt, edge))
                output.append(next_pt)
    
    if len(output) < 3:
        return None
    
    return np.array(output, dtype=np.float64)


def _transform_segments_to_crop(
    segments: List[np.ndarray],
    coords: Tuple[int, int, int, int],
    img_h: int, img_w: int,
    min_area_ratio: float = 0.3
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Transform polygon segments from original image to crop coordinates.
    
    Args:
        segments: List of [N, 2] arrays with normalized (x, y) points
        coords: (x1, y1, x2, y2) crop coordinates in pixels
        img_h, img_w: Original image dimensions
        min_area_ratio: Minimum fraction of original area to keep segment
        
    Returns:
        new_segments: List of transformed segments (normalized to crop)
        valid_mask: Boolean array indicating which segments survived
    """
    x1, y1, x2, y2 = coords
    crop_w, crop_h = x2 - x1, y2 - y1
    
    new_segments = []
    valid_mask = np.zeros(len(segments), dtype=bool)
    
    for idx, seg in enumerate(segments):
        if len(seg) < 3:
            continue
        
        # Denormalize to absolute coordinates
        poly = seg.copy().astype(np.float64)
        poly[:, 0] *= img_w
        poly[:, 1] *= img_h
        
        # Calculate original area
        original_area = _polygon_area(poly)
        if original_area < 1e-6:
            continue
        
        # Clip to crop bounds
        clipped = _clip_polygon_to_rect(poly, x1, y1, x2, y2)
        
        if clipped is None or len(clipped) < 3:
            continue
        
        # Check area threshold
        clipped_area = _polygon_area(clipped)
        if clipped_area / original_area < min_area_ratio:
            continue
        
        # Transform to crop coordinates (normalized)
        clipped[:, 0] = (clipped[:, 0] - x1) / crop_w
        clipped[:, 1] = (clipped[:, 1] - y1) / crop_h
        
        # Clip to [0, 1]
        clipped = np.clip(clipped, 0, 1)
        
        new_segments.append(clipped.astype(np.float32))
        valid_mask[idx] = True
    
    return new_segments, valid_mask


def _transform_segments_for_letterbox(
    segments: List[np.ndarray],
    ratio: Tuple[float, float],
    pad: Tuple[int, int],
    img_h: int, img_w: int,
    target_size: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Transform polygon segments for letterboxed full image.
    
    Returns:
        new_segments: List of transformed segments
        valid_mask: Boolean array indicating which input segments were kept
    """
    scale = ratio[0]
    pad_w, pad_h = pad
    
    new_segments = []
    valid_mask = np.zeros(len(segments), dtype=bool)
    
    for i, seg in enumerate(segments):
        if len(seg) < 3:
            continue
        
        new_seg = seg.copy().astype(np.float64)
        
        # Transform: original normalized -> letterbox normalized
        new_seg[:, 0] = (seg[:, 0] * img_w * scale + pad_w) / target_size
        new_seg[:, 1] = (seg[:, 1] * img_h * scale + pad_h) / target_size
        
        # Clip to [0, 1]
        new_seg = np.clip(new_seg, 0, 1)
        
        new_segments.append(new_seg.astype(np.float32))
        valid_mask[i] = True
    
    return new_segments, valid_mask


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


@nb.jit(nopython=True, fastmath=True, cache=True)
def _generate_random_coords_numba(
    img_h: int,
    img_w: int,
    boxes_xyxy: np.ndarray,
    num_crops: int,
    crop_size: int,
    object_crop_prob: float,
    random_vals: np.ndarray,
    box_indices: np.ndarray,
    jitter_vals: np.ndarray,
    pos_vals: np.ndarray,
) -> np.ndarray:
    """
    Numba-accelerated random crop coordinate generation.
    
    Random values are pre-generated by numpy for numba compatibility.
    """
    max_x = max(0, img_w - crop_size)
    max_y = max(0, img_h - crop_size)
    has_boxes = len(boxes_xyxy) > 0
    
    coords = np.empty((num_crops, 4), dtype=np.int64)
    
    for i in range(num_crops):
        if random_vals[i] < object_crop_prob and has_boxes:
            # Center on random object with jitter
            box = boxes_xyxy[box_indices[i] % len(boxes_xyxy)]
            cx = (box[0] + box[2]) * 0.5
            cy = (box[1] + box[3]) * 0.5
            jitter_x = jitter_vals[i, 0] * crop_size * 0.6 - crop_size * 0.3
            jitter_y = jitter_vals[i, 1] * crop_size * 0.6 - crop_size * 0.3
            x1 = int(cx - crop_size * 0.5 + jitter_x)
            y1 = int(cy - crop_size * 0.5 + jitter_y)
        else:
            # Random position
            x1 = int(pos_vals[i, 0] * max_x) if max_x > 0 else 0
            y1 = int(pos_vals[i, 1] * max_y) if max_y > 0 else 0
        
        # Clamp to image bounds
        x1 = max(0, min(x1, max_x))
        y1 = max(0, min(y1, max_y))
        coords[i, 0] = x1
        coords[i, 1] = y1
        coords[i, 2] = min(x1 + crop_size, img_w)
        coords[i, 3] = min(y1 + crop_size, img_h)
    
    return coords


@lru_cache(maxsize=4096)
def cached_grid_coords(img_h: int, img_w: int, crop_h: int, crop_w: int, overlap_ratio: float) -> Tuple[Tuple[int, ...], ...]:
    """Cached grid coordinates (hashable tuple output for LRU cache)."""
    coords = _calculate_grid_coords(img_h, img_w, crop_h, crop_w, overlap_ratio)
    return tuple(tuple(int(c) for c in coord) for coord in coords)


@lru_cache(maxsize=4096)
def cached_grid_count(img_h: int, img_w: int, crop_h: int, crop_w: int, overlap_ratio: float) -> int:
    """Cached grid slice count."""
    return _count_grid_slices(img_h, img_w, crop_h, crop_w, overlap_ratio)


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

        # sahi=True MUST be passed to super() so load_image() does NOT resize
        # during cache_images() — we need original resolution for cropping
        kwargs["sahi"] = True

        # Warn about RAM usage: SAHI caches full-resolution images
        cache_val = kwargs.get("cache", None)
        if cache_val == "ram" or cache_val is True:
            LOGGER.warning(
                "WARNING ⚠️ SAHI + cache='ram' stores full-resolution images in RAM. "
                "Consider cache='low-ram' for SAHI to save memory."
            )

        super().__init__(img_path=img_path, *args, **kwargs)
        self._cache_image_shapes()
        self.slice_indices = self._precompute_slices()
        
        # Sort by image index for better cache locality
        if self.use_slicing:
            self.slice_indices.sort(key=lambda x: x[0])

        # Per-worker image cache (initialized lazily)
        self._worker_image_cache: Optional[Dict[int, np.ndarray]] = None
        self._worker_id: Optional[int] = None
        self._cache_size_bytes: int = 0

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
            x1 = min(x1, max(0, actual_w - self.crop_size))
            y1 = min(y1, max(0, actual_h - self.crop_size))
            x2 = min(x1 + self.crop_size, actual_w)
            y2 = min(y1 + self.crop_size, actual_h)
            coords = (x1, y1, x2, y2)

            crop_im = im[y1:y2, x1:x2].copy()
            crop_h, crop_w = crop_im.shape[:2]
            
            # Compute what LetterBox will do to this crop
            target = self.crop_size
            scale = min(target / crop_h, target / crop_w)
            new_h, new_w = int(crop_h * scale), int(crop_w * scale)
            pad_h, pad_w = target - new_h, target - new_w
            pad_top, pad_left = pad_h // 2, pad_w // 2
            
            resized_shape = (new_h, new_w)  # Shape after resize, before padding
            ratio_pad = ((scale, scale), (pad_left, pad_top))

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
            "segments": crop_labels.get("segments", []), 
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
    ) -> Dict[str, Any]:
        """Transform labels for letterboxed full image."""
        n_cls_cols = len(self.nc) if isinstance(self.nc, (list, tuple)) else 1
        
        bboxes = labels.get("bboxes", np.zeros((0, 4), dtype=np.float32))
        cls = labels.get("cls", np.zeros((0, n_cls_cols), dtype=np.float32))
        segments = labels.get("segments", [])

        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes, dtype=np.float32) if bboxes is not None and len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        if not isinstance(cls, np.ndarray):
            cls = np.array(cls, dtype=np.float32) if cls is not None and len(cls) > 0 else np.zeros((0, n_cls_cols), dtype=np.float32)

        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(-1, 4) if len(bboxes) > 0 else np.zeros((0, 4), dtype=np.float32)
        if cls.ndim == 1:
            cls = cls.reshape(-1, n_cls_cols) if len(cls) > 0 else np.zeros((0, n_cls_cols), dtype=np.float32)

        if len(bboxes) == 0:
            return {
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "cls": cls.astype(np.float32),
                "segments": [],
            }

        ratio, (pad_w, pad_h) = ratio_pad
        scale = ratio[0]
        target = self.crop_size

        # For segmentation: filter by segments first to keep sync
        if segments and len(segments) > 0 and len(segments) == len(bboxes):
            new_segments, valid_mask = _transform_segments_for_letterbox(
                segments, ratio, (pad_w, pad_h), img_h, img_w, target
            )
            
            # Use same mask for bboxes
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                return {
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "cls": np.zeros((0, n_cls_cols), dtype=np.float32),
                    "segments": [],
                }
            
            bboxes = bboxes[valid_indices]
            cls = cls[valid_indices]
        else:
            new_segments = []

        # Transform bboxes
        new_bboxes = np.zeros_like(bboxes)
        new_bboxes[:, 0] = (bboxes[:, 0] * img_w * scale + pad_w) / target
        new_bboxes[:, 1] = (bboxes[:, 1] * img_h * scale + pad_h) / target
        new_bboxes[:, 2] = bboxes[:, 2] * img_w * scale / target
        new_bboxes[:, 3] = bboxes[:, 3] * img_h * scale / target

        return {
            "bboxes": new_bboxes.astype(np.float32),
            "cls": cls.astype(np.float32),
            "segments": new_segments,
        }



    def _transform_labels_to_crop(
        self,
        labels: Dict[str, Any],
        coords: Tuple[int, int, int, int],
        img_h: int,
        img_w: int,
    ) -> Dict[str, Any]:
        """Transform labels from full image to crop coordinates."""
        x1, y1, x2, y2 = coords
        n_cls_cols = len(self.nc) if isinstance(self.nc, (list, tuple)) else 1

        bboxes = labels.get("bboxes", np.array([]))
        cls = labels.get("cls", np.array([]))
        segments = labels.get("segments", [])

        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes) if bboxes is not None and len(bboxes) > 0 else np.zeros((0, 4))
        if not isinstance(cls, np.ndarray):
            cls = np.array(cls) if cls is not None and len(cls) > 0 else np.zeros((0, n_cls_cols))

        if cls.ndim == 1 and len(cls) > 0:
            cls = cls.reshape(-1, 1) if n_cls_cols == 1 else cls.reshape(-1, n_cls_cols) if len(cls) % n_cls_cols == 0 else cls.reshape(-1, 1)

        if len(bboxes) == 0:
            return {
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "cls": np.zeros((0, n_cls_cols), dtype=np.float32),
                "segments": [],
            }

        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(-1, 4)

        if cls.ndim == 1:
            cls = cls.reshape(-1, 1)
        
        if cls.shape[1] != n_cls_cols and cls.shape[1] == 1 and n_cls_cols > 1:
            new_cls_arr = np.zeros((len(cls), n_cls_cols), dtype=cls.dtype)
            new_cls_arr[:, 0] = cls[:, 0]
            cls = new_cls_arr

        # For segmentation: filter segments first, then use the same mask for bboxes
        if segments and len(segments) > 0 and len(segments) == len(bboxes):
            # Transform segments and get valid mask
            new_segments, segment_valid_mask = _transform_segments_to_crop(
                segments, coords, img_h, img_w, self.min_object_coverage
            )
            
            # Use segment_valid_mask to filter bboxes too
            valid_indices = np.where(segment_valid_mask)[0]
            
            if len(valid_indices) == 0:
                return {
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "cls": np.zeros((0, n_cls_cols), dtype=np.float32),
                    "segments": [],
                }
            
            # Transform only valid bboxes to crop coordinates
            crop_w, crop_h = x2 - x1, y2 - y1
            boxes_xyxy = _xywh_to_xyxy(bboxes[valid_indices].astype(np.float64), img_h, img_w)
            
            new_bboxes = np.zeros((len(valid_indices), 4), dtype=np.float32)
            for i, box in enumerate(boxes_xyxy):
                # Clip to crop bounds
                new_x1 = max(box[0] - x1, 0)
                new_y1 = max(box[1] - y1, 0)
                new_x2 = min(box[2] - x1, crop_w)
                new_y2 = min(box[3] - y1, crop_h)
                
                # Convert to normalized xywh
                new_bboxes[i, 0] = (new_x1 + new_x2) / 2 / crop_w
                new_bboxes[i, 1] = (new_y1 + new_y2) / 2 / crop_h
                new_bboxes[i, 2] = (new_x2 - new_x1) / crop_w
                new_bboxes[i, 3] = (new_y2 - new_y1) / crop_h
            
            new_cls = cls[valid_indices].astype(np.float32)
            
        else:
            # Detection only or segments don't match bboxes - use numba-accelerated bbox filtering
            boxes_xyxy = _xywh_to_xyxy(bboxes.astype(np.float64), img_h, img_w)
            cls_flat = cls[:, 0].astype(np.float64) if cls.ndim > 1 else cls.astype(np.float64)
            
            new_bboxes, filtered_cls = _filter_bboxes(
                boxes_xyxy, cls_flat, x1, y1, x2, y2, self.min_object_coverage
            )
            
            if len(new_bboxes) == 0:
                return {
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                    "cls": np.zeros((0, n_cls_cols), dtype=np.float32),
                    "segments": [],
                }
            
            new_bboxes = new_bboxes.astype(np.float32)
            # Restore multi-column cls if needed
            if n_cls_cols > 1:
                new_cls = np.zeros((len(filtered_cls), n_cls_cols), dtype=np.float32)
                new_cls[:, 0] = filtered_cls
            else:
                new_cls = filtered_cls.reshape(-1, 1).astype(np.float32)
            new_segments = []

        # Format outputs
        new_bboxes = new_bboxes.astype(np.float32)
        if len(new_bboxes) == 0:
            return {
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "cls": np.zeros((0, n_cls_cols), dtype=np.float32),
                "segments": [],
            }
        
        return {
            "bboxes": new_bboxes,
            "cls": new_cls,
            "segments": new_segments,
        }


    def _get_cached_image(self, img_idx: int) -> np.ndarray:
        """Get image with per-worker LRU caching and memory-aware eviction."""
        import torch
        
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info else -1
        
        # Reset cache if worker changed
        if self._worker_image_cache is None or self._worker_id != current_worker_id:
            self._worker_image_cache = {}
            self._worker_id = current_worker_id
            self._cache_size_bytes = 0
        
        if img_idx in self._worker_image_cache:
            return self._worker_image_cache[img_idx]
        
        im, _, _ = self.load_image(img_idx)
        if im is None:
            raise FileNotFoundError(f"Image not found: {self.im_files[img_idx]}")
        
        im_size = im.nbytes
        
        # FIFO eviction - evict until we have room
        while len(self._worker_image_cache) >= self._buffer_size:
            oldest_key = next(iter(self._worker_image_cache))
            old_im = self._worker_image_cache.pop(oldest_key)
            self._cache_size_bytes -= old_im.nbytes
            del old_im
        
        self._worker_image_cache[img_idx] = im
        self._cache_size_bytes += im_size
        
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
            n_cls_tasks=1 if self.single_cls else len(self.nc),
        ))
        return transforms

    def on_epoch_end(self):
        """Regenerate random crops for next epoch."""
        self._epoch += 1
        
        if not self.use_slicing:
            self.slice_indices = self._precompute_slices()
            LOGGER.info(f"{colorstr('SAHIDataset')}: Regenerated {len(self.slice_indices)} random crops for epoch {self._epoch}")

    def _log_config(self):
        """Log dataset configuration."""
        mode = "train" if self.augment else "val"
        task = "segment" if self.use_segments else "detect"
        lines = [
            f"\n{colorstr('SAHIDataset')} ({mode}, {task}):",
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


# ==================== Random Crop Generation ====================

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
    
    Wrapper that pre-generates random values for numba-accelerated core.
    """
    # Pre-generate all random values with numpy (numba-compatible)
    rng = np.random.default_rng(seed)
    random_vals = rng.random(num_crops)
    box_indices = rng.integers(0, max(1, len(bboxes)), size=num_crops)
    jitter_vals = rng.random((num_crops, 2))
    pos_vals = rng.random((num_crops, 2))
    
    # Convert bboxes to xyxy format
    boxes_xyxy = _xywh_to_xyxy(bboxes.astype(np.float64), img_h, img_w) if len(bboxes) > 0 else np.empty((0, 4), dtype=np.float64)
    
    # Call numba-accelerated function
    coords = _generate_random_coords_numba(
        img_h, img_w, boxes_xyxy, num_crops, crop_size, object_crop_prob,
        random_vals, box_indices, jitter_vals, pos_vals
    )
    
    return [tuple(c) for c in coords]
