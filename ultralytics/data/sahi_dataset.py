from functools import lru_cache

import math
from multiprocessing.pool import ThreadPool
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numba as nb

from ultralytics.utils import NUM_THREADS, LOGGER, TQDM, colorstr

from .augment import Compose, Format, LetterBox, crop_transforms, crop_val_transforms, v8_transforms
from .dataset import YOLODataset


class SAHIDataset(YOLODataset):  # only for bboxes, TODO: keypoints and masks
    """
    A dataset class that supports Slicing Aided Hyper Inference (SAHI) strategies for training and validation.

    This class extends the standard YOLODataset to allow slicing of images into smaller crops using configurable cutting
    strategies such as grid-based slicing or random cropping. It is useful for training models whose performance during
    inference with SAHI should be improved by exposing them to similar crop-based training samples.

    Attributes:
        cut_strategy (str): Strategy used to crop images. Options are:
            - 'grid': Slice image into a grid of fixed-size crops with optional overlap (similar to SAHI).
            - 'random_crop': Random crops from the image. The number of crops per image is determined by sampling_rate.
        crop_size (int): Size (width and height) of each crop in pixels.
        overlap_ratio (float): Overlap ratio between adjacent slices when using 'grid' strategy (0 to 1).
        sampling_rate (float): For 'random_crop' strategy, determines how many random crops to generate relative to the
                               number of grid slices an image would produce. E.g., 0.3 means 30% of the number of grid slices.
        use_slicing (bool): Whether slicing is being used (True if cut_strategy is 'grid').
        slice_indices (List[Tuple[int, int]]): Precomputed list of (image_index, slice_index) pairs for all slices.
        avg_slices (float): Average number of slices per image.
        min_slices (int): Minimum number of slices across all images.
        max_slices (int): Maximum number of slices across all images.

    Methods:
        _precompute_slices: Precomputes slice indices for all images using parallel processing.
        __len__: Returns the total number of slices if slicing is used, otherwise returns the number of original images.
        get_image_and_label: Retrieves a single processed image and its corresponding label.
        _get_grid_slice: Generates a single slice from an image using grid slicing.
        _filter_and_transform_annotations: Filters and transforms annotations to match the current slice.
        build_transforms: Builds data augmentation and preprocessing transformations based on the slicing strategy.
    """

    def __init__(
        self,
        img_path: str,
        cut_strategy: str = "grid",
        crop_size: int = 640,
        overlap_ratio: float = 0,  # now minimal overlap
        sampling_rate: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initializes the SAHIDataset instance.

        Args:
            img_path (str): Path/paths to the directory containing images.
            cut_strategy (str): Strategy for slicing images ('grid' or 'random_crop'). Default is 'grid'.
            crop_size (int): Size of each crop (square). Default is 640.
            overlap_ratio (float): Minimal desired fraction of overlap between adjacent slices (used for 'grid', also calculate num of crops for 'random_crop'). Default is 0.
            sampling_rate (float): When using 'random_crop', this defines the ratio of random crops per image compared to
                                   how many slices that image would generate under the 'grid' strategy. Default is 1.0.
            *args, **kwargs: Additional arguments passed to the parent YOLODataset class.
        """
        self.cut_strategy = cut_strategy
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.sampling_rate = sampling_rate
        self.use_slicing = cut_strategy == "grid"
        super().__init__(img_path=img_path, *args, **kwargs)

        # logging
        task = "train" if self.augment else "val"
        prefix = colorstr(f"SAHIDataset for {task}")
        info = [f":\n  cut_strategy: {self.cut_strategy}", f"\n  crop_size: {self.crop_size}"]
        # overlap_ratio only for grid
        if self.cut_strategy == "grid":
            info.append(f"\n  Minimal overlap_ratio: {self.overlap_ratio}")

        info.append(f"\n  total_images: {self.ni}")
        LOGGER.info(prefix + "".join(info))

        # slice_indices store (img_idx, slice_idx) for 'random_crop' or
        # (img_idx, slice_idx, (start_x, start_y, end_x, end_y)) for 'grid'
        self.slice_indices = self._precompute_slices()  # Список (img_idx, slice_idx)

    def _precompute_slices(self) -> List[Tuple[int, Any]]:
        """
        Precompute (img_idx, slice_idx) pairs and collect statistics.
        For 'grid', it also computes and stores the coordinates of each slice.
        If cut_strategy is 'random_crop', returns fewer slices per image based on sampling_rate.
        """

        @nb.jit(
            nb.types.Tuple(
                (
                    nb.int64[:, :],  # x: 2D int64 array
                    nb.int64[:, :],  # y: 2D int64 array
                )
            )(
                nb.int64[:],  # x: 1D int64 array
                nb.int64[:],  # y: 1D int64 array
            ),
            nopython=True,
            fastmath=True,
            parallel=False,
            inline="always",
        )
        def meshgrid2d_ij(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Numba-compiled meshgrid2d. Same as np.meshgrid(x, y) but faster.

            Args:
                x: 1D int64 array
                y: 1D int64 array

            Returns:
                tuple[np.ndarray, np.ndarray]: 2D int64 arrays
            """

            shape = (x.size, y.size)  # Matrix dimensions: rows=x, columns=y
            xx = np.empty(shape, dtype=x.dtype)
            yy = np.empty(shape, dtype=y.dtype)

            # Broadcast x along columns (axis=1)
            xx[...] = x[:, np.newaxis]  # Reshape x to (x.size, 1)

            # Broadcast y along rows (axis=0)
            yy[...] = y[np.newaxis, :]  # Reshape y to (1, y.size)

            return xx, yy

        @lru_cache(maxsize=64)
        @nb.jit(
            nopython=True,
            fastmath=True,
            parallel=False,
            inline="always",
        )
        def calculate_slices_coordinates(
            imgsz: tuple[int, int],
            crop_size: tuple[int, int],
            overlap_ratio: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Adjust the image size to be divisible by crop size and overlap.

            Args:
                imgsz (tuple[int, int]): Image size - (height, width).
                crop_size (tuple[int, int]): Crop size - (height, width).
                overlap_ratio (float): Overlap ratio.

            Returns:
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: List of slice coordinates.
            """
            img_h, img_w = imgsz
            crop_h, crop_w = crop_size
            overlap_h, overlap_w = int(overlap_ratio * crop_h), int(overlap_ratio * crop_w)

            # If the image is smaller than the crop size, use the whole image as a
            # single slice so downstream code still receives valid coordinates.
            if img_h <= crop_h and img_w <= crop_w:
                y1_flat = np.array([0], dtype=np.int64)
                x1_flat = np.array([0], dtype=np.int64)
                y2_flat = np.array([img_h], dtype=np.int64)
                x2_flat = np.array([img_w], dtype=np.int64)
                return y1_flat, x1_flat, y2_flat, x2_flat

            step_h = crop_h - overlap_h
            step_w = crop_w - overlap_w

            # Calculate number of steps using integer arithmetic
            n_steps_h = 1 if img_h <= crop_h else (img_h - crop_h + step_h - 1) // step_h + 1
            n_steps_w = 1 if img_w <= crop_w else (img_w - crop_w + step_w - 1) // step_w + 1

            # Generate starting positions
            y1_base = np.arange(0, n_steps_h) * step_h
            x1_base = np.arange(0, n_steps_w) * step_w

            # Adjust last row/column
            if img_h > crop_h and n_steps_h > 1:
                y1_base[-1] = img_h - crop_h
            if img_w > crop_w and n_steps_w > 1:
                x1_base[-1] = img_w - crop_w

            # Create meshgrid and flatten
            y1_grid, x1_grid = meshgrid2d_ij(y1_base, x1_base)
            y1_flat, x1_flat = y1_grid.flatten(), x1_grid.flatten()

            # Calculate end positions
            y2_flat = np.minimum(y1_flat + crop_h, img_h)
            x2_flat = np.minimum(x1_flat + crop_w, img_w)

            return x1_flat, y1_flat, x2_flat, y2_flat

        def wrapped_calculate_slices_coordinates(idx: int):
            """
            Wrapper for calculate_slices_coordinates.
            """
            coordinates = calculate_slices_coordinates(
                self.labels[idx]["shape"],
                (self.crop_size, self.crop_size),
                self.overlap_ratio,
            )
            
            return idx, np.stack(coordinates, axis=1)

        @lru_cache(maxsize=64)
        @nb.jit(
            nopython=True,
            fastmath=True,
            parallel=False,
            inline="always",
        )
        def calculate_number_of_slices(
            imgsz: tuple[int, int],
            crop_size: tuple[int, int],
            overlap_ratio: float,
        ) -> int:
            """
            Calculate the number of slices for an image.

            Args:
                imgsz (tuple[int, int]): Image size - (height, width).
                crop_size (tuple[int, int]): Crop size - (height, width).
                overlap_ratio (float): Overlap ratio.

            Returns:
                int: Number of slices.
            """
            img_h, img_w = imgsz
            crop_h, crop_w = crop_size

            if img_h <= crop_h and img_w <= crop_w:
                return 1

            overlap_h, overlap_w = int(overlap_ratio * crop_h), int(overlap_ratio * crop_w)

            # stride is how far the sliding window moves each step
            stride_h = crop_h - overlap_h
            stride_w = crop_w - overlap_w

            # number of window positions in each direction
            n_h = max(math.ceil((img_h - crop_h) / stride_h) + 1, 1)
            n_w = max(math.ceil((img_w - crop_w) / stride_w) + 1, 1)

            return n_h * n_w

        def wrapped_calculate_number_of_slices(idx: int):
            """
            Wrapper for calculate_number_of_slices.
            """
            ns = calculate_number_of_slices(
                self.labels[idx]["shape"],
                (self.crop_size, self.crop_size),
                self.overlap_ratio,
            )
            if ns <= 0:
                raise ValueError(f"Check your crop size and overlap ratio. Image size: {self.labels[idx]['shape']}, crop size: {self.crop_size}, overlap ratio: {self.overlap_ratio}")

            return idx, ns

        desc = f"{colorstr('SAHI Calculating slices')}"
        slice_indices: List[Tuple[int, Any]] = []
        slices_per_image: List[int] = []

        with ThreadPool(NUM_THREADS) as pool:
            # Thread-pool map over all images
            results = pool.imap(
                func=wrapped_calculate_slices_coordinates if self.cut_strategy == "grid" else wrapped_calculate_number_of_slices,
                iterable=range(self.ni)
            )

            pbar = TQDM(results, total=self.ni, desc=desc)
            for idx, calculated_data in pbar:
                if self.cut_strategy == "random_crop":
                    sampled_slices = max(1, round(calculated_data * self.sampling_rate))
                    slice_indices.extend([(idx, s) for s in range(sampled_slices)])
                    slices_per_image.append(sampled_slices)
                else:  # grid
                    for s_idx, coords in enumerate(calculated_data):
                        slice_indices.append((idx, s_idx, coords))
                    slices_per_image.append(len(calculated_data))

                # Store statistics
                avg_slices = sum(slices_per_image) / len(slices_per_image)
                min_slices = min(slices_per_image)
                max_slices = max(slices_per_image)

                pbar.desc = f"{desc}: min {min_slices}, max {max_slices}, avg {avg_slices:.2f} per image"

            pbar.close()

        return slice_indices

    def __len__(self):
        return len(self.slice_indices)

    def get_image_and_label(self, index: int) -> Dict[str, Any]:
        if self.use_slicing:
            return self._get_grid_slice(index)
        else:
            img_idx, _ = self.slice_indices[index]
            label = deepcopy(self.labels[img_idx])
            label.pop("shape", None)
            label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(img_idx)
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            )
            return self.update_labels_info(label)

    def _get_grid_slice(self, index: int) -> Dict[str, Any]:
        """
        Generate a single slice based on precomputed coordinates.

        Args:
            index (int): Index of the slice in `self.slice_indices`.

        Returns:
            Dict[str, Any]: Dictionary containing sliced image and filtered labels.
        """
        img_idx, slice_idx, slice_bbox_coords = self.slice_indices[index]
        start_x, start_y, end_x, end_y = slice_bbox_coords
        im, (h0, w0), _ = self.load_image(img_idx)
        labels = deepcopy(self.labels[img_idx])

        slice_im = im[start_y:end_y, start_x:end_x]
        if slice_im.shape[0] == 0 or slice_im.shape[1] == 0:
            raise ValueError(f"Slice image is empty. Image size: {im.shape}, slice coords: {slice_bbox_coords}, overlap_ratio: {self.overlap_ratio}")
        slice_bbox = [start_x, start_y, end_x, end_y]

        slice_labels = self._filter_and_transform_annotations(labels, slice_bbox, h0, w0)
        n_attrs = labels["cls"].shape[1] if len(labels["cls"]) > 0 else 1

        labels.update(
            {
                "img": slice_im,
                "ori_shape": (h0, w0),
                "resized_shape": slice_im.shape[:2],
                "ratio_pad": (1.0, 1.0),
                "cls": slice_labels["cls"],
                "original_img_idx": img_idx,  # Index of original image
                "slice_idx": slice_idx,        # Index of slice within image
                "slice_coords": slice_bbox_coords,  # Coordinates of this crop in original image
                "original_im_file": self.im_files[img_idx],  # Original image path
            }
        )

        if slice_labels["bboxes"].size == 0:
            labels["bboxes"] = np.empty((0, 4), dtype=np.float32)
            if labels["cls"].size == 0:
                labels["cls"] = np.empty((0, n_attrs), dtype=np.float32)
        else:
            bboxes = np.array(slice_labels["bboxes"], dtype=np.float32)
            if bboxes.ndim != 2 or bboxes.shape[1] != 4:
                raise ValueError(f"Expected shape (N, 4), got {bboxes.shape}")
            labels["bboxes"] = bboxes

        if "segments" in labels:
            labels["segments"] = slice_labels.get("segments", [])
        if "keypoints" in labels:
            labels["keypoints"] = slice_labels.get("keypoints", None)

        return self.update_labels_info(labels)

    def _filter_and_transform_annotations(
        self, labels: Dict[str, Any], slice_bbox: List[int], h0: int, w0: int
    ) -> Dict[str, Union[np.ndarray, List]]:
        """
        Filter and transform annotations to match current slice coordinates.

        Args:
            labels (Dict[str, Any]): Original labels dictionary.
            slice_bbox (List[int]): Bounding box of the slice in pixel coordinates [x_min, y_min, x_max, y_max].
            h0 (int): Original image height.
            w0 (int): Original image width.

        Returns:
            Dict[str, Union[np.ndarray, List]]: Transformed labels within the slice.
        """
        x_min, y_min, x_max, y_max = slice_bbox
        x_crop_size, y_crop_size = x_max - x_min, y_max - y_min
        slice_labels = {"cls": [], "bboxes": []}
        n_attrs = labels["cls"].shape[1] if len(labels["cls"]) > 0 else 1        
        for i in range(len(labels["bboxes"])):
            cls = labels["cls"][i]
            bbox = labels["bboxes"][i]  # (cx, cy, w, h) normalized

            cx, cy, w, h = bbox
            x1 = (cx - w / 2) * w0
            y1 = (cy - h / 2) * h0
            x2 = (cx + w / 2) * w0
            y2 = (cy + h / 2) * h0

            inter_x1 = max(x1, x_min)
            inter_y1 = max(y1, y_min)
            inter_x2 = min(x2, x_max)
            inter_y2 = min(y2, y_max)

            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                continue  # No intersection

            new_x1 = max(x1 - x_min, 0)
            new_y1 = max(y1 - y_min, 0)
            new_x2 = min(x2 - x_min, x_crop_size)
            new_y2 = min(y2 - y_min, y_crop_size)

            cx_new = (new_x1 + new_x2) / 2 / x_crop_size
            cy_new = (new_y1 + new_y2) / 2 / y_crop_size
            w_new = (new_x2 - new_x1) / x_crop_size
            h_new = (new_y2 - new_y1) / y_crop_size

            slice_labels["cls"].append(cls)
            slice_labels["bboxes"].append([cx_new, cy_new, w_new, h_new])

        if len(slice_labels["cls"]) > 0:
            slice_labels["cls"] = np.array(slice_labels["cls"], dtype=np.float32)
        else:
            slice_labels["cls"] = np.zeros((0, n_attrs), dtype=np.float32)
        
        slice_labels["bboxes"] = np.array(slice_labels["bboxes"], dtype=np.float32)
        if len(slice_labels["bboxes"]) == 0:
            slice_labels["bboxes"] = np.zeros((0, 4), dtype=np.float32)
        
        return slice_labels

    def build_transforms(self, hyp: Optional[Dict[str, Any]] = None) -> Compose:
        if isinstance(hyp.scale_range, str):
            hyp.scale_range = hyp.scale_range.strip("()")
            x, y = hyp.scale_range.split(",")
            hyp.scale_range = (float(x), float(y))
        if self.cut_strategy == "grid":
            if self.augment:
                transforms = v8_transforms(
                    dataset=self,
                    imgsz=self.crop_size,
                    hyp=hyp,
                    stretch=False,
                )
            else:
                transforms = Compose([LetterBox(new_shape=(self.crop_size, self.crop_size), scaleup=False)])
        elif self.cut_strategy == "random_crop":
            if self.augment:
                transforms = crop_transforms(
                    dataset=self,
                    imgsz=self.crop_size,
                    hyp=hyp,
                )
            else:
                transforms = crop_val_transforms(
                    dataset=self,
                    imgsz=self.crop_size,
                    hyp=hyp,
                )
        else:
            raise ValueError(f"Unknown cut strategy: {self.cut_strategy}")

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio if hyp else 0.5,
                mask_overlap=hyp.overlap_mask if hyp else False,
                bgr=hyp.bgr if hyp and self.augment else 0.0,
                n_cls_tasks=1 if self.single_cls else len(self.nc),
            )
        )

        return transforms
