from sahi.slicing import slice_image
from sahi.utils.coco import Coco
import numpy as np
import cv2
from copy import deepcopy
from .dataset import YOLODataset

from .augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
    crop_transforms,
    crop_val_transforms,
)

class SAHIDataset(YOLODataset):
    def __init__(
        self,
        img_path,
        cut_strategy="grid",
        crop_size=640,
        overlap_ratio=0.2,
        *args,
        **kwargs
    ):
        """Initialize SAHI-aware dataset."""
        self.cut_strategy = cut_strategy
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.use_slicing = cut_strategy == "grid"
        super().__init__(img_path=img_path, *args, **kwargs)
        if self.use_slicing:
            self.slice_indices = self._precompute_slices()  # Список (img_idx, slice_idx)

    def _precompute_slices(self):
        """Предвычисляем количество слайсов на каждое изображение для grid стратегии."""
        slice_indices = []
        for idx in range(self.ni):
            # Загружаем изображение (без resize)
            im, (h0, w0), _ = self.load_image(idx)
            num_slices_h = int(np.ceil(h0 / (self.crop_size * (1 - self.overlap_ratio))))
            num_slices_w = int(np.ceil(w0 / (self.crop_size * (1 - self.overlap_ratio))))
            total_slices = num_slices_h * num_slices_w
            slice_indices.extend([(idx, s) for s in range(total_slices)])
        return slice_indices

    def __len__(self):
        if self.use_slicing:
            return len(self.slice_indices)
        else:
            return super().__len__()

    def get_image_and_label(self, index):
        """Возвращает изображения и соответствующие ему аннотации."""
        if self.use_slicing:
            return self._get_grid_slice(index)
        else:
            label = deepcopy(self.labels[index])
            label.pop("shape", None)
            label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
            label["ratio_pad"] = (
                label["resized_shape"][0] / label["ori_shape"][0],
                label["resized_shape"][1] / label["ori_shape"][1],
            ) 
            return self.update_labels_info(label)

    def _get_grid_slice(self, index):
        """Генерирует слайс по сетке и фильтрует аннотации."""
        img_idx, slice_idx = self.slice_indices[index]
        im, (h0, w0), _ = self.load_image(img_idx)
        labels = deepcopy(self.labels[img_idx])

        # Генерация слайсов через SAHI
        sliced_image = slice_image(
            image=im,
            slice_height=self.crop_size,
            slice_width=self.crop_size,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
        )

        # Выбираем нужный слайс
        slice_obj = sliced_image[slice_idx]
        slice_im = slice_obj.image
        slice_bbox = slice_obj.bbox  # (x_min, y_min, x_max, y_max)

        # Преобразуем аннотации под координаты слайса
        slice_labels = self._filter_and_transform_annotations(labels, slice_bbox, h0, w0)

        # Обновляем метаданные
        label = {
            "im_file": labels["im_file"],
            "shape": slice_im.shape[:2],
            "cls": slice_labels["cls"],
            "bboxes": slice_labels["bboxes"],
            "segments": slice_labels.get("segments", []),
            "keypoints": slice_labels.get("keypoints", None),
        }

        # Добавляем оригинальный размер и коэффициенты масштабирования
        label["ori_shape"] = (h0, w0)
        label["resized_shape"] = slice_im.shape[:2]
        label["ratio_pad"] = (1.0, 1.0)  # Нет ресайза
        return label

    def _filter_and_transform_annotations(self, labels, slice_bbox, h0, w0):
        """Фильтрует и трансформирует аннотации под координаты слайса."""
        x_min, y_min, x_max, y_max = slice_bbox
        img_area = h0 * w0
        slice_labels = {"cls": [], "bboxes": [], "segments": []}

        for i in range(len(labels["bboxes"])):
            cls = labels["cls"][i]
            bbox = labels["bboxes"][i]  # (cx, cy, w, h) в нормализованных координатах

            # Переводим в пиксельные координаты
            cx, cy, w, h = bbox
            x1 = (cx - w / 2) * w0
            y1 = (cy - h / 2) * h0
            x2 = (cx + w / 2) * w0
            y2 = (cy + h / 2) * h0

            # Проверяем, пересекается ли бокс со слайсом
            inter_x1 = max(x1, x_min)
            inter_y1 = max(y1, y_min)
            inter_x2 = min(x2, x_max)
            inter_y2 = min(y2, y_max)

            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                continue  # Нет пересечения

            # Обрезаем бокс под слайс
            new_x1 = max(x1 - x_min, 0)
            new_y1 = max(y1 - y_min, 0)
            new_x2 = min(x2 - x_min, self.crop_size)
            new_y2 = min(y2 - y_min, self.crop_size)

            # Переводим в нормализованные координаты относительно слайса
            cx_new = (new_x1 + new_x2) / 2 / self.crop_size
            cy_new = (new_y1 + new_y2) / 2 / self.crop_size
            w_new = (new_x2 - new_x1) / self.crop_size
            h_new = (new_y2 - new_y1) / self.crop_size

            slice_labels["cls"].append(cls)
            slice_labels["bboxes"].append([cx_new, cy_new, w_new, h_new])

        slice_labels["cls"] = np.array(slice_labels["cls"], dtype=np.float32)
        slice_labels["bboxes"] = np.array(slice_labels["bboxes"], dtype=np.float32)
        return slice_labels
    
    def build_transforms(self, hyp=None):
        """Builds and appends transforms based on cut_strategy."""
        if self.cut_strategy == "grid" and self.augment:
            transforms = v8_transforms(
                dataset=self,
                imgsz=self.crop_size,
                hyp=hyp,
                stretch=False
            )
        elif self.cut_strategy == "random_crop" and self.augment:
            transforms = crop_transforms(
                dataset=self, 
                imgsz=self.crop_size, 
                hyp = hyp,
                )
        elif not self.augment:
            transforms = crop_val_transforms(
                dataset=self, 
                imgsz=self.crop_size, 
                hyp = hyp,
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
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )

        return transforms