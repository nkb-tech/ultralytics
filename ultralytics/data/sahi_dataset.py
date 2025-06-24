from sahi.slicing import slice_image
from sahi.utils.coco import Coco
import numpy as np
import cv2
from copy import deepcopy
from .dataset import YOLODataset

class YOLOSAHIDataset(YOLODataset):
    def __init__(
        self,
        img_path,
        cut_strategy="grid",
        slice_size=640,
        overlap_ratio=0.2,
        *args,
        **kwargs
    ):
        """Initialize SAHI-aware dataset."""
        super().__init__(img_path=img_path, *args, **kwargs)
        self.cut_strategy = cut_strategy
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.slice_indices = self._precompute_slices()  # Список (img_idx, slice_idx)

    def _precompute_slices(self):
        """Предвычисляем количество слайсов на каждое изображение для grid стратегии."""
        slice_indices = []
        for idx in range(self.ni):
            # Загружаем изображение (без resize)
            im, (h0, w0), _ = self.load_image(idx)
            num_slices_h = int(np.ceil(h0 / (self.slice_size * (1 - self.overlap_ratio))))
            num_slices_w = int(np.ceil(w0 / (self.slice_size * (1 - self.overlap_ratio))))
            total_slices = num_slices_h * num_slices_w
            slice_indices.extend([(idx, s) for s in range(total_slices)])
        return slice_indices

    def __len__(self):
        """Возвращаем общее количество слайсов."""
        return len(self.slice_indices)

    def get_image_and_label(self, index):
        """Возвращает кроп изображения и соответствующие ему аннотации."""
        if self.cut_strategy == "grid":
            return self._get_grid_slice(index)
        elif self.cut_strategy == "random_crop":
            return self._get_random_crop(index)
        else:
            raise ValueError(f"Unknown cut strategy: {self.cut_strategy}")

    def _get_grid_slice(self, index):
        """Генерирует слайс по сетке и фильтрует аннотации."""
        img_idx, slice_idx = self.slice_indices[index]
        im, (h0, w0), _ = self.load_image(img_idx)
        labels = deepcopy(self.labels[img_idx])

        # Генерация слайсов через SAHI
        sliced_image = slice_image(
            image=im,
            slice_height=self.slice_size,
            slice_width=self.slice_size,
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
            new_x2 = min(x2 - x_min, self.slice_size)
            new_y2 = min(y2 - y_min, self.slice_size)

            # Переводим в нормализованные координаты относительно слайса
            cx_new = (new_x1 + new_x2) / 2 / self.slice_size
            cy_new = (new_y1 + new_y2) / 2 / self.slice_size
            w_new = (new_x2 - new_x1) / self.slice_size
            h_new = (new_y2 - new_y1) / self.slice_size

            slice_labels["cls"].append(cls)
            slice_labels["bboxes"].append([cx_new, cy_new, w_new, h_new])

        slice_labels["cls"] = np.array(slice_labels["cls"], dtype=np.float32)
        slice_labels["bboxes"] = np.array(slice_labels["bboxes"], dtype=np.float32)
        return slice_labels