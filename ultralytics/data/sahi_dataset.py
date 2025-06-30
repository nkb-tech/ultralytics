from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial

import numpy as np

from ultralytics.utils import LOGGER, colorstr

from .augment import Compose, Format, LetterBox, crop_transforms, crop_val_transforms, v8_transforms
from .dataset import YOLODataset


class SAHIDataset(YOLODataset):
    def __init__(self, img_path, cut_strategy="grid", crop_size=640, overlap_ratio=0, *args, **kwargs):
        """Initialize SAHI-aware dataset."""
        self.cut_strategy = cut_strategy
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.use_slicing = cut_strategy == "grid"
        super().__init__(img_path=img_path, *args, **kwargs)

        # logging
        task = "train" if self.augment else "val"
        prefix = colorstr(f"SAHIDataset for {task}")
        info = [f":\n  cut_strategy: {self.cut_strategy}", f"\n  crop_size: {self.crop_size}"]
        # overlap_ratio only for grid
        if self.cut_strategy == "grid":
            info.append(f"\n  overlap_ratio: {self.overlap_ratio}")

        info.append(f"\n  total_images: {self.ni}")
        LOGGER.info(prefix + "".join(info))

        if self.use_slicing:
            self.slice_indices = self._precompute_slices()  # Список (img_idx, slice_idx)
            crop_info = (
                f"  Total crops: {len(self.slice_indices)}\n"
                f"  Avg crops/image: {self.avg_slices:.1f}\n"
                f"  Min/Max crops/image: {self.min_slices}/{self.max_slices}"
            )
            LOGGER.info(crop_info)

    def _precompute_slices(self):
        """Предвычисляем (img_idx, slice_idx) параллельно и собираем статистику."""

        def process_single_image(idx, crop_size, overlap_ratio, load_image_func):
            im, (h0, w0), _ = load_image_func(idx)

            step_x = int(crop_size * (1 - overlap_ratio))
            step_y = int(crop_size * (1 - overlap_ratio))

            cols = max(1, (w0 - crop_size) // step_x + 1)
            rows = max(1, (h0 - crop_size) // step_y + 1)

            total_slices = cols * rows
            return idx, total_slices

        # Параллельная обработка
        with ThreadPoolExecutor() as executor:
            process_fn = partial(
                process_single_image,
                crop_size=self.crop_size,
                overlap_ratio=self.overlap_ratio,
                load_image_func=self.load_image,
            )
            results = list(executor.map(process_fn, range(self.ni)))

        # Разбираем результаты
        slice_indices = []
        slices_per_image = []

        for idx, total_slices in results:
            slice_indices.extend([(idx, s) for s in range(total_slices)])
            slices_per_image.append(total_slices)

        # Сохраняем статистику
        self.avg_slices = sum(slices_per_image) / len(slices_per_image)
        self.min_slices = min(slices_per_image)
        self.max_slices = max(slices_per_image)

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
        """Генерирует один слайс по координатам"""
        img_idx, slice_idx = self.slice_indices[index]
        im, (h0, w0), _ = self.load_image(img_idx)
        labels = deepcopy(self.labels[img_idx])

        # Расчет количества столбцов и строк
        step_x = int(self.crop_size * (1 - self.overlap_ratio))
        step_y = int(self.crop_size * (1 - self.overlap_ratio))
        cols = max(1, (w0 - self.crop_size) // step_x + 1)

        # Определение координат текущего слайса
        row = slice_idx // cols
        col = slice_idx % cols

        start_x = col * step_x
        start_y = row * step_y

        # Обрезка изображения
        end_x = min(start_x + self.crop_size, w0)
        end_y = min(start_y + self.crop_size, h0)
        slice_im = im[start_y:end_y, start_x:end_x]
        slice_bbox = [start_x, start_y, end_x, end_y]

        # Фильтрация аннотаций
        slice_labels = self._filter_and_transform_annotations(labels, slice_bbox, h0, w0)

        # Обновление метаданных
        labels.update(
            {
                "img": slice_im,
                "ori_shape": (h0, w0),
                "resized_shape": slice_im.shape[:2],
                "ratio_pad": (1.0, 1.0),
                "cls": slice_labels["cls"],
            }
        )

        if slice_labels["bboxes"].size == 0:  # Если список пуст
            labels["bboxes"] = np.empty((0, 4), dtype=np.float32)  # Форма (0, 4)
        else:
            # Преобразуем список в массив и проверяем форму
            bboxes = np.array(slice_labels["bboxes"], dtype=np.float32)
            if bboxes.ndim != 2 or bboxes.shape[1] != 4:
                raise ValueError(f"Ожидаемая форма (N, 4), получена {bboxes.shape}")
            labels["bboxes"] = bboxes

        if "segments" in labels:
            labels["segments"] = slice_labels.get("segments", [])
        if "keypoints" in labels:
            labels["keypoints"] = slice_labels.get("keypoints", None)

        return self.update_labels_info(labels)

    def _filter_and_transform_annotations(self, labels, slice_bbox, h0, w0):
        """Фильтрует и трансформирует аннотации под координаты слайса."""
        x_min, y_min, x_max, y_max = slice_bbox
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
        if self.cut_strategy == "grid":
            if self.augment:
                transforms = v8_transforms(dataset=self, imgsz=self.crop_size, hyp=hyp, stretch=False)
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
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )

        return transforms
