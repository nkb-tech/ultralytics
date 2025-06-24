from typing import Any, Dict, List, Tuple, Union
import numpy as np

ALBU_AVAILABLE = False

try:
    import albumentations as A
    ALBU_AVAILABLE = True
    from albumentations import AtLeastOneBBoxRandomCrop
    from albumentations.core.transforms_interface import DualTransform
    from albumentations.augmentations.crops.transforms import CropSizeError
    import random
except:
    ALBU_AVAILABLE = False

class SafeFixedRandomCrop(AtLeastOneBBoxRandomCrop):
    """Гарантированный кроп size×size c хотя бы одним bbox внутри.
    Не применяется, если изображение меньше заданного размера."""
    
    def __init__(
        self,
        size: int = 640,
        erosion_factor: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(
            height=size,
            width=size,
            erosion_factor=erosion_factor,
            p=p,
        )

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        image_height, image_width = params["shape"][:2]
        if image_height < self.height or image_width < self.width:
            return {"crop_coords": (0, 0, image_width, image_height)}
        return super().get_params_dependent_on_data(params, data)

class RandomCropLarge(DualTransform):
    """
    Срабатывает с вероятностью p, НО только если картинка ≥ `threshold` по обеим осям.
    """

    def __init__(
        self,
        crop_size: int = 1024,
        threshold: int = 1024, # определяем с насколько больших изображений мы будем делать случайный кроп 
        erosion_factor: float = 0.0,
        p: float = 1.0
    ):
        super().__init__(p=p)
        self.crop_size = crop_size
        self.threshold = threshold
        self.random_crop = A.RandomCrop(height=crop_size, width=crop_size, p=1.0) # используется для получения background 
        self.safe_fixed_crop = SafeFixedRandomCrop(
            size=crop_size,
            erosion_factor=erosion_factor,
            p=1.0
        )

    def _use_random_crop(self, height: int, width: int) -> bool:
        """решает, можно ли брать жесткий RandomCrop(640×640)"""
        return (
                height >= self.crop_size and width >= self.crop_size  # кроп поместится
        ) and (
                height > self.threshold or width > self.threshold  # картинка достаточно «большая»
        )
        
    def apply(self, img, **params):
        height, width = img.shape[:2]
        if self._use_random(height, width):
            return self.random_crop.apply(img, **params)
        else:
            return self.safe_fixed_crop.apply(img, **params)

    def apply_to_bboxes(self, bboxes: list[list[float]], **params) -> list[list[float]]:
        height, width = params["shape"][:2]
        if self._use_random(height, width):
            return self.random_crop.apply_to_bboxes(bboxes, **params)
        else:
            return self.safe_fixed_crop.apply_to_bboxes(bboxes, **params)

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        height, width = params["shape"][:2]
        if self._use_random(height, width):
            return self.random_crop.get_params_dependent_on_data(params, data)
        else:
            return self.safe_fixed_crop.get_params_dependent_on_data(params, data)

    def get_transform_init_args_names(self):
        return ("crop_size", "threshold")

class SahiCropsTransform(DualTransform):
    def __init__(
        self,
        crop_size: int = 640,
        threshold: int = 1024,
        erosion_factor: float = 0.0,
        p: float = 1.0,
        bg_crop_prob: float = 0.1,  # Вероятность использовать RandomCropLarge для background'а
        always_apply: bool = False
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.crop_size = crop_size
        self.threshold = threshold
        self.erosion_factor = erosion_factor
        self.bg_crop_prob = bg_crop_prob
        
        # Создаем базовые трансформы
        self.random_crop = RandomCropLarge(height=crop_size, width=crop_size, p=1.0)
        self.safe_fixed_crop = SafeFixedRandomCrop(
            size=crop_size,
            erosion_factor=erosion_factor,
            p=1.0
        )

    def _use_random(self, height: int, width: int, has_boxes: bool) -> bool:
        """
        Решает, использовать ли RandomCropLarge.
        Возвращает True только если:
        1. Изображение достаточно большое (height > threshold или width > threshold)
        2. Есть боксы, но мы хотим получить background (случайный выбор по bg_crop_prob)
        """
        if not has_boxes:
            return False  # Если нет боксов, не используем SafeFixedRandomCrop
            
        return (
            height >= self.crop_size and width >= self.crop_size and
            (height > self.threshold or width > self.threshold) and
            random.random() < self.bg_crop_prob
        )

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Применяет кроп к изображению."""
        height, width = img.shape[:2]
        has_boxes = len(params.get('bboxes', [])) > 0
        
        if self._use_random(height, width, has_boxes):
            return self.random_crop.apply(img, **params)
        else:
            return self.safe_fixed_crop.apply(img, **params)

    def apply_to_bboxes(self, bboxes: List[List[float]], **params) -> List[List[float]]:
        """Применяет кроп к bounding box'ам."""
        height, width = params["shape"][:2]
        has_boxes = len(bboxes) > 0
        
        if self._use_random(height, width, has_boxes):
            return self.random_crop.apply_to_bboxes(bboxes, **params)
        else:
            return self.safe_fixed_crop.apply_to_bboxes(bboxes, **params)

    def get_params_dependent_on_data(self, params: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Возвращает параметры кропа, зависящие от данных."""
        height, width = params["shape"][:2]
        bboxes = data.get("bboxes", [])
        has_boxes = len(bboxes) > 0
        
        if self._use_random(height, width, has_boxes):
            return self.random_crop.get_params_dependent_on_data(params, data)
        else:
            return self.safe_fixed_crop.get_params_dependent_on_data(params, data)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("crop_size", "threshold", "erosion_factor", "bg_crop_prob")