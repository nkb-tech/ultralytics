from typing import Any
ALBU_AVAILABLE = False

try:
    import albumentations as A
    ALBU_AVAILABLE = True
    from albumentations import AtLeastOneBBoxRandomCrop
    from albumentations.core.transforms_interface import DualTransform

except:
    ALBU_AVAILABLE = False

class SafeFixedRandomCrop(AtLeastOneBBoxRandomCrop):
    """Гарантированный кроп size×size c хотя бы одним bbox внутри.
    Не применяется, если изображение меньше заданного размера."""
    
    def __init__(
        self,
        crop_size: int = 640,
        erosion_factor: float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(
            height=crop_size,
            width=crop_size,
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
            crop_size=crop_size,
            erosion_factor=erosion_factor,
            p=1.0
        )

    def _use_random(self, height: int, width: int) -> bool:
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

         