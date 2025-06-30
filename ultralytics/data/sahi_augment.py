from typing import Any

try:
    import albumentations as A

    ALBU_AVAILABLE = True
    from albumentations import AtLeastOneBBoxRandomCrop
    from albumentations.core.transforms_interface import DualTransform
except:
    ALBU_AVAILABLE = False


class SafeFixedRandomCrop(AtLeastOneBBoxRandomCrop):
    """
    A transform that applies a random crop of fixed size with bbox inside only if the image is large enough.

    If the image is smaller than the requested crop size, no cropping is applied.
    When the image is large enough, this transform ensures that at least one bounding box remains inside the crop.

    Args:
        crop_size (int): The size of the square crop in pixels (height and width). Default: 640.
        erosion_factor (float): Erosion factor applied to bounding boxes before computing the crop. Helps avoid too tight crops. Default: 0.0.
        p (float): Probability of applying the transform. Default: 1.0.
    """

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
    A dual transform that applies either a strict random crop or a safe crop that ensures at least one bounding box is preserved.

    This transform applies `albumentations.RandomCrop` if the image is larger than both `crop_size` and `threshold`.
    Otherwise, it falls back to `SafeFixedRandomCrop`, which guarantees that at least one bounding box remains inside the crop.

    Args:
        crop_size (int): The size of the square crop in pixels (height and width). Default: 1024.
        threshold (int): Minimum dimension (either height or width) for the image to be considered "large" and allow strict random cropping. Default: 1024.
        erosion_factor (float): Erosion factor applied to bounding boxes before computing the crop. Helps avoid too tight crops. Default: 0.0.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(self, crop_size: int = 1024, threshold: int = 1024, erosion_factor: float = 0.0, p: float = 1.0):
        super().__init__(p=p)
        self.crop_size = crop_size
        self.threshold = threshold
        self.random_crop = A.RandomCrop(height=crop_size, width=crop_size, p=1.0)  # for background crops
        self.safe_fixed_crop = SafeFixedRandomCrop(crop_size=crop_size, erosion_factor=erosion_factor, p=1.0)

    def _use_random(self, height: int, width: int) -> bool:
        """Determines whether to use the strict RandomCrop based on image dimensions."""
        return (height >= self.crop_size and width >= self.crop_size) and (
            height > self.threshold or width > self.threshold  # big enough for background
        )

    def apply(
        self,
        img,
        **params,
    ):
        height, width = img.shape[:2]
        if self._use_random(height, width):
            return self.random_crop.apply(img, **params)
        else:
            return self.safe_fixed_crop.apply(img, **params)

    def apply_to_bboxes(
        self,
        bboxes: list[list[float]],
        **params,
    ) -> list[list[float]]:
        height, width = params["shape"][:2]
        if self._use_random(height, width):
            return self.random_crop.apply_to_bboxes(bboxes, **params)
        else:
            return self.safe_fixed_crop.apply_to_bboxes(bboxes, **params)

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        height, width = params["shape"][:2]
        if self._use_random(height, width):
            return self.random_crop.get_params_dependent_on_data(params, data)
        else:
            return self.safe_fixed_crop.get_params_dependent_on_data(params, data)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_size", "threshold")
