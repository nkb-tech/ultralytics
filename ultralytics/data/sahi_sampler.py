from typing import Dict, Iterator, List
from collections import defaultdict
import random

from torch.utils.data import Sampler


class SAHIBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups crops from the same image together.
    
    Training randomness is maintained by:
    - Shuffling the order of images
    - Shuffling crops within each image
    
    Args:
        slice_indices: List of (img_idx, slice_idx, coords) from SAHIDataset
        batch_size: Number of samples per batch
        drop_last: Drop incomplete final batch
        shuffle: Shuffle image order (True for training)
        group_shuffle: Shuffle crops within each image
    """
    
    def __init__(
        self,
        slice_indices: List[tuple],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        group_shuffle: bool = True,
    ):
        self.slice_indices = slice_indices
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.group_shuffle = group_shuffle
        self.image_groups = self._build_image_groups()
    
    def _build_image_groups(self) -> Dict[int, List[int]]:
        """Map image index -> list of dataset indices for that image."""
        groups = defaultdict(list)
        for dataset_idx, (img_idx, _, _) in enumerate(self.slice_indices):
            groups[img_idx].append(dataset_idx)
        return dict(groups)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches with optimized image locality."""
        image_indices = list(self.image_groups.keys())
        
        if self.shuffle:
            random.shuffle(image_indices)
        
        # Flatten with image locality preserved
        all_indices = []
        for img_idx in image_indices:
            group = self.image_groups[img_idx].copy()
            if self.group_shuffle:
                random.shuffle(group)
            all_indices.extend(group)
        
        # Yield fixed-size batches
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
    
    def __len__(self) -> int:
        n = len(self.slice_indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
