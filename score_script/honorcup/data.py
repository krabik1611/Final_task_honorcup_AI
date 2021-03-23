import os
import random
from typing import Optional, Tuple

import cv2
import torch
from torch.utils.data import Dataset

__all__ = ["SRDataset"]


class SRDataset(Dataset):
    def __init__(
            self,
            hr_dir: str,
            lr_dir: str,
            crop_size: Optional[int] = None,
            length: Optional[int] = None) -> None:
        self._hr_dir = hr_dir
        self._lr_dir = lr_dir
        self._crop_size = crop_size
        self._length = length

        samples = []
        for name in os.listdir(lr_dir):
            if not name.endswith(".png"):
                continue
            if not os.path.exists(os.path.join(hr_dir, name)):
                raise RuntimeError(f"File {name} does not exist in {hr_dir}")
            samples.append(name)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples) if not self._length else self._length

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self._samples[item % len(self._samples)]
        lr_image = cv2.imread(os.path.join(self._lr_dir, name))
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.imread(os.path.join(self._hr_dir, name))
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        if hr_image.shape != (lr_image.shape[0] * 2, lr_image.shape[1] * 2, lr_image.shape[2]):
            raise RuntimeError(f"Shapes of LR and HR images mismatch for sample {name}")

        lr_image = torch.from_numpy(lr_image).permute(2, 0, 1).float() / 255.
        hr_image = torch.from_numpy(hr_image).permute(2, 0, 1).float() / 255.

        if self._crop_size is not None:
            x_start = random.randint(0, lr_image.shape[1] - self._crop_size)
            y_start = random.randint(0, lr_image.shape[2] - self._crop_size)

            lr_image = lr_image[
                       :,
                       x_start:x_start + self._crop_size,
                       y_start:y_start + self._crop_size]
            hr_image = hr_image[
                       :,
                       x_start * 2:x_start * 2 + self._crop_size * 2,
                       y_start * 2:y_start * 2 + self._crop_size * 2]
        return lr_image, hr_image
