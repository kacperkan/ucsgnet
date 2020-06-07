import math
import typing as t

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Grayscale,
    Resize,
    ToTensor,
)
from typing_extensions import Literal

from ucsgnet.common import SEED


def get_simple_2d_transforms() -> t.Callable[
    [t.Union[Image.Image, np.ndarray]], torch.Tensor
]:
    return Compose([Resize(64), Grayscale(), ToTensor()])


class HdfsDataset3D(Dataset):
    def __init__(
        self, path: str, points_per_sample: int, seed: int, size: int
    ):
        self.path = path
        self.points_per_sample = points_per_sample
        self.size = size

        self._rng = np.random.RandomState(seed)

        with h5py.File(self.path, "r") as h5_file:
            self._voxels = h5_file["voxels"][:]
            self._values = h5_file[f"values_{self.size}"][:]
            self._points = h5_file[f"points_{self.size}"][:]

    def __len__(self) -> int:
        return len(self._voxels)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        vox = self._voxels[index].astype(np.float32)
        cube = self._values[index].astype(np.float32)
        points = self._points[index].astype(np.float32)

        vox = torch.from_numpy(vox.transpose((3, 0, 1, 2)))
        sampled_gt = torch.from_numpy(cube)
        sampled_points = torch.from_numpy((points + 0.5) / 256 - 0.5)

        where_ones = sampled_points[sampled_gt[:, 0] == 1]

        min_coords = where_ones.min(dim=0)[0]
        max_coords = where_ones.max(dim=0)[0]

        x_min, x_max = min_coords[0], max_coords[0]
        y_min, y_max = min_coords[1], max_coords[1]
        z_min, z_max = min_coords[2], max_coords[2]

        bounding_dims = (
            torch.tensor(
                (z_max - z_min, y_max - y_min, x_max - x_min),
                dtype=torch.float32,
            )
            + 0.5
        ) / 256 - 0.5
        bounding_vol = bounding_dims.prod()

        return vox, sampled_points, sampled_gt, bounding_vol


def process_single_2d_image(
    image: np.ndarray,
    transforms: t.Optional[
        t.Callable[[np.ndarray], t.Union[torch.Tensor, np.ndarray]]
    ],
) -> t.Tuple[torch.Tensor, ...]:
    height, width = image.shape[0], image.shape[1]
    current_max_distance = max(width, height) * math.sqrt(2)

    thresholded = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 0).astype(
        np.uint8
    )
    distances = cv2.distanceTransform(1 - thresholded, cv2.DIST_L2, maskSize=0)
    coords = np.stack(
        np.meshgrid(range(width), range(height)), axis=-1
    ).reshape(
        (-1, 2)
    )  # -> N, (x, y)
    distances = distances[coords[:, 1], coords[:, 0]]

    y_indices, x_indices = np.where(thresholded == 1)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    bounding_h = (y_max - y_min) / current_max_distance
    bounding_w = (x_max - x_min) / current_max_distance

    bounding_volume = bounding_h * bounding_w

    dim = max(height, width)

    coords = coords.astype(np.float32)
    coords = (coords + 0.5) / dim - 0.5

    if transforms is not None:
        image = Image.fromarray(image, mode="RGB")
        image = transforms(image)
    else:
        image = torch.from_numpy(image).float() / 255
    coords = torch.from_numpy(coords).float()

    distances = (distances <= 0).astype(np.float32)
    return image, coords, distances, bounding_volume


class CADDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        data_split: Literal["train", "valid", "test"],
        transforms: t.Optional[t.Callable[[np.ndarray], torch.Tensor]] = None,
    ):

        super().__init__()

        self.h5_file_path = h5_file_path
        self.transforms = transforms
        self.data_split = data_split

        if data_split == "train":
            self.data_key = "train_images"
        elif data_split == "valid":
            self.data_key = "val_images"
        else:
            self.data_key = "test_images"

        with h5py.File(self.h5_file_path, "r") as h5_file:
            self._images = h5_file[self.data_key][:]

        self.__cache = {}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        if index in self.__cache:
            return self.__cache[index]
        image = self._images[index].astype(np.uint8) * 255
        image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)
        image, coords, distances, bounding_volume = process_single_2d_image(
            image, self.transforms
        )

        self.__cache[index] = (image, coords, distances, bounding_volume)

        return image, coords, distances, bounding_volume


class SimpleDataset(Dataset):
    def __init__(
        self,
        image_paths: t.Sequence[str],
        points_with_distances_paths: t.Optional[t.Sequence[str]],
        points_per_sample: t.Optional[int],
        transforms: t.Optional[t.Callable[[np.ndarray], torch.Tensor]] = None,
        verbose: bool = False,
    ):

        super().__init__()

        self.image_paths = image_paths
        self.points_per_sample = points_per_sample
        self.points_with_distances_paths = points_with_distances_paths
        self.transforms = transforms

        self._rng = np.random.RandomState(SEED)
        if verbose:
            print(f"Loaded {len(image_paths)} paths")

        self.__cache = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, ...]:
        if index in self.__cache:
            return self.__cache[index]
        image = cv2.imread(self.image_paths[index])
        image, coords, distances, bounding_volume = process_single_2d_image(
            image, self.transforms
        )
        self.__cache[index] = (image, coords, distances, bounding_volume)

        return image, coords, distances, bounding_volume
