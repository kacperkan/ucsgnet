import argparse
import math
import typing
from pathlib import Path

import cv2
import numpy as np
import torch

from ucsgnet.ucsgnet.shape_evaluators import (
    CircleSphereEvaluator,
    ShapeEvaluator,
    SquareCubeEvaluator,
)


def param_to_tensors(
    x: float, y: float, rotation: float, *params: float
) -> typing.Tuple[torch.Tensor, ...]:
    translation = torch.tensor([x, y], dtype=torch.float32).reshape((1, 1, 2))
    rotation = torch.tensor(rotation, dtype=torch.float32).reshape((1, 1, 1))
    params = torch.tensor(params, dtype=torch.float32).reshape(
        (1, 1, len(params))
    )
    return params, translation, rotation


class Creator:
    def __init__(self, eval_type: ShapeEvaluator, width: int, height: int):
        self.eval_type = eval_type
        self.width = width
        self.height = height

    def generate_points(self) -> torch.Tensor:
        grid = (
            torch.stack(
                torch.meshgrid(
                    [torch.arange(0, self.width), torch.arange(0, self.height)]
                ),
                dim=-1,
            )
            .float()
            .permute((1, 0, 2))
            .reshape((-1, 2))
        )
        grid -= torch.tensor(
            [self.height / 2, self.width / 2], dtype=torch.float
        )
        grid /= self.width
        return grid

    def get_binarized_image(self, distances: torch.Tensor) -> np.ndarray:
        distances = (
            (distances <= 0)
            .float()
            .reshape((self.height, self.width))
            .detach()
            .cpu()
            .numpy()
        )
        distances = distances.astype(np.float32)
        return distances

    def get_single_shape(
        self, rotation: float, x: float, y: float, *params: float
    ) -> np.ndarray:
        params, translation, rotation = param_to_tensors(
            x, y, rotation, *params
        )
        points = self.generate_points()
        shape = self.eval_type.evaluate_points(
            params,
            self.eval_type.transform_points(points, translation, rotation),
        )
        return self.get_binarized_image(shape)


circle_creator = Creator(CircleSphereEvaluator(1, 2), 128, 128)
box_creator = Creator(SquareCubeEvaluator(1, 2), 128, 128)


def diff(shape_1: np.ndarray, shape_2: np.ndarray) -> np.ndarray:
    return (shape_1 - shape_2).clip(0, 1)


def union(shape_1: np.ndarray, shape_2: np.ndarray) -> np.ndarray:
    return (shape_1 + shape_2).clip(0, 1)


def intersection(shape_1: np.ndarray, shape_2: np.ndarray) -> np.ndarray:
    return (shape_1 + shape_2 - 1).clip(0, 1)


def get_path(out_path: Path, name: str) -> str:
    return (out_path / name).with_suffix(".png").as_posix()


def to_byte(shape: np.ndarray) -> np.ndarray:
    shape = (shape * 255).astype(np.uint8)
    shape = cv2.copyMakeBorder(
        shape, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=128
    )
    return shape


def generate_visualization(output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    shapes = {
        "box_1": box_creator.get_single_shape(
            math.pi / 4, -0.2, 0.0, 0.2, 0.2
        ),
        "box_2": box_creator.get_single_shape(0, -0.2, 0.25, 0.3, 0.25),
        "box_3": box_creator.get_single_shape(0, -0.2, 0.2, 0.2, 0.2),
        "box_4": box_creator.get_single_shape(0, 0.35, 0.2, 0.05, 0.2),
        "circle_1": circle_creator.get_single_shape(0, 0.05, 0.15, 0.15),
        "circle_2": circle_creator.get_single_shape(0, 0.35, 0.0, 0.15),
        "circle_3": circle_creator.get_single_shape(0, 0.35, 0.35, 0.05),
        "circle_4": circle_creator.get_single_shape(0, 0.35, -0.1, 0.2),
    }

    for key, shape in shapes.items():
        cv2.imwrite(get_path(output_path, key), to_byte(shape))

    # box 1 box 2
    cv2.imwrite(
        get_path(output_path, "box_1_box_2_diff"),
        to_byte(diff(shapes["box_1"], shapes["box_2"])),
    )
    cv2.imwrite(
        get_path(output_path, "box_2_box_1_diff"),
        to_byte(diff(shapes["box_2"], shapes["box_1"])),
    )
    cv2.imwrite(
        get_path(output_path, "box_1_box_2_inter"),
        to_byte(intersection(shapes["box_1"], shapes["box_2"])),
    )
    cv2.imwrite(
        get_path(output_path, "box_1_box_2_union"),
        to_byte(union(shapes["box_1"], shapes["box_2"])),
    )

    # box 3 box 4
    cv2.imwrite(
        get_path(output_path, "box_3_box_4_union"),
        to_byte(union(shapes["box_3"], shapes["box_4"])),
    )
    cv2.imwrite(
        get_path(output_path, "box_3_box_4_inter"),
        to_byte(intersection(shapes["box_3"], shapes["box_4"])),
    )
    cv2.imwrite(
        get_path(output_path, "box_3_box_4_diff"),
        to_byte(diff(shapes["box_3"], shapes["box_4"])),
    )
    cv2.imwrite(
        get_path(output_path, "box_4_box_3_diff"),
        to_byte(diff(shapes["box_4"], shapes["box_3"])),
    )

    # circle 4 circle 2
    cv2.imwrite(
        get_path(output_path, "circle_4_circle_2_inter"),
        to_byte(intersection(shapes["circle_4"], shapes["circle_2"])),
    )
    cv2.imwrite(
        get_path(output_path, "circle_4_circle_2_diff"),
        to_byte(diff(shapes["circle_4"], shapes["circle_2"])),
    )
    cv2.imwrite(
        get_path(output_path, "circle_2_circle_4_diff"),
        to_byte(diff(shapes["circle_2"], shapes["circle_4"])),
    )
    cv2.imwrite(
        get_path(output_path, "circle_4_circle_2_union"),
        to_byte(union(shapes["circle_2"], shapes["circle_4"])),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate example 2D visualizations of a single relation "
            "prediction layer "
        )
    )

    parser.add_argument("--out_path", help="Output folder path")

    args = parser.parse_args()
    generate_visualization(args.out_path)


if __name__ == "__main__":
    main()
