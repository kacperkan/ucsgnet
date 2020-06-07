import argparse
import math
import typing as t
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from ucsgnet.ucsgnet.shape_evaluators import (
    CircleSphereEvaluator,
    SquareCubeEvaluator,
)

BIN_UNION_OP = (1, 1, 0)
BIN_INT_OP = (1, 1, -1)
BIN_DIFF_OP = (1, -1, 0)

CIRCLE = CircleSphereEvaluator(1, 2)
SQUARE = SquareCubeEvaluator(1, 2)


def param_to_tensor(
    x: float, y: float, *params: float
) -> t.Tuple[torch.Tensor, ...]:
    translation = torch.tensor([x, y], dtype=torch.float32).reshape((1, 1, 2))
    rotation = torch.tensor(math.pi / 4, dtype=torch.float32).reshape(
        (1, 1, 1)
    )
    params = torch.tensor(params, dtype=torch.float32).reshape(
        (1, 1, len(params))
    )
    return params, translation, rotation


def visualize_single(distances: torch.Tensor, width: int, height: int):
    distances = (
        distances.clamp_min(0).reshape((height, width)).detach().cpu().numpy()
    )
    distances = distances.astype(np.float32)

    plt.figure()
    plt.imshow(distances, cmap="gray")
    plt.show()


def get_image(distances: torch.Tensor, width: int, height: int) -> np.ndarray:
    distances = (
        distances.clamp_min(0).reshape((height, width)).detach().cpu().numpy()
    )
    distances = distances.astype(np.float32)
    return distances


def get_binarized_image(
    distances: torch.Tensor, width: int, height: int
) -> np.ndarray:
    distances = (
        distances.float().reshape((height, width)).detach().cpu().numpy()
    )
    distances = distances.astype(np.float32)
    return distances


def generate_points(width: int, height: int) -> torch.Tensor:
    grid = (
        torch.stack(
            torch.meshgrid([torch.arange(0, width), torch.arange(0, height)]),
            dim=-1,
        )
        .float()
        .reshape((-1, 2))
    )
    grid -= torch.tensor([height / 2, width / 2], dtype=torch.float)
    grid /= height
    return grid


def fun_on_binary(
    x: torch.Tensor, dist_1: float, dist_2: float, dist_3: float
):
    x = x.unsqueeze(-1)

    dist_op = torch.tensor(
        [dist_1, dist_2, dist_3], dtype=torch.float32
    ).reshape((1, 1, -1, 1))
    probas = torch.tensor([1, 1, 1]).reshape(dist_op.size())
    x = dist_op * x * probas

    return x.sum(dim=2)


def visualize_bin() -> t.Dict[str, np.ndarray]:
    width, height = 256, 256
    points = generate_points(width, height)
    params, translation, rotation = param_to_tensor(0.0, 0.0, 0.25)

    shape_1 = CIRCLE.evaluate_points(
        params, CIRCLE.transform_points(points, translation, rotation)
    ).permute(0, 2, 1)
    shape_1 = (shape_1 <= 0).float()

    params, translation, rotation = param_to_tensor(0.12, 0.12, 0.25)
    shape_2 = CIRCLE.evaluate_points(
        params, CIRCLE.transform_points(points, translation, rotation)
    ).permute(0, 2, 1)
    shape_2 = (shape_2 <= 0).float()

    union_distances = (shape_1 + shape_2).clamp(0, 1)
    inter_distances = (shape_1 + shape_2 - 1).clamp(0, 1)
    diff_distances = (shape_1 - shape_2).clamp(0, 1)
    diff_rev_distances = (shape_2 - shape_1).clamp(0, 1)

    return {
        "shape_1": get_binarized_image(shape_1, width, height),
        "shape_2": get_binarized_image(shape_2, width, height),
        "union": get_image(union_distances, width, height),
        "intersection": get_image(inter_distances, width, height),
        "difference_a_b": get_image(diff_distances, width, height),
        "difference_b_a": get_image(diff_rev_distances, width, height),
    }


def produce_visualizations(output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=False)

    visualizations = visualize_bin()
    for name, vis in visualizations.items():
        cv2.imwrite(
            (output_dir / name).with_suffix(".png").as_posix(),
            (vis * 255).astype(np.uint8),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Script producing example binary operations on binary sets"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help=(
            "Output directory for visualizations. If does not exist, then it "
            "is created first."
        ),
        required=True,
    )

    args = parser.parse_args()
    produce_visualizations(args.out_dir)


if __name__ == "__main__":
    main()
