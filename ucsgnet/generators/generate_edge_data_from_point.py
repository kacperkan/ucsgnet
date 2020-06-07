# Code adapted from: https://git.io/JfeXK
import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

from ucsgnet.utils import read_point_normal_ply_file, write_ply_point_normal


def to_tensor(data: np.ndarray) -> torch.Tensor:
    data = torch.from_numpy(data)
    if torch.cuda.is_available():
        return data.cuda()
    return data


def get_points(vertices: torch.Tensor, normals: torch.Tensor) -> np.ndarray:
    # compute Chamfer distance and Normal consistency
    num_of_points = vertices.shape[0]
    points_mat_1 = vertices.view((num_of_points, 1, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    points_mat_2 = vertices.view((1, num_of_points, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    dist = (points_mat_1 - points_mat_2).pow(2).sum(dim=2)
    closest_index = (dist < 0.0001).int()

    normals_mat_1 = normals.view((num_of_points, 1, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    normals_mat_2 = normals.view((1, num_of_points, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    prod = (normals_mat_1 * normals_mat_2).sum(dim=2)
    all_edge_index = (prod.abs() < 0.1).int()

    edge_index = (closest_index * all_edge_index).max(dim=1)[0]
    points = torch.cat((vertices, normals), dim=1)
    points = points[edge_index > 0.5].detach().cpu().numpy()

    np.random.shuffle(points)
    return points[:4096]


def generate_edges(
    valid_shapes_file: str,
    ground_truth_folder: str,
    reconstruction_folder: str,
    out_dir: str,
):
    with open(valid_shapes_file, "r") as f:
        eval_list = f.readlines()

    eval_list = [item.strip().split("/") for item in eval_list]
    reconstruction_folder = Path(reconstruction_folder)
    out_dir = Path(out_dir)
    ground_truth_folder = Path(ground_truth_folder)

    with tqdm.trange(len(eval_list)) as pbar:
        for idx in pbar:
            class_name = eval_list[idx][0]
            object_name = eval_list[idx][1]
            pbar.set_postfix_str(f"{class_name}/{object_name}")

            # read pred
            ply_file = (
                reconstruction_folder
                / class_name
                / object_name
                / "pred_pc.ply"
            )
            pred_vertices, pred_normals = read_point_normal_ply_file(ply_file)
            pred_points = get_points(
                to_tensor(pred_vertices), to_tensor(pred_normals)
            )

            gt_file = (
                ground_truth_folder / class_name / object_name
            ).with_suffix(".ply")
            gt_pc_vertices, gt_pc_normals = read_point_normal_ply_file(
                gt_file.as_posix()
            )

            gt_points = get_points(
                to_tensor(gt_pc_vertices), to_tensor(gt_pc_normals)
            )

            object_out_folder = out_dir / class_name
            object_out_folder.mkdir(exist_ok=True, parents=True)

            write_ply_point_normal(
                (object_out_folder / f"{object_name}_pred.ply"), pred_points
            )

            write_ply_point_normal(
                (object_out_folder / f"{object_name}_true.ply"), gt_points
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Script generates edges from points of predicted and ground "
            "truth shapes"
        )
    )

    parser.add_argument(
        "--valid_shapes_file",
        type=str,
        required=True,
        help="A file containing all valid shapes",
    )
    parser.add_argument(
        "--ground_truth_folder",
        type=str,
        required=True,
        help="A folder path containing all ground truth points",
    )
    parser.add_argument(
        "--reconstruction_folder",
        type=str,
        required=True,
        help="A folder path containing reconstructed objects from predictions",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="An output directory that will containg generated edges",
    )

    args = parser.parse_args()
    generate_edges(**vars(args))


if __name__ == "__main__":
    main()
