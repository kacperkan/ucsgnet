# Code adapted from: https://git.io/Jfeua
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import tqdm

from ucsgnet.common import Evaluation3D
from ucsgnet.utils import read_point_normal_ply_file


def to_tensor(data: np.ndarray) -> torch.Tensor:
    data = torch.from_numpy(data).float()
    if torch.cuda.is_available():
        return data.cuda()
    return data


def get_chamfer_distance_and_normal_consistency(
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    gt_normals: torch.Tensor,
    pred_normals: torch.Tensor,
) -> Tuple[float, float]:
    gt_num_points = gt_points.shape[0]
    pred_num_points = pred_points.shape[0]

    points_gt_matrix = gt_points.unsqueeze(1).expand(
        [gt_points.shape[0], pred_num_points, gt_points.shape[-1]]
    )
    points_pred_matrix = pred_points.unsqueeze(0).expand(
        [gt_num_points, pred_points.shape[0], pred_points.shape[-1]]
    )

    distances = (points_gt_matrix - points_pred_matrix).pow(2).sum(dim=-1)
    match_pred_gt = distances.argmin(dim=0)
    match_gt_pred = distances.argmin(dim=1)

    dist_pred_gt = (pred_points - gt_points[match_pred_gt]).pow(2).sum(dim=-1).mean()
    dist_gt_pred = (gt_points - pred_points[match_gt_pred]).pow(2).sum(dim=-1).mean()

    normals_dot_pred_gt = (
        (pred_normals * gt_normals[match_pred_gt]).sum(dim=1).abs().mean()
    )

    normals_dot_gt_pred = (
        (gt_normals * pred_normals[match_gt_pred]).sum(dim=1).abs().mean()
    )
    chamfer_distance = dist_pred_gt + dist_gt_pred
    normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2

    return chamfer_distance.item(), normal_consistency.item()


def get_cd_nc_for_points(
    ground_truth_point_surface: Path,
    reconstructed_shapes_folder: Path,
    class_name: str,
    object_name: str,
) -> Tuple[float, float]:
    # read ground truth point cloud
    gt_file = (ground_truth_point_surface / class_name / object_name).with_suffix(
        ".ply"
    )
    gt_pc_vertices, gt_pc_normals = read_point_normal_ply_file(gt_file.as_posix())

    # read preds
    pred_file = reconstructed_shapes_folder / class_name / object_name / "pred_pc.ply"
    pred_pc_vertices, pred_pc_normals = read_point_normal_ply_file(pred_file.as_posix())

    # compute Chamfer distance and Normal consistency
    return get_chamfer_distance_and_normal_consistency(
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_vertices),
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_normals),
    )


def get_cd_nc_for_edges(
    edges_data_folder: Path, class_name: str, object_name: str
) -> Tuple[float, float]:
    # read ground truth point cloud
    gt_file = edges_data_folder / class_name / f"{object_name}_true.ply"
    gt_pc_vertices, gt_pc_normals = read_point_normal_ply_file(gt_file.as_posix())

    # read preds
    pred_file = edges_data_folder / class_name / f"{object_name}_pred.ply"
    pred_pc_vertices, pred_pc_normals = read_point_normal_ply_file(pred_file.as_posix())

    # compute Chamfer distance and Normal consistency
    return get_chamfer_distance_and_normal_consistency(
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_vertices),
        to_tensor(gt_pc_vertices),
        to_tensor(pred_pc_normals),
    )


def evaluate(
    valid_shape_names_file: str,
    reconstructed_shapes_folder: str,
    raw_shapenet_data_folder: str,
    edges_data_folder: str,
    ground_truth_point_surface: str,
    out_folder: str,
):
    with open(valid_shape_names_file, "r") as f:
        eval_list = f.readlines()
    eval_list = [item.strip().split("/") for item in eval_list]
    print(f"Num objects: {len(eval_list)}")

    reconstructed_shapes_folder = Path(reconstructed_shapes_folder)
    ground_truth_point_surface = Path(ground_truth_point_surface)
    edges_data_folder = Path(edges_data_folder)
    raw_shapenet_data_folder = Path(raw_shapenet_data_folder)

    out_per_obj = defaultdict(dict)
    category_chamfer_distance_sum = defaultdict(float)
    category_normal_consistency_sum = defaultdict(float)

    edge_category_chamfer_distance_sum = defaultdict(float)
    edge_category_normal_consistency_sum = defaultdict(float)

    category_count = defaultdict(int)
    mean_metrics = defaultdict(float)
    total_entries = 0

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    errors = []

    with tqdm.trange(len(eval_list)) as pbar:
        for idx in pbar:
            class_name = eval_list[idx][0]
            object_name = eval_list[idx][1]
            pbar.set_postfix_str(f"{class_name}/{object_name}")

            cat_id = Evaluation3D.CATEGORY_IDS.index(class_name)

            # cd and nc for edges
            try:
                edge_cd, edge_nc = get_cd_nc_for_edges(
                    edges_data_folder, class_name, object_name
                )
            except RuntimeError:
                errors.append((class_name, object_name))
                continue

            edge_category_chamfer_distance_sum[cat_id] += edge_cd
            edge_category_normal_consistency_sum[cat_id] += edge_nc

            out_per_obj[class_name][object_name] = {}

            out_per_obj[class_name][object_name]["chamfer_distance_edge"] = edge_cd
            out_per_obj[class_name][object_name]["normal_consistency_edge"] = edge_nc

            mean_metrics["chamfer_distance_edge"] += edge_cd
            mean_metrics["normal_consistency_edge"] += edge_nc

            # cd and nc
            points_cd, points_nc = get_cd_nc_for_points(
                ground_truth_point_surface,
                reconstructed_shapes_folder,
                class_name,
                object_name,
            )

            category_chamfer_distance_sum[cat_id] += points_cd
            category_normal_consistency_sum[cat_id] += points_nc

            out_per_obj[class_name][object_name] = {
                "id": cat_id,
                "chamfer_distance": points_cd,
                "normal_consistency": points_nc,
            }

            mean_metrics["chamfer_distance"] += points_cd
            mean_metrics["normal_consistency"] += points_nc

            category_count[cat_id] += 1
            total_entries += 1

    print(f"{len(errors)} error shapes")

    per_category = {
        Evaluation3D.CATEGORY_NAMES[cat_id]: {
            "chamfer_distance": category_chamfer_distance_sum[cat_id]
            / category_count[cat_id],
            "normal_consistency": category_normal_consistency_sum[cat_id]
            / category_count[cat_id],
            "chamfer_distance_edge": edge_category_chamfer_distance_sum[cat_id]
            / category_count[cat_id],
            "normal_consistency_edge": edge_category_normal_consistency_sum[cat_id]
            / category_count[cat_id],
        }
        for cat_id in category_chamfer_distance_sum.keys()
    }

    with open(out_folder / "per_category_metrics.json", "w") as f:
        json.dump(per_category, f, indent=4)

    with open(out_folder / "per_object_metrics.json", "w") as f:
        json.dump(out_per_obj, f, indent=4)

    with open(out_folder / "errors.txt", "w") as f:
        f.write("\n".join(comp[0] + "/" + comp[1] for comp in errors))

    mean_metrics = {
        name: metric / total_entries for name, metric in mean_metrics.items()
    }
    with open(out_folder / "mean_metrics.json", "w") as f:
        json.dump(mean_metrics, f, indent=4)
    print(mean_metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate standard metrics on 3D autencoding task"
    )
    parser.add_argument(
        "--valid_shape_names_file",
        help="A file path containing names of shapes to validate on",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--reconstructed_shapes_folder",
        help="A folder path containing reconstructed shapes and point clouds",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--raw_shapenet_data_folder",
        help="A folder path to ground truth ShapeNet shapes",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ground_truth_point_surface",
        help="A folder path to ground truth sampled points of ShapeNet shapes",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--edges_data_folder",
        help=(
            "A folder path to edges data generated with "
            "`generate_edge_data_from_point.py`"
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_folder",
        help="A folder path where metrics will be saved",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    evaluate(**vars(args))


if __name__ == "__main__":
    main()
