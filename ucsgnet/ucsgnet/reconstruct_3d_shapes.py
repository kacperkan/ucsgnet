import argparse
import copy
import os
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import mcubes
import numpy as np
import pymesh
import torch
import tqdm
import trimesh.creation
import trimesh.repair

from ucsgnet.mesh_utils import sample_points_polygon
from ucsgnet.ucsgnet.net_3d import Net
from ucsgnet.ucsgnet.shape_evaluators import (
    CircleSphereEvaluator,
    SquareCubeEvaluator,
)
from ucsgnet.utils import (
    quat_to_rot_matrix_numpy,
    write_ply_point_normal,
    write_ply_triangle,
)


def to_trimesh(mesh: pymesh.Mesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)


class OpType(Enum):
    NONE = "none"
    UNION = "union"
    INTER = "inter"
    DIFF = "diff"

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class ObjType(Enum):
    BOX = "box"
    SPHERE = "sphere"


@dataclass
class CSGEntry:
    obj: t.Optional[pymesh.Mesh]
    op: OpType
    used: bool
    left_index: int = -1
    right_index: int = -1

    def set_used(self):
        self.used = True

    def set_unused(self):
        self.used = False

    def set_obj(self, obj: pymesh.Mesh):
        self.obj = obj

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        result.obj = pymesh.form_mesh(
            self.obj.vertices.copy(), self.obj.faces.copy()
        )
        result.used = self.used
        result.left_index = self.left_index
        result.right_index = self.right_index
        result.op = self.op
        return result


def transform_mesh(
    mesh: pymesh.Mesh, translation: np.ndarray, quaternion: np.ndarray
) -> pymesh.Mesh:
    # negated_rot = trimesh.transformations.quaternion_inverse(rot)
    bbox = mesh.bbox
    centroid = 0.5 * (bbox[0] + bbox[1])
    vertices = mesh.vertices
    rotation_matrix = quat_to_rot_matrix_numpy(quaternion)

    # rotation_matrix = trimesh.transformations.quaternion_matrix(
    #     negated_rot
    # )
    # transform = trimesh.transformations.concatenate_matrices(
    #     rotation_matrix, translation_matrix
    # )
    # primitive_mesh = primitive_mesh.apply_transform(
    #     trimesh.transformations.concatenate_matrices(
    #         rotation_matrix.T, translation_matrix.T
    #     ).T
    # )
    vertices = rotation_matrix.T.dot((vertices - centroid).T).T + centroid
    vertices = vertices + translation[None]
    mesh = pymesh.form_mesh(vertices, mesh.faces)

    return mesh


class CSGComposer:
    def __init__(self, sphere_complexity: int):
        self.per_layer_combinations: t.Dict[
            int, t.List[CSGEntry]
        ] = defaultdict(list)
        self.sphere_complexity = sphere_complexity

    def add_box(
        self, params: np.ndarray, translation: np.ndarray, rotation: np.ndarray
    ):
        min_point = -params
        max_point = params
        box = pymesh.generate_box_mesh(min_point, max_point)
        box = transform_mesh(box, translation, rotation)
        self.per_layer_combinations[0].append(
            CSGEntry(box, OpType.NONE, used=False)
        )

    def add_sphere(
        self, params: np.ndarray, translation: np.ndarray, rotation: np.ndarray
    ):
        sphere = pymesh.generate_icosphere(
            params[0],
            np.array([0, 0, 0], dtype=np.float32),
            refinement_order=self.sphere_complexity,
        )
        sphere = transform_mesh(sphere, translation, rotation)
        self.per_layer_combinations[0].append(
            CSGEntry(sphere, OpType.NONE, used=False)
        )

    def combine_shapes(
        self,
        op_left_index: int,
        op_right_index: int,
        cur_layer_index: int,
        op_type: OpType,
    ):
        self.per_layer_combinations[cur_layer_index].append(
            CSGEntry(None, op_type, False, op_left_index, op_right_index)
        )

    def refine_connections(self):
        last_layer = len(self.per_layer_combinations.keys()) - 1
        for obj in self.per_layer_combinations[last_layer]:
            if obj.op == OpType.UNION:
                obj.set_used()
            else:
                obj.set_unused()

        layers = list(range(len(self.per_layer_combinations.keys())))[::-1][
            :-1
        ]
        for layer_index in layers:
            layer = self.per_layer_combinations[layer_index]
            prev_layer = self.per_layer_combinations[layer_index - 1]
            were_used = set()
            for entry in layer:
                if not entry.used:
                    if entry.left_index not in were_used:
                        prev_layer[entry.left_index].set_unused()
                    if entry.right_index not in were_used:
                        prev_layer[entry.right_index].set_unused()
                else:
                    prev_layer[entry.left_index].set_used()
                    prev_layer[entry.right_index].set_used()
                    were_used.add(entry.left_index)
                    were_used.add(entry.right_index)

    def redirect_base_primitives_to_layer(self, layer_index: int):
        if layer_index == 0:
            return
        for entry in self.per_layer_combinations[0]:
            self.per_layer_combinations[layer_index].append(
                copy.deepcopy(entry)
            )

    def commit_meshes(self):
        layers = list(self.per_layer_combinations.keys())[1:]

        for layer_index in layers:
            curr_layer = self.per_layer_combinations[layer_index]
            prev_layer = self.per_layer_combinations[layer_index - 1]
            for entry in curr_layer:
                if not entry.used:
                    continue
                left_entry = prev_layer[entry.left_index]
                right_entry = prev_layer[entry.right_index]

                if entry.op == OpType.UNION:
                    out_obj = pymesh.boolean(
                        left_entry.obj, right_entry.obj, operation="union"
                    )
                elif entry.op == OpType.DIFF:
                    out_obj = pymesh.boolean(
                        left_entry.obj, right_entry.obj, operation="difference"
                    )
                elif entry.op == OpType.INTER:
                    out_obj = pymesh.boolean(
                        left_entry.obj,
                        right_entry.obj,
                        operation="intersection",
                    )
                elif entry.op == OpType.NONE:
                    continue
                else:
                    raise ValueError("Unknown operation: {}".format(entry.op))

                entry.obj = out_obj

    def get_shapes_with_names(self) -> t.List[t.Tuple[str, trimesh.Trimesh]]:
        name_template = "layer-{}_op-{}_index-{}_left-op-{}_right-op-{}.obj"
        output = []
        for layer, entries in self.per_layer_combinations.items():
            for index, entry in enumerate(entries):
                if not entry.used:
                    continue
                name = name_template.format(
                    layer, entry.op, index, entry.left_index, entry.right_index
                )

                obj = to_trimesh(entry.obj)
                trimesh.repair.fix_inversion(obj)
                output.append((name, obj))
        return output


class VoxelReconstructor:
    def __init__(self, model: Net, size: int, sphere_complexity: int):
        self.model = model
        self.size = size
        self.sphere_complexity = sphere_complexity
        self.coords: Optional[torch.Tensor] = None
        self._num_shapes_per_type = model.net.shapes_per_type

        self.initiate()

    def initiate(self):
        dima = self.size
        dim = self.size

        test_point_batch_size = (
            self.size * self.size * self.size
        )  # do not change

        aux_x = np.zeros([dima, dima, dima], np.uint8)
        aux_y = np.zeros([dima, dima, dima], np.uint8)
        aux_z = np.zeros([dima, dima, dima], np.uint8)

        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    aux_x[i, j, k] = i * multiplier
                    aux_y[i, j, k] = j * multiplier
                    aux_z[i, j, k] = k * multiplier
        self.coords = np.zeros([multiplier3, dima, dima, dima, 3], np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[
                        i * multiplier2 + j * multiplier + k, :, :, :, 0
                    ] = (aux_x + i)
                    self.coords[
                        i * multiplier2 + j * multiplier + k, :, :, :, 1
                    ] = (aux_y + j)
                    self.coords[
                        i * multiplier2 + j * multiplier + k, :, :, :, 2
                    ] = (aux_z + k)
        self.coords = (self.coords + 0.5) / dim - 0.5

        self.coords = np.reshape(
            self.coords, [multiplier3, test_point_batch_size, 3]
        )
        self.coords = torch.from_numpy(self.coords).float()

        if torch.cuda.is_available():
            self.coords = self.coords.cuda()

    def retrieve_primitives(
        self,
        translation_params: np.ndarray,
        rotation_params: np.ndarray,
        shape_params: t.List[t.Tuple[str, np.ndarray]],
        composer: CSGComposer,
    ):
        translation_params = translation_params[0]
        rotation_params = rotation_params[0]
        for i, (trans, rot) in enumerate(
            zip(translation_params, rotation_params)
        ):
            type_index = i // self._num_shapes_per_type
            object_index = i % self._num_shapes_per_type

            shape_type_all_parameters = shape_params[type_index][1][0]
            single_shape_params = shape_type_all_parameters[object_index]
            shape_part = self.model.net.evaluator_.parts[type_index]
            if isinstance(shape_part, SquareCubeEvaluator):
                composer.add_box(single_shape_params, trans, rot)
            elif isinstance(shape_part, CircleSphereEvaluator):
                composer.add_sphere(single_shape_params, trans, rot)
            else:
                raise ValueError(
                    "Unknown shape type: {}".format(
                        shape_part.__class__.__name__
                    )
                )

    def get_meshes_with_names(
        self,
        shape_params: t.List[
            t.Tuple[str, np.ndarray]
        ],  # -> List[name, [1, num_shapes, num_params]]
        translation_params: np.ndarray,  # -> 1, num_shapes, 3
        rotation_params: np.ndarray,
        # -> 1, num_shapes, 4 (quaternions)
        sampled_shapes: t.List[np.ndarray],
        # -> List[1, 2, broadcast_points, num_in_shapes, num_out_shapes]
    ) -> t.List[t.Tuple[str, pymesh.Mesh]]:
        composer = CSGComposer(self.sphere_complexity)
        self.retrieve_primitives(
            translation_params, rotation_params, shape_params, composer
        )

        for layer_index, sampled_shapes_layer in enumerate(sampled_shapes):
            sampled_shapes_layer = sampled_shapes_layer[0, :, 0]
            sampled_shapes_layer_indices: np.ndarray = np.argmax(
                sampled_shapes_layer, axis=1
            )

            first_sampled_shape_indices = sampled_shapes_layer_indices[0]
            second_sampled_shape_indices = sampled_shapes_layer_indices[1]

            # union
            for left_index, right_index in zip(
                first_sampled_shape_indices, second_sampled_shape_indices
            ):
                composer.combine_shapes(
                    left_index, right_index, layer_index + 1, OpType.UNION
                )

            # inter
            for left_index, right_index in zip(
                first_sampled_shape_indices, second_sampled_shape_indices
            ):
                composer.combine_shapes(
                    left_index, right_index, layer_index + 1, OpType.INTER
                )

            # diff a - b
            for left_index, right_index in zip(
                first_sampled_shape_indices, second_sampled_shape_indices
            ):
                composer.combine_shapes(
                    left_index, right_index, layer_index + 1, OpType.DIFF
                )

            # diff b - a
            for left_index, right_index in zip(
                first_sampled_shape_indices, second_sampled_shape_indices
            ):
                composer.combine_shapes(
                    right_index, left_index, layer_index + 1, OpType.DIFF
                )

            if layer_index < len(sampled_shapes) - 1:
                composer.redirect_base_primitives_to_layer(layer_index + 1)
        composer.refine_connections()
        composer.commit_meshes()
        meshes_list = composer.get_shapes_with_names()
        # meshes_list[-1][1].show()
        # for name, primitive in meshes_list:
        #     if "layer-0" in name:
        #         primitive)export("primitives/{}".format(name))

        # meshes_list[-1][1].show()
        #
        # import ipdb
        #
        # ipdb.set_trace()
        return meshes_list

    def reconstruct_single(
        self, voxels: torch.Tensor
    ) -> t.Tuple[
        t.List[t.Tuple[np.ndarray, np.ndarray]],
        t.List[t.Tuple[np.ndarray, np.ndarray]],
        t.List[np.ndarray],
        t.List[t.List[t.Tuple[str, pymesh.Mesh]]],
    ]:
        pred_reconstructions = []
        true_reconstructions = []
        points_normals = []
        csg_path = []
        for vox in voxels:
            vox_pred = (
                self.model(vox[None], self.coords)
                .detach()
                .cpu()
                .numpy()
                .reshape((64, 64, 64))
            )

            params = [
                (pair[0], pair[1].detach().cpu().numpy())
                for pair in (
                    self.model.net.evaluator_.get_all_last_predicted_parameters_of_shapes()
                )
            ]
            trans = (
                self.model.net.evaluator_.get_all_translation_vectors()
                .detach()
                .cpu()
                .numpy()
            )
            rot = (
                self.model.net.evaluator_.get_all_rotation_vectors()
                .detach()
                .cpu()
                .numpy()
            )
            sampled_shapes = [
                layer.last_sampled_shapes.detach().cpu().numpy()
                for layer in self.model.net.csg_layers_
            ]

            # meshes
            out_meshes = self.get_meshes_with_names(
                params, trans, rot, sampled_shapes
            )
            csg_path.append(out_meshes)
            if out_meshes[-1][1].vertices.shape[0] == 0:
                vertices, triangles = mcubes.marching_cubes(vox_pred, 0.5)
                vertices = (vertices - 0.5) / self.size - 0.5
            else:
                vertices, triangles = (
                    out_meshes[-1][1].vertices,
                    out_meshes[-1][1].faces,
                )
            pred_reconstructions.append((vertices, triangles))

            vox_numpy = vox.detach().cpu().numpy().reshape((64, 64, 64))
            true_vertices, true_triangles = mcubes.marching_cubes(
                vox_numpy, 0.5
            )
            true_vertices = (true_vertices - 0.5) / self.size - 0.5
            true_reconstructions.append((true_vertices, true_triangles))

            # points
            sampled_points_normals = sample_points_polygon(
                vertices.astype(np.float32),
                triangles.astype(np.int32),
                vox_pred,
                16000,
            )
            sampled_points_normals = torch.from_numpy(sampled_points_normals)
            if torch.cuda.is_available():
                sampled_points_normals = sampled_points_normals.cuda()

            sample_points_value = self.model(
                vox[None], sampled_points_normals[None, :, :3]
            ).reshape((1, -1, 1))

            sampled_points_normals = (
                sampled_points_normals[sample_points_value[0, :, 0] > 1e-4]
                .detach()
                .cpu()
                .numpy()
            )

            np.random.shuffle(sampled_points_normals)
            points_normals.append(sampled_points_normals[:4096])
            self.model.net.clear_retained_codes_and_params()

        return (
            pred_reconstructions,
            true_reconstructions,
            points_normals,
            csg_path,
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstructs all shapes in the dataset by predicting values at "
            "each 3D point and then thresholding"
        ),
        add_help=False,
    )

    parser.add_argument(
        "--weights_path", required=True, help="Path to the model to load"
    )
    parser.add_argument(
        "--size", type=int, help="Data size to be used", required=True
    )
    parser.add_argument(
        "--processed",
        dest="processed_data_path",
        type=str,
        help="Base folder of processed data",
        required=True,
    )
    parser.add_argument(
        "--valid",
        dest="valid_file",
        type=str,
        help="Path to valid HDF5 file with the valid data",
        required=True,
    )
    parser.add_argument(
        "--valid_shape_names",
        type=str,
        help=(
            "Path to valid text file with the names for each data point in "
            "the valid dataset"
        ),
        required=True,
    )
    parser.add_argument(
        "--sphere_complexity",
        type=int,
        help="Number of segments lat/lon of the sphere",
        required=False,
        default=16,
    )

    parser = Net.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    out_dir = Path("data") / "3d_reconstructions"
    out_dir.mkdir(exist_ok=True)

    model = Net.load_from_checkpoint(args.weights_path)
    model.build("", args.valid_file, args.processed_data_path, 64)
    model.turn_fine_tuning_mode()
    model.freeze()
    model.hparams.batch_size = 1
    if torch.cuda.is_available():
        model = model.cuda()
    with open(
        os.path.join(args.processed_data_path, args.valid_shape_names)
    ) as f:
        file_names = f.read().split("\n")

    reconstructor = VoxelReconstructor(
        model, args.size, args.sphere_complexity
    )

    loader = model.val_dataloader()
    current_object_index = 0
    for _, batch in enumerate(tqdm.tqdm(loader)):
        voxels = batch[0]
        should_repeat = False
        temp_object_index = current_object_index
        for _ in enumerate(voxels):
            model_name, instance_name = os.path.split(
                file_names[temp_object_index]
            )
            current_out_dir = out_dir / model_name / instance_name
            if (
                not current_out_dir.exists()
                or not (current_out_dir / "csg_path").exists()
                or not (current_out_dir / "true_vox.npy").exists()
                or not (current_out_dir / "pred_mesh.ply").exists()
                or not (current_out_dir / "pred_mesh.obj").exists()
                or not (current_out_dir / "pred_pc.ply").exists()
            ):
                should_repeat = True
            temp_object_index += 1
        if not should_repeat:
            current_object_index += len(voxels)
            continue
        if torch.cuda.is_available():
            voxels = voxels.cuda()
        (
            pred_reconstsructions,
            true_reconstructions,
            points_normals,
            csg_paths,
        ) = reconstructor.reconstruct_single(voxels)

        voxels = voxels.detach().cpu().numpy()
        for i, true in enumerate(voxels):
            pred_vertices, pred_triangles = pred_reconstsructions[i]
            true_vertices, true_triangles = true_reconstructions[i]

            model_name, instance_name = os.path.split(
                file_names[current_object_index]
            )
            current_out_dir = out_dir / model_name / instance_name
            csg_path_dir = current_out_dir / "csg_path"

            csg_path_dir.mkdir(exist_ok=True, parents=True)

            np.save(current_out_dir / f"true_vox", true)

            write_ply_triangle(
                current_out_dir / "pred_mesh.ply",
                pred_vertices,
                pred_triangles,
            )

            trimesh.Trimesh(pred_vertices, pred_triangles).export(
                current_out_dir / "pred_mesh.obj"
            )

            write_ply_triangle(
                current_out_dir / "true_mesh.ply",
                true_vertices,
                true_triangles,
            )

            write_ply_point_normal(
                current_out_dir / "pred_pc.ply", points_normals[i]
            )

            try:
                for name, mesh in csg_paths[i]:
                    trimesh.Trimesh(mesh.vertices, mesh.faces).export(
                        csg_path_dir / name
                    )
            except ValueError:
                print("Wrong shape")

            current_object_index += 1


if __name__ == "__main__":
    main()
