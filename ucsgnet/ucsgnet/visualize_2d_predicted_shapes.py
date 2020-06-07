import argparse
import copy
import json
import os
import shutil
import typing as t
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
import PIL.ImageOps
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from ucsgnet.dataset import (
    CADDataset,
    SimpleDataset,
    get_simple_2d_transforms,
)
from ucsgnet.ucsgnet.net_2d import Net, RelationLayer
from ucsgnet.ucsgnet.shape_evaluators import (
    CircleSphereEvaluator,
    ShapeEvaluator,
    SquareCubeEvaluator,
)

OUT_DIR = Path("paper-stuff/2d-shapes-visualization")
OUT_DIR.mkdir(exist_ok=True)

ORIGINAL_WIDTH = 64
ORIGINAL_HEIGHT = 64
SCALE_FACTOR = 8
NUM_PIXELS = ORIGINAL_HEIGHT * ORIGINAL_WIDTH
WAS_USED_COLOR = (0, 255, 0)
SOFTMAX_THRESHOLD = 0.01
LINE_WIDTH = 9

SCALE = max(ORIGINAL_WIDTH * SCALE_FACTOR, ORIGINAL_HEIGHT * SCALE_FACTOR)


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
    RECTANGLE = "rect"
    CIRCLE = "circ"


def detach_tensor(arg: torch.Tensor) -> np.ndarray:
    return arg.cpu().detach().numpy()


def detach_tensors(*args: torch.Tensor) -> t.Tuple[np.ndarray, ...]:
    array = tuple(arg.cpu().detach().numpy() for arg in args)
    return array


def numpy_to_list_of_tuples(arg: np.ndarray) -> t.List[t.Tuple[float, ...]]:
    return [tuple(elems) for elems in arg]


def polygon_width(
    self: PIL.ImageDraw.ImageDraw,
    points: t.List[t.Tuple[float, ...]],
    fill: t.Union[str, int, t.Tuple[int, ...]],
    width: int,
):
    points = points + [points[0]]
    self.line(points, fill=fill, width=width)
    for point in points:
        point = np.array(point)
        self.ellipse(
            (tuple(point - width // 2), tuple(point + width // 2)), fill=fill
        )


PIL.ImageDraw.ImageDraw.polygon_width = polygon_width


class _Sampler:
    def __init__(self):
        colormap = plt.get_cmap("hsv")
        indices = np.random.permutation(256)

        self._cur_index = -1
        self._colors = colormap(np.linspace(0, 1, 256))[indices]
        np.random.shuffle(self._colors)

    def next_color(self):
        self._cur_index = (self._cur_index + 1) % len(self._colors)
        return tuple((self._colors[self._cur_index] * 255).astype(np.int)[:-1])


COLOR_SAMPLER = _Sampler()


def transform_to_the_original_scale(points: np.ndarray) -> np.ndarray:
    return (points + 0.5) * SCALE - 0.5


def transform_points(
    points: np.ndarray, translation: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    rotation = rotation.item()
    rot_matrix = np.array(
        [
            [np.cos(rotation), np.sin(rotation)],
            [-np.sin(rotation), np.cos(rotation)],
        ]
    )
    points = points @ rot_matrix.T + translation
    points = transform_to_the_original_scale(points)
    return points


def get_transformed_circle_points(
    original_radius, translate: np.ndarray, rotation: np.ndarray
) -> t.List[t.Tuple[float, ...]]:
    radius = original_radius * SCALE
    center = np.array((0, 0))
    center = transform_points(center[None], translate, rotation)
    points = numpy_to_list_of_tuples(
        np.concatenate((center - radius, center + radius), axis=0)
    )
    return points


def get_transformed_rectangle_points(
    parameters: np.ndarray, translation: np.ndarray, rotation: np.ndarray
) -> t.List[t.Tuple[float, ...]]:
    w, h = parameters

    left_top = -w, -h
    right_top = w, -h

    left_bottom = -w, h
    right_bottom = w, h
    points = np.array((left_top, right_top, right_bottom, left_bottom))
    points = numpy_to_list_of_tuples(
        transform_points(points, translation, rotation)
    )
    return points


class AbstractComposer:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def draw_circle(
        self,
        params: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,  # mentioned for compatibility only
        corresponding_attention_weights: t.Optional[t.List[np.ndarray]],
    ):
        raise NotImplementedError

    def draw_rectangle(
        self,
        params: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        corresponding_attention_weights: t.Optional[t.List[np.ndarray]],
    ):
        raise NotImplementedError


class Composer(AbstractComposer):
    def __init__(
        self,
        orig_img: np.ndarray,
        prediction: np.ndarray,
        height: int,
        width: int,
    ):
        super().__init__(height, width)

        orig_img = cv2.cvtColor(255 - orig_img, cv2.COLOR_BGR2BGRA)
        prediction = cv2.cvtColor(255 - prediction, cv2.COLOR_BGR2BGRA)

        # original img
        self.orig_img_ = PIL.Image.fromarray(orig_img, mode="RGBA")
        self.orig_img_ = self.orig_img_.resize(
            size=(
                self.orig_img_.size[0] * SCALE_FACTOR,
                self.orig_img_.size[1] * SCALE_FACTOR,
            ),
            resample=PIL.Image.BILINEAR,
        )
        self.orig_img_used_only_ = self.orig_img_.copy()

        # predictions
        self.pred_img_ = PIL.Image.fromarray(prediction, mode="RGBA")
        self.pred_img_ = self.pred_img_.resize(
            size=(
                self.pred_img_.size[0] * SCALE_FACTOR,
                self.pred_img_.size[1] * SCALE_FACTOR,
            ),
            resample=PIL.Image.BILINEAR,
        )
        self.pred_img_used_only_ = self.pred_img_.copy()

        # clear imgs
        self.img_ = PIL.Image.new(
            "RGBA", (self.width, self.height), (255, 255, 255)
        )
        self.img_used_only_ = PIL.Image.new(
            "RGBA", (self.width, self.height), (255, 255, 255)
        )

        # canvas
        self.canvas_ = PIL.ImageDraw.Draw(self.img_, "RGBA")
        self.canvas_used_only_ = PIL.ImageDraw.Draw(
            self.img_used_only_, "RGBA"
        )
        self.canvas_orig_ = PIL.ImageDraw.Draw(self.orig_img_, "RGBA")
        self.canvas_orig_used_only_ = PIL.ImageDraw.Draw(
            self.orig_img_used_only_, "RGBA"
        )
        self.canvas_pred_ = PIL.ImageDraw.Draw(self.pred_img_, "RGBA")
        self.canvas_pred_used_only_ = PIL.ImageDraw.Draw(
            self.pred_img_used_only_, "RGBA"
        )

    @classmethod
    def was_shape_used(
        cls, index: int, attention_weights: t.List[np.ndarray]
    ) -> bool:
        for attention_weight_layer in attention_weights:
            argmaxes = attention_weight_layer.argmax(axis=1)
            maxes = np.max(attention_weight_layer, axis=1)
            if np.any(argmaxes == index):
                if np.any(maxes[argmaxes == index] >= SOFTMAX_THRESHOLD):
                    return True
        return False

    def draw_circle(
        self,
        params: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,  # mentioned for compatibility only
        corresponding_attention_weights: t.Optional[t.List[np.ndarray]],
    ):
        for i, (p, t, r) in enumerate(zip(params, translation, rotation)):
            points = get_transformed_circle_points(p, t, r)
            outline_color = COLOR_SAMPLER.next_color()

            self.canvas_orig_.ellipse(
                points, outline=outline_color, width=LINE_WIDTH
            )
            self.canvas_pred_.ellipse(
                points, outline=outline_color, width=LINE_WIDTH
            )

            if self.was_shape_used(i, corresponding_attention_weights):
                self.canvas_.ellipse(
                    points, outline=WAS_USED_COLOR, width=LINE_WIDTH
                )
                self.canvas_used_only_.ellipse(
                    points, outline=(0, 0, 0), width=LINE_WIDTH
                )
                self.canvas_orig_used_only_.ellipse(
                    points, outline=(168, 0, 0, 255), width=LINE_WIDTH
                )
                self.canvas_pred_used_only_.ellipse(
                    points, outline=(168, 0, 0, 255), width=LINE_WIDTH
                )
            else:
                self.canvas_.ellipse(
                    points, outline=(0, 0, 0), width=LINE_WIDTH
                )

    def draw_rectangle(
        self,
        params: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        corresponding_attention_weights: t.Optional[t.List[np.ndarray]],
    ):
        for i, (p, t, r) in enumerate(zip(params, translation, rotation)):
            points = get_transformed_rectangle_points(p, t, r)
            outline_color = COLOR_SAMPLER.next_color()
            self.canvas_orig_.polygon_width(
                points, fill=outline_color, width=LINE_WIDTH
            )
            self.canvas_pred_.polygon_width(
                points, fill=outline_color, width=LINE_WIDTH
            )
            if self.was_shape_used(i, corresponding_attention_weights):
                self.canvas_.polygon_width(
                    points, fill=WAS_USED_COLOR, width=LINE_WIDTH
                )
                self.canvas_used_only_.polygon_width(
                    points, fill=(0, 0, 0), width=LINE_WIDTH
                )
                self.canvas_orig_used_only_.polygon_width(
                    points, fill=(168, 0, 0), width=LINE_WIDTH
                )
                self.canvas_pred_used_only_.polygon_width(
                    points, fill=(168, 0, 0), width=LINE_WIDTH
                )
            else:
                self.canvas_.polygon_width(
                    points, fill=(0, 0, 0), width=LINE_WIDTH
                )


class BaseShapeComposer(AbstractComposer):
    def __init__(self, shape_img: np.ndarray, height: int, width: int):
        super().__init__(height, width)
        self.shape_img = PIL.Image.fromarray(shape_img, mode="RGB")
        self.shape_img = PIL.ImageOps.invert(self.shape_img)
        self.canvas = PIL.ImageDraw.Draw(self.shape_img, "RGB")

    def draw_circle(
        self,
        params: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        corresponding_attention_weights: t.Optional[t.List[np.ndarray]],
    ):
        points = get_transformed_circle_points(params, translation, rotation)
        outline_color = COLOR_SAMPLER.next_color()
        self.canvas.ellipse(points, outline=outline_color, width=LINE_WIDTH)

    def draw_rectangle(
        self,
        params: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
        corresponding_attention_weights: t.Optional[t.List[np.ndarray]],
    ):
        points = get_transformed_rectangle_points(
            params, translation, rotation
        )
        outline_color = COLOR_SAMPLER.next_color()
        self.canvas.polygon_width(points, fill=outline_color, width=LINE_WIDTH)


def aggregate_shape_weights_from_model(
    model: Net, part_index: int
) -> t.List[np.ndarray]:
    weights = []
    per_layer_out_shapes = model.net.out_shapes_per_layer * 4
    shapes_per_type = model.net.shapes_per_type
    for index, layer in enumerate(
        model.net.csg_layers_
    ):  # type: (int, RelationLayer)
        param = layer.last_sampled_shapes
        param = param.detach().cpu().numpy()[0, :, 0]
        if index == 0:
            weights.append(
                param[
                    :,
                    part_index
                    * shapes_per_type : (part_index + 1)
                    * shapes_per_type,
                ]
            )
        else:
            weights.append(
                param[
                    :,
                    per_layer_out_shapes
                    + part_index
                    * shapes_per_type : (
                        (part_index + 1) * shapes_per_type
                        + per_layer_out_shapes
                    ),
                ]
            )
    return weights


def visualize_primitives_from_each_layer(
    model: Net,
    original_images: np.ndarray,
    predictions: np.ndarray,
    out_dir: Path,
):
    layer: RelationLayer
    part: ShapeEvaluator

    drawer = Composer(
        original_images,
        predictions,
        ORIGINAL_HEIGHT * SCALE_FACTOR,
        ORIGINAL_WIDTH * SCALE_FACTOR,
    )

    for i, part in enumerate(model.net.evaluator_):
        parameters = detach_tensor(part.last_parameters)[0]
        parameters = np.squeeze(parameters)
        translation = detach_tensor(part.translation_vector_prediction)[0]
        rotation = detach_tensor(part.rotation_params_prediction)[0]

        part_weights = aggregate_shape_weights_from_model(model, i)

        if isinstance(part, CircleSphereEvaluator):
            drawer.draw_circle(parameters, translation, rotation, part_weights)
        elif isinstance(part, SquareCubeEvaluator):
            drawer.draw_rectangle(
                parameters, translation, rotation, part_weights
            )
        else:
            raise ValueError(
                "Unsupported evaluator: {}".format(part.__class__.__name__)
            )
    drawer.img_.save((out_dir / f"marked_primitives.png").as_posix())
    drawer.img_used_only_.save(
        (out_dir / f"marked_primitives_used_only.png").as_posix()
    )
    drawer.orig_img_.save(
        (out_dir / f"marked_primitives_with_gt.png").as_posix()
    )
    drawer.orig_img_used_only_.save(
        (out_dir / f"marked_primitives_with_gt_used_only.png").as_posix()
    )
    drawer.pred_img_.save(
        (out_dir / f"marked_primitives_with_pred.png").as_posix()
    )
    drawer.pred_img_used_only_.save(
        (out_dir / f"marked_primitives_with_pred_used_only.png").as_posix()
    )


@dataclass
class CSGEntry:
    obj: np.ndarray
    op: OpType
    used: bool
    left_index: int = -1
    right_index: int = -1

    def set_used(self):
        self.used = True

    def set_unused(self):
        self.used = False


class CSGComposer:
    def __init__(
        self,
        height: int,
        width: int,
        original_img: np.ndarray,
        prediction: np.ndarray,
    ):
        self.height = height
        self.width = width
        self.base_shapes_imgs_: t.List[np.ndarray] = []
        self.per_layer_combinations: t.Dict[int, t.List[CSGEntry]] = {}

        self.original_img = PIL.Image.fromarray(
            255 - original_img, "RGB"
        ).resize(
            size=(
                original_img.shape[0] * SCALE_FACTOR,
                original_img.shape[1] * SCALE_FACTOR,
            ),
            resample=PIL.Image.BILINEAR,
        )
        self.prediction = PIL.Image.fromarray(255 - prediction, "RGB").resize(
            size=(
                prediction.shape[0] * SCALE_FACTOR,
                prediction.shape[1] * SCALE_FACTOR,
            ),
            resample=PIL.Image.BILINEAR,
        )

        self.original_img_canvas = PIL.ImageDraw.Draw(self.original_img)
        self.prediction_canvas = PIL.ImageDraw.Draw(self.prediction)

        self._base_shapes_params = []

    def draw_circle(
        self,
        params: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,  # mentioned for compatibility only
    ):
        for i, (p, t, r) in enumerate(zip(params, translation, rotation)):
            points = get_transformed_circle_points(p, t, r)
            img = PIL.Image.new("1", (self.width, self.height), 0)

            canvas = PIL.ImageDraw.Draw(img, "1")
            canvas.ellipse(points, fill=1)
            self.base_shapes_imgs_.append(np.array(img, dtype=np.float32))
            self._base_shapes_params.append((points, ObjType.CIRCLE))

    def draw_rectangle(
        self, params: np.ndarray, translation: np.ndarray, rotation: np.ndarray
    ):
        for i, (p, t, r) in enumerate(zip(params, translation, rotation)):
            points = get_transformed_rectangle_points(p, t, r)
            img = PIL.Image.new("1", (self.width, self.height), 0)

            canvas = PIL.ImageDraw.Draw(img, "1")
            canvas.polygon(points, fill=1)
            self.base_shapes_imgs_.append(np.array(img).astype(np.float32))
            self._base_shapes_params.append((points, ObjType.RECTANGLE))

    def initiate_first_layer_with_base_shapes(self):
        self.per_layer_combinations = {
            0: [
                CSGEntry(elem, OpType.NONE, False)
                for elem in self.base_shapes_imgs_
            ]
        }

    def combine_shapes(
        self,
        op_left_index: int,
        op_right_index: int,
        cur_layer_index: int,
        op_type: OpType,
    ):
        prev_layer_index = cur_layer_index - 1
        op_left = self.per_layer_combinations[prev_layer_index][
            op_left_index
        ].obj
        op_right = self.per_layer_combinations[prev_layer_index][
            op_right_index
        ].obj

        self.per_layer_combinations[prev_layer_index][op_left_index].set_used()
        self.per_layer_combinations[prev_layer_index][
            op_right_index
        ].set_used()

        if op_type == OpType.UNION:
            out = (op_left + op_right).clip(0, 1)
        elif op_type == OpType.DIFF:
            out = (op_left - op_right).clip(0, 1)
        elif op_type == OpType.INTER:
            out = (op_left + op_right - 1).clip(0, 1)
        else:
            raise ValueError("Unknown OP type: {}".format(op_type))

        if cur_layer_index not in self.per_layer_combinations:
            self.per_layer_combinations[cur_layer_index] = []
        self.per_layer_combinations[cur_layer_index].append(
            CSGEntry(out, op_type, False, op_left_index, op_right_index)
        )

    def refine_connections(self):
        layers = list(self.per_layer_combinations.keys())[::-1][1:-1]
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

    def commit_to_canvas(self):
        objects = self.per_layer_combinations[0]
        for ind, (obj, (points, obj_type)) in enumerate(
            zip(objects, self._base_shapes_params)
        ):
            if obj.used:
                if obj_type == ObjType.RECTANGLE:
                    self.prediction_canvas.polygon_width(
                        points, fill=(255, 0, 0), width=LINE_WIDTH
                    )
                    self.original_img_canvas.polygon_width(
                        points, fill=(255, 0, 0), width=LINE_WIDTH
                    )
                elif obj_type == ObjType.CIRCLE:
                    self.prediction_canvas.ellipse(
                        points, outline=(255, 0, 0), width=LINE_WIDTH
                    )
                    self.original_img_canvas.ellipse(
                        points, outline=(255, 0, 0), width=LINE_WIDTH
                    )
                else:
                    raise ValueError(
                        "Unknown object type: {}".format(obj_type.value)
                    )

    def __len__(self) -> int:
        return len(self.base_shapes_imgs_)


def visualize_reconstruction_path(
    model: Net, out_dir: Path, original_img: np.ndarray, prediction: np.ndarray
):
    out_out_dir = out_dir / "csg_path"
    out_out_dir.mkdir(parents=True, exist_ok=True)
    csg_composer = CSGComposer(
        SCALE_FACTOR * ORIGINAL_HEIGHT,
        SCALE_FACTOR * ORIGINAL_WIDTH,
        original_img,
        prediction,
    )
    for i, part in enumerate(model.net.evaluator_):
        parameters = detach_tensor(part.last_parameters)[0]
        parameters = np.squeeze(parameters)
        translation = detach_tensor(part.translation_vector_prediction)[0]
        rotation = detach_tensor(part.rotation_params_prediction)[0]

        if isinstance(part, CircleSphereEvaluator):
            csg_composer.draw_circle(parameters, translation, rotation)
        elif isinstance(part, SquareCubeEvaluator):
            csg_composer.draw_rectangle(parameters, translation, rotation)

    csg_composer.initiate_first_layer_with_base_shapes()
    for layer_index, relation_layer in enumerate(
        model.net.csg_layers_
    ):  # type: (int, RelationLayer)
        mask = (
            relation_layer.last_sampled_shapes.detach().cpu().numpy()[0, :, 0]
        )
        operands = np.argmax(mask, axis=1)
        # to match output way in the relation layer

        # union
        for out_shape_index in range(operands.shape[-1]):
            op_left_index, op_right_index = operands[:, out_shape_index]
            csg_composer.combine_shapes(
                op_left_index, op_right_index, layer_index + 1, OpType.UNION
            )

        # inter
        for out_shape_index in range(operands.shape[-1]):
            op_left_index, op_right_index = operands[:, out_shape_index]
            csg_composer.combine_shapes(
                op_left_index, op_right_index, layer_index + 1, OpType.INTER
            )

        # diff a - b
        for out_shape_index in range(operands.shape[-1]):
            op_left_index, op_right_index = operands[:, out_shape_index]
            csg_composer.combine_shapes(
                op_left_index, op_right_index, layer_index + 1, OpType.DIFF
            )

        # diff b - a
        for out_shape_index in range(operands.shape[-1]):
            op_left_index, op_right_index = operands[:, out_shape_index]
            csg_composer.combine_shapes(
                op_right_index, op_left_index, layer_index + 1, OpType.DIFF
            )

        # csg_composer.redirect_base_primitives_to_layer(layer_index + 1)

    csg_composer.refine_connections()
    csg_composer.commit_to_canvas()
    name_template = "layer-{}_op-{}_index-{}_left-op-{}_right-op-{}.png"
    for layer, objects in csg_composer.per_layer_combinations.items():
        for ind, obj in enumerate(objects):
            if obj.used or (
                layer == len(model.net.csg_layers_) and obj.op == OpType.UNION
            ):
                cv2.imwrite(
                    (
                        out_out_dir
                        / name_template.format(
                            layer, obj.op, ind, obj.left_index, obj.right_index
                        )
                    ).as_posix(),
                    ((1 - obj.obj) * 255).astype(np.uint8),
                )

    csg_composer.original_img.save(
        (out_out_dir / "ground-truth-used.png").as_posix()
    )
    csg_composer.prediction.save(
        (out_out_dir / "prediction-used.png").as_posix()
    )


def visualize_input_and_output_at_given_layer(
    intermediate_results: t.List[torch.Tensor], model: Net, out_dir: Path
):
    first_layer_input_data: t.Optional[torch.Tensor] = None

    for layer_index in range(len(intermediate_results)):
        to_visualize: t.List[t.Tuple[np.ndarray, bool]] = []

        if layer_index == 0:
            first_layer_input_data = intermediate_results[layer_index]
            current_to_iterate = intermediate_results[layer_index]
        else:
            current_to_iterate = torch.cat(
                (intermediate_results[layer_index], first_layer_input_data),
                dim=-1,
            )

        relation_layer: RelationLayer = model.net.csg_layers_[layer_index]
        mask = (
            relation_layer.last_sampled_shapes.detach().cpu().numpy()[0, :, 0]
        )

        for output_shape_index in range(current_to_iterate.shape[-1]):
            img = detach_tensor(
                model.binarize(current_to_iterate[..., output_shape_index])
            ).reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))

            was_used = np.any(
                (mask[:, output_shape_index] > SOFTMAX_THRESHOLD).astype(
                    np.float32
                )
                != 0
            )
            img = (
                cv2.resize(img, (512, 512), cv2.INTER_LANCZOS4) * 255
            ).astype(np.uint8)

            to_visualize.append((img, was_used))

        # saving the visualization
        for output_shape_index, (image, was_used) in enumerate(to_visualize):
            postfix = "_used" if was_used else "_not_used"
            out_path = (
                out_dir
                / f"intermediate_result_{layer_index}_{output_shape_index}_"
                f"{postfix}.png"
            ).as_posix()
            cv2.imwrite(out_path, image)


def visualize_combined_first_layer_results_with_primitives(
    model: Net, intermediate_results: t.List[torch.Tensor], out_dir: Path
):
    first_layer_input_data = intermediate_results[0]

    for output_shape_index in range(first_layer_input_data.shape[-1]):
        img = detach_tensor(
            model.binarize(first_layer_input_data[..., output_shape_index])
        ).reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))

        img = (cv2.resize(img, (512, 512), cv2.INTER_LANCZOS4) * 255).astype(
            np.uint8
        )
        img = np.expand_dims(img, axis=-1).repeat(3, axis=-1)

        base_shape_composer = BaseShapeComposer(
            img, ORIGINAL_HEIGHT * SCALE_FACTOR, ORIGINAL_WIDTH * SCALE_FACTOR
        )

        part_index = output_shape_index // model.net.shapes_per_type
        object_index = output_shape_index % model.net.shapes_per_type

        part = model.net.evaluator_.parts[part_index]
        parameters = detach_tensor(part.last_parameters[0, object_index])
        parameters = np.squeeze(parameters)
        translation = detach_tensor(
            part.translation_vector_prediction[0, object_index]
        )
        rotation = detach_tensor(
            part.rotation_params_prediction[0, object_index]
        )

        if isinstance(part, SquareCubeEvaluator):
            base_shape_composer.draw_rectangle(
                parameters, translation, rotation, None
            )
        elif isinstance(part, CircleSphereEvaluator):
            base_shape_composer.draw_circle(
                parameters, translation, rotation, None
            )

        out_path = (
            out_dir / f"shape_with_primitive_{output_shape_index}.png"
        ).as_posix()
        cv2.imwrite(out_path, np.array(base_shape_composer.shape_img))


def visualize_synthetic_dataset(model: Net, img_path: str):
    loader = DataLoader(
        SimpleDataset(
            [img_path],
            points_per_sample=NUM_PIXELS,
            points_with_distances_paths=None,
            transforms=get_simple_2d_transforms(),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    image, points, trues, _ = next(iter(loader))

    (
        output,
        predicted_shapes_distances,
        intermediate_results,
        scaled_distances,
    ) = model(
        image,
        points,
        return_distances_to_base_shapes=True,
        return_intermediate_output_csg=True,
        return_scaled_distances_to_shapes=True,
    )

    binarized = model.binarize(output)
    mse_loss = F.mse_loss(binarized, trues).item()
    img_name = Path(img_path).with_suffix("").name
    out_dir = OUT_DIR / "synthetic" / f"{mse_loss}_{img_name}"
    out_dir.mkdir(exist_ok=True, parents=True)

    np_image = cv2.imread(img_path)
    shutil.copy2(img_path, (out_dir / "ground-truth.png").as_posix())

    binarized = (
        binarized.squeeze()
        .detach()
        .cpu()
        .numpy()
        .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
    )
    np_output = (
        output.squeeze()
        .detach()
        .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
        .unsqueeze(dim=-1)
        .expand(([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3]))
        .cpu()
        .numpy()
    )
    np_output = (np_output * 255).astype(np.uint8)
    intermediate_prediction = intermediate_results[-1]

    cv2.imwrite(
        (out_dir / "binarized.png").as_posix(),
        (binarized * 255).astype(np.uint8),
    )

    cv2.imwrite(
        (
            out_dir
            / f"intermediate_result_{len(model.net.csg_layers_)}_0_used.png"
        ).as_posix(),
        (detach_tensor(intermediate_prediction)[..., 0] * 255)
        .astype(np.uint8)
        .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH)),
    )

    visualize_primitives_from_each_layer(model, np_image, np_output, out_dir)
    visualize_input_and_output_at_given_layer(
        intermediate_results[:-1], model, out_dir
    )
    visualize_combined_first_layer_results_with_primitives(
        model, intermediate_results[:-1], out_dir
    )
    visualize_reconstruction_path(model, out_dir, np_image, np_output)


def visualize_cad_dataset(model: Net, h5_file_path: str):
    loader = DataLoader(
        CADDataset(
            h5_file_path,
            data_split="valid",
            transforms=get_simple_2d_transforms(),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for data_index, (image, points, trues, _) in enumerate(tqdm.tqdm(loader)):
        (
            output,
            predicted_shapes_distances,
            intermediate_results,
            scaled_distances,
        ) = model(
            image,
            points,
            return_distances_to_base_shapes=True,
            return_intermediate_output_csg=True,
            return_scaled_distances_to_shapes=True,
        )

        binarized = model.binarize(output)
        mse_loss = F.mse_loss(binarized, trues).item()
        out_dir = OUT_DIR / "cad" / f"{mse_loss}_{data_index}"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_dir.mkdir(exist_ok=True)

        np_image = (
            image.detach().cpu().numpy().squeeze(axis=(0, 1)) * 255
        ).astype(np.uint8)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)

        cv2.imwrite((out_dir / "ground-truth.png").as_posix(), np_image)

        binarized = (
            binarized.squeeze()
            .detach()
            .cpu()
            .numpy()
            .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
        )
        np_output = (
            output.squeeze()
            .detach()
            .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH))
            .unsqueeze(dim=-1)
            .expand(([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 3]))
            .cpu()
            .numpy()
        )
        np_output = (np_output * 255).astype(np.uint8)
        intermediate_prediction = intermediate_results[-1]

        cv2.imwrite(
            (out_dir / "binarized.png").as_posix(),
            (binarized * 255).astype(np.uint8),
        )

        cv2.imwrite(
            (
                out_dir / f"intermediate_result_"
                f"{len(model.net.csg_layers_)}_0_used.png"
            ).as_posix(),
            (detach_tensor(intermediate_prediction)[..., 0] * 255)
            .astype(np.uint8)
            .reshape((ORIGINAL_HEIGHT, ORIGINAL_WIDTH)),
        )

        visualize_primitives_from_each_layer(
            model, np_image, np_output, out_dir
        )
        visualize_input_and_output_at_given_layer(
            intermediate_results[:-1], model, out_dir
        )
        visualize_combined_first_layer_results_with_primitives(
            model, intermediate_results[:-1], out_dir
        )
        visualize_reconstruction_path(model, out_dir, np_image, np_output)


def visualize_multiple(
    checkpoint_path: str, data_path: str, valid: str, data_type: str
):
    model = Net.load_from_checkpoint(checkpoint_path)
    model = model.eval()
    model.freeze()
    model.turn_fine_tuning_mode()

    if data_type == "synthetic":
        with open(valid) as f:
            paths = json.load(f)

        for sample in tqdm.tqdm(paths):
            visualize_synthetic_dataset(model, os.path.join(data_path, sample))

    elif data_type == "cad":
        visualize_cad_dataset(model, os.path.join(data_path, valid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Path to ckpt path",
        dest="checkpoint_path",
    )
    parser.add_argument(
        "--valid",
        required=True,
        type=str,
        help="Path to a file containing valid data",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to the folder containing the data",
        dest="data_path",
    )
    parser.add_argument(
        "--data_type",
        required=True,
        type=str,
        choices=["synthetic", "cad"],
        help="Data to be used to reconstruc images",
    )

    args = parser.parse_args()
    visualize_multiple(**vars(args))


if __name__ == "__main__":
    main()
