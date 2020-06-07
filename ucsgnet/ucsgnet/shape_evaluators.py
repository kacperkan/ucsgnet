import abc
import math
import typing as t

import torch
import torch.nn as nn
from ucsgnet.utils import quat_to_rot_matrix


class ShapeEvaluator(nn.Module):
    def __init__(
        self, num_parameters: int, num_of_shapes: int, num_dimensions: int
    ):
        super().__init__()
        num_rotation_parameters = 1 if num_dimensions == 2 else 4

        self.num_of_shapes = num_of_shapes
        self.num_parameters = num_parameters
        self.num_dimensions = num_dimensions
        self._ef_features = 32
        self.layers_ = nn.Linear(
            self._ef_features * 64,
            self.num_of_shapes
            * (self.num_parameters + num_dimensions + num_rotation_parameters),
        )
        self._translation_vector_prediction: t.Optional[torch.Tensor] = None
        self._rotation_vector_prediction: t.Optional[torch.Tensor] = None
        self._last_parameters: t.Optional[torch.Tensor] = None

    def forward(
        self, code: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        batch_size = code.shape[0]
        all_parameters = self.layers_(code).reshape(
            batch_size, self.num_of_shapes, -1
        )
        shape_params = all_parameters[:, :, : self.num_parameters]
        translation = all_parameters[
            :,
            :,
            self.num_parameters : self.num_parameters + self.num_dimensions,
        ]
        rotation = all_parameters[
            :, :, self.num_parameters + self.num_dimensions :
        ]

        self._last_parameters = shape_params
        self._translation_vector_prediction = translation
        self._rotation_vector_prediction = rotation

        points = self.transform_points(points, translation, rotation)
        return self.evaluate_points(shape_params, points)

    def transform_points(
        self,
        points: torch.Tensor,
        translation: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        rotation = rotation.unsqueeze(dim=-2)

        if self.num_dimensions == 2:
            rotation_matrices = rotation.new_zeros(
                rotation.shape[:-1] + (2, 2)
            )
            rotation = rotation[..., 0]
            rotation_matrices[..., 0, 0] = rotation.cos()
            rotation_matrices[..., 0, 1] = rotation.sin()
            rotation_matrices[..., 1, 0] = -rotation.sin()
            rotation_matrices[..., 1, 1] = rotation.cos()
        else:
            rotation_matrices = quat_to_rot_matrix(rotation)
            rotation_matrices = rotation_matrices.transpose(-2, -1)

        points = points - translation.unsqueeze(dim=-2)
        points = (rotation_matrices * points.unsqueeze(dim=-1)).sum(dim=-2)
        return points

    @abc.abstractmethod
    def evaluate_points(
        self, parameters: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    def translation_vector_prediction(self) -> torch.Tensor:
        return self._translation_vector_prediction

    @property
    def rotation_params_prediction(self) -> torch.Tensor:
        return self._rotation_vector_prediction

    @property
    def last_predicted_parameters_of_shape(self) -> torch.Tensor:
        return self._last_parameters

    def clear_translation_vector_prediction(self):
        self._translation_vector_prediction = None

    @property
    def last_parameters(self):
        return self._last_parameters

    @abc.abstractmethod
    def get_volume(self) -> t.Union[float, torch.Tensor]:
        ...


class CircleSphereEvaluator(ShapeEvaluator):
    def __init__(self, num_of_shapes: int, num_dimensions: int):
        super().__init__(1, num_of_shapes, num_dimensions)

    def evaluate_points(
        self, parameters: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        lengths = points.norm(dim=-1)
        return lengths - parameters

    def get_volume(self) -> t.Union[float, torch.Tensor]:
        if self._last_parameters is None:
            return 0.0

        if self.num_dimensions == 3:
            return 4 / 3 * self._last_parameters.pow(3).sum(dim=-1) * math.pi
        elif self.num_dimensions == 2:
            return self._last_parameters.pow(2).sum(dim=-1) * math.pi
        else:
            raise ValueError(
                f"Not supported num of dimensions: {self.num_dimensions}"
            )


class SquareCubeEvaluator(ShapeEvaluator):
    def __init__(self, num_of_shapes: int, num_dimensions: int):
        super().__init__(num_dimensions, num_of_shapes, num_dimensions)

    def evaluate_points(
        self, parameters: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        q_points = points.abs() - parameters.unsqueeze(dim=-2)

        lengths = (q_points.max(torch.zeros_like(q_points))).norm(dim=-1)
        zeros_points = torch.zeros_like(lengths)
        xs = q_points[..., 0]
        ys = q_points[..., 1]
        if self.num_dimensions > 2:
            zs = q_points[..., 2]
            filling = ys.max(zs).max(xs).min(zeros_points)
        else:
            filling = ys.max(xs).min(zeros_points)
        return lengths + filling

    def get_volume(self) -> t.Union[float, torch.Tensor]:
        return self._last_parameters.prod(dim=-1)


class ConeEvaluator(ShapeEvaluator):
    def __init__(self, num_of_shapes: int, num_dimensions: int):
        super().__init__(3, num_of_shapes, num_dimensions)

    def evaluate_points(
        self, parameters: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:

        trig_angle = parameters.unsqueeze(dim=-2)
        if self.num_dimensions > 2:
            q = torch.cat(
                (
                    points[..., [0, 2]].norm(dim=-1, p=2, keepdim=True),
                    points[..., [1]],
                ),
                dim=-1,
            )
            d1 = -q[..., 1] - trig_angle[..., -1]
            d2 = (q * trig_angle[..., [0, 1]]).sum(dim=-1).max(q[..., 1])
            d1d2 = torch.stack((d1, d2), dim=-1).clamp_min(0)
            return d1d2.norm(p=2, dim=-1) + d1.max(d2).clamp_max(0)

        q = torch.cat(
            (
                points[..., [0]].norm(dim=-1, p=2, keepdim=True),
                points[..., [1]],
            ),
            dim=-1,
        )
        d1 = -q[..., 1] - trig_angle[..., -1]
        d2 = (q * trig_angle[..., [0, 1]]).sum(dim=-1).max(q[..., 1])
        d1d2 = torch.stack((d1, d2), dim=-1).clamp_min(0)
        return d1d2.norm(p=2, dim=-1) + d1.max(d2).clamp_max(0)

    def get_volume(self) -> t.Union[float, torch.Tensor]:
        # TODO: this is currently copied from the shpere
        return self._last_parameters.prod(dim=-1)


class CompundEvaluator(nn.Module):
    def __init__(self, parts: t.Sequence[ShapeEvaluator]):
        super().__init__()
        self.parts: nn.ModuleList = nn.ModuleList(parts)

    def forward(self, x: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        return torch.cat([part(x, points) for part in self.parts], dim=1)

    def __len__(self) -> int:
        return len(self.parts)

    def get_all_translation_vectors(self) -> torch.Tensor:
        return torch.cat(
            [part.translation_vector_prediction for part in self.parts], dim=1
        )

    def get_all_rotation_vectors(self) -> torch.Tensor:
        return torch.cat(
            [part.rotation_params_prediction for part in self.parts], dim=1
        )

    def get_all_last_predicted_parameters_of_shapes(
        self,
    ) -> t.List[t.Tuple[str, torch.Tensor]]:
        return [
            (part.__class__.__name__, part.last_predicted_parameters_of_shape)
            for part in self.parts
        ]

    def clear_translation_vectors(self):
        for part in self.parts:
            part.clear_translation_vector_prediction()

    def __iter__(self) -> ShapeEvaluator:
        for part in self.parts:
            yield part

    def enumerate_indices(self) -> t.Iterator[t.Tuple[int, int]]:
        offset = 0
        for i, part in enumerate(self.parts):
            yield offset, offset + part.num_of_shapes
            offset += part.num_of_shapes


class PlanesEvaluator(nn.Module):
    def __init__(self, num_of_planes: int, num_of_dimensions: int):
        super().__init__()
        self._ef_features = 32
        self.num_of_planes = num_of_planes
        self.num_of_dimensions = num_of_dimensions
        self.layers_ = nn.Linear(
            self._ef_features * 64,
            self.num_of_planes * (self.num_of_dimensions + 1),
        )
        self._last_parameters: t.Optional[torch.Tensor] = None

    def forward(self, code: torch.Tensor, points: torch.Tensor):
        batch_size = code.shape[0]
        coords_shape = tuple(points.shape[:-1]) + (1,)
        additional_coords = points.new_ones(coords_shape)
        extended_points = torch.cat((points, additional_coords), dim=-1)
        parameters = self.layers_(code).reshape(
            (batch_size, self.num_of_planes, self.num_of_dimensions + 1)
        )

        distances = extended_points.bmm(parameters.permute(0, 2, 1))

        self._last_parameters = parameters
        return distances

    @property
    def last_predicted_parameters(self) -> torch.Tensor:
        return self._last_parameters


def create_compound_evaluator(
    use_planes: bool, shapes_per_type: int, num_dimensions: int
) -> t.Union[CompundEvaluator, PlanesEvaluator]:
    if use_planes:
        return PlanesEvaluator(shapes_per_type, num_dimensions)

    return CompundEvaluator(
        [
            CircleSphereEvaluator(shapes_per_type, num_dimensions),
            SquareCubeEvaluator(shapes_per_type, num_dimensions),
        ]
    )
