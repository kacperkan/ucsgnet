import typing as t

import torch
import torch.nn as nn

from ucsgnet.common import RNN_LATENT_SIZE, TrainingStage
from ucsgnet.ucsgnet.csg_layers import RelationLayer, Scaler
from ucsgnet.ucsgnet.extractors import FeatureExtractor
from ucsgnet.ucsgnet.shape_evaluators import CompundEvaluator, PlanesEvaluator


class CSGNet(nn.Module):
    def __init__(
        self,
        extractor: FeatureExtractor,
        decoder: nn.Module,
        evaluator: t.Union[PlanesEvaluator, CompundEvaluator],
        shapes_per_type: int,
        out_shapes_per_layer: int,
        binarizing_threshold: float,
        num_csg_layers: int,
    ):
        super().__init__()
        self.encoder_ = extractor
        self.decoder_ = decoder
        self.evaluator_ = evaluator
        self.shapes_per_type = shapes_per_type
        self.out_shapes_per_layer = out_shapes_per_layer
        self.binarizing_threshold = binarizing_threshold
        self.use_planes = isinstance(self.evaluator_, PlanesEvaluator)
        self.evaluators_count = 1 if self.use_planes else len(self.evaluator_)
        self.num_output_shapes_from_evaluator = (
            self.shapes_per_type * self.evaluators_count
        )

        layers = []
        in_shapes = self.num_output_shapes_from_evaluator
        out_shapes = self.out_shapes_per_layer
        self.scaler_ = Scaler()

        num_layers = num_csg_layers
        for i in range(num_layers):
            if i == num_layers - 1:
                out_shapes = 1
            layers.append(
                RelationLayer(
                    in_shapes,
                    out_shapes,
                    self.binarizing_threshold,
                    extractor.out_features,
                )
            )
            in_shapes = out_shapes * 4 + self.num_output_shapes_from_evaluator

        self.csg_layers_: nn.ModuleList[RelationLayer] = nn.ModuleList(layers)
        self._base_mode = TrainingStage.INITIAL_TRAINING

        self.gru_encoder = nn.GRUCell(
            input_size=extractor.out_features,
            hidden_size=RNN_LATENT_SIZE,
            bias=True,
        )
        self._gru_hidden_state = nn.Parameter(
            torch.Tensor(1, RNN_LATENT_SIZE), requires_grad=True
        )
        self._retained_latent_code: t.Optional[torch.Tensor] = None
        self._retained_shape_params: t.Optional[torch.Tensor] = None

        nn.init.constant_(self._gru_hidden_state, 0.01)

    def turn_fine_tuning_mode(self):
        self.switch_mode(TrainingStage.FINE_TUNING)

    def turn_initial_training_mode(self):
        self.switch_mode(TrainingStage.INITIAL_TRAINING)

    def switch_mode(self, new_mode: TrainingStage):
        self._base_mode = new_mode
        self.scaler_.switch_mode(new_mode)
        for layer in self.csg_layers_:  # type: RelationLayer
            layer.switch_mode(new_mode)

    def clear_retained_codes_and_params(self):
        self._retained_shape_params = None
        self._retained_latent_code = None

    def forward(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        *,
        return_distances_to_base_shapes: bool = False,
        return_intermediate_output_csg: bool = False,
        return_scaled_distances_to_shapes: bool = False,
        retain_latent_code: bool = False,
        retain_shape_params: bool = False
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]:
        batch_size = images.shape[0]
        if retain_latent_code and self._retained_latent_code is not None:
            code = self._retained_latent_code
        else:
            code = self.encoder_(images)
            if retain_latent_code:
                self._retained_latent_code = code

        if retain_shape_params and self._retained_shape_params is not None:
            shape_params = self._retained_shape_params
        else:
            shape_params = self.decoder_(code)
            if retain_shape_params:
                self._retained_shape_params = shape_params

        if self.use_planes:
            base_shapes = self.evaluator_(
                shape_params, points
            )  # -> batch, num_points, num_shapes
        else:
            points = points.unsqueeze(
                dim=1
            )  # broadcasting for different of shapes

            base_shapes = self.evaluator_(
                shape_params, points
            )  # -> batch, num_shapes, num_points

            base_shapes = base_shapes.permute(
                (0, 2, 1)
            )  # -> batch, num_points, num_shapes

        scaled_shapes = 1 - self.scaler_(base_shapes)
        last_distances = scaled_shapes
        partial_distances = [last_distances]

        code = self.gru_encoder(
            code,
            self._gru_hidden_state.expand(
                [batch_size, self._gru_hidden_state.shape[1]]
            ),
        )

        for index, csg_layer in enumerate(
            self.csg_layers_
        ):  # type: (int, RelationLayer)
            if index > 0:
                last_distances = torch.cat(
                    (last_distances, scaled_shapes), dim=-1
                )
            last_distances = csg_layer(last_distances, code)
            partial_distances.append(last_distances)

            code = self.gru_encoder(
                csg_layer.emit_parameters(batch_size), code
            )

        last_distances = last_distances[..., 0]  # taking union
        distances = last_distances.clamp(0, 1)
        outputs = [distances]
        if return_distances_to_base_shapes:
            outputs.append(base_shapes)
        if return_intermediate_output_csg:
            outputs.append(partial_distances)
        if return_scaled_distances_to_shapes:
            outputs.append(scaled_shapes)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def get_latent_codes_for_each_layer(
        self, images: torch.Tensor, points: torch.Tensor
    ) -> t.Dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        code = self.encoder_(images)
        shape_params = self.decoder_(code)

        if self.use_planes:
            base_shapes = self.evaluator_(
                shape_params, points
            )  # -> batch, num_points, num_shapes
        else:
            points = points.unsqueeze(
                dim=1
            )  # broadcasting for different of shapes

            base_shapes = self.evaluator_(
                shape_params, points
            )  # -> batch, num_shapes, num_points

            base_shapes = base_shapes.permute(
                (0, 2, 1)
            )  # -> batch, num_points, num_shapes

        scaled_shapes = 1 - self.scaler_(base_shapes)
        last_distances = scaled_shapes
        partial_distances = [last_distances]

        code = self.gru_encoder(
            code,
            self._gru_hidden_state.expand(
                [batch_size, self._gru_hidden_state.shape[1]]
            ),
        )
        codes = {"base": code}
        emits = {}

        csg_layer: RelationLayer
        for index, csg_layer in enumerate(self.csg_layers_):
            if index > 0:
                last_distances = torch.cat(
                    (last_distances, scaled_shapes), dim=-1
                )
            last_distances = csg_layer(last_distances, code)
            partial_distances.append(last_distances)

            emitted_parameters = csg_layer.emit_parameters(batch_size)
            code = self.gru_encoder(emitted_parameters, code)
            codes["layer_{}".format(index)] = code
            emits["emits_{}".format(index)] = emitted_parameters

        return codes, emits
