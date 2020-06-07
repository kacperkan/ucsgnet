import typing as t

import torch
import torch.nn as nn

from ucsgnet.common import FLOAT_EPS, RNN_LATENT_SIZE, TrainingStage


def parametrized_clipping_function(
    x: torch.Tensor,
    m: torch.Tensor,
    min_value: float = 0,
    max_value: float = 1,
) -> torch.Tensor:
    return (x / m.clamp_min(FLOAT_EPS)).clamp(min_value, max_value)


def gumbel_softmax(
    probabilities: torch.Tensor,
    temperature: t.Union[torch.Tensor, float],
    dim: int,
) -> torch.Tensor:
    samples = torch.rand_like(probabilities).clamp_min(FLOAT_EPS)
    samples = -torch.log(-torch.log(samples))
    if isinstance(temperature, torch.Tensor):
        temperature = temperature.clamp_min(1e-3)
    return (
        ((probabilities + FLOAT_EPS).log() + samples) / temperature
    ).softmax(dim=dim)


class Scaler(nn.Module):
    def __init__(self, min_value: float = 0, max_value: float = 1):
        super().__init__()
        self.m = nn.Parameter(torch.Tensor(1, 1, 1), requires_grad=True)
        nn.init.constant_(self.m, 1)

        self.min_value = min_value
        self.max_value = max_value

        self._base_mode = TrainingStage.INITIAL_TRAINING

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inp -> batch size, num_points, num_shapes
        return parametrized_clipping_function(
            x, self.m, self.min_value, self.max_value
        )

    def switch_mode(self, new_mode: TrainingStage):
        self._base_mode = new_mode


class RelationLayer(nn.Module):
    def __init__(
        self,
        num_in_shapes: int,
        num_out_shapes: int,
        binarizing_threshold: float,
        latent_size: int,
    ):
        super().__init__()

        self.num_in_shapes = num_in_shapes
        self.num_out_shapes = num_out_shapes
        self.binarizing_threshold = binarizing_threshold
        self.latent_size = latent_size

        self.composition_vector_1_ = nn.Parameter(
            torch.Tensor(RNN_LATENT_SIZE, num_in_shapes, num_out_shapes),
            requires_grad=True,
        )
        self.composition_vector_2_ = nn.Parameter(
            torch.Tensor(RNN_LATENT_SIZE, num_in_shapes, num_out_shapes),
            requires_grad=True,
        )

        nn.init.normal_(self.composition_vector_1_, std=0.1)
        nn.init.normal_(self.composition_vector_2_, std=0.1)

        self.temperature_ = nn.Parameter(torch.Tensor(1), requires_grad=True)

        nn.init.constant_(self.temperature_, 2)

        # these components are used for losses.py
        self._base_mode = TrainingStage.INITIAL_TRAINING
        self.last_mask: t.Optional[torch.Tensor] = None
        self.last_sampled_shapes: t.Optional[torch.Tensor] = None
        self.operations_before_clamping: t.Optional[torch.Tensor] = None
        self.operations_after_clamping: t.Optional[torch.Tensor] = None

        self.left_op_mask: t.Optional[torch.Tensor] = None
        self.right_op_mask: t.Optional[torch.Tensor] = None

        self.switch_mode(TrainingStage.INITIAL_TRAINING)

        self.parameter_encoder = nn.Sequential(
            nn.Linear(num_in_shapes * 2 * num_out_shapes, latent_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(latent_size, latent_size),
        )

    def forward(
        self, x: torch.Tensor, latent_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        x -> batch, num_points, num_shapes
        """
        batch_size = x.shape[0]
        mask_l = (
            (
                latent_vector[:, :, None, None]
                * self.composition_vector_1_[None]
            )
            .sum(dim=1)
            .softmax(dim=-2)
        )
        mask_r = (
            (
                latent_vector[:, :, None, None]
                * self.composition_vector_2_[None]
            )
            .sum(dim=1)
            .softmax(dim=-2)
        )
        mask = torch.stack([mask_l, mask_r], dim=1).unsqueeze(dim=2)

        self.left_op_mask = mask_l
        self.right_op_mask = mask_r

        # according to the https://arxiv.org/abs/1907.11065
        self.last_mask = mask
        mask = gumbel_softmax(
            mask, self.temperature_.clamp(FLOAT_EPS, 2), dim=-2
        )
        self.last_sampled_shapes = mask
        per_head_mask = mask.split(split_size=1, dim=1)
        x = x.unsqueeze(-1)

        partial_res = []
        for head in per_head_mask:  # type: torch.Tensor
            head = head.squeeze(dim=1)
            partial_res.append((x * head).sum(dim=2))
        x = torch.stack(partial_res, dim=2)
        x = torch.cat(
            [
                x[:, :, 0] + x[:, :, 1],  # union
                x[:, :, 0] + x[:, :, 1] - 1,  # intersection
                x[:, :, 0] - x[:, :, 1],  # diff
                x[:, :, 1] - x[:, :, 0],  # diff
            ],
            dim=-1,
        )
        self.operations_before_clamping = x
        x = x.clamp(0, 1)
        self.operations_after_clamping = x
        return x

    def emit_parameters(self, batch_size: int) -> torch.Tensor:
        attention_weights = self.last_sampled_shapes.view(
            (batch_size, 2 * self.num_in_shapes * self.num_out_shapes)
        )
        return self.parameter_encoder(attention_weights)

    def switch_mode(self, new_mode: TrainingStage):
        self._base_mode = new_mode
        self.temperature_.requires_grad = (
            new_mode == TrainingStage.INITIAL_TRAINING
        )
