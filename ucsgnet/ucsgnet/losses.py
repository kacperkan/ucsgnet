import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

from ucsgnet.common import FLOAT_EPS, TrainingStage
from ucsgnet.ucsgnet.csg_layers import RelationLayer, Scaler
from ucsgnet.ucsgnet.shape_evaluators import (
    CompundEvaluator,
    PlanesEvaluator,
    ShapeEvaluator,
)


def get_positive_parameter_loss(evaluator: CompundEvaluator) -> float:
    loss = 0
    for part in evaluator:  # type: ShapeEvaluator
        loss += (
            (-part.last_parameters).clamp_min(0).sum(dim=(1, 2)).mean(dim=0)
        )
    return loss


def get_recon_loss(preds: torch.Tensor, trues: torch.Tensor) -> torch.Tensor:
    # input -> batch_size, num_points
    return F.mse_loss(preds, trues)


def get_scaling_loss(scaler: Scaler) -> torch.Tensor:
    return (scaler.m.abs().clamp_min(FLOAT_EPS) - FLOAT_EPS).sum()


def get_temperature_decreasing_loss(
    csg_layers: nn.ModuleList,
) -> t.Union[float, torch.Tensor]:
    loss = 0
    for layer in csg_layers:  # type: RelationLayer
        loss += (
            layer.temperature_.abs().clamp_min(FLOAT_EPS) - FLOAT_EPS
        ).sum()
    return loss


def get_translation_loss(
    shape_evaluator: CompundEvaluator,
) -> t.Union[torch.Tensor, float]:
    loss = 0
    for part in shape_evaluator.parts:  # type: ShapeEvaluator
        loss += (
            (part.translation_vector_prediction.norm(dim=-1) - 0.5).relu() ** 2
        ).mean()
    return loss


def get_composite_loss(
    preds: torch.Tensor,
    trues: torch.Tensor,
    max_volumes: torch.Tensor,
    points: torch.Tensor,
    intermediate_results: torch.Tensor,
    csg_layers: nn.ModuleList,
    shape_evaluator: t.Union[CompundEvaluator, PlanesEvaluator],
    stage: TrainingStage,
    uses_planes: bool,
    step: int,
    scaler: Scaler,
) -> t.Tuple[
    t.Union[torch.Tensor, float], t.Dict[str, t.Union[torch.Tensor, float]]
]:
    total_loss = 0

    recon_loss = get_recon_loss(preds, trues)
    out = {"rec": recon_loss}
    total_loss += recon_loss

    if scaler.m.item() <= 0.05:
        temp = 1e-1 * get_temperature_decreasing_loss(csg_layers)
        out["temp"] = temp
        total_loss += temp

    scaling_loss = 1e-1 * get_scaling_loss(scaler)
    out["scale"] = scaling_loss
    total_loss += scaling_loss

    trans_loss = 1e-1 * get_translation_loss(shape_evaluator)
    out["trans"] = trans_loss
    total_loss += trans_loss

    if not uses_planes:
        pos_param_loss = get_positive_parameter_loss(shape_evaluator)
        out["pos_param"] = pos_param_loss
        total_loss += pos_param_loss

    return total_loss, out
