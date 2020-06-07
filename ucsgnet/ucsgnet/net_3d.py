import argparse
import os
import typing as t
from collections import OrderedDict, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from ucsgnet.common import TrainingStage
from ucsgnet.dataset import HdfsDataset3D
from ucsgnet.ucsgnet.csg_layers import RelationLayer
from ucsgnet.ucsgnet.extractors import Decoder, Extractor3D
from ucsgnet.ucsgnet.losses import get_composite_loss
from ucsgnet.ucsgnet.metrics import mse
from ucsgnet.ucsgnet.model import CSGNet
from ucsgnet.ucsgnet.shape_evaluators import create_compound_evaluator


class Net(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.net = CSGNet(
            Extractor3D(),
            Decoder(),
            create_compound_evaluator(
                self.hparams.use_planes,
                self.hparams.shapes_per_type,
                self.hparams.num_dimensions,
            ),
            self.hparams.shapes_per_type,
            self.hparams.out_shapes_per_layer,
            self.hparams.weight_binarizing_threshold,
            self.hparams.num_csg_layers,
        )

        self.train_file_: t.Optional[str] = None
        self.valid_file_: t.Optional[str] = None
        self.data_path_: t.Optional[str] = None
        self.current_data_size_: t.Optional[int] = None

        self.__optimizers: t.Optional[t.Sequence[Optimizer]] = None
        self._base_mode = TrainingStage.INITIAL_TRAINING

        (
            trainable_params_count,
            non_trainable_params_count,
        ) = self.num_of_parameters

        self.original_csg_layers_temperatures_: t.Dict[int, torch.Tensor] = {}

        print("Num of trainable params: {}".format(trainable_params_count))
        print(
            "Num of not trainable params: {}".format(
                non_trainable_params_count
            )
        )

    def turn_fine_tuning_mode(self):
        self.switch_mode(TrainingStage.FINE_TUNING)

    def turn_initial_training_mode(self):
        self.switch_mode(TrainingStage.INITIAL_TRAINING)

    def switch_mode(self, new_mode: TrainingStage):
        self._base_mode = new_mode
        self.net.switch_mode(new_mode)

    def build(
        self, train_file: str, valid_file: str, data_path: str, data_size: int
    ):
        self.train_file_ = train_file
        self.valid_file_ = valid_file
        self.data_path_ = data_path
        self.current_data_size_ = data_size

    @property
    def num_of_parameters(self) -> t.Tuple[int, int]:
        total_trainable_params = 0
        total_nontrainable_params = 0

        for param in self.parameters(recurse=True):
            if param.requires_grad:
                total_trainable_params += np.prod(param.shape)
            else:
                total_nontrainable_params += np.prod(param.shape)
        return total_trainable_params, total_nontrainable_params

    def on_train_start(self):
        if self._base_mode == TrainingStage.FINE_TUNING:
            for i, layer in enumerate(
                self.net.csg_layers_
            ):  # type: (int, RelationLayer)
                self.original_csg_layers_temperatures_[
                    i
                ] = layer.temperature_.data

    def forward(
        self,
        images: torch.Tensor,
        points: torch.Tensor,
        *,
        return_distances_to_base_shapes: bool = False,
        return_intermediate_output_csg: bool = False,
        return_scaled_distances_to_shapes: bool = False,
        retain_latent_code: bool = False,
        retain_shape_params: bool = False,
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, ...]]:
        return self.net(
            images,
            points,
            return_distances_to_base_shapes=return_distances_to_base_shapes,
            return_intermediate_output_csg=return_intermediate_output_csg,
            return_scaled_distances_to_shapes=return_scaled_distances_to_shapes,
            retain_shape_params=retain_shape_params,
            retain_latent_code=retain_latent_code,
        )

    def training_step(
        self,
        batch: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> t.Dict[str, t.Any]:
        self.logger.train()
        image, points, trues, bounding_volume = batch
        predictions, distances_to_base_shapes, intermediate_results = self(
            image,
            points,
            return_distances_to_base_shapes=True,
            return_intermediate_output_csg=True,
        )
        predictions = predictions.squeeze(dim=-1)
        trues = trues.squeeze(dim=-1)
        total_loss, partial_losses_dict = get_composite_loss(
            predictions,
            trues,
            bounding_volume,
            points,
            intermediate_results,
            self.net.csg_layers_,
            self.net.evaluator_,
            self._base_mode,
            self.net.use_planes,
            self.global_step,
            self.net.scaler_,
        )

        tqdm_dict = {
            "train_loss": total_loss,
            "train_predictions_avg": predictions.mean(),
            **{
                "train_" + key: value.mean()
                for key, value in partial_losses_dict.items()
            },
            **{
                f"lr_{i}": torch.tensor(
                    optimizer.param_groups[0]["lr"],
                    dtype=torch.float,
                    device=predictions.device,
                )
                for i, optimizer in enumerate(self.__optimizers)
            },
        }

        logger_dict = {
            "loss": total_loss,
            "predictions_avg": predictions.mean(),
            **partial_losses_dict,
            **{
                f"lr_{i}": torch.tensor(
                    optimizer.param_groups[0]["lr"],
                    dtype=torch.float,
                    device=predictions.device,
                )
                for i, optimizer in enumerate(self.__optimizers)
            },
        }

        output = OrderedDict(
            {"loss": total_loss, "progress_bar": tqdm_dict, "log": logger_dict}
        )

        self._log_train_step_to_tensorboard()
        return output

    def _log_train_step_to_tensorboard(self):
        if self.hparams.use_planes:
            self.logger.log_histogram(
                f"planes_params",
                self.net.evaluator_.last_predicted_parameters,
                self.global_step,
            )
        else:
            for j, (name, tensor) in enumerate(
                self.net.evaluator_.get_all_last_predicted_parameters_of_shapes()
            ):
                self.logger.log_histogram(
                    f"evaluate_{name}_0_{j}", tensor, self.global_step
                )

            translation_vectors = (
                self.net.evaluator_.get_all_translation_vectors()
            )
            self.logger.log_histogram(
                f"translation_x_0",
                translation_vectors[..., 0],
                self.global_step,
            )
            self.logger.log_histogram(
                f"translation_y_0",
                translation_vectors[..., 1],
                self.global_step,
            )
            if self.hparams.num_dimensions == 3:
                self.logger.log_histogram(
                    f"translation_z_0",
                    translation_vectors[..., 2],
                    self.global_step,
                )

        for i, layer in enumerate(self.net.csg_layers_):  # type: RelationLayer
            self.logger.log_histogram(
                f"rel_layer_dist_temp_{i}/vals",
                layer.temperature_,
                self.global_step,
            )

        self.logger.log_histogram(
            "scaler/m", self.net.scaler_.m, self.global_step
        )

    def compute_latent_code(self, input_voxels: torch.Tensor) -> torch.Tensor:
        return self.net.encoder_(input_voxels)

    def training_step_end(
        self, aggregate_data: t.Dict[str, t.Any]
    ) -> t.Dict[str, t.Any]:
        return {
            "loss": aggregate_data["loss"].mean(),
            "progress_bar": {
                x: y.mean() for x, y in aggregate_data["progress_bar"].items()
            },
            "log": {x: y.mean() for x, y in aggregate_data["log"].items()},
        }

    def validation_step(
        self,
        batch: t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> t.Dict[str, t.Any]:
        image, points, trues, bounding_volume = batch
        predictions, distances_to_base_shapes, intermediate_results = self(
            image,
            points,
            return_distances_to_base_shapes=True,
            return_intermediate_output_csg=True,
        )
        predictions = predictions.squeeze(dim=-1)
        trues = trues.squeeze(dim=-1)
        total_loss, partial_losses_dict = get_composite_loss(
            predictions,
            trues,
            bounding_volume,
            points,
            intermediate_results,
            self.net.csg_layers_,
            self.net.evaluator_,
            self._base_mode,
            self.net.use_planes,
            self.global_step,
            self.net.scaler_,
        )

        logger_dict = {
            "loss": total_loss,
            **partial_losses_dict,
            "mse": mse(self.binarize(predictions), trues),
        }

        output = OrderedDict({"loss": total_loss, "log": logger_dict})

        return output

    def validation_step_end(
        self, aggregate_data: t.Dict[str, t.Any]
    ) -> t.Dict[str, t.Any]:
        return {
            "loss": aggregate_data["loss"].mean(),
            "log": {x: y.mean() for x, y in aggregate_data["log"].items()},
        }

    def validation_end(
        self, outputs: t.List[t.Dict[str, t.Any]]
    ) -> t.Dict[str, t.Any]:
        self.logger.valid()
        means = defaultdict(int)
        for output in outputs:
            for key, value in output["log"].items():
                means[key] += value
        means = {key: value / len(outputs) for key, value in means.items()}
        logger_dict = means
        tqdm_dict = {
            "valid_" + key: value.mean().item() for key, value in means.items()
        }
        result = {
            "valid_loss": means["loss"].mean().item(),
            "progress_bar": tqdm_dict,
            "log": logger_dict,
        }
        return result

    def configure_optimizers(
        self,
    ) -> t.Tuple[t.Sequence[Optimizer], t.Sequence[optim.lr_scheduler.StepLR]]:
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )

        self.__optimizers = [optimizer]
        return [optimizer], []

    def _dataloader(self, training: bool) -> DataLoader:
        batch_size = self.hparams.batch_size
        a_file = self.train_file_ if training else self.valid_file_
        points_to_sample = 16 * 16 * 16
        if self.current_data_size_ == 64:
            points_to_sample *= 4
        loader = DataLoader(
            dataset=HdfsDataset3D(
                os.path.join(self.data_path_, a_file),
                points_to_sample,
                self.hparams.seed,
                size=self.current_data_size_,
            ),
            batch_size=batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=0,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(False)

    @classmethod
    def binarize(cls, predictions: torch.Tensor) -> torch.Tensor:
        return (predictions >= 0.5).float()

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser])

        parser.add_argument(
            "--num_dimensions",
            help="Number of dimensions to be evaulated on",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--shapes_per_type",
            help=(
                "Number of shapes per type, ex. 64 will create 64 squares and "
                "64 circles"
            ),
            type=int,
            default=32,
        )
        parser.add_argument(
            "--lr",
            help="Learning rate of the optimizer",
            type=float,
            default=1e-3,
        )

        parser.add_argument(
            "--beta1",
            help="Beta_1 parameter of the Adam optimizer",
            type=float,
            default=0.5,
        )

        parser.add_argument(
            "--beta2",
            help="Beta_2 parameter of the Adam optimizer",
            type=float,
            default=0.99,
        )
        parser.add_argument(
            "--batch_size", help="Batch size", type=int, default=16
        )
        parser.add_argument(
            "--out_shapes_per_layer",
            type=int,
            help="Number of output shapes per layer",
            default=16,
        )
        parser.add_argument(
            "--weight_binarizing_threshold",
            type=float,
            help=(
                "Thresholding value for weights. If weight > `threshold` "
                "then it is set to 1. If -`threshold` < weight <= "
                "`threshold then set 0 and to -1 otherwise."
            ),
            default=0.1,
        )
        parser.add_argument(
            "--use_planes",
            action="store_true",
            help=(
                "Whether use normal shapes (circles, squares etc.) or "
                "planes that are combined later. Note, that for planes, "
                "it is recommended to set `shapes_per_type` much higher"
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Seed for RNG to sample points to predict distance",
        )
        parser.add_argument(
            "--num_csg_layers",
            type=int,
            help="Number of relation prediction layers",
            default=3,
        )

        return parser
