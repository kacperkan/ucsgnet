import argparse
import typing as t
from collections import OrderedDict, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from ucsgnet.common import THREADS, TrainingStage
from ucsgnet.dataset import SimpleDataset, get_simple_2d_transforms
from ucsgnet.ucsgnet.csg_layers import RelationLayer
from ucsgnet.ucsgnet.extractors import Decoder, Extractor2D
from ucsgnet.ucsgnet.losses import get_composite_loss
from ucsgnet.ucsgnet.metrics import mse
from ucsgnet.ucsgnet.model import CSGNet
from ucsgnet.ucsgnet.shape_evaluators import create_compound_evaluator
from ucsgnet.utils import get_simple_dataset_paths_from_config


class Net(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams = hparams
        self.net = CSGNet(
            Extractor2D(),
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

        self.train_split_config_: t.Optional[str] = None
        self.valid_split_config_: t.Optional[str] = None
        self.data_path_: t.Optional[str] = None

        self.__optimizers: t.Optional[t.Sequence[Optimizer]] = None
        self._base_mode = TrainingStage.INITIAL_TRAINING

        (
            trainable_params_count,
            non_trainable_params_count,
        ) = self.num_of_parameters

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
        self, train_split_config: str, valid_split_config: str, data_path: str
    ):
        self.train_split_config_ = train_split_config
        self.valid_split_config_ = valid_split_config
        self.data_path_ = data_path

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

        if self.hparams.use_planes:
            self.logger.log_histogram(
                f"planes_params",
                self.net.evaluator_.last_predicted_parameters.reshape((-1,)),
                self.global_step,
            )
        else:
            for j, (name, tensor) in enumerate(
                self.net.evaluator_.get_all_last_predicted_parameters_of_shapes()
            ):
                self.logger.log_histogram(
                    f"evaluate_{name}_0_{j}",
                    tensor.reshape((-1,)),
                    self.global_step,
                )

            translation_vectors = (
                self.net.evaluator_.get_all_translation_vectors()
            )
            self.logger.log_histogram(
                f"translation_x_0",
                translation_vectors[..., 0].reshape((-1,)),
                self.global_step,
            )
            self.logger.log_histogram(
                f"translation_y_0",
                translation_vectors[..., 1].reshape((-1,)),
                self.global_step,
            )
            if self.hparams.num_dimensions == 3:
                self.logger.log_histogram(
                    f"translation_z_0",
                    translation_vectors[..., 2].reshape((-1,)),
                    self.global_step,
                )

        for i, layer in enumerate(self.net.csg_layers_):  # type: RelationLayer
            self.logger.log_histogram(
                f"rel_layer_dist_temp_{i}/vals",
                layer.temperature_.reshape((-1,)),
                self.global_step,
            )

        self.logger.log_histogram(
            "scaler/m", self.net.scaler_.m.reshape((-1,)), self.global_step
        )

        tqdm_dict = {
            "train_loss": total_loss,
            "train_predictions_avg": predictions.mean(),
            **{
                "train_" + key: value
                for key, value in partial_losses_dict.items()
            },
            **{
                f"lr_{i}": torch.tensor(
                    optimizer.param_groups[0]["lr"], dtype=torch.float
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
                    optimizer.param_groups[0]["lr"], dtype=torch.float
                )
                for i, optimizer in enumerate(self.__optimizers)
            },
        }

        output = OrderedDict(
            {"loss": total_loss, "progress_bar": tqdm_dict, "log": logger_dict}
        )

        return output

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
            "valid_" + key: value.item() for key, value in means.items()
        }
        result = {
            "valid_loss": means["loss"],
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

    def _dataloader_simple(
        self, training: bool, split_path: str
    ) -> DataLoader:
        batch_size = self.hparams.batch_size
        renders = get_simple_dataset_paths_from_config(
            self.data_path_, split_path
        )
        transforms = get_simple_2d_transforms()

        loader = DataLoader(
            dataset=SimpleDataset(
                renders,
                None,
                self.hparams.points_per_sample_in_batch,
                transforms,
            ),
            batch_size=batch_size,
            shuffle=training,
            drop_last=training,
            num_workers=THREADS,
        )
        return loader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader_simple(True, self.train_split_config_)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader_simple(False, self.valid_split_config_)

    def __next_elem_from_loader(
        self, loader: DataLoader
    ) -> t.Tuple[torch.Tensor, ...]:
        images, coords, distances, _ = next(iter(loader))
        if self.on_gpu:
            images = images.cuda()
            coords = coords.cuda()
            distances = distances.cuda()
        return images, coords, distances

    def on_epoch_end(self):
        val_loader = self.val_dataloader()
        (images, coords, distances) = self.__next_elem_from_loader(val_loader)

        images = images[:16]
        coords = coords[:16]
        distances = distances[:16]

        b, c, h, w = images.shape
        final_predictions = self(images, coords).reshape((b, c, h, w))

        input_images = torchvision.utils.make_grid(images, normalize=True)
        gt = torchvision.utils.make_grid(
            distances.view_as(images), normalize=True
        )
        pred_grid = torchvision.utils.make_grid(
            final_predictions, normalize=True
        )

        binarized_pred_grid = torchvision.utils.make_grid(
            self.binarize(final_predictions), normalize=True
        )

        self.logger.experiment.add_image(
            "input_images", input_images, self.current_epoch
        )
        self.logger.experiment.add_image("gt", gt, self.current_epoch)
        self.logger.experiment.add_image(
            "reconstruction", pred_grid, self.current_epoch
        )
        self.logger.experiment.add_image(
            "binarized_pred", binarized_pred_grid, self.current_epoch
        )

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
            default=8,
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
            "--points",
            type=int,
            dest="points_per_sample_in_batch",
            help="Number of SDF samples per sample in a batch.",
            default=1024,
        )
        parser.add_argument(
            "--sampling_count",
            type=int,
            help="Num of sampling to perform in relational layers",
            default=5,
        )
        parser.add_argument(
            "--out_shapes_per_layer",
            type=int,
            help="Number of output shapes per layer",
            default=2,
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
            "--num_csg_layers",
            type=int,
            help="Number of relation prediction layers",
            default=2,
        )

        return parser
