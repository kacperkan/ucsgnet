import argparse
import json
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from ucsgnet.callbacks import ModelCheckpoint
from ucsgnet.loggers import TensorBoardLogger
from ucsgnet.ucsgnet.net_3d import Net

MAX_NB_EPOCHS = 501


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training code for a CSG-Net.", add_help=False
    )
    parser.add_argument(
        "--train",
        dest="train_file",
        type=str,
        help="Path to training HDF5 file with the training data",
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
        "--processed",
        dest="processed_data_path",
        type=str,
        help="Base folder of processed data",
        required=True,
    )
    parser.add_argument(
        "--pretrained_path",
        dest="checkpoint_path",
        type=str,
        help=(
            "If provided, then it assumes pretraining and continuation of "
            "training"
        ),
        default="",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment",
        default="test",
    )
    parser.add_argument(
        "--full_resolution_only",
        action="store_true",
        default=False,
        help="Whether use the resolution 64 only instead of whole training",
    )
    parser = Net.add_model_specific_args(parser)
    return parser.parse_args()


def training(
    model: Net,
    experiment_name: str,
    args: argparse.Namespace,
    data_size: int,
    train_file: str,
    valid_file: str,
    processed_data_path: str,
    is_fine_tuning: bool,
):
    model_saving_path = os.path.join("models", experiment_name, "initial")
    model.build(train_file, valid_file, processed_data_path, data_size)
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path, exist_ok=True)

    with open(os.path.join(model_saving_path, "params.json"), "w") as f:
        json.dump(vars(args), f)
    logger = TensorBoardLogger(
        os.path.join(model_saving_path, "logs"), log_train_every_n_step=200
    )

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(model_saving_path, "ckpts", "model.ckpt"),
        monitor="valid_loss",
        period=2,
    )

    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        distributed_backend="dp",
        default_save_path=model_saving_path,
        logger=logger,
        max_epochs=(
            MAX_NB_EPOCHS if not is_fine_tuning else MAX_NB_FINE_TUNING_EPOCHS
        ),
        early_stop_callback=EarlyStopping(
            monitor="valid_loss", patience=40, verbose=True
        ),
        checkpoint_callback=checkpointer,
        progress_bar_refresh_rate=1,
    )
    # fitting
    trainer.fit(model)


def train(args: argparse.Namespace):
    model = Net(args)

    if args.checkpoint_path and len(args.checkpoint_path) > 0:
        print(f"Loading pretrained model from: {args.checkpoint_path}")
        model = model.load_from_checkpoint(args.checkpoint_path)

    if not args.full_resolution_only:
        training(
            model,
            args.experiment_name + "_16",
            args,
            16,
            args.train_file,
            args.valid_file,
            args.processed_data_path,
            False,
        )
        training(
            model,
            args.experiment_name + "_32",
            args,
            32,
            args.train_file,
            args.valid_file,
            args.processed_data_path,
            False,
        )
    training(
        model,
        args.experiment_name + "_64",
        args,
        64,
        args.train_file,
        args.valid_file,
        args.processed_data_path,
        False,
    )


if __name__ == "__main__":
    train(get_args())
