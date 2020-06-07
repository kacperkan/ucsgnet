import argparse
import json
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer

from ucsgnet.callbacks import ModelCheckpoint
from ucsgnet.loggers import TensorBoardLogger
from ucsgnet.ucsgnet.net_2d import Net

MAX_NB_EPOCHS = 251


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training code for a CSG-Net.", add_help=False
    )
    parser.add_argument(
        "--train",
        dest="train_split_config",
        type=str,
        help="Path to training split of samples with one of generators",
        required=True,
    )
    parser.add_argument(
        "--valid",
        dest="valid_split_config",
        type=str,
        help="Path to training split of samples generated with of generators",
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
    parser = Net.add_model_specific_args(parser)
    return parser.parse_args()


def training(
    model: pl.LightningModule, experiment_name: str, args: argparse.Namespace
):
    model_saving_path = os.path.join("models", experiment_name, "initial")
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
        period=10,
    )

    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        distributed_backend="dp",
        default_save_path=model_saving_path,
        logger=logger,
        max_epochs=MAX_NB_EPOCHS,
        early_stop_callback=None,
        checkpoint_callback=checkpointer,
        progress_bar_refresh_rate=1,
    )
    # fitting
    trainer.fit(model)


def train(args: argparse.Namespace):
    model = Net(args)
    model.build(
        args.train_split_config,
        args.valid_split_config,
        args.processed_data_path,
    )

    if args.checkpoint_path and len(args.checkpoint_path) > 0:
        print(f"Loading pretrained model from: {args.checkpoint_path}")
        model = model.load_from_checkpoint(args.checkpoint_path)
    training(model, args.experiment_name + "_main", args)


if __name__ == "__main__":
    train(get_args())
