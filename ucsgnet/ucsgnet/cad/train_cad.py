import argparse

from ucsgnet.ucsgnet.cad.net_cad import Net
from ucsgnet.ucsgnet.train_2d import training

MAX_NB_EPOCHS = 170


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training code for a CSG-Net.", add_help=False
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        help="Path to h5 file containing data",
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


def train(args: argparse.Namespace):
    model = Net(args)
    model.build(args.data_path)

    if args.checkpoint_path and len(args.checkpoint_path) > 0:
        print(f"Loading pretrained model from: {args.checkpoint_path}")
        model = model.load_from_checkpoint(args.checkpoint_path)
    training(model, args.experiment_name + "_main", args)


if __name__ == "__main__":
    train(get_args())
