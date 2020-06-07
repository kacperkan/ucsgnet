import argparse
import json
import math
import os
import typing as t

import torch
import tqdm
from torch.utils.data import DataLoader

from ucsgnet.common import TrainingStage
from ucsgnet.ucsgnet.cad.net_cad import Net
from ucsgnet.ucsgnet.metrics import chamfer, iou


def eval_on_single_loader(
    model: Net, loader: DataLoader, prefix: str
) -> t.Dict[str, float]:
    all_predictions = []
    all_truths = []
    for batch in tqdm.tqdm(loader):
        image, points, trues, _ = batch
        if torch.cuda.is_available():
            image = image.cuda()
            points = points.cuda()
            trues = trues.cuda()
        predictions = model(image, points)
        predictions = model.binarize(predictions)
        all_predictions.append(predictions)
        all_truths.append(trues)

    predictions = torch.cat(all_predictions, dim=0)
    truths = torch.cat(all_truths, dim=0)

    num_samples = predictions.shape[0]
    per_dim_size = int(math.sqrt(predictions.shape[1]))

    predictions = (
        predictions.reshape((num_samples, per_dim_size, per_dim_size))
        .detach()
        .cpu()
        .numpy()
    )
    truths = (
        truths.reshape((num_samples, per_dim_size, per_dim_size))
        .detach()
        .cpu()
        .numpy()
    )

    chamfer_dist = chamfer(predictions, truths).mean(axis=0).item()
    iou_val = iou(predictions, truths).mean(axis=0).item()

    return {prefix + "_chamfer": chamfer_dist, prefix + "_iou": iou_val}


def evaluate(args: argparse.Namespace):
    net = Net.load_from_checkpoint(args.weights_path)
    net.build(args.data_path)
    net = net.eval()
    net.freeze()
    net.switch_mode(TrainingStage.FINE_TUNING)
    if torch.cuda.is_available():
        net = net.cuda()
    val_dataloader = net.val_dataloader()

    metrics = eval_on_single_loader(net, val_dataloader, "valid")
    metrics_path = args.out_dir
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    with open(os.path.join(metrics_path, "metrics_2d.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(json.dumps(metrics, indent=4))


def main():
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Script evaluating model on the valid dataset",
    )
    parser.add_argument(
        "--weights_path",
        required=True,
        help="Path to the *.ckpt path",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to h5 file of cad samples",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Output directory for metrics",
        required=True,
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
