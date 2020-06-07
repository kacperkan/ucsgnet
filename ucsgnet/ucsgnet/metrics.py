import cv2
import numpy as np
import torch
import torch.nn.functional as F


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_pred, y_true, reduction="mean")


def iou(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    iou_val = (y_true * y_pred).sum(axis=(1, 2)) / (
        (y_true + y_pred).clip(0, 1).sum(axis=(1, 2)) + 1.0
    )
    return iou_val


def chamfer(images1: np.ndarray, images2: np.ndarray) -> np.ndarray:
    """Taken from:https://git.io/JfIpC"""
    # Convert in the opencv data format
    images1 = (images1 * 255).astype(np.uint8)
    images2 = (images2 * 255).astype(np.uint8)
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size ** 2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (
            (summ1[i] == 0)
            or (summ2[i] == 0)
            or (summ1[i] == filled_value)
            or (summ2[i] == filled_value)
        ):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size ** 2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3
        )

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size ** 2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3
        )
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (np.sum(E2, (1, 2)) + 1) + np.sum(
        D2 * E1, (1, 2)
    ) / (np.sum(E1, (1, 2)) + 1)
    # TODO make it simpler
    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 16
    return distances
