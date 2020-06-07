import abc
import logging
import multiprocessing as mp
import random
import typing as t
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

RENDER_SERVER = "http://localhost:8000/render"

SEED = 1337
THREADS = mp.cpu_count()

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ConfigType = t.Dict[str, t.Dict[str, t.Any]]

logging.basicConfig(level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.FileHandler("logs/log.log", mode="a")
    handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(stream_handler)

    return logger


FLOAT_EPS = torch.finfo(torch.float32).eps
RNN_LATENT_SIZE = 256


class Evaluation3D:
    CATEGORY_LIST = [
        "02691156_airplane",
        "02828884_bench",
        "02933112_cabinet",
        "02958343_car",
        "03001627_chair",
        "03211117_display",
        "03636649_lamp",
        "03691459_speaker",
        "04090263_rifle",
        "04256520_couch",
        "04379243_table",
        "04401088_phone",
        "04530566_vessel",
    ]
    CATEGORY_IDS = [name[:8] for name in CATEGORY_LIST]
    CATEGORY_NAMES = [name.split("_")[1] for name in CATEGORY_LIST]
    NUM_POINTS = 4096

    CATEGORY_COUNTS = [
        809,
        364,
        315,
        1500,
        1356,
        219,
        464,
        324,
        475,
        635,
        1702,
        211,
        388,
    ]


class TrainingStage(Enum):
    INITIAL_TRAINING = 0
    FINE_TUNING = 1
    LATENT_CODE_OPTIMIZATION = 2


class FeatureExtractor(abc.ABC, nn.Module):
    @property
    @abc.abstractmethod
    def out_features(self):
        pass
