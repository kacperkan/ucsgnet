import typing as t

import torch
import torch.nn as nn

from ucsgnet.common import FeatureExtractor


class Conv2DWithActivation(nn.Sequential):
    def __init__(
        self, in_filters: int, out_filters: int, padding: str = "same"
    ):
        super().__init__()

        padding = 2 if padding == "same" else 0
        self.add_module(
            "conv",
            nn.Conv2d(
                in_filters, out_filters, 4, 2, padding=padding, bias=True
            ),
        )
        self.add_module("lrelu", nn.LeakyReLU(0.01))


class Conv2DWithoutActivation(nn.Sequential):
    def __init__(
        self, in_filters: int, out_filters: int, padding: str = "same"
    ):
        super().__init__()

        padding = 2 if padding == "same" else 0
        self.add_module(
            "conv",
            nn.Conv2d(
                in_filters, out_filters, 4, 2, padding=padding, bias=True
            ),
        )


class Conv2DWithoutBias(nn.Sequential):
    def __init__(self, in_filters, out_filters: int, padding: str = "same"):
        super().__init__()
        padding = 2 if padding == "same" else 0
        self.add_module(
            "conv",
            nn.Conv2d(
                in_filters, out_filters, 4, 2, padding=padding, bias=False
            ),
        )


class Conv3DWithActivation(nn.Sequential):
    def __init__(self, in_filters, out_filters: int, padding: str = "same"):
        super().__init__()
        padding = 2 if padding == "same" else 0
        self.add_module(
            "conv",
            nn.Conv3d(
                in_filters, out_filters, 4, 2, padding=padding, bias=False
            ),
        )
        self.add_module("lrelu", nn.LeakyReLU(0.01))


class ConvTransposed3DWithActivation(nn.Sequential):
    def __init__(self, in_filters, out_filters: int, padding: str = "same"):
        super().__init__()
        padding = 2 if padding == "same" else 0
        self.add_module(
            "conv_transposed",
            nn.ConvTranspose2d(
                in_filters, out_filters, 4, 2, padding=padding, bias=False
            ),
        )
        self.add_module("lrelu", nn.LeakyReLU(0.01))


class LinearWithActivation(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.add_module(
            "linear", nn.Linear(in_features, out_features, bias=True)
        )
        self.add_module("lrelu", nn.LeakyReLU(0.01))


class Flatten(nn.Module):
    def __init__(self, shape: t.Sequence[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = (
            x.shape[0] if self.shape[0] == -1 else self.shape[0],
        ) + tuple(self.shape[1:])
        return x.view(out_shape)


class Extractor2D(FeatureExtractor):
    def __init__(self):
        super().__init__()

        self._ef_features = 32
        self.out_features_ = self._ef_features * 8
        self.layers = nn.Sequential(
            Conv2DWithActivation(1, self._ef_features),
            Conv2DWithActivation(self._ef_features, self._ef_features * 2),
            Conv2DWithActivation(self._ef_features * 2, self._ef_features * 4),
            Conv2DWithActivation(self._ef_features * 4, self._ef_features * 8),
            Conv2DWithoutActivation(
                self._ef_features * 8, self._ef_features * 8, padding="valid"
            ),
            Flatten((-1, self._ef_features * 8)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).view(x.size()[0], -1)

    @property
    def out_features(self) -> int:
        return self.out_features_


def _init_conv_without_bias(
    layer: t.Union[nn.Conv2d, nn.Conv3d]
) -> t.Union[nn.Conv2d, nn.Conv3d]:
    nn.init.xavier_uniform_(layer.weight)
    return layer


def _init_conv_with_bias(
    layer: t.Union[nn.Conv2d, nn.Conv3d]
) -> t.Union[nn.Conv2d, nn.Conv3d]:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0)
    return layer


def _init_linear_with_bias(layer: nn.Linear) -> nn.Linear:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0)
    return layer


class _ResnetBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        stride = 1 if in_features == out_features else 2
        self.conv1 = nn.Conv2d(
            in_features, out_features, 3, stride, 1, bias=False
        )
        self.lrelu1 = nn.LeakyReLU(0.01)
        self.conv2 = nn.Conv2d(out_features, out_features, 3, 1, 1, bias=False)
        self.lrelu2 = nn.LeakyReLU(0.01)

        self.linear_transform = lambda x: x
        if stride > 1:
            self.linear_transform = nn.Conv2d(
                in_features, out_features, 1, stride, 0, bias=False
            )
            _init_conv_without_bias(self.linear_transform)
        _init_conv_without_bias(self.conv1)
        _init_conv_without_bias(self.conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu1(self.conv1(x))
        out = self.lrelu2(self.conv2(out) + self.linear_transform(x))
        return out


class ExtractorSVR(FeatureExtractor):
    def __init__(self):
        super().__init__()

        self._ef_dim = 32
        self.latent_dim = self._ef_dim * 8
        self.conv_init = _init_conv_with_bias(
            nn.Conv2d(1, self._ef_dim, 7, 2, 3, bias=True)
        )
        self.lrelu_init = nn.LeakyReLU(0.01)

        self.block_1_1 = _ResnetBlock(self._ef_dim, self._ef_dim)
        self.block_1_2 = _ResnetBlock(self._ef_dim, self._ef_dim)

        self.block_2_1 = _ResnetBlock(self._ef_dim, self._ef_dim * 2)
        self.block_2_2 = _ResnetBlock(self._ef_dim * 2, self._ef_dim * 2)

        self.block_3_1 = _ResnetBlock(self._ef_dim * 2, self._ef_dim * 4)
        self.block_3_2 = _ResnetBlock(self._ef_dim * 4, self._ef_dim * 4)

        self.block_4_1 = _ResnetBlock(self._ef_dim * 4, self._ef_dim * 8)
        self.block_4_2 = _ResnetBlock(self._ef_dim * 8, self._ef_dim * 8)

        self.conv_9 = _init_conv_with_bias(
            nn.Conv2d(self._ef_dim * 8, self._ef_dim * 16, 4, 2, 1, bias=True)
        )
        self.lrelu_9 = nn.LeakyReLU(0.01)
        self.conv_10 = _init_conv_with_bias(
            nn.Conv2d(self._ef_dim * 16, self._ef_dim * 16, 4, 2, 0, bias=True)
        )
        self.lrelu_10 = nn.LeakyReLU(0.01)
        self.flatten = Flatten((-1, self._ef_dim * 16))

        self.linear = nn.Sequential(
            _init_linear_with_bias(
                nn.Linear(self._ef_dim * 16, self._ef_dim * 16, bias=True)
            ),
            nn.LeakyReLU(0.01),
            _init_linear_with_bias(
                nn.Linear(self._ef_dim * 16, self._ef_dim * 16, bias=True)
            ),
            nn.LeakyReLU(0.01),
            _init_linear_with_bias(
                nn.Linear(self._ef_dim * 16, self._ef_dim * 16, bias=True)
            ),
            nn.LeakyReLU(0.01),
            _init_linear_with_bias(
                nn.Linear(self._ef_dim * 16, self.latent_dim, bias=True)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu_init(self.conv_init(x))
        out = self.block_1_2(self.block_1_1(out))
        out = self.block_2_2(self.block_2_1(out))
        out = self.block_3_2(self.block_3_1(out))
        out = self.block_4_2(self.block_4_1(out))

        out = self.lrelu_9(self.conv_9(out))
        out = self.lrelu_10(self.conv_10(out))
        out = self.flatten(out)

        return self.linear(out)

    @property
    def out_features(self) -> int:
        return self.latent_dim


class Extractor3D(FeatureExtractor):
    def __init__(self):
        super().__init__()

        self._ef_dim = 32

        self.layers = nn.Sequential(
            _init_conv_with_bias(
                nn.Conv3d(1, self._ef_dim, 4, 2, 1, bias=True)
            ),
            nn.LeakyReLU(0.01),
            _init_conv_with_bias(
                nn.Conv3d(self._ef_dim, self._ef_dim * 2, 4, 2, 1, bias=True)
            ),
            nn.LeakyReLU(0.01),
            _init_conv_with_bias(
                nn.Conv3d(
                    self._ef_dim * 2, self._ef_dim * 4, 4, 2, 1, bias=True
                )
            ),
            nn.LeakyReLU(0.01),
            _init_conv_with_bias(
                nn.Conv3d(
                    self._ef_dim * 4, self._ef_dim * 8, 4, 2, 1, bias=True
                )
            ),
            nn.LeakyReLU(0.01),
            _init_conv_with_bias(
                nn.Conv3d(
                    self._ef_dim * 8, self._ef_dim * 8, 4, 2, 0, bias=True
                )
            ),
            Flatten((-1, self._ef_dim * 8)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    @property
    def out_features(self) -> int:
        return self._ef_dim * 8


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._ef_dim = 32
        self.layers = nn.Sequential(
            _init_linear_with_bias(
                nn.Linear(self._ef_dim * 8, self._ef_dim * 16)
            ),
            nn.LeakyReLU(0.01),
            _init_linear_with_bias(
                nn.Linear(self._ef_dim * 16, self._ef_dim * 32)
            ),
            nn.LeakyReLU(0.01),
            _init_linear_with_bias(
                nn.Linear(self._ef_dim * 32, self._ef_dim * 64)
            ),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
