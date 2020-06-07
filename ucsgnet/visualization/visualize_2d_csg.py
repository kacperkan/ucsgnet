import argparse
import random
import typing as t
from pathlib import Path

import PIL.Image
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ucsgnet.generators.generate_csg_dataset import Shape
from ucsgnet.generators.generate_simpler_csg import (
    Circle,
    Diff,
    Drawer,
    Or,
    Square,
)

np.random.seed(707)
random.seed(707)


class CircleWithSampling(Circle):
    @staticmethod
    def sample(
        count: int, height: int, width: int
    ) -> t.Tuple[PIL.Image.Image, ...]:
        out = []
        for i in range(count):
            figure = Circle(height, width)
            out.append(figure.to_img())
        return tuple(out)


class Sampler(Drawer):
    def draw(self) -> PIL.Image.Image:
        pass

    def __init__(self, height: int, width: int, figure: t.Type[Shape]):
        super().__init__(height, width)

        self.figure = figure

    def sample(self, count: int) -> t.Tuple[PIL.Image.Image, ...]:
        longest_dim = max(self.width, self.height)
        out = []
        for i in range(count):
            y = np.random.uniform(self.min_y_, self.max_y_)
            x = np.random.uniform(self.min_x_, self.max_x_)
            size = np.random.uniform(0.3 * longest_dim, 0.85 * longest_dim)
            instance = self.figure(self.height, self.width)
            shape = instance.draw(x, y, size).to_img()
            out.append(shape)
        return tuple(out)


class CustomizedMoonShapeWithSquare(Drawer):
    def draw(self) -> t.Tuple[PIL.Image.Image, ...]:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)

        longest_dim = max(self.height, self.width)

        c1_size = np.random.uniform(0.6 * longest_dim, 0.85 * longest_dim)
        c2_size = np.random.uniform(0.5 * c1_size, 0.8 * c1_size)

        rand_ratio = np.random.uniform(0.3, 1.0)
        rand_angle = np.random.uniform(0, np.pi * 2)

        main_endpoint = np.array([c1_size / 2 * rand_ratio, 0])
        rot_matrix = np.array(
            [
                [np.cos(rand_angle), -np.sin(rand_angle)],
                [np.sin(rand_angle), np.cos(rand_angle)],
            ]
        )

        coords = main_endpoint.dot(rot_matrix.T)
        c2_x = coords[0] + x
        c2_y = coords[1] + y

        c1 = Circle(self.height, self.width).draw(x, y, c1_size).to_img()
        c2 = (
            Circle(self.height, self.width)
            .draw(coords[0] + x, coords[1] + y, c2_size)
            .to_img()
        )

        moon_shape = Diff(c1, c2).consume()
        sq = (
            Square(self.height, self.width)
            .draw(c2_x, c2_y, c2_size / 2)
            .to_img()
        )

        return Or(moon_shape, sq).consume(), moon_shape, sq, c1, c2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def get_parts_visualization() -> t.Tuple[
    t.Tuple[np.ndarray, ...], t.Tuple[np.ndarray, ...]
]:
    thresholded_parts = CustomizedMoonShapeWithSquare(512, 512).draw()
    thresholded_parts = [
        np.array(part).astype(np.uint8) for part in thresholded_parts
    ]
    return (
        tuple(thresholded_parts),
        _process_generated_images(thresholded_parts),
    )


def _process_generated_images(
    parts: t.List[np.ndarray],
) -> t.Tuple[np.ndarray]:
    distances = []
    for part in parts:
        distance = cv2.distanceTransform(part, cv2.DIST_L2, maskSize=0)
        inside_distance = -cv2.distanceTransform(
            1 - part, cv2.DIST_L2, maskSize=0
        )
        total_distance: np.ndarray = distance + inside_distance
        total_distance = sigmoid(total_distance / 20)
        total_distance = (
            (total_distance - total_distance.min())
            / (total_distance.max() - total_distance.min())
            * 255
        ).astype(np.uint8)

        distances.append(total_distance)
    return tuple(distances)


def get_samples_visualization(
    figure: t.Type[Shape],
) -> t.Tuple[t.Tuple[np.ndarray, ...], t.Tuple[np.ndarray, ...]]:
    sampler = Sampler(512, 512, figure)
    thresholded_parts = sampler.sample(10)
    thresholded_parts = [
        np.array(part).astype(np.uint8) for part in thresholded_parts
    ]
    return (
        tuple(thresholded_parts),
        _process_generated_images(thresholded_parts),
    )


def generate_visualizations(path: Path):
    if not path.exists():
        path.mkdir()
    thrs, parts = get_parts_visualization()
    for i, (thr, part) in enumerate(zip(thrs, parts)):
        fig = Figure(figsize=(8, 8), dpi=100, frameon=False)
        # A canvas must be manually attached to the figure (pyplot would
        # automatically
        # do it).  This is done by instantiating the canvas with the figure as
        # argument.
        canvas = FigureCanvasAgg(fig)

        ax = fig.add_axes(
            [0, 0, 1, 1]
        )  # position: left, bottom, width, height
        ax.set_axis_off()
        ax.axis("off")
        ax.imshow(part, cmap="seismic")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        datum = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        datum = datum[..., ::-1][..., 1:]
        datum = cv2.resize(datum, (512, 512))

        cv2.imwrite((path / f"{i}_thr.png").as_posix(), thr * 255)
        cv2.imwrite((path / f"{i}_dist.png").as_posix(), datum)

    for i, (thr, part) in enumerate(zip(*get_samples_visualization(Square))):
        fig = Figure(figsize=(8, 8), dpi=100, frameon=False)
        # A canvas must be manually attached to the figure (pyplot would
        # automatically
        # do it).  This is done by instantiating the canvas with the figure as
        # argument.
        canvas = FigureCanvasAgg(fig)

        ax = fig.add_axes(
            [0, 0, 1, 1]
        )  # position: left, bottom, width, height
        ax.set_axis_off()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.imshow(part, cmap="seismic")
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        datum = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        datum = datum[..., ::-1][..., 1:]
        datum = cv2.resize(datum, (512, 512))

        cv2.imwrite(
            (path / f"{i}_example_rectangle_thr.png").as_posix(), thr * 255
        )
        cv2.imwrite(
            (path / f"{i}_example_rectangle_dist.png").as_posix(), datum
        )

    for i, (thr, part) in enumerate(zip(*get_samples_visualization(Circle))):
        fig = Figure(figsize=(8, 8), dpi=100, frameon=False)
        # A canvas must be manually attached to the figure (pyplot would
        # automatically
        # do it).  This is done by instantiating the canvas with the figure as
        # argument.
        canvas = FigureCanvasAgg(fig)

        ax = fig.add_axes(
            [0, 0, 1, 1]
        )  # position: left, bottom, width, height
        ax.set_axis_off()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.imshow(part, cmap="seismic")
        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()

        datum = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        datum = datum[..., ::-1][..., 1:]
        datum = cv2.resize(datum, (512, 512))

        cv2.imwrite(
            (path / f"{i}_example_circle_thr.png").as_posix(), thr * 255
        )
        cv2.imwrite((path / f"{i}_example_circle_dist.png").as_posix(), datum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", default="paper-stuff/parts", type=Path)

    args = parser.parse_args()
    generate_visualizations(args.out_folder)

    print(f"Visualizations have been saved in {args.out_folder.absolute()}")


if __name__ == "__main__":
    main()
