import abc
import argparse
import math
import scipy.stats
from pathlib import Path

import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import numpy as np
import tqdm

np.random.seed(1337)


class Tree:
    def __init__(self, depth: int, height: int, width: int):
        self.depth = depth
        self.height = height
        self.width = width

        self.nodes_ = []
        self._shapes = [
            Rectangle,
            Circle,
            Cross,
            RotatedDiamond,
            HollowRotatedDiamond,
            Triangle,
        ]

        self._operators = [And, Or, Diff]

        self.initialize()

    def initialize(self):
        self.nodes_ = []

    def generate_random_structure(self):
        for _ in range(2 ** self.depth):
            node: Shape = np.random.choice(self._shapes)(
                self.height, self.width
            )
            node.randomize()
            self.nodes_.append(node.to_img())

    def consume_randomly(self):
        while len(self.nodes_) > 1:
            indices = np.arange(0, len(self.nodes_))
            node_1_idx = np.random.choice(indices)
            node_1 = self.nodes_.pop(node_1_idx)

            indices = np.arange(0, len(self.nodes_))
            node_2_idx = np.random.choice(indices)
            node_2 = self.nodes_.pop(node_2_idx)

            rand_op = np.random.choice(self._operators)(node_1, node_2)
            new_node = rand_op.consume()
            self.nodes_.append(new_node)

        return self.nodes_[0]


class Shape(abc.ABC):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

        self.area_ = self.height * self.width
        self.img_ = PIL.Image.new("1", (self.width, self.height), 0)
        self.canvas_ = PIL.ImageDraw.Draw(self.img_, "1")

    def draw(self, x: int, y: int, size: int) -> "Shape":
        raise NotImplementedError

    def to_img(self) -> PIL.Image.Image:
        return self.img_

    def randomize(self):
        size = np.random.uniform(self.area_ * 0.002, self.area_ * 0.015)
        x = np.random.uniform(size / 2, self.width - size / 2)
        y = np.random.uniform(size / 2, self.height - size / 2)

        self.draw(x, y, size)


class Rectangle(Shape):
    def draw(self, x: int, y: int, size: int) -> "Shape":
        width = size
        height = int(size / 3)

        rect_x1 = x - width // 2
        rect_y1 = y - height // 2

        rect_x2 = x + width // 2
        rect_y2 = y + height // 2

        self.canvas_.rectangle(
            ((rect_x1, rect_y1), (rect_x2, rect_y2)), fill="white"
        )

        return self


class CustomRectangle(Shape):
    def __init__(self, height: int, width: int, hw_ratio: float):
        super().__init__(height, width)

        self.hw_ratio = hw_ratio

    def draw(self, x: int, y: int, size: int) -> "Shape":
        width = size
        height = self.hw_ratio * width

        rect_x1 = x - width // 2
        rect_y1 = y - height // 2

        rect_x2 = x + width // 2
        rect_y2 = y + height // 2

        self.canvas_.rectangle(
            ((rect_x1, rect_y1), (rect_x2, rect_y2)), fill="white"
        )

        return self


class Square(Shape):
    def draw(self, x: int, y: int, size: int) -> "Shape":
        width = size
        height = size

        rect_x1 = x - width // 2
        rect_y1 = y - height // 2

        rect_x2 = x + width // 2
        rect_y2 = y + height // 2
        self.canvas_.rectangle(
            ((rect_x1, rect_y1), (rect_x2, rect_y2)), fill="white"
        )

        return self


class Circle(Shape):
    def draw(self, x: int, y: int, size: int) -> "Shape":
        radius = size // 2
        left_up_point = x - radius, y - radius
        right_down_point = (x + radius, y + radius)
        self.canvas_.ellipse((left_up_point, right_down_point), fill="white")
        return self


class Cross(Shape):
    def draw(self, x: int, y: int, size: int) -> "Shape":
        width = size
        height = int(size / 3)

        rect_1_x1 = x - width // 2
        rect_1_y1 = y - height // 2

        rect_1_x2 = x + width // 2
        rect_1_y2 = y + height // 2

        rect_2_x1 = x - height // 2
        rect_2_y1 = y - width // 2

        rect_2_x2 = x + height // 2
        rect_2_y2 = y + width // 2

        self.canvas_.rectangle(
            ((rect_1_x1, rect_1_y1), (rect_1_x2, rect_1_y2)), fill="white"
        )
        self.canvas_.rectangle(
            ((rect_2_x1, rect_2_y1), (rect_2_x2, rect_2_y2)), fill="white"
        )

        return self


class RotatedDiamond(Shape):
    def draw(self, x: int, y: int, size: int) -> "Shape":
        top_corner = (x, y - size // 2)
        right_corner = (x + size // 2, y)
        left_corner = (x - size // 2, y)
        bottom_corner = (x, y + size // 2)
        self.canvas_.polygon(
            (top_corner, right_corner, bottom_corner, left_corner),
            fill="black",
        )

        return self


class HollowRotatedDiamond(Shape):
    def _draw_diamond(self, x: int, y: int, size: int, color: str):
        top_corner = (x, y - size // 2)
        right_corner = (x + size // 2, y)
        left_corner = (x - size // 2, y)
        bottom_corner = (x, y + size // 2)
        self.canvas_.polygon(
            (top_corner, right_corner, bottom_corner, left_corner), fill=color
        )

    def draw(self, x: int, y: int, size: int) -> "Shape":
        self._draw_diamond(size, x, y, "white")
        self._draw_diamond(int(size * 0.6), x, y, "black")

        return self


class Triangle(Shape):
    def draw(self, x: int, y: int, size: int) -> "Shape":
        top_point = (x, y - size * math.sqrt(3) / 2)
        left_point = (x - size * math.sqrt(3) / 2, y + size * math.sqrt(3) / 6)
        right_point = (
            x + size * math.sqrt(3) / 2,
            y + size * math.sqrt(3) / 6,
        )

        self.canvas_.polygon(
            (top_point, right_point, left_point), fill="white"
        )

        return self


class Operator:
    def __init__(self, shape_1: PIL.Image.Image, shape_2: PIL.Image.Image):
        self.shape_1 = shape_1
        self.shape_2 = shape_2

    def consume(self) -> PIL.Image.Image:
        raise NotImplementedError


class IncrementalAnd(Operator):
    def consume(self) -> PIL.Image.Image:
        return PIL.ImageChops.logical_or(
            self.shape_1,
            PIL.ImageChops.logical_and(self.shape_1, self.shape_2),
        )


class And(Operator):
    def consume(self) -> PIL.Image.Image:
        return PIL.ImageChops.logical_and(self.shape_1, self.shape_2)


class Or(Operator):
    def consume(self) -> PIL.Image.Image:
        return PIL.ImageChops.logical_or(self.shape_1, self.shape_2)


class Diff(Operator):
    def consume(self) -> PIL.Image.Image:
        common_area = PIL.ImageChops.logical_and(self.shape_1, self.shape_2)
        new_image = PIL.ImageChops.difference(self.shape_1, common_area)
        return new_image


class Drawer:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.area_ = self.height * self.width

        self._shapes = [
            Rectangle,
            Triangle,
            Circle,
            Cross,
            RotatedDiamond,
            HollowRotatedDiamond,
        ]
        self._operators = [Diff, Or, IncrementalAnd]

        self._main_shape = None
        self._last_shape = None

    def initialize(self):
        self._main_shape = None
        self._last_shape = None

    def synthesize_random_shape(self):
        size = np.random.uniform(self.area_ * 0.002, self.area_ * 0.015)
        x = np.random.uniform(size / 2, self.width - size / 2)
        y = np.random.uniform(size / 2, self.height - size / 2)

        shape = np.random.choice(self._shapes)(self.height, self.width)
        shape.draw(x, y, size)
        img = shape.to_img()

        if self._main_shape is None:
            self._main_shape = img
        else:
            self._last_shape = img

    def synthesize_operator(self):
        if self._last_shape is None:
            return

        to_compose = (self._main_shape, self._last_shape)
        operator_cls = np.random.choice(self._operators)
        operator = operator_cls(*to_compose)
        self._main_shape = operator.consume()

    def generate_random_composition(self, num_steps: int):
        self.initialize()
        self.synthesize_random_shape()
        for i in range(num_steps):
            self.synthesize_random_shape()
            self.synthesize_operator()

    def to_img(self) -> PIL.Image.Image:
        return self._main_shape


def generate_single_image(
    height: int, width: int, trials: int
) -> PIL.Image.Image:
    img = None
    for i in range(trials):
        x = np.arange(5, 12)
        x_u, x_l = x + 0.5, x - 0.5
        prob = scipy.stats.norm.cdf(x_u, scale=5) - scipy.stats.norm.cdf(
            x_l, scale=5
        )
        prob = prob / prob.sum()
        depth = np.random.choice(x, p=prob)
        tree = Tree(depth, height, width)
        tree.generate_random_structure()
        img = tree.consume_randomly()
        img = PIL.ImageChops.invert(img)
        if not np.all(np.asarray(img)):
            return img

    return img


def generate_images(out_folder: str, n: int, height: int, width: int):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    for i in tqdm.trange(n):
        image = generate_single_image(height, width, 100)
        image.save(out_folder / f"{i}.png", format="png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_folder", help="Folder where images will be saved")
    parser.add_argument(
        "num_images", type=int, help="Number of images to produce"
    )
    parser.add_argument(
        "--height", type=int, help="Height of produced images", default=64
    )
    parser.add_argument(
        "--width", type=int, help="Width of produced images", default=64
    )

    args = parser.parse_args()

    generate_images(args.out_folder, args.num_images, args.height, args.width)


if __name__ == "__main__":
    main()
