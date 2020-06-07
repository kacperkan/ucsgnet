import abc
import argparse
import math
import shutil
from pathlib import Path
from typing import List

import PIL.Image
import PIL.ImageDraw
import numpy as np
import tqdm
import random

from .generate_cross_diamond_dataset import Drawer as CrossDiamondDrawer
from .generate_csg_dataset import Circle, Diff, Or, Square, Triangle

np.random.seed(0)
random.seed(0)


class Drawer(abc.ABC):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

        self.min_y_ = int(0.3 * self.height)
        self.max_y_ = int(0.7 * self.height)

        self.min_x_ = int(0.3 * self.width)
        self.max_x_ = int(0.7 * self.width)

        self.area_ = self.height * self.width

    def draw(self) -> PIL.Image.Image:
        raise NotImplementedError


class TriangleDiffCircle(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)
        t_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        c_size = np.random.uniform(0.7 * t_size, 0.9 * t_size)
        triangle = (
            Triangle(self.height, self.width).draw(x, y, t_size).to_img()
        )
        circle = (
            Circle(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, c_size)
            .to_img()
        )

        return Diff(triangle, circle).consume()


class TriangleDiffRectangle(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)
        t_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        r_size = np.random.uniform(0.7 * t_size, 0.9 * t_size)
        triangle = (
            Triangle(self.height, self.width).draw(x, y, t_size).to_img()
        )
        rect = (
            Square(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, r_size)
            .to_img()
        )

        return Diff(triangle, rect).consume()


class TriangleDiffRecantglePlusRectangle(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)
        t_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        r_size = np.random.uniform(0.7 * t_size, 0.9 * t_size)
        r_smaller_size = np.random.uniform(0.4 * t_size, 0.9 * t_size)
        triangle = (
            Triangle(self.height, self.width).draw(x, y, t_size).to_img()
        )
        rect = (
            Square(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, r_size)
            .to_img()
        )
        rect_smaller = (
            Square(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, r_smaller_size)
            .to_img()
        )
        out = Or(Diff(triangle, rect).consume(), rect_smaller).consume()

        return out


class TriangleDiffRecantglePlusCircle(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)
        t_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        r_size = np.random.uniform(0.7 * t_size, 0.9 * t_size)
        c_smaller_size = np.random.uniform(0.4 * t_size, 0.9 * t_size)
        triangle = (
            Triangle(self.height, self.width).draw(x, y, t_size).to_img()
        )
        rect = (
            Square(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, r_size)
            .to_img()
        )
        circle = (
            Circle(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, c_smaller_size)
            .to_img()
        )
        out = Or(Diff(triangle, rect).consume(), circle).consume()

        return out


class TriangleDiffCirclePlusRectangle(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)
        t_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        c_size = np.random.uniform(0.7 * t_size, 0.9 * t_size)
        r_smaller_size = np.random.uniform(0.4 * t_size, 0.9 * t_size)
        triangle = (
            Triangle(self.height, self.width).draw(x, y, t_size).to_img()
        )
        circle = (
            Circle(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, c_size)
            .to_img()
        )
        rect_smaller = (
            Square(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, r_smaller_size)
            .to_img()
        )
        out = Or(Diff(triangle, circle).consume(), rect_smaller).consume()

        return out


class TriangleDiffCirclePlusCircle(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)
        t_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        c_size = np.random.uniform(0.7 * t_size, 0.9 * t_size)
        c_smaller_size = np.random.uniform(0.4 * t_size, 0.9 * t_size)
        triangle = (
            Triangle(self.height, self.width).draw(x, y, t_size).to_img()
        )
        circle = (
            Circle(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, c_size)
            .to_img()
        )
        circle_smaller = (
            Circle(self.height, self.width)
            .draw(x, y - math.sqrt(3) / 12 * t_size, c_smaller_size)
            .to_img()
        )
        out = Or(Diff(triangle, circle).consume(), circle_smaller).consume()

        return out


class Donut(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)
        c1_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        c2_size = np.random.uniform(0.5 * c1_size, 0.8 * c1_size)
        c1 = Circle(self.height, self.width).draw(x, y, c1_size).to_img()
        c2 = Circle(self.height, self.width).draw(x, y, c2_size).to_img()

        return Diff(c1, c2).consume()


class MultipleDonuts(Drawer):
    def draw(self) -> PIL.Image.Image:
        n = np.random.choice(np.arange(2, 4))
        global_img = Donut(self.height, self.width).draw()
        for n in range(n - 1):
            aux = Donut(self.height, self.width).draw()
            global_img = Or(global_img, aux).consume()
        return global_img


class Arc(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)

        c1_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
        c2_size = np.random.uniform(0.7 * c1_size, 0.8 * c1_size)

        width = c1_size - c2_size

        coords = (
            (x - c1_size / 2, y - c1_size / 2),
            (x + c1_size / 2, y + c1_size / 2),
        )

        start_angle = np.random.uniform(0, 360)
        end_angle = (np.random.uniform(180, 359) + start_angle) % 360
        if end_angle < start_angle:
            end_angle = 360 - end_angle

        img = PIL.Image.new("1", (self.width, self.height), 0)

        d = PIL.ImageDraw.ImageDraw(img)
        d.arc(coords, start_angle, end_angle, width=int(width), fill="white")

        return img


class MultipleArcs(Drawer):
    def draw(self) -> PIL.Image.Image:
        n = 2
        global_img = Arc(self.height, self.width).draw()
        for n in range(n - 1):
            aux = Arc(self.height, self.width).draw()
            global_img = Or(global_img, aux).consume()
        return global_img


class CrossDiamond(Drawer):
    def draw(self) -> PIL.Image.Image:
        dr = CrossDiamondDrawer(self.height, self.width)
        dr.put_rotated_hollow_diamond_randomly()
        dr.put_cross_randomly()
        dr.put_rotated_diamond_randomly()

        return dr.to_img()


class MoonShape(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)

        c1_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
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

        c1 = Circle(self.height, self.width).draw(x, y, c1_size).to_img()
        c2 = (
            Circle(self.height, self.width)
            .draw(coords[0] + x, coords[1] + y, c2_size)
            .to_img()
        )

        return Diff(c1, c2).consume()


class MoonShapeWithSquare(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)

        c1_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
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

        return Or(moon_shape, sq).consume()


class MoonShapeWithCircle(Drawer):
    def draw(self) -> PIL.Image.Image:
        y = np.random.uniform(self.min_y_, self.max_y_)
        x = np.random.uniform(self.min_x_, self.max_x_)

        c1_size = np.random.uniform(0.005 * self.area_, 0.015 * self.area_)
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
            Circle(self.height, self.width)
            .draw(c2_x, c2_y, c2_size / 2)
            .to_img()
        )

        return Or(moon_shape, sq).consume()


def generate(n: int, out_path: Path, height: int, width: int):
    if out_path.exists():
        shutil.rmtree(out_path.as_posix())
    out_path.mkdir(exist_ok=True, parents=True)

    possible_drawers: List[Drawer] = [
        TriangleDiffCircle(height, width),
        TriangleDiffRectangle(height, width),
        Donut(height, width),
        Arc(height, width),
        MoonShape(height, width),
        # repeated to increase probability
        MoonShapeWithSquare(height, width),
        TriangleDiffCirclePlusCircle(height, width),
        TriangleDiffCirclePlusRectangle(height, width),
        TriangleDiffRecantglePlusCircle(height, width),
        TriangleDiffRecantglePlusRectangle(height, width),
    ]

    for i in tqdm.trange(n):
        img = np.random.choice(possible_drawers).draw()
        img.save((out_path / f"{i}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_path", type=Path)
    parser.add_argument("-n", "--number", type=int, default=100)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    generate(args.number, args.out_path, args.height, args.width)


if __name__ == "__main__":
    main()
