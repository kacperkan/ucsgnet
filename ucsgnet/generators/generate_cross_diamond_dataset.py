import argparse
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
import numpy as np
import tqdm

np.random.seed(1337)


class Drawer:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

        self.area_ = self.height * self.width
        self.img_ = PIL.Image.new("L", (self.width, self.height), 255)
        self.canvas_ = PIL.ImageDraw.Draw(self.img_, "L")

    def put_cross(self, size: int, x: int, y: int):
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
            ((rect_1_x1, rect_1_y1), (rect_1_x2, rect_1_y2)), fill="black"
        )
        self.canvas_.rectangle(
            ((rect_2_x1, rect_2_y1), (rect_2_x2, rect_2_y2)), fill="black"
        )

    def put_rotated_diamond(
        self, size: int, x: int, y: int, color: str = "black"
    ):
        top_corner = (x, y - size // 2)
        right_corner = (x + size // 2, y)
        left_corner = (x - size // 2, y)
        bottom_corner = (x, y + size // 2)
        self.canvas_.polygon(
            (top_corner, right_corner, bottom_corner, left_corner), fill=color
        )

    def put_rotated_hollow_diamond(self, size: int, x: int, y: int):
        self.put_rotated_diamond(size, x, y, "black")
        self.put_rotated_diamond(int(size * 0.6), x, y, "white")

    def to_img(self) -> PIL.Image.Image:
        return self.img_

    def put_cross_randomly(self):
        random_cross_size = np.random.uniform(
            self.area_ * 0.005, self.area_ * 0.012
        )
        random_cross_x_position = np.random.uniform(
            0.4 * self.width, 0.6 * self.width
        )
        random_cross_y_position = np.random.uniform(
            random_cross_size / 2, self.height - random_cross_size / 2
        )
        self.put_cross(
            int(random_cross_size),
            int(random_cross_x_position),
            int(random_cross_y_position),
        )

    def put_rotated_diamond_randomly(self):
        random_size = np.random.uniform(self.area_ * 0.003, self.area_ * 0.010)
        random_x_position = np.random.uniform(
            random_size / 2, 0.3 * self.width
        )
        random_y_position = np.random.uniform(
            random_size / 2, self.height - random_size / 2
        )
        self.put_rotated_diamond(
            int(random_size), int(random_x_position), int(random_y_position)
        )

    def put_rotated_hollow_diamond_randomly(self):
        random_size = np.random.uniform(self.area_ * 0.003, self.area_ * 0.010)
        random_x_position = np.random.uniform(
            0.6 * self.width, self.width - random_size / 2
        )
        random_y_position = np.random.uniform(
            random_size / 2, self.height - random_size / 2
        )
        self.put_rotated_hollow_diamond(
            int(random_size), int(random_x_position), int(random_y_position)
        )


def generate_single_image(height: int, width: int) -> PIL.Image.Image:
    drawer = Drawer(height, width)
    drawer.put_rotated_diamond_randomly()
    drawer.put_rotated_hollow_diamond_randomly()
    drawer.put_cross_randomly()
    return drawer.to_img()


def generate_images(out_folder: str, n: int, height: int, width: int):
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    for i in tqdm.trange(n):
        image = generate_single_image(height, width)
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
