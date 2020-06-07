import json
import typing as t
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


# methods below are blatantly copied from
# https://github.com/czq142857/BSP-NET-pytorch/blob/master/utils.py


def get_simple_dataset_paths_from_config(
    processed_data_path: str, split_config_path: str
) -> t.List[str]:
    with open(split_config_path) as f:
        config = json.load(f)

    processed_data_path = Path(processed_data_path)
    renders = [(processed_data_path / path).as_posix() for path in config]

    return renders


def write_ply_point_normal(
    name: str, vertices: np.ndarray, normals: t.Optional[np.ndarray] = None
):
    fout = open(name, "w")
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(
                str(vertices[ii, 0])
                + " "
                + str(vertices[ii, 1])
                + " "
                + str(vertices[ii, 2])
                + " "
                + str(vertices[ii, 3])
                + " "
                + str(vertices[ii, 4])
                + " "
                + str(vertices[ii, 5])
                + "\n"
            )
    else:
        for ii in range(len(vertices)):
            fout.write(
                str(vertices[ii, 0])
                + " "
                + str(vertices[ii, 1])
                + " "
                + str(vertices[ii, 2])
                + " "
                + str(normals[ii, 0])
                + " "
                + str(normals[ii, 1])
                + " "
                + str(normals[ii, 2])
                + "\n"
            )
    fout.close()


def write_ply_triangle(name: str, vertices: np.ndarray, triangles: np.ndarray):
    fout = open(name, "w")
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(triangles)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(
            str(vertices[ii, 0])
            + " "
            + str(vertices[ii, 1])
            + " "
            + str(vertices[ii, 2])
            + "\n"
        )
    for ii in range(len(triangles)):
        fout.write(
            "3 "
            + str(triangles[ii, 0])
            + " "
            + str(triangles[ii, 1])
            + " "
            + str(triangles[ii, 2])
            + "\n"
        )
    fout.close()


def quat_to_rot_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    rot_matrix = quaternions.new_zeros(quaternions.shape[:-1] + (3, 3))
    s = 1 / quaternions.norm(dim=-1, p=2).pow(2)

    qr, qi, qj, qk = quaternions.split(split_size=1, dim=-1)
    qr = qr[..., 0]
    qi = qi[..., 0]
    qj = qj[..., 0]
    qk = qk[..., 0]

    qi2 = qi.pow(2)
    qj2 = qj.pow(2)
    qk2 = qk.pow(2)

    rot_matrix[..., 0, 0] = 1 - 2 * s * (qj2 + qk2)
    rot_matrix[..., 0, 1] = 2 * s * (qi * qj - qk * qr)
    rot_matrix[..., 0, 2] = 2 * s * (qi * qk + qj * qr)

    rot_matrix[..., 1, 0] = 2 * s * (qi * qj + qk * qr)
    rot_matrix[..., 1, 1] = 1 - 2 * s * (qi2 + qk2)
    rot_matrix[..., 1, 2] = 2 * s * (qj * qk - qi * qr)

    rot_matrix[..., 2, 0] = 2 * s * (qi * qk - qj * qr)
    rot_matrix[..., 2, 1] = 2 * s * (qj * qk + qi * qr)
    rot_matrix[..., 2, 2] = 1 - 2 * s * (qi2 + qj2)

    return rot_matrix


def quat_to_rot_matrix_numpy(quaternions: np.ndarray) -> np.ndarray:
    rot_matrix = np.zeros(quaternions.shape[:-1] + (3, 3))
    s = 1 / (np.linalg.norm(quaternions, ord=2, axis=-1) ** 2)

    qr, qi, qj, qk = np.split(quaternions, indices_or_sections=4, axis=-1)
    qr = qr[..., 0]
    qi = qi[..., 0]
    qj = qj[..., 0]
    qk = qk[..., 0]

    qi2 = qi ** 2
    qj2 = qj ** 2
    qk2 = qk ** 2

    rot_matrix[..., 0, 0] = 1 - 2 * s * (qj2 + qk2)
    rot_matrix[..., 0, 1] = 2 * s * (qi * qj - qk * qr)
    rot_matrix[..., 0, 2] = 2 * s * (qi * qk + qj * qr)

    rot_matrix[..., 1, 0] = 2 * s * (qi * qj + qk * qr)
    rot_matrix[..., 1, 1] = 1 - 2 * s * (qi2 + qk2)
    rot_matrix[..., 1, 2] = 2 * s * (qj * qk - qi * qr)

    rot_matrix[..., 2, 0] = 2 * s * (qi * qk - qj * qr)
    rot_matrix[..., 2, 1] = 2 * s * (qj * qk + qi * qr)
    rot_matrix[..., 2, 2] = 1 - 2 * s * (qi2 + qj2)

    return rot_matrix


def read_point_normal_ply_file(
    shape_file: str,
) -> t.Tuple[np.ndarray, np.ndarray]:
    file = open(shape_file, "r")
    lines = file.readlines()

    start = 0
    vertex_num = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num, 3], np.float32)
    normals = np.zeros([vertex_num, 3], np.float32)
    for i in range(vertex_num):
        line = lines[i + start].split()
        vertices[i, 0] = float(line[0])  # X
        vertices[i, 1] = float(line[1])  # Y
        vertices[i, 2] = float(line[2])  # Z
        normals[i, 0] = float(line[3])  # normalX
        normals[i, 1] = float(line[4])  # normalY
        normals[i, 2] = float(line[5])  # normalZ
    return vertices, normals


def ply2obj(ply_path: str, obj_path: str):
    ply = PlyData.read(ply_path)

    with open(obj_path, "w") as f:
        f.write("# OBJ file\n")

        verteces = ply["vertex"]

        for v in verteces:
            p = [v["x"], v["y"], v["z"]]
            if "red" in v and "green" in v and "blue" in v:
                c = [v["red"] / 256, v["green"] / 256, v["blue"] / 256]
            else:
                c = [0, 0, 0]
            a = p + c
            f.write("v %.6f %.6f %.6f %.6f %.6f %.6f \n" % tuple(a))

        for v in verteces:
            if "nx" in v and "ny" in v and "nz" in v:
                n = (v["nx"], v["ny"], v["nz"])
                f.write("vn %.6f %.6f %.6f\n" % n)

        for v in verteces:
            if "s" in v and "t" in v:
                t = (v["s"], v["t"])
                f.write("vt %.6f %.6f\n" % t)

        if "face" in ply:
            for i in ply["face"]["vertex_index"]:
                f.write("f")
                for j in range(i.size):
                    # ii = [ i[j]+1 ]
                    ii = [i[j] + 1, i[j] + 1, i[j] + 1]
                    # f.write(" %d" % tuple(ii) )
                    f.write(" %d/%d/%d" % tuple(ii))
                f.write("\n")
