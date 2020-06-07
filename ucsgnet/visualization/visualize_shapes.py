import argparse
import json
import os
import tempfile
import typing as t
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
import tqdm
import trimesh
import trimesh.repair
from collections import OrderedDict

OBJECTS_TO_RENDER = OrderedDict(
    {
        "02691156": [
            "d18592d9615b01bbbc0909d98a1ff2b4",
            "d18f2aeae4146464bd46d022fd7d80aa",
            "d199612c22fe9313f4fb6842b3610149",
        ],
        "02828884": [
            "c8298f70c094a9fd25d3c528e036a532",
            "c83b3192c338527a2056b4bd5d870b47",
            "c8802eaffc7e595b2dc11eeca04f912e",
        ],
        "02933112": [
            "aff3488d05343a89e42b7a6468e7283f",
            "b06b351b939e279bc5ff6d1af2135fc9",
            "b0709afab8a3d9ce7e65d4ecde1c77ce",
        ],
        "02958343": [
            "cbc6e31c744ef872b34a6368f13f8b72",
            "cbc946b4f4c022305e524bb779a94481",
            "cbd0b4a0d264c856f35cb5c94d4343bf",
        ],
        "03001627": [
            "cbc47018135fc1b1462977c6d3c24550",
            "cbc5e6fce716e48ea28e529ba1f4836e",
            "cbc76d55a04d5b2e1d9a8cea064f5297",
        ],
        "03211117": [
            "d7ab9503d7f6dac6b4382097c3e8bcf7",
            "d7b87d0083bf5568fd28950562697757",
            "d8142f27c166cc21f103e4fb531505b4",
        ],
        "03636649": [
            "d13f1adad399c9f1ea93fe4e1ab627a2",
            "d153ae6d65b31e00fcb8d8c6d4df8143",
            "d16bb6b2f26084556acbef8d3bef8f28",
        ],
        "03691459": [
            "c90cbb0458648665da49a3feeb6532eb",
            "c91e878553979be9c5c378bd9e63485",
            "c91f926711d5e8261d485f425cc21556",
        ],
        "04090263": [
            "ca25a955adaa031812b38b1d99376c0b",
            "ca2bafb1ba4b97a1683e3750b53385d5",
            "ca4e431c75a8242445e0c3a4b827d51a",
        ],
        "04256520": [
            "cc644fad0b76a441d84c7dc40ac6d743",
            "cc7b690e4d86b471397aad305ec14786",
            "cc906e84c1a985fe80db6871fa4b6f35",
        ],
        "04379243": [
            "cd44665771f7b7d2b2000d40d3899456",
            "cd4e8748514e028642d23b95defe1ce5",
            "cd5f235344ff4c10d5b24cafb84903c7",
        ],
        "04401088": [
            "d2bce5fd1fd04beeb3db4664acec42ef",
            "d2f3eb92a31649647c17b7a9bb17a24",
            "d37afdca0c48251044b992023e0d3ef0",
        ],
        "04530566": [
            "cd65ea1bb0e091d5a1ea2dd0a4cf317e",
            "cd67f7d1ba943b162f84cb7932f866fd",
            "cdaff2fe98efb90058a8952c93ff9829",
        ],
    }
)


xml_head = """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="1000"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="2,2,2" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="ldrfilm">
                <integer name="width" value="480"/>
                <integer name="height" value="480"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <float name="alpha" value="0.1"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="0.63,0.61,0.58"/>
        </bsdf>

        <shape type="rectangle">
            <transform name="toWorld">
                <lookat origin="0,-11,5" target="0,0,0" /> 
                <scale x="0.5" y="0.5" z="0.5" />
            </transform>
            <emitter type="area">
                <spectrum name="radiance" value="40"/>
            </emitter> 
        </shape>

    """

xml_tail = """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="200" y="200" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
            <emitter type="area">
                <spectrum name="radiance" value="1"/>
            </emitter> 
        </shape>

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="20" y="20" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="0.8,0.8,0.8"/>
            </emitter>
        </shape>
    </scene>
    """

xml_obj = """
        <shape type="obj">
            <string name="filename" value="meshes/mesh.obj"/>
            <boolean name="faceNormals" value="true" />
            <bsdf type="twosided">
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="{},{},{}"/>
                </bsdf>
            </bsdf>
            <transform name="toWorld">
                <rotate x="1" angle="90" />
                <rotate z="1" angle="-90" />
            </transform>
        </shape>
    """

xml_obj_shapenet = """
        <shape type="obj">
            <string name="filename" value="meshes/mesh.obj"/>
            <boolean name="faceNormals" value="true" />
            <bsdf type="twosided">
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="{},{},{}"/>
                </bsdf>
            </bsdf>
            <transform name="toWorld">
                <rotate x="1" angle="90" />
                <rotate z="1" angle="-90" />
            </transform>
        </shape>
    """


def decode_image(byte_data: t.List[float]) -> np.ndarray:
    byte_data = np.asarray(byte_data, dtype=np.uint8)[..., np.newaxis]
    img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
    return img


def as_mesh(scene_or_mesh: t.Union[trimesh.Trimesh, trimesh.Scene]):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def modify_shapenet_file_to_temp_file(obj_path: str, temp_file: t.IO):
    obj: trimesh.Trimesh = trimesh.load_mesh(obj_path)
    obj = as_mesh(obj)
    obj.export(temp_file, file_type="obj")


def render_single_obj(obj_path: str, is_shapenet: bool) -> np.ndarray:
    if is_shapenet:
        xml_segments = [
            xml_head,
            xml_obj_shapenet.format("244", "244", "244"),
            xml_tail,
        ]
    else:
        xml_segments = [
            xml_head,
            xml_obj.format("244", "244", "244"),
            xml_tail,
        ]
    xml_content = str.join("", xml_segments)
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".zip", delete=False
    ) as temp_file:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", delete=False
        ) as temp_xml_file:
            temp_xml_file.write(xml_content)

        with zipfile.ZipFile(temp_file.name, mode="w") as zip_file:
            zip_file.write(temp_xml_file.name, arcname="scene.xml")
            if is_shapenet:
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".obj", delete=False
                ) as obj_temp_file:
                    modify_shapenet_file_to_temp_file(obj_path, obj_temp_file)
                zip_file.write(
                    obj_temp_file.name, os.path.join("meshes", "mesh.obj")
                )
                os.unlink(obj_temp_file.name)
            else:
                zip_file.write(obj_path, os.path.join("meshes", "mesh.obj"))

    try:
        with open(temp_file.name, "rb") as f:
            result = requests.post(
                "http://mitsuba:8000/render_zip", files={"zip": f}
            )
    except requests.exceptions.ConnectionError:
        with open(temp_file.name, "rb") as f:
            result = requests.post(
                "http://localhost:8000/render_zip", files={"zip": f}
            )
    os.unlink(temp_file.name)
    os.unlink(temp_xml_file.name)
    data = json.loads(result.content)
    an_img = decode_image(data)

    return an_img


def visualize_shapes(in_folder: str, out_folder: str, raw_folder: str):
    out_folder = Path(out_folder)
    shapes = Path(in_folder)

    for object_name, instances in tqdm.tqdm(OBJECTS_TO_RENDER.items()):
        for instance_name in tqdm.tqdm(instances):
            ground_truth_path = os.path.join(
                raw_folder,
                object_name,
                instance_name,
                "models",
                "model_normalized.obj",
            )
            if not os.path.exists(ground_truth_path):
                continue

            out_dir = out_folder / object_name / instance_name
            out_dir.mkdir(parents=True, exist_ok=True)

            obj = shapes / object_name / instance_name / "pred_mesh.obj"
            try:
                img = render_single_obj(obj.as_posix(), False)
                cv2.imwrite(
                    (out_dir / obj.with_suffix(".png").name).as_posix(),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )

                orig_img = render_single_obj(ground_truth_path, True)
                cv2.imwrite(
                    (out_dir / "ground_truth.png").as_posix(),
                    cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR),
                )
                for csg in tqdm.tqdm(
                    list(
                        (
                            shapes / object_name / instance_name / "csg_path"
                        ).glob("*.obj")
                    )
                ):
                    (out_dir / "csg_path").mkdir(exist_ok=True)
                    img = render_single_obj(csg.as_posix(), False)
                    cv2.imwrite(
                        (
                            out_dir / "csg_path" / csg.with_suffix(".png").name
                        ).as_posix(),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    )
            except TypeError:
                print("Invalid shape {}/{}".format(object_name, instance_name))
                continue
            except ValueError:
                print("Invalid shape {}/{}".format(object_name, instance_name))
                continue


def main():
    parser = argparse.ArgumentParser(
        description="Code for generating visualizations of shapes"
    )
    parser.add_argument(
        "--raw_folder", help="Input folder for the ShapeNet dataset", type=str,
    )
    parser.add_argument(
        "--in_folder",
        help="Input folder to generate visualizations from",
        type=str,
    )
    parser.add_argument(
        "--out_folder",
        help=(
            "Output folder for visualizations. Paths of visualization will "
            "be the same as for objects "
        ),
        type=str,
    )

    args = parser.parse_args()
    visualize_shapes(args.in_folder, args.out_folder, args.raw_folder)


if __name__ == "__main__":
    main()
