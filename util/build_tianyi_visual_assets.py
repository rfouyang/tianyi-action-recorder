"""Build a render-only Tianyi MJCF and decimated visual meshes with Blender.

Run from the project root:

    blender -b -noaudio --factory-startup \
      --python util/build_tianyi_visual_assets.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
import xml.etree.ElementTree as element_tree
from pathlib import Path

try:
    import bpy
except ModuleNotFoundError as error:  # pragma: no cover - executed by Blender
    raise SystemExit("Run this builder with Blender, not the project Python interpreter") from error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "asset" / "tianyi2"
SOURCE_MJCF = ASSET_DIR / "tianyi2_pos.xml"
SOURCE_VISUAL_URDF = ASSET_DIR / "tianyi2.0_urdf_with_hands.urdf"
SOURCE_MESH_DIR = ASSET_DIR / "meshes"
OUTPUT_MESH_DIR = ASSET_DIR / "meshes_visual_optimized"
OUTPUT_MJCF = ASSET_DIR / "tianyi2_visual.xml"
OUTPUT_METADATA = ASSET_DIR / "visual_model_metadata.json"
OUTPUT_URDF_MESH_METADATA = ASSET_DIR / "visual_urdf_mesh_metadata.json"
TARGET_TRIANGLES_PER_MESH = 12_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def optimize_mesh(source_path: Path, output_path: Path) -> tuple[int, int]:
    clear_scene()
    bpy.ops.wm.stl_import(
        filepath=str(source_path),
        global_scale=1.0,
        forward_axis="Y",
        up_axis="Z",
        use_mesh_validate=True,
    )
    imported_objects = tuple(
        candidate for candidate in bpy.context.selected_objects if candidate.type == "MESH"
    )
    if len(imported_objects) != 1:
        raise RuntimeError(
            f"Expected one mesh object in {source_path}; got {len(imported_objects)}"
        )
    mesh_object = imported_objects[0]
    bpy.context.view_layer.objects.active = mesh_object
    source_triangles = len(mesh_object.data.polygons)

    optimized_triangles = source_triangles
    for _ in range(3):
        if optimized_triangles <= TARGET_TRIANGLES_PER_MESH:
            break
        modifier = mesh_object.modifiers.new(name="MuJoCo visual decimation", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = TARGET_TRIANGLES_PER_MESH / optimized_triangles
        modifier.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        next_triangle_count = len(mesh_object.data.polygons)
        if next_triangle_count >= optimized_triangles:
            break
        optimized_triangles = next_triangle_count

    for polygon in mesh_object.data.polygons:
        polygon.use_smooth = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.obj_export(
        filepath=str(output_path),
        check_existing=False,
        export_selected_objects=True,
        export_uv=False,
        export_normals=True,
        export_colors=False,
        export_materials=False,
        export_triangulated_mesh=True,
        apply_modifiers=True,
        forward_axis="Y",
        up_axis="Z",
    )
    optimized_triangles = len(mesh_object.data.polygons)
    return source_triangles, optimized_triangles


def build_visual_mjcf() -> None:
    tree = element_tree.parse(SOURCE_MJCF)
    root = tree.getroot()
    root.attrib["model"] = "tianyi2_visual_optimized"

    mesh_name_mapping: dict[str, str] = {}
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("Source MJCF has no asset section")
    for mesh in asset.findall("mesh"):
        original_name = mesh.attrib["name"]
        stem = Path(original_name).stem
        optimized_name = f"visual_{stem}"
        mesh_name_mapping[original_name] = optimized_name
        mesh.attrib["name"] = optimized_name
        mesh.attrib["file"] = f"./meshes_visual_optimized/{stem}.obj"

    for parent in root.iter():
        for child in tuple(parent):
            if child.tag == "geom" and child.attrib.get("class") == "collision":
                parent.remove(child)

    for geom in root.findall(".//geom"):
        mesh_name = geom.attrib.get("mesh")
        if mesh_name in mesh_name_mapping:
            geom.attrib["mesh"] = mesh_name_mapping[mesh_name]

    for material in asset.findall("material"):
        if material.attrib.get("name") == "base_material":
            material.attrib.update(
                rgba="0.68 0.73 0.77 1",
                specular="0.32",
                shininess="0.38",
            )
    element_tree.SubElement(
        asset,
        "texture",
        name="visual_sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.16 0.21 0.28",
        rgb2="0.55 0.63 0.7",
        width="512",
        height="3072",
    )

    visual = element_tree.Element("visual")
    element_tree.SubElement(
        visual,
        "global",
        offwidth="1280",
        offheight="1280",
    )
    element_tree.SubElement(
        visual,
        "quality",
        shadowsize="2048",
        offsamples="2",
        numslices="24",
        numstacks="16",
        numquads="2",
    )
    element_tree.SubElement(
        visual,
        "headlight",
        ambient="0.42 0.42 0.42",
        diffuse="0.72 0.72 0.72",
        specular="0.18 0.18 0.18",
    )
    asset_index = tuple(root).index(asset)
    root.insert(asset_index, visual)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Source MJCF has no worldbody section")
    floor = worldbody.find("geom[@name='floor']")
    if floor is not None:
        floor.attrib["rgba"] = "0.34 0.39 0.44 1"
    worldbody.insert(
        0,
        element_tree.Element(
            "light",
            name="visual_key_light",
            directional="true",
            castshadow="true",
            pos="2.5 2.0 3.5",
            dir="-0.5 -0.35 -1",
            diffuse="0.72 0.75 0.8",
            specular="0.25 0.25 0.25",
        ),
    )
    worldbody.insert(
        1,
        element_tree.Element(
            "light",
            name="visual_fill_light",
            directional="true",
            castshadow="false",
            pos="-2 -2 2.5",
            dir="0.5 0.4 -1",
            diffuse="0.28 0.31 0.36",
            specular="0.08 0.08 0.08",
        ),
    )

    element_tree.indent(tree, space="  ")
    tree.write(OUTPUT_MJCF, encoding="utf-8", xml_declaration=False)


def mjcf_visual_mesh_names() -> set[str]:
    """Return the 25 body meshes used by the offline MuJoCo renderer."""
    root = element_tree.parse(SOURCE_MJCF).getroot()
    return {
        Path(mesh.attrib["file"].removeprefix("./")).name for mesh in root.findall("./asset/mesh")
    }


def urdf_visual_mesh_names() -> set[str]:
    """Return every body and Inspire-hand visual mesh used by Viser."""
    root = element_tree.parse(SOURCE_VISUAL_URDF).getroot()
    return {
        Path(mesh.attrib["filename"]).name for mesh in root.findall("./link/visual/geometry/mesh")
    }


def main() -> None:
    OUTPUT_MESH_DIR.mkdir(parents=True, exist_ok=True)
    body_mesh_names = mjcf_visual_mesh_names()
    viser_mesh_names = urdf_visual_mesh_names()
    if not body_mesh_names < viser_mesh_names:
        raise RuntimeError(
            "The Inspire-hand visual mesh set must contain all body-only MuJoCo meshes"
        )

    mesh_records = []
    for source_name in sorted(viser_mesh_names):
        source_path = SOURCE_MESH_DIR / source_name
        if not source_path.is_file():
            raise RuntimeError(f"Missing pinned Tianyi visual mesh: {source_path}")
        output_path = OUTPUT_MESH_DIR / f"{source_path.stem}.obj"
        source_triangles, optimized_triangles = optimize_mesh(source_path, output_path)
        mesh_records.append(
            {
                "source": source_path.name,
                "output": output_path.name,
                "source_triangles": source_triangles,
                "optimized_triangles": optimized_triangles,
                "source_sha256": sha256(source_path),
                "sha256": sha256(output_path),
            }
        )
        print(
            f"{source_path.name}: {source_triangles} -> {optimized_triangles} triangles",
            flush=True,
        )

    build_visual_mjcf()
    body_mesh_records = [record for record in mesh_records if record["source"] in body_mesh_names]
    metadata = {
        "schema_version": 1,
        "purpose": "render_only",
        "source_mjcf": SOURCE_MJCF.name,
        "source_mjcf_sha256": sha256(SOURCE_MJCF),
        "visual_mjcf": OUTPUT_MJCF.name,
        "visual_mjcf_sha256": sha256(OUTPUT_MJCF),
        "generator": f"Blender {bpy.app.version_string}",
        "target_triangles_per_mesh": TARGET_TRIANGLES_PER_MESH,
        "source_triangle_count": sum(item["source_triangles"] for item in body_mesh_records),
        "optimized_triangle_count": sum(item["optimized_triangles"] for item in body_mesh_records),
        "meshes": body_mesh_records,
    }
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    urdf_mesh_metadata = {
        "schema_version": 1,
        "purpose": "viser_body_and_inspire_hands",
        "source_urdf": SOURCE_VISUAL_URDF.name,
        "source_urdf_sha256": sha256(SOURCE_VISUAL_URDF),
        "generator": f"Blender {bpy.app.version_string}",
        "target_triangles_per_mesh": TARGET_TRIANGLES_PER_MESH,
        "mesh_count": len(mesh_records),
        "source_triangle_count": sum(item["source_triangles"] for item in mesh_records),
        "optimized_triangle_count": sum(item["optimized_triangles"] for item in mesh_records),
        "meshes": mesh_records,
    }
    OUTPUT_URDF_MESH_METADATA.write_text(
        json.dumps(urdf_mesh_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Optimized "
        f"{metadata['source_triangle_count']} -> {metadata['optimized_triangle_count']} "
        f"triangles; wrote {OUTPUT_MJCF.relative_to(PROJECT_ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:  # pragma: no cover - Blender command-line failure path
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
