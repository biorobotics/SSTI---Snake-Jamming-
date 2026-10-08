"""Reproduce the current anchoring scenes from preserved robot/mesh assets.

Reference files are never overwritten. Generated variants share their source
geometry/inertia, with explicitly selected actuation and collision profiles.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REFERENCES = (
    "REU_snake_robot_stable_1.xml", "REU_snake_robot.xml",
    "REU_snake_Downstream.xml", "xml_anchor_scene.xml",
    "xml_anchor_scene_box.xml", "xml_wrench_sampleing.xml",
)
MESH_PATH = Path("REU_Snake/Mesh/ReU_Mesh.STL")


def element(root, tag, name):
    return next(node for node in root.iter(tag) if node.get("name") == name)


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=False)


def five_module_robot(four_module_path):
    root = ET.parse(four_module_path).getroot()
    tail = copy.deepcopy(element(root, "body", "mod_3_in"))
    for node in tail.iter():
        if "name" in node.attrib:
            node.set("name", node.get("name").replace("mod_3", "mod_4")
                     .replace("joint_3", "joint_4").replace("mesh3", "mesh4"))
    element(tail, "geom", "mesh4").set("material", "blue")
    element(root, "body", "mod_3_out").append(tail)
    servo = copy.deepcopy(element(root, "position", "servo_3_pos"))
    servo.set("name", "servo_4_pos")
    servo.set("joint", "joint_4")
    root.find("actuator").append(servo)
    sensor = copy.deepcopy(element(root, "jointactuatorfrc", "joint_3_torque_sensor"))
    sensor.set("name", "joint_4_torque_sensor")
    sensor.set("joint", "joint_4")
    root.find("sensor").append(sensor)
    return root


def torque_variant(reference, collision):
    root = copy.deepcopy(reference)
    actuator = root.find("actuator")
    actuator.clear()
    for i in range(1, 5):
        ET.SubElement(actuator, "motor", {
            "name": f"motor_{i}_torque", "joint": f"joint_{i}", "gear": "1",
            "ctrllimited": "true", "ctrlrange": "-10 10",
            "forcelimited": "true", "forcerange": "-10 10",
        })
    for geom in root.iter("geom"):
        kind, name = geom.get("type"), geom.get("name", "")
        enabled = (
            kind == "mesh" if collision == "stl_only" else
            kind == "cylinder" and
            (collision == "all_cylinders" or name.endswith("_in"))
        )
        if not enabled:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    element(root, "geom", "mesh4").set("material", "blue")
    return root


def collision_bounds(model, data):
    lows, highs = [], []
    for g in range(model.ngeom):
        matrix = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start+count] @ matrix.T + center
            low, high = vertices.min(axis=0), vertices.max(axis=0)
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis = matrix[:, 2]
            radius, half_length = model.geom_size[g, :2]
            extent = radius*np.sqrt(np.maximum(0, 1-axis**2)) + half_length*np.abs(axis)
            low, high = center-extent, center+extent
        else:
            raise ValueError(f"Unsupported reference geom: {model.geom(g).name}")
        lows.append(low)
        highs.append(high)
    return np.min(lows, axis=0), np.max(highs, axis=0)


def base_pose(reference, signed_q):
    model = mujoco.MjModel.from_xml_path(str(reference))
    data = mujoco.MjData(model)
    # The central hinge's world axis is -y in this preserved CAD hierarchy.
    # A signed q/2 rotation of the free base aligns the bend bisector vertically.
    data.qpos[3:7] = [np.cos(signed_q/4), 0, np.sin(signed_q/4), 0]
    data.qpos[model.joint("joint_2").qposadr[0]] = signed_q
    mujoco.mj_forward(model, data)
    low, high = collision_bounds(model, data)
    data.qpos[0] -= (low[0]+high[0])/2
    data.qpos[2] += 0.005-low[2]  # Explicit 5 mm floor clearance.
    return data.qpos[:7].copy()


def scene(robot, q, base, gap, profile, model_name, key_name, tail=None):
    root = ET.Element("mujoco", {"model": model_name})
    root.append(ET.Comment(
        "q is central-axis hinge deflection. Wall face gap is "
        "2*r + L*cos((pi-q)/2). Geometry/contact parameters are not calibrated."
    ))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "scale", {"contactheight": "0.05", "contactwidth": "0.01"})
    ET.SubElement(root, "option", {
        "cone": "elliptic", "noslip_iterations": "5",
        "timestep": "0.002", "gravity": "0 0 -9.81",
    })
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", {
        "type": "skybox", "builtin": "gradient", "rgb1": ".3 .5 .7",
        "rgb2": "0 0 0", "width": "32", "height": "32",
    })
    ET.SubElement(assets, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker",
        "width": "512", "height": "512", "rgb1": ".1 .2 .3", "rgb2": ".2 .3 .4",
    })
    ET.SubElement(assets, "material", {
        "name": "grid", "texture": "grid", "texrepeat": "1 1",
        "texuniform": "true", "reflectance": ".2",
    })
    ET.SubElement(root, "include", {"file": robot})
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(world, "geom", {
        "name": "floor", "size": "5 5 .05", "type": "plane",
        "material": "grid", "group": "2",
    })
    for i, sign in ((1, 1), (2, -1)):
        wall = ET.SubElement(world, "body", {
            "name": f"Wall{i}", "pos": f"{sign*(gap/2+0.01):.17g} 0 0.2",
        })
        ET.SubElement(wall, "geom", {
            "name": f"Wall_{i}", "type": "box", "size": "0.01 0.11 0.5",
            "contype": "1", "conaffinity": "2", "group": "2",
            "rgba": "0.8 0.5 0.2 0.2",
        })
    contact = ET.SubElement(root, "contact")
    count = 4 if tail is None else 5
    for i in range(count):
        if profile == "all_cylinders":
            geoms = [f"mod_{i}_{suffix}" for suffix in ("in", "link", "out")]
        elif profile == "in_only":
            geoms = [f"mod_{i}_in"]
        else:
            geoms = [f"mesh{i}"]
        for geom in geoms:
            for wall in (1, 2):
                ET.SubElement(contact, "pair", {
                    "name": f"{geom}_wall{wall}", "geom1": f"Wall_{wall}", "geom2": geom,
                    "condim": "3", "friction": "1.35 1.35 1.35",
                    "solimp": "0.99 0.99 0.001", "solref": "0.004 1",
                })
    hinges = [0, q, 0] + ([] if tail is None else [tail])
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", {
        "name": key_name,
        "qpos": " ".join(f"{value:.17g}" for value in [*base, *hinges]),
        "ctrl": " ".join("0" for _ in hinges),
    })
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-assets", type=Path, default=ASSETS)
    parser.add_argument("--q", type=float, default=0.65, help="Central-axis deflection magnitude [rad]")
    parser.add_argument("--L", type=float, default=0.1021, help="Analytical arm length in wall formula [m]")
    parser.add_argument("--r", type=float, default=0.0167, help="Analytical radius in wall formula [m]")
    args = parser.parse_args()
    if not (0 < args.q < np.pi/2 and args.L > 0 and args.r > 0):
        parser.error("Require 0 < q < pi/2, L > 0, and r > 0")
    output = args.output_assets.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output != ASSETS.resolve():
        for name in REFERENCES:
            shutil.copy2(ASSETS/name, output/name)
        (output/MESH_PATH).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ASSETS/MESH_PATH, output/MESH_PATH)
    reference = output/"REU_snake_robot_stable_1.xml"
    five = five_module_robot(reference)
    proxy = ET.parse(output/"REU_snake_robot.xml").getroot()
    robots = {
        "REU_snake_robot_stable_5modules.xml": five,
        "REU_snake_robot_stable_5modules_torque.xml": torque_variant(five, "stl_only"),
        "REU_snake_robot_torque_proxy.xml": torque_variant(proxy, "in_only"),
        "REU_snake_robot_torque_proxy_all_cylinders.xml": torque_variant(proxy, "all_cylinders"),
    }
    generated = []
    for name, xml in robots.items():
        write_xml(xml, output/name)
        generated.append(name)
    default_q = np.isclose(args.q, 0.65, rtol=0, atol=1e-12)
    tag = "q065" if default_q else f"q{args.q:.4f}".replace(".", "p")
    key = "q065" if default_q else "anchor_init"
    prefix = f"xml_anchor_{tag}"
    gap = 2*args.r + args.L*np.cos((np.pi-args.q)/2)
    base_left = base_pose(reference, args.q)
    base_right = base_pose(reference, -args.q)
    configurations = [
        ("left", "REU_snake_robot_stable_1.xml", args.q, base_left, "stl_only", None),
        ("right", "REU_snake_robot_stable_1.xml", -args.q, base_right, "stl_only", None),
        ("5modules_plus45", "REU_snake_robot_stable_5modules.xml", args.q, base_left, "stl_only", np.pi/4),
        ("5modules_minus45", "REU_snake_robot_stable_5modules.xml", args.q, base_left, "stl_only", -np.pi/4),
        ("5modules_torque_convex", "REU_snake_robot_stable_5modules_torque.xml", args.q, base_left, "stl_only", np.pi/4),
        ("5modules_torque_proxy", "REU_snake_robot_torque_proxy.xml", args.q, base_left, "in_only", np.pi/4),
        ("5modules_torque_proxy_all_cylinders", "REU_snake_robot_torque_proxy_all_cylinders.xml", args.q, base_left, "all_cylinders", np.pi/4),
    ]
    for suffix, robot, signed_q, base, profile, tail in configurations:
        name = f"{prefix}_{suffix}.xml"
        xml = scene(robot, signed_q, base, gap, profile, f"anchoring_{tag}_{suffix}", key, tail)
        write_xml(xml, output/name)
        generated.append(name)
    for name in generated:
        mujoco.MjModel.from_xml_path(str(output/name))
    manifest = {
        "q_deflection_rad": args.q, "L_m": args.L, "r_m": args.r,
        "wall_face_gap_m": float(gap), "keyframe": key, "generated": generated,
        "warning": "Changing L/r changes wall spacing; it does not rescale the preserved CAD model.",
        "references_preserved": list(REFERENCES), "MuJoCo": mujoco.__version__,
    }
    (output/"generated_assets_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
