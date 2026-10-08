"""Report summed world-frame proxy contact wrenches per robot-geom/wall pair."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from contact_wrench import aggregate

PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT = PACKAGE/"assets/xml_anchor_q065_5modules_torque_proxy.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT)
    parser.add_argument("--keyframe", default="q065")
    parser.add_argument("--seconds", type=float, default=0.)
    parser.add_argument("--output", type=Path, default=PACKAGE/"outputs/contact_wrenches.json")
    args = parser.parse_args()
    if not 0 <= args.seconds <= 10:
        parser.error("--seconds must be in [0,10]")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key < 0:
        parser.error(f"Missing keyframe {args.keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = 0.
    for _ in range(round(args.seconds/model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    pairs, floor, violation = aggregate(model, data)
    rows = []
    for (geom, wall), values in pairs.items():
        f = values["force"]
        magnitude = float(np.linalg.norm(f))
        rows.append({
            "robot_geom": geom, "wall": wall, "n_solver_points": values["n_points"],
            "force_world_N": f.tolist(),
            "force_magnitude_N": magnitude,
            "force_direction_world": (f/magnitude).tolist() if magnitude > 1e-9 else None,
            "moment_world_Nm": values["moment"].tolist(),
            "moment_reference_world_m": values["reference"].tolist(),
        })
    result = {
        "model": args.model.name, "time_s": float(data.time),
        "force_convention": "Force applied to the snake; contact-frame forces summed in world coordinates",
        "contact_pairs": rows, "floor_force_on_robot_world_N": floor.tolist(),
        "maximum_contact_cone_violation_N": violation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
