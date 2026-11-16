"""Six planar joints of the archived ReU12 under MuJoCo position control."""

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
import yaml

from collision import Geometry, MODULES, OBSTACLE_MASK, PLANE, make_obstacle
from observations import BOX_SLOTS, OBS_SIZE, observation
from rewards import reward_components


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ACTIVE = ("head_anchor_hinge", "joint_2", "joint_4", "joint_6", "joint_8", "joint_10")
HARD_GAPS = (10, 8, 7, 6)


def load_config(path=HERE / "config.yaml"):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if "time_discount" in config:
        half_life = float(config["time_discount"]["half_life_s"])
        dt = config["environment"]["timestep"] * config["environment"]["substeps"]
        if not math.isfinite(half_life) or half_life <= 0 or dt <= 0:
            raise ValueError("Time discount requires a positive half-life and control interval")
        config["ppo"]["gamma"] = math.exp(-math.log(2.) * dt / half_life)
    return config


def read_cases():
    return {gap: json.loads((ASSETS / f"case_{gap:02d}cm.json").read_text()) for gap in HARD_GAPS}


def make_maps(seed, count, max_obstacles):
    rng = np.random.default_rng(seed)
    return tuple(tuple((float(rng.uniform(.25, .75)), float(side * rng.uniform(.13, .38)),
                        float(rng.uniform(.045, .14)), float(rng.uniform(.035, .10)),
                        float(rng.uniform(-.55, .55)))
                       for side in rng.choice([-1., 1.], int(rng.integers(1, max_obstacles + 1))))
                 for _ in range(count))


class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, stage="A", map_seed=0, gap=None, fixture=None):
        super().__init__()
        self.cfg = dict((config or load_config())["environment"])
        if stage not in ("A", "B", "C", "D") or (gap is not None and (stage != "D" or gap not in HARD_GAPS)):
            raise ValueError("Invalid stage or corridor gap")
        self.stage, self.gap = stage, gap
        self.fixture = fixture
        if fixture is not None and (stage != "D" or gap not in (None, fixture["gap_cm"])):
            raise ValueError("A corridor fixture requires the matching Stage D scenario")
        self.template = (ASSETS / "archived_base.xml").read_bytes()
        root = ET.fromstring(self.template)
        self.mesh_asset = (HERE.parents[1] / "mujoco_sim/assets/REU_Snake/Mesh/ReU_Mesh.STL").read_bytes()
        self.cases = read_cases()
        self.base_boxes = []
        self.height = None
        for box in root.findall("./worldbody/geom"):
            if box.get("name", "").startswith("map_obstacle_"):
                x, y, z = map(float, box.get("pos").split())
                sx, _, sz = map(float, box.get("size").split())
                self.base_boxes.append((-z, x, 2 * sz, 2 * sx, 0.))
                self.height = y
        if self.height is None or len(self.base_boxes) != 6:
            raise ValueError("Unexpected archived obstacle template")
        self.maps = make_maps(map_seed, self.cfg["random_map_count"], self.cfg["max_training_obstacles"])
        self.simulations = {}
        self.action_space = gym.spaces.Box(-1., 1., (6,), np.float32)
        self.observation_space = gym.spaces.Box(-1., 1., (OBS_SIZE,), np.float32)

    def build_sim(self, boxes):
        root = ET.fromstring(self.template)
        root.find("compiler").attrib.update(meshdir="", fusestatic="false")
        root.find("option").set("timestep", str(self.cfg["timestep"]))
        ET.SubElement(root.find("option"), "flag", filterparent="disable")
        world = root.find("worldbody")
        for geom in world.findall("geom"):
            if geom.get("name", "").startswith("map_obstacle_"):
                world.remove(geom)
        for body in world.iter("body"):
            for joint in body.findall("joint"):
                if joint.get("name") not in ACTIVE:
                    body.remove(joint)
        margin = self.cfg["contact_clearance_m"]
        mesh_only = self.cfg.get("mesh_only_collision", False)
        for geom in world.iter("geom"):
            name = geom.get("name", "")
            match = re.match(r"(?:mod_|mesh)(\d+)", name)
            if match:
                if mesh_only and geom.get("type") != "mesh":
                    geom.attrib.update(contype="0", conaffinity="0")
                    continue
                module = int(match[1])
                mask = OBSTACLE_MASK | sum(1 << i for i in range(MODULES) if abs(i - module) >= 2)
                geom.attrib.update(contype=str(1 << module), conaffinity=str(mask), margin=str(margin))
        endpoint = world.find(".//body[@name='mod_0_in']")
        endpoint_center = endpoint.find("geom[@name='mod_0_in']").get("pos")
        ET.SubElement(endpoint, "site", name="reaching_marker_M1", type="sphere", pos=endpoint_center,
                      size=".009", rgba="1 .22 .55 1")
        for index, box in enumerate(boxes):
            make_obstacle(world, index, box, self.height, margin)
        root.remove(root.find("actuator"))
        keyframes = root.find("keyframe")
        if keyframes is not None:
            root.remove(keyframes)
        actuator = ET.SubElement(root, "actuator")
        for name in ACTIVE:
            joint = world.find(f".//joint[@name='{name}']")
            limit = self.cfg["actuator_limit_nm"]
            ET.SubElement(actuator, "position", name=f"position_{name}", joint=name, gear="1",
                          kp=str(self.cfg["position_kp"]), kv=str(self.cfg["position_kv"]),
                          ctrllimited="true", ctrlrange=joint.get("range"),
                          forcelimited="true", forcerange=f"{-limit} {limit}")
        model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"),
                                          {"ReU_Mesh.STL": self.mesh_asset})
        data = mj.MjData(model)
        robot, modules = [], []
        for geom in range(model.ngeom):
            match = re.match(r"(?:mod_|mesh)(\d+)", model.geom(geom).name)
            if match and (not mesh_only or model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH):
                robot.append(geom)
                modules.append(int(match[1]))
        obstacles = [model.geom(f"rl_obstacle_{i}").id for i in range(len(boxes))]
        expected_geoms = MODULES if mesh_only else 48
        if (model.nq != 6 or model.nv != 6 or model.nu != 6 or len(robot) != expected_geoms
                or set(modules) != set(range(MODULES))):
            raise ValueError("Archived robot does not match the six-DOF specification")
        return model, data, Geometry(model, data, robot, modules, obstacles)

    def use_boxes(self, boxes):
        self.boxes = tuple(tuple(box) for box in boxes)
        if len(self.boxes) > BOX_SLOTS:
            raise ValueError("Too many obstacles for fixed-length observations")
        if self.boxes not in self.simulations:
            self.simulations[self.boxes] = self.build_sim(self.boxes)
        self.m, self.d, self.geometry = self.simulations[self.boxes]
        joints = [self.m.joint(name).id for name in ACTIVE]
        self.q_indices = self.m.jnt_qposadr[joints]
        self.v_indices = self.m.jnt_dofadr[joints]
        self.bounds = self.m.jnt_range[joints]
        self.joint_scale = np.max(np.abs(self.bounds), axis=1)
        self.actuator_limits = self.m.actuator_forcerange[:, 1]
        self.marker_id = self.m.site("reaching_marker_M1").id
        self.head_id = self.m.site("HEAD_module_12").id
        self.backbone_ids = [self.m.geom(f"mod_{i}_in").id for i in range(11, -1, -1)]
        self.rne_force = np.zeros(self.m.nv)

    @property
    def q(self):
        return self.d.qpos[self.q_indices]

    @property
    def dq(self):
        return self.d.qvel[self.v_indices]

    @property
    def marker(self):
        return PLANE @ self.d.site_xpos[self.marker_id]

    def set_state(self, q):
        mj.mj_resetData(self.m, self.d)
        self.d.qpos[self.q_indices] = q
        self.d.ctrl[:] = q
        mj.mj_forward(self.m, self.d)

    def geometric_failure(self):
        return (self.geometry.contact(self.cfg["contact_clearance_m"])
                or (None if self.geometry.in_box() else "out_of_bounds"))

    def task(self):
        if self.stage == "D":
            gap = self.gap if self.gap is not None else int(self.np_random.choice(HARD_GAPS))
            case = self.fixture if self.fixture is not None else self.cases[gap]
            walls = [(*wall[:2], case["wall_length_m"], case["wall_thickness_m"], wall[2])
                     for wall in case["walls_LT_theta_rad"]]
            return [*self.base_boxes, *walls], case["q_start_rad"], case["goal_LT_m"], case["gap_cm"]
        boxes = {"A": (), "B": self.base_boxes[:2], "C": self.maps[int(self.np_random.integers(len(self.maps)))]}
        return boxes[self.stage], np.zeros(6), None, None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for _ in range(80):
            boxes, start, goal, self.case_gap = self.task()
            self.use_boxes(boxes)
            self.set_state(start)
            if self.geometric_failure():
                if self.stage == "D":
                    raise ValueError(f"Archived corridor start is invalid for gap {self.case_gap}")
                continue
            if goal is None:
                goal = self.sample_goal(self.marker.copy())
                if goal is None:
                    continue
            else:
                case = self.fixture if self.fixture is not None else self.cases[self.case_gap]
                self.set_state(case["target_witness_q_rad"])
                if self.geometric_failure():
                    raise ValueError(f"Archived target witness is invalid for gap {self.case_gap}")
                np.testing.assert_allclose(self.marker, goal, rtol=0., atol=1e-10)
            self.set_state(start)
            self.goal = np.asarray(goal, dtype=float)
            self.steps = 0
            self.actual_torque = self.d.actuator_force.copy()
            self.estimated_torque = np.zeros(6)
            self.command = self.q
            self.initial_distance = float(np.linalg.norm(self.marker - self.goal))
            self.minimum_distance = self.initial_distance
            self.link_clearance = self.geometry.link_clearance(self.cfg["clearance_scale_m"])
            return observation(self, self.link_clearance), self.info("running")
        raise RuntimeError("Could not sample a valid start and target for this stage")

    def sample_goal(self, start_marker):
        for _ in range(80):
            q = self.np_random.uniform(-.70, .70, 6)
            self.set_state(q)
            marker = self.marker
            if .065 <= np.linalg.norm(marker - start_marker) <= .52 and not self.geometric_failure():
                return marker
        return None

    def rnea_demand(self, command):
        # Feedforward torque for a critically damped reference acceleration.
        omega = self.cfg["rnea_omega_rad_s"]
        desired = omega**2 * (command - self.q) - 2. * omega * self.dq
        actual = self.d.qacc.copy()
        try:
            self.d.qacc[self.v_indices] = desired
            mj.mj_rne(self.m, self.d, 1, self.rne_force)
        finally:
            self.d.qacc[:] = actual
        return (self.rne_force - self.d.qfrc_passive)[self.v_indices]

    def step(self, action):
        previous_distance = float(np.linalg.norm(self.marker - self.goal))
        self.steps += 1
        command = self.q + self.cfg["action_scale_rad"] * np.clip(action, -1., 1.)
        self.command = np.clip(command, self.bounds[:, 0], self.bounds[:, 1])
        self.estimated_torque = self.rnea_demand(self.command)
        torque_limit = bool(np.any(np.abs(self.estimated_torque) > self.cfg["rnea_limit_nm"]))
        contacts = {}
        allow_obstacle_contact = self.cfg.get("allow_obstacle_contact", False)
        collision_events = 0
        obstacle_contact_substeps = 0
        touching = allow_obstacle_contact and "collision" in self.geometry.contacts(self.cfg["contact_clearance_m"])
        joint_limit = out_of_bounds = False
        saturated = np.zeros(6, dtype=bool)
        if not torque_limit:
            self.d.ctrl[:] = self.command
            for _ in range(self.cfg["substeps"]):
                mj.mj_step(self.m, self.d)
                self.actual_torque = self.d.actuator_force.copy()
                saturated |= np.abs(self.actual_torque) >= self.actuator_limits - 1e-9
                contacts = self.geometry.contacts(self.cfg["contact_clearance_m"])
                if allow_obstacle_contact:
                    collision_events += int("collision" in contacts and not touching)
                    touching = "collision" in contacts
                    obstacle_contact_substeps += int(touching)
                joint_limit = bool(np.any(self.q < self.bounds[:, 0] - 1e-9)
                                   or np.any(self.q > self.bounds[:, 1] + 1e-9))
                if "self_collision" in contacts or (contacts and not allow_obstacle_contact) or joint_limit:
                    break
                mj.mj_kinematics(self.m, self.d)
                out_of_bounds = not self.geometry.in_box()
                if out_of_bounds:
                    break
            mj.mj_forward(self.m, self.d)
        final_contacts = self.geometry.contacts(self.cfg["contact_clearance_m"])
        if allow_obstacle_contact:
            collision_events += int("collision" in final_contacts and not touching)
        for name, point in final_contacts.items():
            contacts.setdefault(name, point)
        out_of_bounds = out_of_bounds or not self.geometry.in_box()
        distance = float(np.linalg.norm(self.marker - self.goal))
        flags = dict(collision="collision" in contacts, self_collision="self_collision" in contacts,
                     joint_limit=joint_limit, out_of_bounds=out_of_bounds, torque_limit=torque_limit,
                     success=distance < self.cfg["goal_tolerance_m"], timeout=self.steps >= self.cfg["max_steps"])
        status = next((name for name, triggered in flags.items()
                       if triggered and not (allow_obstacle_contact and name == "collision")), "running")
        self.minimum_distance = min(self.minimum_distance, distance)
        self.link_clearance = self.geometry.link_clearance(self.cfg["clearance_scale_m"])
        components = reward_components(self.cfg, previous_distance, distance, status)
        info = self.info(status)
        if allow_obstacle_contact:
            info.update(collision=flags["collision"] or flags["self_collision"] or obstacle_contact_substeps > 0,
                        obstacle_collision_events=collision_events,
                        obstacle_contact_s=obstacle_contact_substeps * self.cfg["timestep"])
        info.update(termination_flags=flags, reward_components=components,
                    collision_point_LT_m=contacts.get(status),
                    position_clip_fraction=float(np.mean(command != self.command)),
                    position_tracking_error_rad=float(np.mean(np.abs(self.command - self.q))),
                    actuator_saturation_fraction=float(np.mean(saturated)))
        return (observation(self, self.link_clearance), sum(components.values()),
                status not in {"running", "timeout"}, status == "timeout", info)

    def info(self, status):
        return dict(status=status, is_success=status == "success", collision=status in {"collision", "self_collision"},
                    torque_violation=status == "torque_limit", distance_m=float(np.linalg.norm(self.marker - self.goal)),
                    gap_cm=self.case_gap, elapsed_sim_s=float(self.d.time),
                    initial_distance_m=self.initial_distance, minimum_distance_m=self.minimum_distance,
                    goal_progress_m=self.initial_distance - float(np.linalg.norm(self.marker - self.goal)))

    def check_anchoring_safety(self, state, reaction_force):
        raise NotImplementedError("Anchoring safety is outside this fixed-pivot baseline")
