"""Focused checks of portable asset loading, regeneration, and force conventions."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
ASSETS = PACKAGE/"assets"
sys.path.insert(0, str(PACKAGE/"scripts"))
from contact_wrench import aggregate


class BaselineAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        output = PACKAGE/"outputs"
        output.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(prefix="asset-rebuild-", dir=output)
        cls.rebuilt = Path(cls.temporary.name)/"assets"
        cls.reference_hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ASSETS/"REU_snake_robot.xml", ASSETS/"REU_snake_robot_stable_1.xml")
        }
        subprocess.run(
            [sys.executable, str(PACKAGE/"scripts/generate_assets.py"),
             "--output-assets", str(cls.rebuilt)],
            check=True, stdout=subprocess.PIPE, text=True,
        )

    @classmethod
    def tearDownClass(cls):
        # The recursive cleanup is restricted to the temporary directory owned
        # by this test, under this package's ignored outputs directory.
        target = Path(cls.temporary.name).resolve()
        if target.parent != (PACKAGE/"outputs").resolve():
            raise RuntimeError("Unexpected test cleanup target")
        cls.temporary.cleanup()

    def test_every_packaged_xml_loads(self):
        for path in sorted(ASSETS.glob("*.xml")):
            with self.subTest(asset=path.name):
                model = mujoco.MjModel.from_xml_path(str(path))
                self.assertGreater(model.nbody, 1)
                self.assertEqual(model.nmesh, 1)

    def test_generator_preserves_references_and_rebuilds_physics(self):
        for name, expected in self.reference_hashes.items():
            self.assertEqual(hashlib.sha256((ASSETS/name).read_bytes()).hexdigest(), expected)
        fields = (
            "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
            "geom_type", "geom_pos", "geom_quat", "geom_size", "geom_contype", "geom_conaffinity",
            "jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "dof_damping", "dof_frictionloss",
            "actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
            "pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp",
            "key_qpos", "key_ctrl",
        )
        names = [
            "xml_anchor_q065_left.xml", "xml_anchor_q065_right.xml",
            "xml_anchor_q065_5modules_plus45.xml", "xml_anchor_q065_5modules_minus45.xml",
            "xml_anchor_q065_5modules_torque_convex.xml",
            "xml_anchor_q065_5modules_torque_proxy.xml",
            "xml_anchor_q065_5modules_torque_proxy_all_cylinders.xml",
        ]
        for name in names:
            original = mujoco.MjModel.from_xml_path(str(ASSETS/name))
            regenerated = mujoco.MjModel.from_xml_path(str(self.rebuilt/name))
            for field in fields:
                with self.subTest(scene=name, field=field):
                    np.testing.assert_allclose(
                        getattr(regenerated, field), getattr(original, field), rtol=0, atol=1e-10,
                    )

    def test_collision_profiles_and_direct_torque_mapping(self):
        for mode in ("proxy", "convex"):
            model = mujoco.MjModel.from_xml_path(str(ASSETS/f"xml_anchor_q065_5modules_torque_{mode}.xml"))
            data = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, data, model.key("q065").id)
            self.assertEqual(model.nu, 4)
            self.assertAlmostEqual(data.qpos[model.joint("joint_2").qposadr[0]], .65)
            self.assertAlmostEqual(data.qpos[model.joint("joint_4").qposadr[0]], np.pi/4)
            inactive = set()
            for g in range(model.ngeom):
                if not model.body(int(model.geom_bodyid[g])).name.startswith("mod_"):
                    continue
                enabled = (model.geom(g).name.endswith("_in") if mode == "proxy"
                           else model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
                if not enabled:
                    inactive.add(g)
                    self.assertEqual(model.geom_contype[g], 0)
                    self.assertEqual(model.geom_conaffinity[g], 0)
            for p in range(model.npair):
                self.assertNotIn(int(model.pair_geom1[p]), inactive)
                self.assertNotIn(int(model.pair_geom2[p]), inactive)
            commands = np.array([.2, -.3, .1, -.15])
            data.ctrl[:] = commands
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(data.actuator_force, commands, atol=1e-12)
            for a in range(model.nu):
                joint = int(model.actuator_trnid[a, 0])
                np.testing.assert_allclose(
                    data.qfrc_actuator[model.jnt_dofadr[joint]], commands[a], atol=1e-12,
                )
            for _ in range(50):
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                for c in data.contact:
                    self.assertNotIn(int(c.geom1), inactive)
                    self.assertNotIn(int(c.geom2), inactive)

    def test_proxy_force_aggregation_matches_engine_body_buffer(self):
        model = mujoco.MjModel.from_xml_path(str(ASSETS/"xml_anchor_q065_5modules_torque_proxy.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("q065").id)
        mujoco.mj_forward(model, data)
        pairs, floor, _ = aggregate(model, data)
        mujoco.mj_rnePostConstraint(model, data)
        # There are no applied external loads, and wall bodies have no robot
        # inertia motion; compare only the summed robot body forces.
        ids = [b for b in range(model.nbody) if model.body(b).name.startswith("mod_")]
        expected = data.cfrc_ext[ids, 3:].sum(axis=0)
        actual = sum((value["force"] for value in pairs.values()), np.zeros(3)) + floor
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        left = pairs[("mod_0_in", "Wall_2")]
        self.assertGreater(left["n_points"], 1)
        self.assertGreater(left["force"][0], 0.)


if __name__ == "__main__":
    unittest.main()
