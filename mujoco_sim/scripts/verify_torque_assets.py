"""Verify actuation conversion and preservation of each source asset's physics."""
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent/'outputs'
ASSETS = HERE.parent/'assets'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    pairs = [('REU_snake_robot_stable_5modules.xml','REU_snake_robot_stable_5modules_torque.xml'),
             ('REU_snake_robot.xml','REU_snake_robot_torque_proxy.xml')]
    for source,destination in pairs:
        original = mujoco.MjModel.from_xml_path(str(ASSETS/source))
        model = mujoco.MjModel.from_xml_path(str(ASSETS/destination))
        for field in ('body_parentid','body_pos','body_quat','body_mass','body_inertia','body_ipos',
                      'jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','dof_damping','dof_frictionloss',
                      'geom_type','geom_pos','geom_quat','geom_size','geom_friction'):
            np.testing.assert_allclose(getattr(model,field),getattr(original,field),rtol=0,atol=1e-12)
        assert model.nu == 4
        mode = 'proxy' if 'proxy' in destination else 'convex'
        for g in range(model.ngeom):
            disabled = (not model.geom(g).name.endswith('_in') if mode == 'proxy'
                        else model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER)
            if disabled:
                assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
            else:
                assert model.geom_contype[g] == original.geom_contype[g]
                assert model.geom_conaffinity[g] == original.geom_conaffinity[g]
        commands = np.array([.2,-.3,.1,-.15])
        data = mujoco.MjData(model)
        data.ctrl[:] = commands
        mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.actuator_force,commands,rtol=0,atol=1e-12)
        row = {'source_asset':source,'torque_asset':destination,'geometry_inertia_joint_parameters_preserved':True,
               'exclusive_collision_profile':mode,'collision_masks_verified':True,
               'test_commands_Nm':commands.tolist(),'actuator_forces_Nm':data.actuator_force.tolist(),'mapping':[]}
        for a in range(model.nu):
            assert np.allclose(model.actuator_gear[a],[1,0,0,0,0,0])
            j = int(model.actuator_trnid[a,0])
            dof = int(model.jnt_dofadr[j])
            np.testing.assert_allclose(data.qfrc_actuator[dof],commands[a],rtol=0,atol=1e-12)
            row['mapping'].append({'actuator':model.actuator(a).name,'joint':model.joint(j).name,
                                   'generalized_joint_torque_Nm':float(data.qfrc_actuator[dof]),
                                   'ctrl_range_Nm':model.actuator_ctrlrange[a].tolist()})
        results.append(row)
    (OUTPUT/'torque_asset_verification.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
