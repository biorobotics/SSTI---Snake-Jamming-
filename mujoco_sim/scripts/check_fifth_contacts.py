"""Inspect +/- pi/4 fifth-module candidates at the q=0.65 keyframe."""
import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent/'outputs'
ASSETS = HERE.parent / 'assets'


def load(scene):
    model = mujoco.MjModel.from_xml_path(str(ASSETS/scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,model.key('q065').id)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    mujoco.mj_forward(model,data)
    return model,data


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference, initial = load('xml_anchor_q065_left.xml')
    rows = []
    for tail in ('plus45','minus45'):
        model,data = load(f'xml_anchor_q065_5modules_{tail}.xml')
        # Kinematics of original four modules must stay exactly the same.
        for i in range(4):
            old = reference.body(f'mod_{i}_body').id
            new = model.body(f'mod_{i}_body').id
            np.testing.assert_allclose(data.xpos[new],initial.xpos[old],atol=1e-12)
            np.testing.assert_allclose(data.xmat[new],initial.xmat[old],atol=1e-12)
            np.testing.assert_allclose(model.body_mass[new],reference.body_mass[old],atol=0)
        angle = float(data.qpos[model.joint('joint_4').qposadr[0]])
        expected_angle = np.pi/4 if tail == 'plus45' else -np.pi/4
        assert abs(angle-expected_angle) < 1e-12
        assert abs(data.qpos[model.joint('joint_2').qposadr[0]]-.65)<1e-12
        wall_ids = [model.geom(k).id for k in ('Wall_1','Wall_2')]
        gap = data.geom_xpos[wall_ids[0],0]-data.geom_xpos[wall_ids[1],0]-sum(model.geom_size[g,0] for g in wall_ids)
        expected_gap = 2*.0167+.1021*np.cos((np.pi-.65)/2)
        assert abs(gap-expected_gap)<1e-12
        fifth_geoms = [g for g in range(model.ngeom)
                       if model.body(int(model.geom_bodyid[g])).name in ('mod_4_in','mod_4_body','mod_4_out')]
        contacts = []
        for c in data.contact:
            if c.dist>0:
                continue
            g1,g2 = int(c.geom1),int(c.geom2)
            if g1 in fifth_geoms and g2 in wall_ids:
                robot,wall = g1,g2
            elif g2 in fifth_geoms and g1 in wall_ids:
                robot,wall = g2,g1
            else:
                continue
            contacts.append({'robot_geom':model.geom(robot).name,'wall':model.geom(wall).name,
                             'signed_distance_m':float(c.dist),'position':c.pos.tolist()})
        mesh = model.geom('mesh4').id
        asset = int(model.geom_dataid[mesh])
        start,count = int(model.mesh_vertadr[asset]),int(model.mesh_vertnum[asset])
        vertices = model.mesh_vert[start:start+count]@data.geom_xmat[mesh].reshape(3,3).T+data.geom_xpos[mesh]
        xmin,xmax = float(vertices[:,0].min()),float(vertices[:,0].max())
        left_face,right_face = -gap/2,gap/2
        beyond_wall = max(0.,left_face-xmin,xmax-right_face)
        walls = sorted({c['wall'] for c in contacts})
        rows.append({'candidate':tail,'joint_4_angle_rad':angle,'q_initial_rad':.65,
                     'wall_face_gap_m':gap,'fifth_wall_contacts':contacts,'fifth_contact_wall_names':walls,
                     'n_fifth_contact_walls':len(walls),'fifth_mesh_x_bounds_m':[xmin,xmax],
                     'fifth_mesh_extent_beyond_inner_wall_m':beyond_wall,
                     'fifth_max_contact_penetration_m':max([-c['signed_distance_m'] for c in contacts],default=0.),
                     'original_four_module_kinematics_preserved':True})
    one_wall = [r for r in rows if r['n_fifth_contact_walls']==1]
    if not one_wall:
        raise RuntimeError('Neither candidate has fifth-module contact with exactly one wall')
    selected = min(one_wall,key=lambda r:r['fifth_mesh_extent_beyond_inner_wall_m'])
    result = {'selection':selected['candidate'],
              'criterion':'exactly one fifth-module contact wall; smallest extent outside the inner wall faces',
              'scope':'initial keyframe geometry only, not a proof of maintained anchoring contact',
              'candidates':rows}
    (OUTPUT/'fifth_module_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
