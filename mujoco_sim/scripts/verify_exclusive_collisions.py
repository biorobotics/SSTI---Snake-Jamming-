"""Check collision masks, explicit pairs, and realized contacts for both modes."""
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
    for mode in ('convex','proxy'):
        path = ASSETS/f'xml_anchor_q065_5modules_torque_{mode}.xml'
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,model.key('q065').id)
        robot = {g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name.startswith('mod_')}
        allowed = int(mujoco.mjtGeom.mjGEOM_MESH if mode=='convex' else mujoco.mjtGeom.mjGEOM_CYLINDER)
        active = {g for g in robot if (int(model.geom_type[g]) == allowed and
                                     (mode != 'proxy' or model.geom(g).name.endswith('_in')))}
        inactive = robot-active
        for g in inactive:
            assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
        for p in range(model.npair):
            assert int(model.pair_geom1[p]) not in inactive
            assert int(model.pair_geom2[p]) not in inactive
        initial_contacts, actual_contacts = [], set()
        tail_walls, max_tail_wall_count = set(), 0
        max_depth = 0.
        for step in range(1501):
            mujoco.mj_forward(model,data)
            current_tail_walls = set()
            for contact in data.contact:
                pair = (int(contact.geom1),int(contact.geom2))
                for g in pair:
                    if g in robot:
                        assert g in active, (mode,model.geom(g).name)
                names = tuple(model.geom(g).name for g in pair)
                actual_contacts.add(names)
                depth = max(0.,-float(contact.dist))
                max_depth = max(max_depth,depth)
                if step == 0:
                    initial_contacts.append({'geoms':names,'signed_distance_m':float(contact.dist)})
                if contact.dist <= 0:
                    for g, other in ((pair[0],pair[1]),(pair[1],pair[0])):
                        if g in robot and model.body(int(model.geom_bodyid[g])).name.startswith('mod_4'):
                            if model.geom(other).name.startswith('Wall_'):
                                current_tail_walls.add(model.geom(other).name)
            tail_walls.update(current_tail_walls)
            max_tail_wall_count = max(max_tail_wall_count,len(current_tail_walls))
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            if step < 1500:
                mujoco.mj_step(model,data)
        rows = {'mode':mode,'scene':str(path),'inactive_robot_geoms':[model.geom(g).name for g in sorted(inactive)],
                'active_robot_geoms':[model.geom(g).name for g in sorted(robot-inactive)],
                'n_explicit_contact_pairs':int(model.npair),'initial_contacts':initial_contacts,
                'contact_pairs_seen_in_3_seconds':sorted(actual_contacts),
                'max_penetration_in_3_seconds_m':max_depth,'tail_contact_walls_seen':sorted(tail_walls),
                'max_tail_wall_count_at_one_snapshot':max_tail_wall_count,
                'no_disabled_geom_contacts':True,'MuJoCo_warning_counts':data.warning.number.tolist()}
        results.append(rows)
    (OUTPUT/'exclusive_collision_verification.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('mode','n_explicit_contact_pairs','no_disabled_geom_contacts',
                     'max_penetration_in_3_seconds_m','tail_contact_walls_seen','max_tail_wall_count_at_one_snapshot',
                     'MuJoCo_warning_counts')} for r in results],indent=2))


if __name__=='__main__':
    main()
