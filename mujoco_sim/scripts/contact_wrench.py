"""Sum all contact forces for each robot geom--wall pair in world coordinates."""
import mujoco
import numpy as np


def aggregate(model, data):
    pairs = {}
    floor_force = np.zeros(3)
    cone_violation = 0.
    for k, c in enumerate(data.contact):
        if c.efc_address < 0:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, k, wrench)
        # c.frame contains the world normal/tangent axes as ROWS.
        rotation = c.frame.reshape(3, 3).T
        f_world = rotation @ wrench[:3]
        m_world = rotation @ wrench[3:]
        mu = np.asarray(c.friction[:2])
        scaled_shear = np.linalg.norm(wrench[1:3] / np.maximum(mu, 1e-12))
        cone_violation = max(cone_violation, float(scaled_shear-wrench[0]), float(-wrench[0]))
        a, b = int(c.geom1), int(c.geom2)
        # MuJoCo's returned wrench acts on geom2; geom1 gets its negative.
        for robot, other, sign in ((a, b, -1.), (b, a, 1.)):
            geom_name = model.geom(robot).name
            other_name = model.geom(other).name
            if not geom_name.startswith('mod_'):
                continue
            if other_name == 'floor':
                floor_force += sign*f_world
            if not other_name.startswith('Wall_'):
                continue
            key = (geom_name, other_name)
            if key not in pairs:
                pairs[key] = {'force': np.zeros(3), 'moment': np.zeros(3),
                              'reference': data.geom_xpos[robot].copy(), 'n_points': 0}
            item = pairs[key]
            item['force'] += sign*f_world
            item['moment'] += np.cross(c.pos-item['reference'], sign*f_world)+sign*m_world
            item['n_points'] += 1
    return pairs, floor_force, max(0., cone_violation)


def channels(model, data):
    pairs, floor, cone = aggregate(model, data)
    out = {'floor_Fx_N': floor[0], 'floor_Fy_N': floor[1], 'floor_Fz_N': floor[2],
           'contact_cone_violation_N': cone,
           'max_penetration_m': max(0., -min((c.dist for c in data.contact), default=0.))}
    for contact, module in enumerate((0, 2, 4), 1):
        geom = f'mod_{module}_in'
        normal_sign = 1. if contact != 2 else -1.
        expected_wall = 'Wall_2' if contact != 2 else 'Wall_1'
        force, moment, count, unexpected = np.zeros(3), np.zeros(3), 0, 0
        for (name, wall), item in pairs.items():
            if name != geom:
                continue
            force += item['force']
            moment += item['moment']
            count += item['n_points']
            unexpected += int(wall != expected_wall)
        out[f'Fn{contact}_absolute_N'] = normal_sign*force[0]
        # Common, signed in-plane shear direction +world z for all sensors.
        # It is NOT silently reversed at the middle wall.
        out[f'Ft{contact}_absolute_N'] = force[2]
        for j, axis in enumerate('xyz'):
            out[f'contact{contact}_F{axis}_world_N'] = force[j]
            out[f'contact{contact}_M{axis}_world_Nm'] = moment[j]
        size = float(np.linalg.norm(force))
        out[f'contact{contact}_resultant_N'] = size
        for j, axis in enumerate('xyz'):
            out[f'contact{contact}_direction_{axis}'] = force[j]/size if size > 1e-9 else np.nan
        out[f'contact{contact}_n_solver_points'] = count
        out[f'contact{contact}_unexpected_walls'] = unexpected
    extra = [item['force'] for (name, _), item in pairs.items() if name not in ('mod_0_in', 'mod_2_in', 'mod_4_in')]
    out['additional_wall_contact_force_N'] = sum(float(np.linalg.norm(f)) for f in extra)
    return out, pairs
