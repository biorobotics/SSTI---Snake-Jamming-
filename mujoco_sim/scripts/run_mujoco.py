"""Minimal existing-asset MuJoCo baseline: inspect, passive run, and viewer.

No legacy launcher is imported: those scripts overwrite MJCF at import time.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import queue
import time

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / 'assets'
DEFAULT_MODEL = ASSETS / 'xml_anchor_scene.xml'

# Existing XML defaults; these are not a measured/preloaded anchoring pose.
INITIAL_QPOS = {'joint_1': 0.0, 'joint_2': 0.0, 'joint_3': 0.0, 'joint_4': 0.0}
DT_OUTPUT = 0.01
JOINT_TEST_AMPLITUDE = 0.05  # radians; purely kinematic, not a torque controller


def name(model, kind, idx):
    return mujoco.mj_id2name(model, kind, int(idx)) or f'unnamed_{int(idx)}'


def fmt(values):
    return np.array2string(np.asarray(values), precision=6, suppress_small=True)


def joint_id(model, joint_name):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if idx < 0:
        raise ValueError(f'Missing joint {joint_name!r} in the selected MJCF')
    return idx


def parse_pose(model, assignments, keyframe=-1):
    pose = {k: v for k, v in INITIAL_QPOS.items()
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, k) >= 0}
    if keyframe >= 0:
        for j in range(model.njnt):
            if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                pose[name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = float(model.key_qpos[keyframe, model.jnt_qposadr[j]])
    for assignment in assignments:
        if '=' not in assignment:
            raise ValueError(f'Use --set-q joint_name=value_rad, got {assignment!r}')
        key, value = assignment.split('=', 1)
        pose[key] = float(value)
    for key, value in pose.items():
        j = joint_id(model, key)
        if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f'{key}: --set-q only supports hinge/slide joints')
        if not np.isfinite(value):
            raise ValueError(f'{key}: initial q must be finite')
        if model.jnt_limited[j] and not model.jnt_range[j, 0] <= value <= model.jnt_range[j, 1]:
            raise ValueError(f'{key}: {value} exceeds joint limits {model.jnt_range[j]}')
    return pose


def reset(model, data, pose, hold=False, keyframe=-1, torques=None):
    if keyframe >= 0:
        mujoco.mj_resetDataKeyframe(model, data, keyframe)
    else:
        mujoco.mj_resetData(model, data)
    for key, value in pose.items():
        j = joint_id(model, key)
        data.qpos[model.jnt_qposadr[j]] = value
    # Zero ctrl is not passive for a position servo at nonzero q. Disable
    # actuation entirely unless explicitly using the existing position servos.
    if torques is not None:
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        mapped = set()
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError('Torque mode requires direct joint motors')
            j = int(model.actuator_trnid[a, 0])
            jname = name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f'{jname}: torque mode requires a hinge joint')
            if not (np.isclose(model.actuator_gainprm[a,0],1.) and
                    np.allclose(model.actuator_biasprm[a],0.) and
                    np.allclose(model.actuator_gear[a],[1.,0.,0.,0.,0.,0.])):
                raise ValueError(f'{jname}: expected gear=1 motor with ctrl in N m')
            value = float(torques.get(jname,0.))
            if not np.isfinite(value):
                raise ValueError(f'{jname}: torque must be finite')
            if model.actuator_ctrllimited[a] and not model.actuator_ctrlrange[a,0] <= value <= model.actuator_ctrlrange[a,1]:
                raise ValueError(f'{jname}: torque {value} N m exceeds configured bounds; not clipped')
            data.ctrl[a] = value
            mapped.add(jname)
        missing = set(torques)-mapped
        if missing:
            raise ValueError(f'No direct torque actuator for joints {sorted(missing)}')
    elif hold:
        model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError('Hold mode expects joint position actuators')
            j = int(model.actuator_trnid[a, 0])
            # Existing position servos have fixed gain kp and bias -kp*q.
            if not np.isclose(model.actuator_gainprm[a, 0], -model.actuator_biasprm[a, 1]):
                raise ValueError(f'Actuator {a} is not recognized as a position servo')
            data.ctrl[a] = data.qpos[model.jnt_qposadr[j]]
    else:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    mujoco.mj_forward(model, data)


def contact_lines(model, data):
    lines = []
    for i, c in enumerate(data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        lines.append(
            f'  {i}: {name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)} / '
            f'{name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)}; pos={fmt(c.pos)}; '
            f'normal geom1->geom2={fmt(c.frame[:3])}; signed distance={c.dist:.7g} m; '
            f'penetration={max(0., -c.dist):.7g} m; active={c.efc_address >= 0}; '
            f'contact-frame wrench={fmt(force)}')
    return lines or ['  none']


def configuration_report(model, data, model_path, pose, hold):
    lines = [f'MuJoCo {mujoco.__version__}', f'MJCF: {model_path}',
             f'Mesh assets: {ASSETS / "REU_Snake/Mesh/ReU_Mesh.STL"}',
             f'timestep={model.opt.timestep:g} s; gravity={fmt(model.opt.gravity)} m/s^2',
             f'nbody={model.nbody}, ngeom={model.ngeom}, njnt={model.njnt}, '
             f'nq={model.nq}, nv={model.nv}, nu={model.nu}',
             f'Initial pose by name={pose}; initial root free qpos={fmt(data.qpos[:7])}',
             f'Actuation: {"DISABLED (truly passive)" if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION) else ("existing position servo hold" if hold else "direct joint torque motors enabled; ctrl in N m")}',
             f'Initial actuator ctrl={fmt(data.ctrl)}',
             'Quaternion order: w,x,y,z. All world positions/axes below are in MuJoCo world coordinates.',
             '\nBODY HIERARCHY / WORLD POSES AT INITIALIZATION']
    for b in range(1, model.nbody):
        parent = name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[b])
        bname = name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        js = range(int(model.body_jntadr[b]), int(model.body_jntadr[b]+model.body_jntnum[b]))
        joints = [name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in js]
        lines.append(f'  {bname}; parent={parent}; joints={joints}; mass={model.body_mass[b]:.6g} kg; '
                     f'world_position={fmt(data.xpos[b])}; world_quat={fmt(data.xquat[b])}')
    lines.append('\nJOINTS (joint anchor/axis in owning body and world)')
    widths = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
              mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
    for j in range(model.njnt):
        q = int(model.jnt_qposadr[j])
        dof = int(model.jnt_dofadr[j])
        jt = mujoco.mjtJoint(int(model.jnt_type[j]))
        acts = [name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)
                if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[a, 0] == j]
        axis = 'n/a (free/ball joint)' if widths[jt] > 1 else f'local={fmt(model.jnt_axis[j])}; world={fmt(data.xaxis[j])}'
        limits = fmt(model.jnt_range[j]) if model.jnt_limited[j] else 'unlimited'
        lines.append(f'  {name(model,mujoco.mjtObj.mjOBJ_JOINT,j)}; body='
                     f'{name(model,mujoco.mjtObj.mjOBJ_BODY,model.jnt_bodyid[j])}; type={jt.name}; '
                     f'axis={axis}; anchor_local={fmt(model.jnt_pos[j])}; '
                     f'anchor_world={fmt(data.xanchor[j])}; qpos={fmt(data.qpos[q:q+widths[jt]])}; '
                     f'limits={limits}; damping={model.dof_damping[dof]:g}; '
                     f'frictionloss={model.dof_frictionloss[dof]:g}; actuators={acts}')
    lines.append('\nACTUATORS')
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        lines.append(f'  {name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)} -> '
                     f'{name(model,mujoco.mjtObj.mjOBJ_JOINT,j)}; gain={fmt(model.actuator_gainprm[a,:3])}; '
                     f'bias={fmt(model.actuator_biasprm[a,:3])}; gear={fmt(model.actuator_gear[a])}; '
                     f'ctrlrange={fmt(model.actuator_ctrlrange[a])}; ctrl_limited={bool(model.actuator_ctrllimited[a])}; '
                     f'forcerange={fmt(model.actuator_forcerange[a])}; force_limited={bool(model.actuator_forcelimited[a])}')
    lines.append('\nGEOMS / COLLISION MASKS (transparency does not disable collision)')
    for g in range(model.ngeom):
        lines.append(f'  {name(model,mujoco.mjtObj.mjOBJ_GEOM,g)}; body='
                     f'{name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g])}; '
                     f'type={mujoco.mjtGeom(int(model.geom_type[g])).name}; group={model.geom_group[g]}; '
                     f'contype={model.geom_contype[g]}; conaffinity={model.geom_conaffinity[g]}; '
                     f'friction(sliding,torsional,rolling)={fmt(model.geom_friction[g])}; '
                     f'condim={model.geom_condim[g]}; world_position={fmt(data.geom_xpos[g])}')
    lines.append('\nEXPLICIT CONTACT PAIRS (5 entries: tangent1,tangent2,torsion,roll1,roll2)')
    for p in range(model.npair):
        lines.append(f'  {name(model,mujoco.mjtObj.mjOBJ_PAIR,p)}: '
                     f'{name(model,mujoco.mjtObj.mjOBJ_GEOM,model.pair_geom1[p])} / '
                     f'{name(model,mujoco.mjtObj.mjOBJ_GEOM,model.pair_geom2[p])}; '
                     f'condim={model.pair_dim[p]}; friction={fmt(model.pair_friction[p])}; '
                     f'solref={fmt(model.pair_solref[p])}; solimp={fmt(model.pair_solimp[p])}')
    lines.extend(['\nCONTACTS AT t=0'] + contact_lines(model, data))
    lines.extend(['\nCONFIGURATION FLAGS / ITEMS TO VERIFY',
                  '  Initial pose is the declared XML/keyframe pose, not a verified formed/preloaded anchor.',
                  '  Adjacent modules have fixed local Euler z=pi/2 rotations: world hinge axes alternate at zero hinge angles.',
                  '  Verify whether the physical snake uses alternating joint axes or is intended to be planar.',
                  '  Collision uses compiled contype/conaffinity and explicit pairs, not geom transparency or display group.',
                  '  No explicit contact exclusions: welded and parent-child filtering still applies by MuJoCo defaults.',
                  '  Body mass/inertia may be explicit or inferred from geoms; the compiled masses are listed above.',
                  '  Source launchers overwrite XML on import; actuator counts must match each selected robot variant.',
                  '  The source launchers turn gravity off; this baseline preserves XML gravity [0,0,-9.81].'])
    walls = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, key) for key in ('Wall_1', 'Wall_2')]
    if min(walls) >= 0:
        g1, g2 = walls
        gap = abs(data.geom_xpos[g1,0]-data.geom_xpos[g2,0])-model.geom_size[g1,0]-model.geom_size[g2,0]
        lines.append(f'  Actual inner wall gap={gap:.9g} m. Check the selected scene definition for intended spacing.')
    mind = min((c.dist for c in data.contact), default=0.)
    lines.append(f'  Max initial penetration={max(0.,-mind):.7g} m. No automatic contact/solver tuning applied.')
    robot_geoms = [g for g in range(model.ngeom) if name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]).startswith('mod_')]
    enabled = sum(bool((int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) or
                       (int(model.geom_contype[b]) & int(model.geom_conaffinity[a])))
                  for i,a in enumerate(robot_geoms) for b in robot_geoms[i+1:])
    lines.append(f'  Robot/robot geom pairs passing collision masks={enabled}; additional welded/parent-body filtering applies.')
    forces = []
    for i in range(data.ncon):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model,data,i,wrench)
        forces.append(wrench[0])
    lines.append(f'  Maximum initial contact normal force={max(forces,default=0.):.7g} N (solver result, not calibrated preload).')
    if max(forces,default=0.) > 100:
        lines.append('  FLAG: initial contact normal force exceeds 100 N; inspect mesh overlap and constraint stiffness before interpreting as physical preload.')
    return '\n'.join(lines)


def passive_check(model, data, seconds, output):
    q0, p0 = data.qpos.copy(), data.xpos.copy()
    body_ids = [b for b in range(1, model.nbody) if name(model,mujoco.mjtObj.mjOBJ_BODY,b).endswith('_body')]
    headings = ['time_s']+[f'qpos_{i}' for i in range(model.nq)]+[f'qvel_{i}' for i in range(model.nv)]
    headings += [f'actuator_force_{name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)}' for a in range(model.nu)]
    headings += [f'{name(model,mujoco.mjtObj.mjOBJ_BODY,b)}_{axis}_m' for b in body_ids for axis in 'xyz']
    headings += ['n_contacts','max_penetration_m']
    peak_velocity, peak_penetration, peak_force = 0., 0., 0.
    last_output = -np.inf
    steps = int(round(seconds/model.opt.timestep))
    finite, time_reset = True, False
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headings)
        for step in range(steps+1):
            mujoco.mj_forward(model, data)
            depth = max(0., -min((c.dist for c in data.contact), default=0.))
            peak_penetration = max(peak_penetration, depth)
            peak_velocity = max(peak_velocity, float(np.max(np.abs(data.qvel))))
            if model.nu:
                peak_force = max(peak_force, float(np.max(np.abs(data.actuator_force))))
            if data.time-last_output >= DT_OUTPUT-.5*model.opt.timestep or step == steps:
                writer.writerow([data.time,*data.qpos,*data.qvel,*data.actuator_force,
                                 *data.xpos[body_ids].ravel(),data.ncon,depth])
                last_output = data.time
            if step == steps:
                break
            tprev = data.time
            mujoco.mj_step(model, data)
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            time_reset = data.time < tprev
            if not finite or time_reset:
                break
    delta = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, delta, 1., q0, data.qpos)
    warnings = {mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number}
    mode = 'passive (actuation disabled)' if model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION) else 'actuation enabled'
    lines = [f'\nSHORT DYNAMICS CHECK ({mode}): {seconds:g} s requested, {data.time:.6g} s reached',
             f'  finite={finite}; unexpected time reset={time_reset}; MuJoCo warnings={warnings}',
             f'  peak |qvel|={peak_velocity:.7g} (mixed m/s and rad/s); final={np.max(np.abs(data.qvel)):.7g}',
             f'  peak actuator force={peak_force:.7g}; peak penetration={peak_penetration:.7g} m',
             f'  maximum link-origin displacement={np.linalg.norm(data.xpos[body_ids]-p0[body_ids],axis=1).max():.7g} m',
             f'  generalized position difference (free translation/rotation then hinges)={fmt(delta)}',
             '  A finite trajectory is numerical stability, not proof of a static anchoring state.']
    if not finite or time_reset or warnings:
        lines.append('  STATUS: numerical warning/failure; inspect output before using the model.')
    else:
        lines.append('  STATUS: numerically stable over the checked interval.')
    lines.append('  Gravity and initial contact penetration can cause settling; no force calibration performed.')
    return '\n'.join(lines)


def kinematic_check(model, pose, output, keyframe=-1):
    """Finite-displacement verification of positive axis and descendant motion."""
    data = mujoco.MjData(model)
    reset(model, data, pose, keyframe=keyframe)
    p0, q0 = data.xpos.copy(), data.qpos.copy()
    rows, lines = [], ['\nJOINT-BY-JOINT KINEMATIC CHECK (+0.05 rad, no dynamics/servo command)']
    joints = sorted({int(model.actuator_trnid[a,0]) for a in range(model.nu)
                     if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT})
    for j in joints:
        jname = name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            lines.append(f'  {jname}: skipped, not a hinge')
            continue
        base = q0[model.jnt_qposadr[j]]
        if model.jnt_limited[j] and base+JOINT_TEST_AMPLITUDE > model.jnt_range[j,1]:
            lines.append(f'  {jname}: skipped, insufficient positive joint range')
            continue
        reset(model, data, pose, keyframe=keyframe)
        axis, anchor = data.xaxis[j].copy(), data.xanchor[j].copy()
        # Exact right-hand rotation for descendant body origins.
        angle = JOINT_TEST_AMPLITUDE
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
        data.qpos[model.jnt_qposadr[j]] += angle
        mujoco.mj_forward(model,data)
        descendants = []
        max_error, max_upstream = 0., 0.
        owner = int(model.jnt_bodyid[j])
        for b in range(1, model.nbody):
            ancestor = b
            while ancestor and ancestor != owner:
                ancestor = int(model.body_parentid[ancestor])
            downstream = ancestor == owner
            expected = anchor+R@(p0[b]-anchor) if downstream else p0[b]
            err = float(np.linalg.norm(data.xpos[b]-expected))
            max_error = max(max_error,err)
            movement = float(np.linalg.norm(data.xpos[b]-p0[b]))
            if not downstream:
                max_upstream = max(max_upstream,movement)
            if name(model,mujoco.mjtObj.mjOBJ_BODY,b).endswith('_body') and downstream:
                descendants.append(name(model,mujoco.mjtObj.mjOBJ_BODY,b))
            rows.append({'joint':jname,'body':name(model,mujoco.mjtObj.mjOBJ_BODY,b),
                         'positive_displacement_rad':angle,'descendant':downstream,
                         'body_movement_m':movement,'right_hand_rotation_error_m':err})
        lines.append(f'  {jname}: world_axis={fmt(axis)}; links={descendants}; '
                     f'max right-hand rotation error={max_error:.3e} m; upstream movement={max_upstream:.3e} m')
    if rows:
        with output.open('w',newline='',encoding='utf-8') as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return '\n'.join(lines)


def view_options():
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    return opt


def configure_display(model):
    # Display-only sizes, independent of dynamics/contact solver parameters.
    model.vis.scale.jointlength = .20
    model.vis.scale.jointwidth = .02
    model.vis.scale.framelength = .20
    model.vis.scale.framewidth = .01
    model.vis.scale.forcewidth = .01
    model.vis.scale.contactheight = .015
    model.vis.scale.contactwidth = .01
    model.vis.scale.com = .08
    model.vis.map.force = .002
    model.vis.global_.offwidth = 1000
    model.vis.global_.offheight = 720
    for g in range(model.ngeom):
        if name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[g]).startswith('mod_'):
            model.geom_group[g] = 0 if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else 1


def scale_force_display(model, data):
    largest = 0.
    for i in range(data.ncon):
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model,data,i,wrench)
        largest = max(largest,float(np.linalg.norm(wrench[:3])))
    # Same proportional visual scale for all arrows. Physical values unchanged.
    model.vis.map.force = min(.002, .04/largest) if largest > 0 else .002


def camera(model, data):
    cam = mujoco.MjvCamera()
    ids = [b for b in range(1,model.nbody) if name(model,mujoco.mjtObj.mjOBJ_BODY,b).endswith('_body')]
    points = data.xpos[ids]
    cam.lookat[:] = (points.min(axis=0)+points.max(axis=0))/2
    cam.distance = max(.45, 2.0*float(np.linalg.norm(np.ptp(points,axis=0)))+.15)
    cam.azimuth, cam.elevation = 90., -8.
    return cam


def snapshot(model, data, output):
    from PIL import Image
    with mujoco.Renderer(model, height=720, width=1000) as renderer:
        renderer.update_scene(data,camera=camera(model,data),scene_option=view_options())
        Image.fromarray(renderer.render()).save(output)


def show_viewer(model, data, pose, args):
    import mujoco.viewer
    actions = queue.SimpleQueue()
    # Use keys not used as simulation controls by the custom loop, except SPACE.
    def key_callback(key):
        actions.put(key)
    paused = args.paused
    joints = sorted({int(model.actuator_trnid[a,0]) for a in range(model.nu)
                     if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT})
    valid_joints = [j for j in joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and
                    (not model.jnt_limited[j] or
                     data.qpos[model.jnt_qposadr[j]]+JOINT_TEST_AMPLITUDE <= model.jnt_range[j,1])]
    nominal = data.qpos.copy()
    test_elapsed = 0.
    visual_mode = 0
    viewer_options = view_options()
    with mujoco.viewer.launch_passive(model,data,key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = camera(model,data).lookat
            viewer.cam.distance = camera(model,data).distance
            viewer.cam.azimuth, viewer.cam.elevation = camera(model,data).azimuth, camera(model,data).elevation
            viewer.opt.flags[:] = viewer_options.flags
        print('VIEWER_OPEN: SPACE pause/play; R reset; F body frames; J axes; C contact points; '
              'N contact forces; M COM; V all/mesh/primitive geoms',flush=True)
        while viewer.is_running():
            started = time.perf_counter()
            with viewer.lock():
                while not actions.empty():
                    key = actions.get()
                    if key == 32:
                        paused = not paused
                    elif key in (ord('R'),ord('r')):
                        reset(model,data,pose,args.hold_position,args.keyframe_id,args.torque_commands)
                        nominal = data.qpos.copy(); test_elapsed = 0.
                    elif key in (ord('F'),ord('f')):
                        viewer.opt.frame = (mujoco.mjtFrame.mjFRAME_BODY if viewer.opt.frame == mujoco.mjtFrame.mjFRAME_NONE
                                            else mujoco.mjtFrame.mjFRAME_NONE)
                    elif key in (ord('V'),ord('v')):
                        visual_mode = (visual_mode+1)%3
                        viewer.opt.geomgroup[0] = visual_mode != 2
                        viewer.opt.geomgroup[1] = visual_mode != 1
                    else:
                        flags = {'J':mujoco.mjtVisFlag.mjVIS_JOINT,'C':mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
                                 'N':mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,'M':mujoco.mjtVisFlag.mjVIS_COM}
                        char = chr(key).upper() if 0 <= key < 128 else ''
                        if char in flags:
                            flag = flags[char]
                            viewer.opt.flags[flag] = not viewer.opt.flags[flag]
                if not paused:
                    if args.joint_test and valid_joints:
                        # Positive excursion -> return -> next joint. Frozen
                        # base, forward kinematics only; not a dynamic hold.
                        segment = int(test_elapsed//4.) % len(valid_joints)
                        phase = (test_elapsed%4.)/4.
                        j = valid_joints[segment]
                        data.qpos[:] = nominal
                        data.qpos[model.jnt_qposadr[j]] += JOINT_TEST_AMPLITUDE*np.sin(np.pi*phase)**2
                        data.qvel[:] = 0
                        test_elapsed += 1/60
                        mujoco.mj_forward(model,data)
                        label = f'KINEMATIC ONLY: {name(model,mujoco.mjtObj.mjOBJ_JOINT,j)} +0.05 rad, then return'
                    else:
                        for _ in range(max(1,round((1/60)/model.opt.timestep))):
                            mujoco.mj_step(model,data)
                        mujoco.mj_forward(model,data)
                        mode = 'TORQUE CONTROL [N m]' if args.torque_control else ('POSITION HOLD' if args.hold_position else 'PASSIVE')
                        label = f'{mode} t={data.time:.2f} s'
                else:
                    label = ('PAUSED: torque motors enabled; ctrl sliders in N m' if args.torque_control
                             else 'PAUSED: inspect declared initial configuration')
            if hasattr(viewer,'set_texts'):
                viewer.set_texts((int(mujoco.mjtFontScale.mjFONTSCALE_100),
                                  int(mujoco.mjtGridPos.mjGRID_BOTTOMLEFT),
                                  label,'SPACE play/pause | R reset | F frames | V geoms'))
            viewer.sync()
            time.sleep(max(0.,1/60-(time.perf_counter()-started)))
    print('VIEWER_CLOSED',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--output',type=Path,default=HERE.parent/'outputs'/'diagnostics')
    parser.add_argument('--set-q',action='append',default=[],metavar='JOINT=RAD')
    parser.add_argument('--keyframe',help='named XML keyframe for initial qpos/ctrl')
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--paused',action='store_true')
    parser.add_argument('--snapshot',action='store_true')
    parser.add_argument('--joint-test',action='store_true')
    actuation = parser.add_mutually_exclusive_group()
    actuation.add_argument('--hold-position',action='store_true',help='enable existing XML position servos explicitly')
    actuation.add_argument('--torque-control',action='store_true',help='enable direct gear=1 joint torque motors')
    parser.add_argument('--torque',action='append',default=[],metavar='JOINT=NM')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 0 < args.seconds <= 10:
        parser.error('--seconds must be >0 and <=10 for this baseline check')
    model_path = args.model.resolve()
    if not model_path.is_file():
        parser.error(f'MJCF not found: {model_path}')
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except (ValueError, OSError) as error:
        parser.error(f'MuJoCo XML load failed. Check include and mesh paths relative to {model_path}: {error}')
    configure_display(model)
    args.keyframe_id = -1
    if args.keyframe:
        args.keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
        if args.keyframe_id < 0:
            parser.error(f'Missing keyframe {args.keyframe!r} in {model_path}')
    try:
        pose = parse_pose(model,args.set_q,args.keyframe_id)
        args.torque_commands = {} if args.torque_control else None
        if args.torque and not args.torque_control:
            raise ValueError('--torque requires --torque-control')
        for assignment in args.torque:
            key,value = assignment.split('=',1)
            args.torque_commands[key] = float(value)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=True)
    data = mujoco.MjData(model)
    try:
        reset(model,data,pose,args.hold_position,args.keyframe_id,args.torque_commands)
    except ValueError as error:
        parser.error(str(error))
    scale_force_display(model,data)
    report = configuration_report(model,data,model_path,pose,args.hold_position)
    if args.snapshot:
        snapshot(model,data,args.output/'initial_configuration.png')
    report += '\n'+passive_check(model,data,args.seconds,args.output/'short_simulation_trace.csv')
    if args.snapshot:
        snapshot(model,data,args.output/'after_short_simulation.png')
    report += '\n'+kinematic_check(model,pose,args.output/'joint_direction_check.csv',args.keyframe_id)
    report_path = args.output/'mujoco_configuration_report.txt'
    report_path.write_text(report+'\n',encoding='utf-8')
    print(report,flush=True)
    print(f'REPORT_SAVED: {report_path}',flush=True)
    if not args.headless:
        reset(model,data,pose,args.hold_position,args.keyframe_id,args.torque_commands)
        show_viewer(model,data,pose,args)


if __name__ == '__main__':
    main()
