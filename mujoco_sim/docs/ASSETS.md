# MuJoCo asset inventory and conventions

The original assets were recovered from the local Research--summer project.
All include and mesh references remain relative to the asset directory.
No absolute workstation paths are required by the packaged scripts.

## Primary generated assets

| Scene | Included robot | Intended use |
|---|---|---|
| xml_anchor_q065_5modules_torque_proxy.xml | REU_snake_robot_torque_proxy.xml | Default: five modules, direct torque, in-cylinder collision only |
| xml_anchor_q065_5modules_torque_convex.xml | REU_snake_robot_stable_5modules_torque.xml | Five modules, direct torque, STL-only collision |
| xml_anchor_q065_5modules_plus45.xml | REU_snake_robot_stable_5modules.xml | Previous mixed-actuation fifth-joint selection reference |
| xml_anchor_q065_5modules_minus45.xml | REU_snake_robot_stable_5modules.xml | Rejected negative fifth-angle candidate |
| xml_anchor_q065_left.xml | REU_snake_robot_stable_1.xml | Previous four-module positive-deflection reference |
| xml_anchor_q065_right.xml | REU_snake_robot_stable_1.xml | Previous four-module negative-deflection reference |
| xml_anchor_q065_5modules_torque_proxy_all_cylinders.xml | REU_snake_robot_torque_proxy_all_cylinders.xml | Archived all-cylinder collision profile |

The four-module reference has two position servos and a central torque motor.
The appended fifth module copies the last module's local geometry and inertia,
adds joint_4 and its actuator/sensor, and leaves the first four modules unchanged.
Active torque variants replace all four actuators with direct gear=1 motors.
The reference source files are not overwritten by generation.

## Preserved original assets

- REU_snake_robot.xml: original five-module STL plus larger cylindrical proxies.
- REU_snake_robot_stable_1.xml: original four-module mesh-based variant.
- REU_snake_Downstream.xml: original 16-module/15-hinge long snake.
- xml_anchor_scene.xml: original five-module anchoring workspace.
- xml_anchor_scene_box.xml: alternative wall/contact configuration; the name
  does not imply rectangular robot collision proxies.
- xml_wrench_sampleing.xml: original long-snake wrench-sampling scene.
- REU_Snake/Mesh/ReU_Mesh.STL: preserved mesh, with the original 0.0254 scale.

The duplicate original REU_snake_xml.xml was identical to
REU_snake_Downstream.xml and is not duplicated.

## Coordinate and force conventions

The two walls' inner surfaces are normal to world +/-x. At the current initial
configuration the principal contacts are module 0/left wall, module 2/right
wall, and module 4/left wall. Their inward normal directions on the snake are
+x, -x, and +x. In-plane shear can be represented along world z; y is the
other wall-tangential direction. Actual solver contact normals can have small
perturbations introduced by multi-contact collision detection.

mujoco.mj_contactForce returns a contact-frame wrench acting on geom2.
contact.frame stores its world basis as rows, so transpose that matrix to
transform to world. Negate the wrench when reporting its action on geom1.
The proxy utility in scripts/contact_wrench.py aggregates all points of the
same robot-geom/wall pair:

    F_world = sum_k F_world_k
    M_world_about_reference =
        sum_k [(p_k - p_reference) cross F_world_k + M_world_k]

Forces are summed, not averaged; vector magnitudes are not summed as scalars.
Force direction is F_world / norm(F_world) only when the norm is nonzero.
Moments use the robot geom center as an explicit reference point.
Contact-induced moment is distinct from the independent actuator joint torque.
Condim=3 supplies normal and two tangential force components; different
point-force arms can still generate a resultant contact-region moment.

## Physical/numerical assumptions

- Gravity is [0,0,-9.81] m/s^2, with a floor at world z=0.
- The analytical wall-spacing radius is 0.0167 m; the original active proxy
  cylinder radius is 0.017018 m. This mismatch is recorded, not calibrated away.
- Wall sliding friction is initially 1.35; wall pair solref=[0.004,1] and
  solimp=[0.99,0.99,0.001]. These inherited values are not material estimates.
- Joint damping/frictionloss and inertia are preserved from each source asset.
- The source body hierarchy has alternating joint axes. It is not silently
  converted into an ideal planar two-body model.
- Robot STL/cylinder shapes remain available for display after disabling their
  collisions. Transparency and display-group toggles are not collision controls.
- Initial angles and wall spacing do not prove a calibrated preload state.

The previous GT parameter-sampling experiments, raw measurements, and large
CSV outputs are intentionally outside this asset package.
