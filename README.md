# SSTI Snake Jamming / Anchoring

Portable MuJoCo assets and a minimal, inspectable simulation baseline. Existing
robot meshes and MJCF hierarchy are reused; reference and previous variants are
retained. No experimental datasets, parameter fitting, RL, or IMU processing are
included in this asset upload.

## Quick start

From the repository root, with Python 3.11+ and an OpenGL-capable desktop:

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Linux/macOS: source .venv/bin/activate
    python -m pip install -r requirements.txt
    python mujoco_sim/scripts/run_q065_torque.py

The official MuJoCo viewer starts paused at the declared configuration. The
default is the **five-module, in-cylinder-only collision-proxy model**, with
four direct torque motors. Use --collision convex for the STL-only variant.
On macOS, use MuJoCo's mjpython launcher for interactive viewer commands.
The current package was validated on Windows with Python 3.13 and MuJoCo 3.9.0.

    python mujoco_sim/scripts/run_q065_torque.py --collision convex
    python mujoco_sim/scripts/run_q065_torque.py --headless --snapshot
    python mujoco_sim/scripts/run_q065_torque.py --torque joint_2=0.2

The last command initializes a 0.2 N m input; Space starts playback. Viewer
actuator Ctrl sliders also accept torque in N m. Gear=1 gives
joint actuator torque = ctrl within the declared +/-10 N m limits.
These are provisional simulation limits, not measured hardware capabilities.

## Current configuration

- Five modules mod_0_body through mod_4_body, connected by four hinges.
- Initial central deflection joint_2 = 0.65 rad.
- Fifth-module connection joint_4 = +pi/4; the -pi/4 candidate is retained
  for inspection but rejected because it starts with severe wall crossing.
- Wall **face-to-face** gap:
  d = 2*r + L*cos((pi-q)/2) = 0.066001427036 m,
  with L=0.1021 m, r=0.0167 m, and q=0.65 rad.
- q here is the chosen central-axis deflection, not an experimentally
  calibrated conversion from HEBI feedback.
- The base is tilted by q/2, centered in the corridor, and given 5 mm floor
  clearance. Initial qpos is a named XML keyframe.

These angles are **initial states**, not setpoints maintained by a position
controller. Zero torque may lead to settling and loss/re-establishment of contact.

## Collision variants

| Variant | Robot collision geometry | Retained visual geometry |
|---|---|---|
| Default proxy | Only mod_i_in cylinders | STL, mod_i_link, and mod_i_out are visual-only |
| Convex | STL mesh convex hulls only | Robot cylinders are visual-only |
| Archived all-cylinder proxy | in/link/out cylinders | STL is visual-only |

The original collision-proxy assets use cylinders, not rectangular robot
boxes; only environmental walls are boxes. Explicit contact pairs are removed
for disabled geoms, so they cannot bypass zero collision masks.
The reference source families have different inertias; comparisons between
them do not isolate collision shape alone.

## Viewer controls

Space: play/pause; R: restore initial keyframe and torque commands;
F: body frames; J: joint axes; C: contact points; N: force arrows; M: center of
mass; V: all / mesh / primitive visibility. Visibility does not change physics.
Force arrows use a proportional display scale and do not alter contact forces.

## Regenerate and verify

    python mujoco_sim/scripts/generate_assets.py
    python mujoco_sim/scripts/verify_torque_assets.py
    python mujoco_sim/scripts/verify_exclusive_collisions.py
    python mujoco_sim/scripts/check_fifth_contacts.py
    python mujoco_sim/scripts/report_contact_wrenches.py
    python -m unittest discover -s mujoco_sim/tests -v

The generator preserves reference files and rebuilds the generated variants.
Use --output-assets /path/to/a/new/assets/folder for a separate reconstruction.
Changing analytical L/r changes wall spacing, not the preserved CAD mesh scale.

## Layout and limitations

mujoco_sim/assets/ contains MJCF and STL; scripts/ contains launch,
generation, and verification entry points; tests/ contains focused checks.
Generated diagnostics are written to ignored mujoco_sim/outputs/.

The baseline loads and runs, but contact penetration and high initialization
forces remain possible. Gravity and the support plane are inherited from the
archived simulation and are not yet aligned/calibrated to the physical acrylic
support. A short finite run is not proof of validated anchoring or Real2Sim
agreement. See [the asset inventory and conventions](mujoco_sim/docs/ASSETS.md)
and [the packaged validation record](mujoco_sim/docs/VALIDATION.md).
