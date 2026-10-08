# Packaged baseline validation

Validated on 2026-10-08 using Windows, Python 3.13.13, MuJoCo 3.9.0,
NumPy 2.4.6, and Pillow 12.2.0.

## Executed checks

- All 17 packaged MJCF/XML files compiled with MuJoCo and resolved the original
  STL and all include references.
- Four focused unittest checks passed: XML loading, generator/reference
  preservation and physical-equivalence reconstruction, collision/torque
  mapping, and proxy contact-force aggregation against the engine body buffer.
- The generator rebuilt all seven current/reference q=0.65 scenes in a separate
  temporary asset directory. Geometry, inertia, joints, actuation, collision
  masks, contact parameters, and initial keyframes matched the packaged scenes
  to numerical tolerance; reference files were not changed.
- All four direct torque commands [0.2,-0.3,0.1,-0.15] N m matched actuator
  forces and generalized hinge torques in both current collision families.
- The convex profile produced no robot-cylinder contacts. The proxy profile
  produced no STL, mod_i_link, or mod_i_out contacts, including explicit pairs.
- Both current five-module torque scenes ran for 3 seconds at zero command
  without MuJoCo warnings and were rendered from the portable package.
- Python syntax and English-only text checks passed. No workstation-specific
  paths are required by the source code.

## Recorded contact diagnostics

| Current profile | Maximum penetration during 3 s | Fifth-module walls seen |
|---|---:|---|
| STL-only convex | 0.811 mm | Wall_2 only |
| mod_i_in-only cylinder proxy | 1.552 mm | Wall_2 only |

Multiple solver points are expected in the cylinder/box contacts. The report
utility sums contact-frame vectors in world coordinates and retains each
point's moment arm; it does not average point forces.

The +pi/4 fifth-joint candidate was selected over -pi/4. In the original
mesh-based selection check their initial fifth-module wall penetrations were
approximately 0.769 mm and 51.989 mm, respectively. Both touched only Wall_2,
so wall-count alone was insufficient; severe wall crossing rejected the
negative candidate.

## Interpretation limits

These checks establish loadability, reproducibility, correct control units,
collision isolation, and consistent force bookkeeping. They do not establish
physical calibration, accurate preload/contact forces, or a static anchor
that maintains the initial angles under zero torque.

The analytical wall-spacing radius and CAD collision radius differ, inherited
joint axes alternate, and the source geometry can start in penetration.
Initialization contact forces can be high. The source families' inertias also
differ, so their comparison is not a pure collision-shape study.
Gravity/support-plane alignment with the physical experiment remains pending.

Generated detailed reports, traces, and images are available under the ignored
outputs directory after running the README commands; they are not committed as
experimental data.
