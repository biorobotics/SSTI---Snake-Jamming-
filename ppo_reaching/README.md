# Planar snake reaching with PPO

This study uses the archived fixed-pivot 12-module ReU model with six active joints. The policy commands six absolute joint positions. MuJoCo runs at 500 Hz, with a 50 Hz position-control update and zero gravity. The reaching benchmark uses a 2 cm goal radius, STL convex contacts, and a 4 N·m limit on both actuator torque and the Newton–Euler estimate.

## Generate maps and evaluate PPO

Run these commands from the repository root. Use a Python 3.10 virtual environment for the pinned reaching dependencies. The repository's root requirements cover the anchoring simulator separately.

```sh
python -m pip install -r ppo_reaching/requirements.txt
python ppo_reaching/map_generator.py --out results/reaching/maps.json --count 12 --seed 20261115
python ppo_reaching/evaluate_maps.py --maps results/reaching/maps.json --out results/reaching/eval
```

The generator writes the reference map and two groups of random variants. It fixes the start, perturbs the goal and the obstacle dimensions and position, and retains cases with collision-free start and goal configurations where the straight joint interpolation is blocked. A valid goal configuration establishes geometric reachability of that pose. The rollout tests whether the controller can execute a path to it.

The evaluation runs the frozen seed-2026 policy, records its strict no-contact success rate by group, and writes CSV metrics and NPZ trajectory traces. CPU inference timing excludes loading the checkpoint and stepping MuJoCo. Add `--render` to produce a reference MP4 and GIF. On headless Linux with EGL support, set `MUJOCO_GL=egl` for rendering. The bundled model reproduces the reference single-obstacle success at 3.62 s in the reference runtime. Load only trusted Stable-Baselines3 checkpoint files.

## Train the reference policy

```sh
python ppo_reaching/clearance_ablation/study.py --out results/reaching/train --group A1 --seed 2026 --budget 262144 --episodes 1 --stl-reward-ablation --obstacle-contact-cost-per-s 8 --torque-limit-nm 4 --save-first-success --initial ppo_reaching/checkpoints/unobstructed_initial.zip
```

Training charges obstacle contact by duration and evaluates at fixed intervals under strict collision termination. The default reward uses Euclidean distance progress, a goal bonus, two command regularizers, contact costs, and other failure penalties. Training uses the fixed reference layout. The generated random layouts are used for held-out evaluation.

## Model and asset sources

- The reaching MJCF in `robot_env/assets/archived_base.xml` is the archived BINSAT corridor scene template. The reaching environment rebuilds its obstacles and position actuators at runtime. It retains the model's inertial parameters and joint damping, including the 3.18 damping in the archived planar joints.
- The reaching environment uses the identical STL already tracked in this repository at `mujoco_sim/assets/REU_Snake/Mesh/ReU_Mesh.STL`.
- The velocity and effort bounds in `configs/joint_limits.json` reproduce the active module-joint limits from the [BINSAT ReU URDF](https://github.com/bshirose/hierarchical_snake_planner/blob/e1a824ad5ebe32cce661e1e76844c90873fe876f/traj_opt/REU_snake_v8.urdf). The pivot is subject to the experiment torque cap and has no matching source velocity bound.
- [SnakeRL](https://github.com/biorobotics/snakerl/blob/0eb071f3a8b65e58fb79a6e2b4166e20028e4b80/snakerl/robot.py) builds a separate simulation model based on the `snakelib-ros2` description. Its default joint damping is 0.1. The reaching checkpoint was trained on the archived MJCF documented above.
