def reward_components(config, previous_distance, distance, status):
    scheme = config.get("reward", {})
    mode = scheme.get("mode", "R1")
    collision = status in {"collision", "self_collision"}
    success = status == "success"
    components = dict(distance=0., progress=0., success_bonus=0., collision_penalty=0., safety_penalty=0.)
    if mode == "R1":
        components["distance"] = -distance
        if collision:
            components["collision_penalty"] = -config["failure_penalty"]
        elif status in {"joint_limit", "out_of_bounds", "torque_limit"}:
            components["safety_penalty"] = -config["failure_penalty"]
    elif mode in {"R2", "R3"}:
        components["progress"] = scheme["progress_weight"] * (previous_distance - distance)
        if mode == "R3":
            components["success_bonus"] = scheme["success_bonus"] if success else 0.
            components["collision_penalty"] = -scheme["collision_penalty"] if collision else 0.
    else:
        raise ValueError(f"Unknown reward mode {mode}")
    return components
