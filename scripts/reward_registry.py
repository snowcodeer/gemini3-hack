import numpy as np

def reward_v1(env, info):
    """Original basic reward (M0-M4)."""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body("Object").id].copy()
    dist_to_target = np.linalg.norm(obj_pos - env.target_pos)
    success = dist_to_target < 0.1
    is_lifted = obj_pos[2] > 0.08
    n_contacts = data.ncon
    
    reward = -0.1
    reward += min(n_contacts, 5) * 0.1
    if is_lifted:
        reward += 2.0
        reward += (1.0 - np.clip(dist_to_target, 0, 1)) * 5.0
    if success:
        reward += 50.0
    return reward, {"success": success, "lifted": is_lifted, "dropped": False}

def reward_v2(env, info):
    """High-reward logic (Unstable peak)."""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body("Object").id].copy()
    dist_to_target = np.linalg.norm(obj_pos - env.target_pos)
    success = dist_to_target < 0.1
    is_lifted = obj_pos[2] > 0.08
    n_contacts = data.ncon
    
    reward = -0.1
    reward += min(n_contacts, 5) * 0.1
    if is_lifted:
        reward += 5.0
        reward += (1.0 - np.clip(dist_to_target, 0, 1)) * 10.0
    if success:
        reward += 500.0
    
    dropped = env.ever_lifted and not is_lifted
    if dropped:
        reward -= 20.0
        
    return reward, {"success": success, "lifted": is_lifted, "dropped": dropped}

def reward_v3(env, info):
    """Stabilized reward (Default)."""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body("Object").id].copy()
    dist_to_target = np.linalg.norm(obj_pos - env.target_pos)
    success = dist_to_target < 0.1
    is_lifted = obj_pos[2] > 0.08
    n_contacts = data.ncon
    
    reward = env.reward_config['time_penalty']
    reward += min(n_contacts, 5) * env.reward_config['contact_scale']
    
    if is_lifted:
        reward += env.reward_config['lift_bonus']
        reward += (1.0 - np.clip(dist_to_target, 0, 1)) * env.reward_config['transport_scale']
    
    if dist_to_target < 0.2:
        reward += (0.2 - dist_to_target) * env.reward_config.get('smooth_near_scale', 50.0)
    
    if success:
        reward += env.reward_config['success_bonus']
        
    dropped = env.ever_lifted and not is_lifted
    if dropped:
        reward += env.reward_config['drop_penalty']
        
    return reward, {"success": success, "lifted": is_lifted, "dropped": dropped}

REWARD_REGISTRY = {
    "v1": reward_v1,
    "v2": reward_v2,
    "v3": reward_v3,
    "stable": reward_v3
}
