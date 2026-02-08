import gymnasium as gym
import gymnasium_robotics
import mujoco

env = gym.make("AdroitHandRelocate-v1")
model = env.unwrapped.model

print(f"Total Geoms: {model.ngeom}")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    body_id = model.geom_bodyid[i]
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    print(f"Geom {i}: {name} (Body: {body_name}) Type: {model.geom_type[i]}")

print(f"\nTotal Bodies: {model.nbody}")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"Body {i}: {name}")
