import gymnasium as gym
import gymnasium_robotics
import numpy as np
import cv2
import mujoco

def debug_axes():
    env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    env.reset()
    
    # Hide table for clarity
    model = env.unwrapped.model
    table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if table_body_id != -1:
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == table_body_id:
                model.geom_rgba[i, 3] = 0.0
                model.geom_pos[i] = [10, 10, 10]

    output_path = "outputs/debug_axes.mp4"
    frames = []
    
    # Define moves: (Name, delta_qpos)
    # qpos[0:3] is x, y, z
    moves = [
        ("Center (0,0,0)", np.array([0.0, 0.0, 0.0])),
        ("Plus X (+0.2,0,0)", np.array([0.2, 0.0, 0.0])),
        ("Minus X (-0.2,0,0)", np.array([-0.2, 0.0, 0.0])),
        ("Plus Y (0,+0.2,0)", np.array([0.0, 0.2, 0.0])),
        ("Minus Y (0,-0.2,0)", np.array([0.0, -0.2, 0.0])),
        ("Plus Z (0,0,+0.2)", np.array([0.0, 0.0, 0.2])),
        ("Minus Z (0,0,-0.2)", np.array([0.0, 0.0, -0.2])),
    ]
    
    duration = 30 # frames per move
    
    height, width = 480, 480
    
    for name, pos in moves:
        for _ in range(duration):
            sim_state = env.unwrapped.data.qpos.copy()
            sim_state[0:3] = pos
            # Freeze fingers?
            
            env.unwrapped.data.qpos[:] = sim_state
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f"Move: {name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw overlay of axes?
            # We will just verify visually from the video content
            
            frames.append(frame)

    # Save
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Saved debug axes video to {output_path}")

if __name__ == "__main__":
    debug_axes()
