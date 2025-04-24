import pybullet
from env.drone_env import DroneEnv
from agents.ddpg_agent import DDPGAgent
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio

if __name__ == "__main__":
    env = DroneEnv(gui=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = 1.0

    agent = DDPGAgent(state_dim, action_dim, action_bound)
    dummy_input = np.zeros((1, state_dim), dtype=np.float32)
    agent.actor(dummy_input)
    agent.actor.load_weights("models/best_ddpg_actor.h5")

    state = env.reset()
    done = False
    positions = []
    frames_2d = []
    frames_3d = []

    step = 0
    max_steps = 500

    while not done and step < max_steps:
        action = agent.select_action(state, noise_scale=0.0)
        next_state, reward, done, _ = env.step(action)

        pos, _ = pybullet.getBasePositionAndOrientation(env.drone)
        positions.append(pos)

        if step % 3 == 0:
            traj = np.array(positions)

            # 2D Plot
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(traj[:, 0], traj[:, 1], '-', color='black', label='Drone Path')
            arrow_idx = np.linspace(0, len(traj) - 2, min(10, len(traj) - 1)).astype(int)
            ax.quiver(traj[arrow_idx, 0], traj[arrow_idx, 1],
                      traj[arrow_idx + 1, 0] - traj[arrow_idx, 0],
                      traj[arrow_idx + 1, 1] - traj[arrow_idx, 1],
                      scale_units='xy', angles='xy', scale=1, width=0.004, color='gray')
            ax.scatter([traj[0, 0]], [traj[0, 1]], c='green', label='Start')
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], c='yellow', label='Current')
            ax.scatter([env.goal_pos[0]], [env.goal_pos[1]], c='blue', label='Goal')

            for obs_id in env.obstacles:
                obs_pos, _ = pybullet.getBasePositionAndOrientation(obs_id)
                ax.scatter(obs_pos[0], obs_pos[1], c='red', s=100)

            ax.set_title("Drone Trajectory (Top-Down)")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True)
            ax.axis("equal")

            fig.canvas.draw()
            image = np.asarray(fig.canvas.renderer.buffer_rgba())
            frames_2d.append(image)
            plt.close(fig)

            # 3D Plot
            fig = plt.figure(figsize=(7, 6))
            ax3d = fig.add_subplot(111, projection='3d')
            ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='black', label='Drone Path')
            ax3d.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', label='Start')
            ax3d.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='yellow', label='Current')
            ax3d.scatter(env.goal_pos[0], env.goal_pos[1], env.goal_pos[2], c='blue', label='Goal')

            for obs_id in env.obstacles:
                obs_pos, _ = pybullet.getBasePositionAndOrientation(obs_id)
                ax3d.scatter(obs_pos[0], obs_pos[1], obs_pos[2], c='red', s=50)

            ax3d.set_xlim(0, 6)
            ax3d.set_ylim(0, 6)
            ax3d.set_zlim(0, 2)
            ax3d.set_title("Drone Trajectory (3D View)")
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.legend()

            fig.canvas.draw()
            image3d = np.asarray(fig.canvas.renderer.buffer_rgba())
            frames_3d.append(image3d)
            plt.close(fig)

        state = next_state
        step += 1

    imageio.mimsave("models/trajectory.gif", frames_2d, fps=5)
    imageio.mimsave("models/trajectory_3d.gif", frames_3d, fps=5)
    env.close()