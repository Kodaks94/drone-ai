from env.drone_env import DroneEnv
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

env = DroneEnv(gui=False)
env.goal_pos = np.array([5.0, 5.0, 1.0])
obs = env.reset()

positions = []

for _ in range(200):
    pos, _ = p.getBasePositionAndOrientation(env.drone)
    to_goal = env.goal_pos - np.array(pos)
    direction = to_goal / (np.linalg.norm(to_goal) + 1e-8)

    p.resetBaseVelocity(env.drone, linearVelocity=direction * 3.0)
    for _ in range(4):  # simulate multiple steps per frame
        p.stepSimulation()
    positions.append(pos)

positions = np.array(positions)
final_pos = positions[-1]
print("Final distance to goal:", np.linalg.norm(env.goal_pos - final_pos))

# Plot the path
plt.plot(positions[:, 0], positions[:, 1], marker='o', label='Path')
plt.scatter([env.goal_pos[0]], [env.goal_pos[1]], c='blue', label='Goal')
plt.scatter([positions[0, 0]], [positions[0, 1]], c='green', label='Start')
plt.scatter([final_pos[0]], [final_pos[1]], c='red', label='End')
plt.axis("equal")
plt.grid()
plt.legend()
plt.title("Drone Motion Toward Goal (No Learning)")
plt.savefig("TESTED.jpg")
