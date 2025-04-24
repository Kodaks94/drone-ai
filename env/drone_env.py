import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
import time
import os


class DroneEnv(gym.Env):
    def __init__(self, gui=False):
        super(DroneEnv, self).__init__()

        self.gui = gui
        self.time_step = 1. / 60.
        self.max_steps = 500  # 8 seconds
        self.goal_pos = np.array([5, 5, 1])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.lidar_size = 8
        obs_high = np.array([1.] * (9 + self.lidar_size),
                            dtype=np.float32)  # 9 = drone position (6) + velocity(2) + goal distance(1)

        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.client = p.connect(p.GUI if gui else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

        self.drone = None
        self.obstacles = []
        self.step_counter = 0

    def reset(self):
        p.resetDebugVisualizerCamera(cameraDistance=8,
                                     cameraYaw=45,
                                     cameraPitch=-45,
                                     cameraTargetPosition=[2.5, 2.5, 1],
                                     physicsClientId=self.client
                                     )
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)

        plane = p.loadURDF("plane.urdf")
        self.drone = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 1])
        self.last_pos = np.array([0, 0, 1])
        self.prev_movement = np.zeros(3)
        self._spawn_obstacles(3)
        self.step_counter = 0
        self.last_dist = np.linalg.norm(self.goal_pos - np.array([0, 0, 1]))
        return self._get_obs()

    def _get_lidar_scan(self, num_rays=8, ray_length=5.):

        pos, _ = p.getBasePositionAndOrientation(self.drone)
        pos = np.array(pos)

        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        from_positions = []
        to_positions = []

        for angle in angles:
            dx = ray_length * np.cos(angle)
            dy = ray_length * np.sin(angle)
            from_pos = pos
            to_pos = pos + np.array([dx, dy, 0])
            from_positions.append(from_pos)
            to_positions.append(to_pos)

        result = p.rayTestBatch(from_positions, to_positions)
        distances = [r[2] * ray_length if r[0] != -1 else ray_length for r in result]

        return np.array(distances) / ray_length

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        velocity = action * 3.0  # Scale max speed
        p.resetBaseVelocity(self.drone, linearVelocity=velocity.tolist())

        for _ in range(4):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)

        self.step_counter += 1

        pos, _ = p.getBasePositionAndOrientation(self.drone)
        done = self._is_done(pos)
        reward = self._get_reward(pos, done)
        obs = self._get_obs()

        return obs, reward, done, {}

    def _spawn_obstacles(self, num_obstacles=10):
        agent_start = np.array([0, 0, 1])
        goal = self.goal_pos
        self.obstacles = []
        for _ in range(num_obstacles):
            t = np.random.uniform(0.3, 1.)
            base_pos = agent_start + t * (goal - agent_start)

            offset = np.random.uniform(-1.5, 1.5, size=3)
            offset[2] = np.random.uniform(0.0, 0.5)

            obs_pos = base_pos + offset

            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=.5, rgbaColor=[1, 0, 0, 1])
            obs = p.createMultiBody(baseMass=0.,
                                    baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=obs_pos.tolist()
                                    )
            self.obstacles.append(obs)

    def _get_obs(self):
        pos, vel = p.getBasePositionAndOrientation(self.drone), p.getBaseVelocity(self.drone)
        pos = np.array(pos[0])
        vel = np.array(vel[0])
        goal_vec = self.goal_pos - pos

        # normalise

        pos /= 10.0
        vel /= 5.0
        goal_vec /= 10.0
        lidar = self._get_lidar_scan(num_rays=self.lidar_size)
        obs = np.concatenate([pos, vel, goal_vec, lidar]).astype(np.float32)
        obs = np.clip(obs, -1., 1.)
        return obs

    def _is_done(self, pos):
        dist_to_goal = np.linalg.norm(self.goal_pos - pos)
        if dist_to_goal < 0.5:
            return True
        if self.step_counter >= self.max_steps:
            return True
        if np.any(np.abs(pos) > 20) or pos[2] < 0:
            return True
        return False

    def _get_reward(self, pos, done):
        pos = np.array(pos)
        to_goal = self.goal_pos - pos
        dist = np.linalg.norm(to_goal)
        to_goal_unit = to_goal / (dist + 1e-8)

        movement = pos - self.last_pos
        smoothness_penalty = np.linalg.norm(movement - self.prev_movement)
        self.prev_movement = movement
        self.last_pos = pos
        progress = self.last_dist - dist
        projection = np.dot(movement, to_goal_unit)
        reward = 10 * projection + 1.5 * progress - 0.05 * dist - 0.5 * smoothness_penalty
        self.last_dist = dist
        collided = any(len(p.getContactPoints(self.drone, obs)) > 0 for obs in self.obstacles)
        if collided:
            reward -= 30
            done = True

        if done and dist < 0.5:
            reward += 100

        return np.clip(reward, -50, 100)

    def close(self):
        p.disconnect(self.client)
