from env.drone_env import DroneEnv
from agents.ddpg_agent import DDPGAgent
import numpy as np
import matplotlib.pyplot as plt
import os


def moving_avg(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode='valid')


if __name__ == "__main__":
    env = DroneEnv(gui=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = 1.0  # from env Box(-1, 1)
    os.makedirs("models", exist_ok=True)

    agent = DDPGAgent(state_dim, action_dim, action_bound)
    episodes = 51
    episode_rewards = []
    best_reward = -np.inf

    for ep in range(episodes):
        env.num_obstacles = min(1 + ep // 100, 5)
        env.goal_pos = np.array([np.random.uniform(4, 6), np.random.uniform(4, 6), 1])

        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.actor.save_weights("models/best_ddpg_actor.h5")
            agent.critic.save_weights("models/best_ddpg_critic.h5")

        print(f"Episode {ep} | Reward: {total_reward:.2f}")

    agent.actor.save_weights("models/ddpg_final_actor.h5")
    agent.critic.save_weights("models/ddpg_final_critic.h5")
    np.save("models/ddpg_episode_rewards.npy", episode_rewards)

    plt.plot(moving_avg(episode_rewards))
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.title("DDPG Training Progress")
    plt.savefig("models/ddpg_reward_curve.png")
    env.close()
