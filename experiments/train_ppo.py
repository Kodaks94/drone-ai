from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent  # assuming PPOAgent is saved here
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def moving_avg(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode='valid')

if __name__ == "__main__":
    env = DroneEnv(gui=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    os.makedirs("models", exist_ok=True)
    agent = PPOAgent(state_dim, action_dim)
    episodes = 51
    episode_rewards = []
    best_reward = -np.inf

    for ep in range(episodes):
        env.num_obstacles = min(1 + ep // 100, 5)
        env.goal_pos = np.array([np.random.uniform(4, 6), np.random.uniform(4, 6), 1])

        state = env.reset()
        done = False
        total_reward = 0
        rewards, states, actions, probs, dones = [], [], [], [], []

        while not done:
            action, prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            probs.append(prob)
            dones.append(done)

            state = next_state
            total_reward += reward

        last_value = agent.critic(tf.convert_to_tensor([state], dtype=tf.float32))[0, 0].numpy()
        returns = agent.compute_returns(rewards, dones, last_value)
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        advantages = np.array(returns) - agent.critic(states_tensor).numpy().flatten()

        agent.train(states_tensor, actions, probs, returns, advantages)

        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.actor.save_weights("models/best_ppo_actor.h5")
            agent.critic.save_weights("models/best_ppo_critic.h5")

        print(f"Episode {ep} | Reward: {total_reward:.2f}")

    agent.actor.save_weights("models/ppo_final_actor.h5")
    agent.critic.save_weights("models/ppo_final_critic.h5")
    np.save("models/ppo_episode_rewards.npy", episode_rewards)

    plt.plot(moving_avg(episode_rewards))
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.title("PPO Training Progress")
    plt.savefig("models/ppo_reward_curve.png")
    env.close()