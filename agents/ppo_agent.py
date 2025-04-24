# PPO Agent and Training Loop for Drone Navigation

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from env.drone_env import DroneEnv
import matplotlib.pyplot as plt

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.logits = layers.Dense(action_dim)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.logits(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.value = layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.value(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_ratio=0.2, lr=3e-4):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.action_dim = action_dim

    def select_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        logits = self.actor(state)
        action_prob = tf.nn.softmax(logits)
        action = np.random.choice(self.action_dim, p=action_prob.numpy()[0])
        return action, action_prob[0, action].numpy()

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    def train(self, states, actions, old_probs, returns, advantages):
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            logits = self.actor(states)
            probs = tf.nn.softmax(logits)
            selected_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_dim), axis=1)
            ratio = selected_probs / (old_probs + 1e-10)

            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(values)))

        actor_grads = tape1.gradient(policy_loss, self.actor.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

