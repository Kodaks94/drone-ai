import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.out = layers.Dense(action_dim, activation='tanh')  # outputs in [-1, 1]

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.out = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)

        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.action_bound = action_bound

    def select_action(self, state, noise_scale=0.1):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        action += noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -1, 1)

    def train(self, batch_size=64):
        if len(self.buffer.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards[:, None] + self.gamma * (1 - dones[:, None]) * target_q
            current_q = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(current_q - target_q))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target_model, source_model):
        target_weights = target_model.get_weights()
        source_weights = source_model.get_weights()
        new_weights = [self.tau * sw + (1 - self.tau) * tw for sw, tw in zip(source_weights, target_weights)]
        target_model.set_weights(new_weights)

    def store(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)