# Drone Navigation with Reinforcement Learning

This project explores reinforcement learning (RL) methods for autonomous drone navigation in a 3D environment using PyBullet. The objective is for a drone to move from a start point (A) to a target point (B) while avoiding randomly placed obstacles.

##  Goals
- Build a custom drone navigation environment in PyBullet.
- Train and compare different RL algorithms:
  - Deep Q-Learning (DQN)
  - Proximal Policy Optimization (PPO)
  - Backpropagation Through Time (BPTT)
- Evaluate each method’s performance on:
  - Reward over time
  - Success rate
  - Collision rate
  - Training speed and convergence

##  Tech Stack
- Python 3
- PyBullet
- NumPy
- PyTorch / TensorFlow (TBD)
- OpenAI Gym interface

##  Methods
We’ll be training agents from scratch, comparing classical and recurrent RL methods. The final output will include visualizations and performance benchmarks.

##  Project Structure
drone-rl/ ├── env/ # Custom PyBullet drone environment ├── agents/ # DQN, PPO, BPTT implementations ├── experiments/ # Training scripts, logs, results ├── utils/ # Helper functions ├── README.md └── requirements.txt


## 🚧 Status
- [x] Repo initialized
- [ ] Environment implemented
- [ ] DQN agent
- [ ] PPO agent
- [ ] BPTT agent
- [ ] Performance analysis

## 👩‍💻 Author
Made with ❤️ by [Mahrad Pisheh Var] - Reinforcement Learning Enthusiast

---
