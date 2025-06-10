import torch
from envs.rocket import Rocket
from sac import SAC

# Test configuration
max_ep_len = 1000
total_test_episodes = 10

# Initialize environment
env = Rocket(max_steps=max_ep_len, task="landing")
state_dim = env.state_dims
action_dim = env.action_dims

# Initialize SAC agent
sac_agent = SAC(state_dim, action_dim)
sac_agent.load("sac_final.pth")

for episode in range(total_test_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_ep_len):
        action = sac_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        env.render()

        if done:
            break

    print(f"Test Episode {episode}, Reward: {episode_reward}")

env.close()