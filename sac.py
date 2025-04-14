import random
from envs.rocket import Rocket
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Rocket(task="landing", max_steps=1000)

# Hyperparameters
alpha = 0.1
tau = 0.002

# Policy Network (pi_theta)
class PolicyNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hidden = nn.Linear(7, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.tanh(outs)
        outs = self.output(outs)
        return outs

# Q-Value Function
class QNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hidden = nn.Linear(9, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = F.relu(self.hidden(x))
        return self.output(x)

pi_model = PolicyNet().to(device)
q_origin_model1 = QNet().to(device)  # Q_phi1
q_origin_model2 = QNet().to(device)  # Q_phi2
q_target_model1 = QNet().to(device)  # Q_phi1'
q_target_model2 = QNet().to(device)  # Q_phi2'
_ = q_target_model1.requires_grad_(False)  # target model doen't need grad
_ = q_target_model2.requires_grad_(False)  # target model doen't need grad

# Pick up action (for each step in episode)
def pick_sample(s):
    with torch.no_grad():
        s_batch = np.expand_dims(s, axis=0)  # (1, 7)
        s_batch = torch.tensor(s_batch, dtype=torch.float64).to(device)

        action = pi_model(s_batch)          # shape: (1, action_dim)
        action = action.squeeze(dim=0)      # shape: (action_dim,)
        action[0] += 1                      # thrust lies between 0.0 and 2.0
        action[1] *= 30                     # gimbal lies between -30 and 30
        return action.cpu().numpy()         # convert to NumPy array


opt_pi = torch.optim.AdamW(pi_model.parameters(), lr=0.0005)

def optimize_theta(states):
    states = torch.tensor(states, dtype=torch.float64).to(device)
    actions = pi_model(states)
    q_values = q_origin_model1(states, actions)
    loss = -q_values.mean()

    opt_pi.zero_grad()
    loss.backward()
    opt_pi.step()

gamma = 0.99

opt_q1 = torch.optim.AdamW(q_origin_model1.parameters(), lr=0.0005)
opt_q2 = torch.optim.AdamW(q_origin_model2.parameters(), lr=0.0005)

def optimize_phi(states, actions, rewards, next_states, dones):
    states = torch.tensor(states, dtype=torch.float64).to(device)
    actions = torch.from_numpy(actions).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float64).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float64).to(device)
    dones = torch.tensor(dones, dtype=torch.float64).unsqueeze(1).to(device)

    # Compute target Q
    with torch.no_grad():
        next_actions = pi_model(next_states)
        q1_next = q_target_model1(next_states, next_actions)
        q2_next = q_target_model2(next_states, next_actions)
        q_min = torch.min(q1_next, q2_next)
        q_target = rewards + gamma * (1.0 - dones) * q_min

    # Update Q1
    q1_pred = q_origin_model1(states, actions)
    loss1 = F.mse_loss(q1_pred, q_target)
    opt_q1.zero_grad()
    loss1.backward()
    opt_q1.step()

    # Update Q2
    q2_pred = q_origin_model2(states, actions)
    loss2 = F.mse_loss(q2_pred, q_target)
    opt_q2.zero_grad()
    loss2.backward()
    opt_q2.step()

def update_target():
    for var, var_target in zip(q_origin_model1.parameters(), q_target_model1.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data
    for var, var_target in zip(q_origin_model2.parameters(), q_target_model2.parameters()):
        var_target.data = tau * var.data + (1.0 - tau) * var_target.data

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, item):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(item)

    def sample(self, batch_size):
        items = random.sample(self.buffer, batch_size)
        states   = [i[0] for i in items]
        actions  = [i[1] for i in items]
        rewards  = [i[2] for i in items]
        n_states = [i[3] for i in items]
        dones    = [i[4] for i in items]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)

buffer = ReplayBuffer(20000)

batch_size = 250

reward_records = []
for i in range(2000):
    # Run episode till done
    s = env.reset()
    done = False
    cum_reward = 0
    while not done:
        a = pick_sample(s)
        s_next, r, done, _ = env.step(a)
        buffer.add([s.tolist(), a, r, s_next.tolist(), float(done)])
        cum_reward += r
        if buffer.length() >= batch_size:
            states, actions, rewards, n_states, dones = buffer.sample(batch_size)
            actions = np.array(actions)
            optimize_theta(states)
            optimize_phi(states, actions, rewards, n_states, dones)
            update_target()
        s = s_next

    # Output total rewards in episode (max 500)
    print("Running episode {} with rewards {}".format(i, cum_reward), end="\r")
    reward_records.append(cum_reward)

    # stop if reward mean > 475.0
    if np.average(reward_records[-50:]) > 475.0:
        break

env.close()
print("\nDone")

import matplotlib.pyplot as plt
# Generate recent 50 interval average
average_reward = []
for idx in range(len(reward_records)):
    avg_list = np.empty(shape=(1,), dtype=int)
    if idx < 50:
        avg_list = reward_records[:idx+1]
    else:
        avg_list = reward_records[idx-49:idx+1]
    average_reward.append(np.average(avg_list))
plt.plot(reward_records)
plt.plot(average_reward)
