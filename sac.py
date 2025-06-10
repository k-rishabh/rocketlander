# sac.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os

print("=" * 90)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device set to : {device}")
print("=" * 90)

torch.set_default_dtype(torch.float32)

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def size(self):
        return len(self.buffer)

class SAC:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau

        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.q1 = QNet(state_dim, action_dim).to(device)
        self.q2 = QNet(state_dim, action_dim).to(device)
        self.q1_target = QNet(state_dim, action_dim).to(device)
        self.q2_target = QNet(state_dim, action_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr_critic)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_action = self.actor(state).cpu().numpy()[0]

        # Add exploration noise (temporary fix)
        noise = np.random.normal(0, 0.1, size=raw_action.shape)
        raw_action += noise

        # Rescale action: thrust in [0, 2], gimbal in [-30, 30]
        action = np.zeros_like(raw_action)
        action[0] = np.clip(raw_action[0] + 1, 0.0, 2.0)
        action[1] = np.clip(raw_action[1] * 30, -30.0, 30.0)

        return action

    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_next, q2_next)

        q1_loss = F.mse_loss(self.q1(states, actions), q_target)
        q2_loss = F.mse_loss(self.q2(states, actions), q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        actor_loss = -self.q1(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=device))
