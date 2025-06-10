# sac_train.py with Moving Average and Std Plot

import os
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from envs.rocket import Rocket
from sac import SAC, ReplayBuffer

def train():
    print("="*90)

    env_name = "RocketLanding"
    task = 'landing'
    render = False
    max_ep_len = 1000
    max_training_timesteps = int(6e6)

    print_freq = max_ep_len * 10
    log_freq = max_ep_len * 2
    save_model_freq = int(1e5)

    batch_size = 256
    gamma = 0.99
    lr_actor = 3e-4
    lr_critic = 3e-4
    tau = 0.005

    random_seed = 0

    print("Training environment: " + env_name)

    env = Rocket(max_steps=max_ep_len, task=task)
    state_dim = env.state_dims
    action_dim = env.action_dims

    log_dir = "SAC_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_dir = f"{log_dir}/{env_name}/"
    os.makedirs(log_dir, exist_ok=True)
    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = f"{log_dir}/SAC_{env_name}_log_{run_num}.csv"
    print("Logging at : " + log_f_name)

    directory = "SAC_preTrained"
    os.makedirs(directory, exist_ok=True)
    directory = f"{directory}/{env_name}/"
    os.makedirs(directory, exist_ok=True)
    checkpoint_path = f"{directory}/SAC_{env_name}_{random_seed}_{run_num}.pth"
    print("Checkpoint path : " + checkpoint_path)

    sac_agent = SAC(state_dim, action_dim, lr_actor, lr_critic, gamma, tau)
    replay_buffer = ReplayBuffer(max_size=100000)

    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    episode_rewards = []
    window_size = 10

    plt.ion()
    fig, ax = plt.subplots()
    plt.show(block=False)
    fig.canvas.flush_events()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('SAC Training Progress')

    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            action = sac_agent.select_action(state)
            print(action)
            next_state, reward, done, _ = env.step(action)

            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            time_step += 1
            current_ep_reward += reward

            if replay_buffer.size() >= batch_size:
                sac_agent.update(replay_buffer, batch_size=batch_size)

            if render and i_episode % 50 == 0:
                env.render()

            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_f.write(f'{i_episode},{time_step},{round(log_avg_reward, 4)}\n')
                log_running_reward, log_running_episodes = 0, 0

            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print(f"Episode: {i_episode}\t Timestep: {time_step}\t Avg Reward: {round(print_avg_reward, 2)}")
                print_running_reward, print_running_episodes = 0, 0

            if time_step % save_model_freq == 0:
                sac_agent.save(checkpoint_path)
                print("Model saved at timestep: ", time_step)

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

        episode_rewards.append(current_ep_reward)

        ax.clear()
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(
                episode_rewards, np.ones(window_size)/window_size, mode='valid')
            moving_std = np.array([
                np.std(episode_rewards[i-window_size+1:i+1]) 
                for i in range(window_size-1, len(episode_rewards))
            ])
            episodes = np.arange(window_size-1, len(episode_rewards))

            ax.plot(episodes, moving_avg, label='Moving Average Reward')
            ax.fill_between(episodes, moving_avg - moving_std, moving_avg + moving_std,
                            color='blue', alpha=0.2, label='Standard Deviation')
        else:
            ax.plot(range(len(episode_rewards)), episode_rewards, label='Episode Reward')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('SAC Training Progress')
        ax.legend()
        plt.draw()
        plt.pause(0.01)

    log_f.close()
    print("Finished training at : ", datetime.now().replace(microsecond=0))

if __name__ == '__main__':
    train()
