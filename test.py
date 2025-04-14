import torch
from envs.rocket import Rocket

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    task = 'landing'  # 'hover' or 'landing'
    max_steps = 800
    env = Rocket(task=task, max_steps=max_steps)

    state = env.reset()
    for step_id in range(max_steps):
        action = env.get_random_action()
        state, reward, done, _ = env.step(action)
        print(state)
        env.render(window_name='test')
        if env.already_crash:
            break