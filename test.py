import torch
from envs.rocket import Rocket

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    task = 'hover'  # 'hover' or 'landing'
    max_steps = 800
    env = Rocket(task=task, max_steps=max_steps)

    state = env.reset()
    for step_id in range(max_steps):
        action = env.get_random_action()
        print(action)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')
        if env.already_crash:
            break