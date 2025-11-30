from env.snake_env import SnakeEnv
import random

if __name__ == "__main__":
    env = SnakeEnv(render_mode=True)
    obs = env.reset()
    done = False

    while not done:
        action = random.randint(0, 2)  # random policy
        obs, reward, done, info = env.step(action)
        env.render(fps=10)

    env.close()