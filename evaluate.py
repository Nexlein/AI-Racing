import os

import pygame
from stable_baselines3 import PPO

from env.racing_env import RacingEnv


def main():
    model_path = "./models/best_model.zip"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train first.")
        return

    env = RacingEnv(render_mode="human")
    model = PPO.load(model_path, env=env)

    obs, _info = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        action, _states = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)

        if terminated or truncated:
            obs, _info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
