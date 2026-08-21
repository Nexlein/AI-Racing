import pygame
from env.racing_env import RacingEnv


def main():
    env = RacingEnv(render_mode="human")
    obs, info = env.reset()

    running = True
    while running:
        throttle = 0.0
        steering = 0.0

        # We need this to allow window close and keyboard events
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            running = False

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle = -1.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = 1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = -1.0

        obs, reward, terminated, truncated, info = env.step([steering, throttle])

        if terminated or truncated:
            print(f"{'Crashed' if terminated else 'Time up'}! Resetting...")
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
