import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from env.car import Car
from env.track import Track


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.width, self.height = 800, 600
        self.track = Track(self.width, self.height)

        # Action space: [steering, throttle]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        # Observation: 5 raycasts + 1 speed
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self.screen = None
        self.clock = None
        self.car = None
        self.steps = 0
        self.max_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car = Car(self.track.start_pos, self.track.start_angle)
        self.steps = 0

        if self.render_mode == "human":
            self._setup_pygame()

        return self._get_obs(), {}

    def _setup_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("AI Racing Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def step(self, action):
        self.steps += 1
        steering, throttle = action

        self.car.step(throttle, steering)

        crashed = self.car.check_collision(self.track.mask)
        truncated = self.steps >= self.max_steps
        terminated = crashed

        # Reward: forward progress proxy (speed). Penalize crash.
        reward = self.car.speed * 0.1
        if crashed:
            reward = -10.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        rays = self.car.get_raycasts(self.track.mask)
        speed = np.array([self.car.speed / self.car.max_speed], dtype=np.float32)
        return np.concatenate((rays, speed))

    def render(self):
        if self.render_mode is None:
            return

        self._setup_pygame()

        self.track.draw(self.screen)
        self.car.draw(self.screen)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
