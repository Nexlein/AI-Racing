from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from env.car import Car
from env.track import Track
from typing import ClassVar


class RacingEnv(gym.Env):
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: str | None = None):
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

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.car: Car | None = None
        self.current_checkpoint = 0
        self.steps = 0
        self.max_steps = 1000

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.car = Car(self.track.start_pos, self.track.start_angle)
        self.current_checkpoint = 0
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

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.car is not None
        self.steps += 1
        steering, throttle = action

        self.car.step(float(throttle), float(steering))

        crashed = self.car.check_collision(self.track.mask)
        truncated = self.steps >= self.max_steps
        terminated = crashed

        reward = -0.1  # Base penalty to encourage speed

        # Checkpoint logic
        cp = self.track.checkpoints[self.current_checkpoint]
        if cp.collidepoint(self.car.pos[0], self.car.pos[1]):
            self.current_checkpoint += 1
            reward += 10.0
            if self.current_checkpoint >= len(self.track.checkpoints):
                reward += 100.0
                terminated = True
                self.current_checkpoint = (
                    0  # Prevent out of bounds if it somehow keeps running
                )

        if crashed:
            reward = -10.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        assert self.car is not None
        rays = self.car.get_raycasts(self.track.mask)
        speed = np.array([self.car.speed / self.car.max_speed], dtype=np.float32)
        return np.concatenate((rays, speed))

    def render(self):
        if self.render_mode is None:
            return

        self._setup_pygame()
        assert self.screen is not None
        assert self.clock is not None
        assert self.car is not None

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
