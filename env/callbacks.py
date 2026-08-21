import json
import os
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from env.racing_env import RacingEnv


class TrajectoryRecordCallback(BaseCallback):
    def __init__(
        self,
        eval_env: RacingEnv,
        save_dir: str,
        record_freq: int = 5000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_dir = save_dir
        self.record_freq = record_freq
        self.trajectories: list[dict[str, Any]] = []
        self.metrics: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq == 0:
            self._record_trajectory()
        return True

    def _record_trajectory(self):
        obs, _ = self.eval_env.reset()
        running = True
        ep_trajectory = []
        steps = 0
        total_reward = 0.0
        terminated = False

        while running:
            assert self.eval_env.car is not None
            ep_trajectory.append(
                {
                    "x": float(self.eval_env.car.pos[0]),
                    "y": float(self.eval_env.car.pos[1]),
                    "angle": float(self.eval_env.car.angle),
                    "radars": [
                        (int(rx), int(ry)) for rx, ry in self.eval_env.car.radars
                    ],
                }
            )

            action, _ = self.model.predict(obs, deterministic=True)  # type: ignore
            if not isinstance(action, np.ndarray):
                action = np.array(action)

            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            total_reward += float(reward)
            steps += 1

            if terminated or truncated:
                running = False

        self.metrics.append(
            {
                "episode": self.num_timesteps,
                "reward": total_reward,
                "steps": steps,
                "crashed": bool(terminated),
            }
        )
        self.trajectories.append(
            {"episode": self.num_timesteps, "steps": ep_trajectory}
        )

        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.save_dir, "metrics.csv"), index=False)
        with open(os.path.join(self.save_dir, "trajectories.json"), "w") as f:
            json.dump(self.trajectories, f)
