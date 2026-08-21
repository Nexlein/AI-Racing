import argparse
import json
import os

import pandas as pd
from stable_baselines3 import PPO

from env.racing_env import RacingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Timestamp of the run (e.g., 2026-08-21_14-30-00)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to evaluate"
    )
    args = parser.parse_args()

    run_dir = os.path.join("artifacts", args.run)
    model_path = os.path.join(run_dir, "train", "models", "best_model.zip")
    eval_dir = os.path.join(run_dir, "eval")

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    os.makedirs(eval_dir, exist_ok=True)

    env = RacingEnv()
    model = PPO.load(model_path, env=env)

    trajectories = []
    metrics = []

    for ep in range(args.episodes):
        obs, _info = env.reset()
        running = True
        steps = 0
        total_reward = 0
        ep_trajectory = []

        while running:
            assert env.unwrapped.car is not None
            ep_trajectory.append(
                {
                    "x": env.unwrapped.car.pos[0],
                    "y": env.unwrapped.car.pos[1],
                    "angle": env.unwrapped.car.angle,
                }
            )

            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                running = False
                metrics.append(
                    {
                        "episode": ep,
                        "reward": total_reward,
                        "steps": steps,
                        "crashed": terminated,
                    }
                )
                trajectories.append({"episode": ep, "steps": ep_trajectory})

    env.close()

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(eval_dir, "metrics.csv"), index=False)

    with open(os.path.join(eval_dir, "trajectories.json"), "w") as f:
        json.dump(trajectories, f)

    print(f"Evaluation complete. Saved to {eval_dir}")
    print(f"Run 'python visualize.py --run {args.run}' to watch the ghost cars.")


if __name__ == "__main__":
    main()
