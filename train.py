import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.racing_env import RacingEnv


def plot_results(log_dir: str, output_file: str):
    csv_path = os.path.join(log_dir, "monitor.csv.monitor.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(log_dir, "monitor.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, skiprows=1)
        plt.figure(figsize=(10, 5))
        plt.plot(
            df["r"].rolling(window=100, min_periods=1).mean(),
            label="Reward (100 ep rolling mean)",
        )
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title("Training Reward")
        plt.legend()
        plt.savefig(output_file)
        plt.close()


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("artifacts", timestamp)
    train_dir = os.path.join(run_dir, "train")
    model_dir = os.path.join(train_dir, "models")
    log_dir = os.path.join(train_dir, "logs")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    shutil.copy("config.yaml", os.path.join(run_dir, "config.yaml"))

    env = RacingEnv()
    env = Monitor(env, os.path.join(log_dir, "monitor.csv"))

    eval_env = RacingEnv()
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=config.get("eval_freq", 5000),
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("save_freq", 10000),
        save_path=model_dir,
        name_prefix="ppo_racing",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=config.get("learning_rate", 0.0003),
    )

    print(f"Starting training. Artifacts in {run_dir}")
    model.learn(
        total_timesteps=config.get("total_timesteps", 100000),
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(os.path.join(model_dir, "ppo_racing_final"))
    plot_results(log_dir, os.path.join(train_dir, "chart.png"))
    print(f"Training complete. Chart saved to {os.path.join(train_dir, 'chart.png')}")


if __name__ == "__main__":
    main()
