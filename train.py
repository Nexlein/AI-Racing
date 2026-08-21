import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from env.racing_env import RacingEnv


def main():
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = RacingEnv()
    env = Monitor(env, log_dir)

    eval_env = RacingEnv()
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_racing",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,
    )

    print("Starting training...")
    model.learn(
        total_timesteps=100000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(os.path.join(model_dir, "ppo_racing_final"))


if __name__ == "__main__":
    main()
