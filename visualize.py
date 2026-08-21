import argparse
import json
import os
import sys
from argparse import Namespace

import pandas as pd
import pygame
import questionary

from env.track import Track


def play_human():
    import numpy as np

    from env.racing_env import RacingEnv

    env = RacingEnv(render_mode="human")
    env.reset()

    running = True
    while running:
        throttle = 0.0
        steering = 0.0

        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            running = False

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle = -1.0

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = 1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = -1.0

        _obs, _reward, terminated, truncated, _info = env.step(
            np.array([steering, throttle], dtype=np.float32)
        )

        if terminated or truncated:
            print(f"{'Crashed' if terminated else 'Time up'}! Resetting...")
            env.reset()

    env.close()


def run_tui() -> Namespace:
    mode = questionary.select("Select Mode:", choices=["eval", "train", "human"]).ask()

    if mode is None:
        sys.exit(0)

    if mode == "human":
        return Namespace(mode="human", run=None, episode="all")

    if not os.path.exists("artifacts"):
        print("No artifacts directory found.")
        sys.exit(1)

    runs = [
        d
        for d in os.listdir("artifacts")
        if os.path.isdir(os.path.join("artifacts", d))
    ]
    runs.sort(reverse=True)

    if not runs:
        print("No runs found in artifacts/")
        sys.exit(1)

    run = questionary.select("Select Run:", choices=runs).ask()

    if run is None:
        sys.exit(0)

    episode = questionary.select(
        "Select Episode Filter:", choices=["best", "all", "custom"]
    ).ask()

    if episode == "custom":
        episode = questionary.text("Enter Episode ID:").ask()

    return Namespace(mode=mode, run=run, episode=episode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", type=str, help="Timestamp of the run (required for eval/train)"
    )
    parser.add_argument(
        "--mode", type=str, default="eval", choices=["eval", "train", "human"]
    )
    parser.add_argument(
        "--episode", type=str, default="all", help="'all', 'best', or ID"
    )

    args = run_tui() if len(sys.argv) == 1 else parser.parse_args()

    if args.mode == "human":
        play_human()
        return

    if not args.run:
        print("--run is required for eval or train modes.")
        return

    run_name = os.path.basename(os.path.normpath(args.run))
    run_dir = os.path.join("artifacts", run_name)
    target_dir = os.path.join(run_dir, args.mode)
    traj_path = os.path.join(target_dir, "trajectories.json")
    metrics_path = os.path.join(target_dir, "metrics.csv")

    if not os.path.exists(traj_path):
        print(f"Trajectories not found at {traj_path}")
        return

    with open(traj_path, "r") as f:
        trajectories = json.load(f)

    df = pd.read_csv(metrics_path)
    true_best_episode = int(df.loc[df["reward"].idxmax(), "episode"])  # type: ignore

    if args.episode == "best":
        highlight_episode = true_best_episode
        trajectories = [t for t in trajectories if t["episode"] == highlight_episode]
    elif args.episode != "all":
        try:
            highlight_episode = int(args.episode)
            trajectories = [
                t for t in trajectories if t["episode"] == highlight_episode
            ]
        except ValueError:
            print("Invalid --episode argument.")
            return
    else:
        highlight_episode = true_best_episode

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 22)

    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"Ghost Cars - Run {args.run}")
    clock = pygame.time.Clock()

    track = Track(800, 600)

    try:
        car_img_base = pygame.image.load("assets/car.png").convert_alpha()
        car_img_base = pygame.transform.scale(car_img_base, (40, 20))

        ghost_img_base = car_img_base.copy()
        ghost_img_base.set_alpha(100)
    except FileNotFoundError:
        car_img_base = pygame.Surface((40, 20), pygame.SRCALPHA)
        car_img_base.fill((50, 200, 50, 255))

        ghost_img_base = pygame.Surface((40, 20), pygame.SRCALPHA)
        ghost_img_base.fill((200, 50, 50, 100))

    step_idx = 0
    running = True
    playback_fps = 60

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    playback_fps = min(300, playback_fps + 30)
                elif event.key == pygame.K_DOWN:
                    playback_fps = max(15, playback_fps - 30)

        track.draw(screen)

        all_done = True
        for traj in trajectories:
            ep = traj["episode"]
            steps = traj["steps"]
            if step_idx < len(steps):
                all_done = False
                state = steps[step_idx]
                surface = car_img_base if ep == highlight_episode else ghost_img_base
                car_img = pygame.transform.rotate(surface, state["angle"])
                rect = car_img.get_rect(center=(state["x"], state["y"]))
                screen.blit(car_img, rect.topleft)

                text_color = (
                    (255, 255, 255) if ep == highlight_episode else (150, 150, 150)
                )
                ep_text = small_font.render(str(ep), True, text_color)
                text_rect = ep_text.get_rect(center=(state["x"], state["y"] - 30))
                screen.blit(ep_text, text_rect)

                if ep == highlight_episode and "radars" in state:
                    car_pos = (state["x"], state["y"])
                    for radar in state["radars"]:
                        pygame.draw.line(screen, (0, 255, 255), car_pos, radar, 1)
                        pygame.draw.circle(screen, (0, 255, 255), radar, 3)

        mode_text = f"Mode: {args.mode.upper()} | Filter: {args.episode} | Highlighted: {highlight_episode} | Speed: {playback_fps} FPS"
        hud_text1 = font.render(mode_text, True, (255, 255, 255))
        hud_text2 = font.render(f"Replay Step: {step_idx}", True, (255, 255, 255))
        screen.blit(hud_text1, (10, 10))
        screen.blit(hud_text2, (10, 40))

        pygame.display.flip()
        clock.tick(playback_fps)

        step_idx += 1
        if all_done:
            step_idx = 0  # Loop playback

    pygame.quit()


if __name__ == "__main__":
    main()
