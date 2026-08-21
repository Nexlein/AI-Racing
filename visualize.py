import argparse
import json
import os

import pandas as pd
import pygame

from env.track import Track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Timestamp of the run")
    args = parser.parse_args()

    run_dir = os.path.join("artifacts", args.run)
    eval_dir = os.path.join(run_dir, "eval")
    traj_path = os.path.join(eval_dir, "trajectories.json")
    metrics_path = os.path.join(eval_dir, "metrics.csv")

    if not os.path.exists(traj_path):
        print(f"Trajectories not found at {traj_path}")
        return

    with open(traj_path, "r") as f:
        trajectories = json.load(f)

    df = pd.read_csv(metrics_path)
    best_episode = int(df.loc[df["reward"].idxmax()]["episode"])

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption(f"Ghost Cars - Run {args.run}")
    clock = pygame.time.Clock()

    track = Track(800, 600)

    ghost_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
    ghost_surface.fill((200, 50, 50, 100))  # Semi-transparent red

    best_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
    best_surface.fill((50, 200, 50, 255))  # Solid green

    step_idx = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        track.draw(screen)

        all_done = True
        for traj in trajectories:
            ep = traj["episode"]
            steps = traj["steps"]
            if step_idx < len(steps):
                all_done = False
                state = steps[step_idx]
                surface = best_surface if ep == best_episode else ghost_surface
                car_img = pygame.transform.rotate(surface, state["angle"])
                rect = car_img.get_rect(center=(state["x"], state["y"]))
                screen.blit(car_img, rect.topleft)

        pygame.display.flip()
        clock.tick(30)

        step_idx += 1
        if all_done:
            step_idx = 0  # Loop playback

    pygame.quit()


if __name__ == "__main__":
    main()
