import argparse
import json
import os

import pandas as pd
import pygame

from env.track import Track


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True, help="Timestamp of the run")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "train"])
    parser.add_argument(
        "--episode", type=str, default="all", help="'all', 'best', or ID"
    )
    args = parser.parse_args()

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
    best_episode = int(df.loc[df["reward"].idxmax()]["episode"])

    if args.episode == "best":
        trajectories = [t for t in trajectories if t["episode"] == best_episode]
    elif args.episode != "all":
        try:
            ep_id = int(args.episode)
            trajectories = [t for t in trajectories if t["episode"] == ep_id]
        except ValueError:
            print("Invalid --episode argument.")
            return

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 36)
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
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

                if ep == best_episode and "radars" in state:
                    car_pos = (state["x"], state["y"])
                    for radar in state["radars"]:
                        pygame.draw.line(screen, (0, 255, 255), car_pos, radar, 1)
                        pygame.draw.circle(screen, (0, 255, 255), radar, 3)

        hud_text = font.render(
            f"Replay Step: {step_idx} | Best Episode: {best_episode}",
            True,
            (255, 255, 255),
        )
        screen.blit(hud_text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

        step_idx += 1
        if all_done:
            step_idx = 0  # Loop playback

    pygame.quit()


if __name__ == "__main__":
    main()
