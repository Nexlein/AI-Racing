# AI Racing

<div align="center">
  <video src="https://github.com/user-attachments/assets/18a41d7b-af1c-45b2-a23d-204d9cb1a5b0" controls="controls" muted="muted" width="800"></video>

![Release](https://img.shields.io/github/v/release/Nexlein/AI-Racing?style=flat-square)
![Build](https://img.shields.io/github/actions/workflow/status/Nexlein/AI-Racing/ci.yml?style=flat-square)

Reinforcement Learning agent that learns to drive a car around a track. Built with **Gymnasium**, **Pygame**, and **Stable Baselines3 (PPO)**.

📖 **[Read the Internal Architecture Documentation](docs/architecture.md)**
</div>

## Features

- Custom kinematic physics and raycast sensors.
- Seeded training (`seed` in `config.yaml`) and isolated artifact tracking for reproducible runs.
- Interactive terminal UI (TUI) for launching replays.
- YouTube-style timeline scrubber and playback speed controls.

## Quick Start

```bash
# 1. Install
uv sync                  # Python 3.12, dependencies and CPU-only torch
uvx pre-commit install   # Ruff, secret scan and commit message checks

# 2. Train the AI (Settings are in config.yaml)
uv run train.py

# 3. Evaluate the best model
uv run evaluate.py --run <timestamp>

# 4. Watch the Replay
uv run visualize.py
```

## Visualization Controls

Running `uv run visualize.py` opens an interactive arrow-key terminal menu. Once the replay starts, use these controls:

- **Up / Down**: Speed up or slow down playback (+/- 30 FPS).
- **Left / Right**: Skip 50 frames backward or forward.
- **Mouse Click/Drag**: Scrub the timeline at the bottom of the window to instantly jump to any time.
- **R**: Export the entire replay to an `.mp4` file in the project root.

*(You can also drive manually to test physics by running `uv run visualize.py --mode human`)*
