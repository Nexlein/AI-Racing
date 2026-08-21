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
- Fully isolated artifact tracking for 100% reproducible training runs.
- Interactive terminal UI (TUI) for launching replays.
- YouTube-style timeline scrubber and playback speed controls.

## Quick Start

```bash
# 1. Install
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Train the AI (Settings are in config.yaml)
python train.py

# 3. Evaluate the best model
python evaluate.py --run <timestamp>

# 4. Watch the Replay
python visualize.py
```

## Visualization Controls

Running `python visualize.py` opens an interactive arrow-key terminal menu. Once the replay starts, use these controls:

- **Up / Down**: Speed up or slow down playback (+/- 30 FPS).
- **Left / Right**: Skip 50 frames backward or forward.
- **Mouse Click/Drag**: Scrub the timeline at the bottom of the window to instantly jump to any time.
- **R**: Export the entire replay to an `.mp4` file in the project root.

*(You can also drive manually to test physics by running `python visualize.py --mode human`)*
