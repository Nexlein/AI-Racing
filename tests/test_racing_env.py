import numpy as np
import pytest

from env.racing_env import RacingEnv

IDLE = np.array([0.0, 0.0], dtype=np.float32)
FORWARD = np.array([0.0, 1.0], dtype=np.float32)


def test_reset_returns_observation_in_space():
    env = RacingEnv()
    obs, info = env.reset(seed=0)

    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info == {}


def test_step_returns_gymnasium_tuple():
    env = RacingEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(FORWARD)

    assert obs.shape == (6,)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info == {}


def test_idle_step_gets_time_penalty():
    env = RacingEnv()
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(IDLE)

    assert reward == pytest.approx(-0.1)
    assert not terminated
    assert not truncated


def test_crash_terminates_with_penalty():
    env = RacingEnv()
    env.reset(seed=0)
    # Full throttle while steering hard into the outer wall.
    action = np.array([1.0, 1.0], dtype=np.float32)
    reward, terminated, truncated = 0.0, False, False
    for _ in range(200):
        _, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert reward == -10.0


def test_checkpoint_gives_reward():
    env = RacingEnv()
    env.reset(seed=0)
    assert env.car is not None
    env.car.pos = list(env.track.checkpoints[0].center)
    _, reward, terminated, _, _ = env.step(IDLE)

    assert reward == pytest.approx(9.9)
    assert env.current_checkpoint == 1
    assert not terminated


def test_episode_truncates_at_max_steps():
    env = RacingEnv()
    env.max_steps = 5
    env.reset(seed=0)
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(IDLE)
        assert not terminated
        assert not truncated
    _, _, terminated, truncated, _ = env.step(IDLE)

    assert not terminated
    assert truncated
