"""
1D Rocket Landing Environment
==============================
A minimal Gymnasium environment that simulates a rocket descending
vertically under gravity. The agent controls a single thruster
(on / off) and must land softly (|velocity| < safe_velocity).

State : [z, v, fuel, mass]
Action: Box(1,) - continuous throttle from 0.0 to 1.0
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from reward import compute_shaping_reward, compute_terminal_reward


class RocketLandingEnv(gym.Env):
    """Reference rocket landing environment used by the scaffold."""

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        # ---- physics constants ----
        self.gravity = 9.8
        self.thrust_force = 400.0
        self.fuel_consumption = 1.0
        self.dry_mass = 10.0
        self.initial_fuel = 20.0
        self.dt = 0.1
        self.max_steps = 500
        self.safe_velocity = 5.0
        self.max_altitude = 500.0

        obs_low = np.array([0.0, -np.inf, 0.0, self.dry_mass], dtype=np.float32)
        obs_high = np.array(
            [self.max_altitude, np.inf, self.initial_fuel, self.dry_mass + self.initial_fuel],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode

        self.z = 0.0
        self.v = 0.0
        self.fuel = 0.0
        self.mass = 0.0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.z = self.np_random.uniform(50.0, 400.0)
        self.v = self.np_random.uniform(-5.0, 5.0)
        self.fuel = self.initial_fuel
        self.mass = self.dry_mass + self.fuel
        self.steps = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        throttle = float(np.clip(np.asarray(action, dtype=np.float32).item(), 0.0, 1.0))

        if throttle > 0.0 and self.fuel > 0:
            fuel_used = min(self.fuel, throttle * self.fuel_consumption * self.dt)
            self.fuel -= fuel_used
            self.mass -= fuel_used
            acceleration = (throttle * self.thrust_force / self.mass) - self.gravity
        else:
            acceleration = -self.gravity

        self.v += acceleration * self.dt
        self.z += self.v * self.dt
        self.steps += 1

        reward = compute_shaping_reward(self.z, self.v)
        terminated = False
        truncated = False

        if self.z <= 0:
            self.z = 0.0
            terminated = True
        elif self.z > self.max_altitude:
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True

        terminal_reward = compute_terminal_reward(
            altitude=self.z,
            velocity=self.v,
            safe_velocity=self.safe_velocity,
            max_altitude=self.max_altitude,
        )
        if terminal_reward is not None:
            reward = terminal_reward

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([self.z, self.v, self.fuel, self.mass], dtype=np.float32)

    def render(self):
        print(
            f"step={self.steps:3d}  "
            f"z={self.z:7.2f} m  "
            f"v={self.v:+7.2f} m/s  "
            f"fuel={self.fuel:5.2f} kg  "
            f"mass={self.mass:5.2f} kg"
        )
