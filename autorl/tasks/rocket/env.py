"""
1D Rocket Landing Environment
==============================
A Gymnasium environment that simulates a rocket descending vertically
under gravity. The agent controls a continuous throttle and must land
softly (|velocity| < safe_velocity) across a wide range of initial
conditions.

State : normalized [altitude, velocity, fuel, mass] in roughly [-1, 1]
Action: Box(1,) - continuous throttle from 0.0 to 1.0
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from reward import compute_shaping_reward, compute_terminal_reward


class RocketLandingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 10}

    # ---- physics (fixed) ----
    GRAVITY = 9.8
    THRUST_FORCE = 400.0
    FUEL_CONSUMPTION = 1.0
    DRY_MASS = 10.0
    DT = 0.1
    SAFE_VELOCITY = 5.0
    MAX_ALTITUDE = 500.0
    MAX_STEPS = 500
    OBS_VELOCITY_SCALE = 50.0

    # ---- initial condition ranges (wide for generalization) ----
    ALTITUDE_RANGE = (10.0, 490.0)
    VELOCITY_RANGE = (-30.0, 10.0)
    FUEL_RANGE = (5.0, 30.0)
    CURRICULUM_EPISODES = 2500
    CURRICULUM_ALTITUDE_RANGE = (40.0, 180.0)
    CURRICULUM_VELOCITY_RANGE = (-12.0, 2.0)
    CURRICULUM_FUEL_RANGE = (18.0, 30.0)
    LOW_FUEL_EPISODE_PROB = 0.35
    LOW_FUEL_EPISODE_PROB_START = 0.1
    LOW_FUEL_ALTITUDE_RANGE = (50.0, 400.0)
    LOW_FUEL_VELOCITY_RANGE = (-12.0, 2.0)
    LOW_FUEL_RANGE = (5.0, 12.0)
    CURRICULUM_LOW_FUEL_ALTITUDE_RANGE = (60.0, 180.0)
    CURRICULUM_LOW_FUEL_VELOCITY_RANGE = (-8.0, 0.0)
    CURRICULUM_LOW_FUEL_RANGE = (8.0, 18.0)
    LOW_FUEL_HIGH_ALTITUDE_EPISODE_PROB = 0.15
    LOW_FUEL_HIGH_ALTITUDE_EPISODE_PROB_START = 0.05
    LOW_FUEL_HIGH_ALTITUDE_RANGE = (280.0, 420.0)
    LOW_FUEL_HIGH_ALTITUDE_VELOCITY_RANGE = (-14.0, 0.0)
    LOW_FUEL_HIGH_ALTITUDE_FUEL_RANGE = (6.0, 12.0)
    CURRICULUM_LOW_FUEL_HIGH_ALTITUDE_RANGE = (180.0, 280.0)
    CURRICULUM_LOW_FUEL_HIGH_ALTITUDE_VELOCITY_RANGE = (-8.0, 0.0)
    CURRICULUM_LOW_FUEL_HIGH_ALTITUDE_FUEL_RANGE = (8.0, 16.0)

    def __init__(self, render_mode=None, scenario=None):
        """
        Args:
            scenario: optional dict with keys {altitude, velocity, fuel}
                      to override random initial conditions (used for eval).
        """
        super().__init__()

        self.gravity = self.GRAVITY
        self.thrust_force = self.THRUST_FORCE
        self.fuel_consumption = self.FUEL_CONSUMPTION
        self.dry_mass = self.DRY_MASS
        self.dt = self.DT
        self.safe_velocity = self.SAFE_VELOCITY
        self.max_altitude = self.MAX_ALTITUDE
        self.max_steps = self.MAX_STEPS

        self.scenario = scenario

        max_fuel = self.FUEL_RANGE[1]
        self.max_fuel = max_fuel
        obs_low = np.full(4, -1.0, dtype=np.float32)
        obs_high = np.full(4, 1.0, dtype=np.float32)
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
        self.training_resets = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.scenario is not None:
            self.z = float(self.scenario["altitude"])
            self.v = float(self.scenario["velocity"])
            self.fuel = float(self.scenario["fuel"])
        else:
            self.z, self.v, self.fuel = self._sample_training_state()
            self.training_resets += 1

        self.mass = self.dry_mass + self.fuel
        self.steps = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _sample_training_state(self):
        """Gradually widen the training distribution over the first resets."""

        progress = min(self.training_resets / self.CURRICULUM_EPISODES, 1.0)

        altitude_range = self._interpolate_range(
            self.CURRICULUM_ALTITUDE_RANGE, self.ALTITUDE_RANGE, progress
        )
        velocity_range = self._interpolate_range(
            self.CURRICULUM_VELOCITY_RANGE, self.VELOCITY_RANGE, progress
        )
        fuel_range = self._interpolate_range(
            self.CURRICULUM_FUEL_RANGE, self.FUEL_RANGE, progress
        )
        low_fuel_probability = (
            self.LOW_FUEL_EPISODE_PROB_START
            + progress * (self.LOW_FUEL_EPISODE_PROB - self.LOW_FUEL_EPISODE_PROB_START)
        )
        low_fuel_high_altitude_probability = (
            self.LOW_FUEL_HIGH_ALTITUDE_EPISODE_PROB_START
            + progress
            * (
                self.LOW_FUEL_HIGH_ALTITUDE_EPISODE_PROB
                - self.LOW_FUEL_HIGH_ALTITUDE_EPISODE_PROB_START
            )
        )

        episode_draw = self.np_random.random()

        if episode_draw < low_fuel_probability:
            altitude = self.np_random.uniform(
                *self._interpolate_range(
                    self.CURRICULUM_LOW_FUEL_ALTITUDE_RANGE,
                    self.LOW_FUEL_ALTITUDE_RANGE,
                    progress,
                )
            )
            velocity = self.np_random.uniform(
                *self._interpolate_range(
                    self.CURRICULUM_LOW_FUEL_VELOCITY_RANGE,
                    self.LOW_FUEL_VELOCITY_RANGE,
                    progress,
                )
            )
            fuel = self.np_random.uniform(
                *self._interpolate_range(
                    self.CURRICULUM_LOW_FUEL_RANGE,
                    self.LOW_FUEL_RANGE,
                    progress,
                )
            )
            return altitude, velocity, fuel

        if episode_draw < low_fuel_probability + low_fuel_high_altitude_probability:
            altitude = self.np_random.uniform(
                *self._interpolate_range(
                    self.CURRICULUM_LOW_FUEL_HIGH_ALTITUDE_RANGE,
                    self.LOW_FUEL_HIGH_ALTITUDE_RANGE,
                    progress,
                )
            )
            velocity = self.np_random.uniform(
                *self._interpolate_range(
                    self.CURRICULUM_LOW_FUEL_HIGH_ALTITUDE_VELOCITY_RANGE,
                    self.LOW_FUEL_HIGH_ALTITUDE_VELOCITY_RANGE,
                    progress,
                )
            )
            fuel = self.np_random.uniform(
                *self._interpolate_range(
                    self.CURRICULUM_LOW_FUEL_HIGH_ALTITUDE_FUEL_RANGE,
                    self.LOW_FUEL_HIGH_ALTITUDE_FUEL_RANGE,
                    progress,
                )
            )
            return altitude, velocity, fuel

        altitude = self.np_random.uniform(*altitude_range)
        velocity = self.np_random.uniform(*velocity_range)
        fuel = self.np_random.uniform(*fuel_range)
        return altitude, velocity, fuel

    @staticmethod
    def _interpolate_range(start_range, end_range, progress):
        low = (1.0 - progress) * start_range[0] + progress * end_range[0]
        high = (1.0 - progress) * start_range[1] + progress * end_range[1]
        return low, high

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

        reward = compute_shaping_reward(
            altitude=self.z,
            velocity=self.v,
            safe_velocity=self.safe_velocity,
            max_altitude=self.max_altitude,
            throttle=throttle,
        )
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
        # Normalize broad state ranges so PPO sees comparable feature scales.
        altitude = (2.0 * (self.z / self.max_altitude)) - 1.0
        velocity = np.clip(self.v / self.OBS_VELOCITY_SCALE, -1.0, 1.0)
        fuel = (2.0 * (self.fuel / self.max_fuel)) - 1.0
        mass = (2.0 * (self.mass / (self.dry_mass + self.max_fuel))) - 1.0
        return np.array([altitude, velocity, fuel, mass], dtype=np.float32)

    def render(self):
        print(
            f"step={self.steps:3d}  "
            f"z={self.z:7.2f} m  "
            f"v={self.v:+7.2f} m/s  "
            f"fuel={self.fuel:5.2f} kg  "
            f"mass={self.mass:5.2f} kg"
        )
