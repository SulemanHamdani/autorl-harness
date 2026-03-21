"""
1D Rocket Landing Environment
==============================
A minimal Gymnasium environment that simulates a rocket descending
vertically under gravity.  The agent controls a single thruster
(on / off) and must land softly (|velocity| < safe_velocity).

State : [z, v, fuel, mass]
Action: Discrete(2)  — 0 = engine off, 1 = thrust
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        # ---- physics constants ----
        self.gravity = 9.8            # m/s²
        self.thrust_force = 400.0     # Newtons
        self.fuel_consumption = 1.0   # kg per second of thrust
        self.dry_mass = 10.0          # kg (rocket without fuel)
        self.initial_fuel = 20.0      # kg
        self.dt = 0.1                 # timestep (seconds)
        self.max_steps = 500
        self.safe_velocity = 5.0      # m/s — soft-landing threshold
        self.max_altitude = 500.0     # m — upper boundary

        # ---- Gymnasium spaces ----
        # Observations: [z, v, fuel, mass]
        #   z    ∈ [0, max_altitude]
        #   v    ∈ [-inf, inf]  (but practically bounded)
        #   fuel ∈ [0, initial_fuel]
        #   mass ∈ [dry_mass, dry_mass + initial_fuel]
        obs_low = np.array([0.0, -np.inf, 0.0, self.dry_mass], dtype=np.float32)
        obs_high = np.array(
            [self.max_altitude, np.inf, self.initial_fuel, self.dry_mass + self.initial_fuel],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Actions: 0 = off, 1 = thrust
        self.action_space = spaces.Discrete(2)

        self.render_mode = render_mode

        # internal state (set properly in reset)
        self.z = 0.0
        self.v = 0.0
        self.fuel = 0.0
        self.mass = 0.0
        self.steps = 0

    # ------------------------------------------------------------------
    # reset — start a new episode
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # step — advance the simulation by one timestep
    # ------------------------------------------------------------------
    def step(self, action):
        # --- apply thrust if requested and fuel available ---
        if action == 1 and self.fuel > 0:
            fuel_used = min(self.fuel, self.fuel_consumption * self.dt)
            self.fuel -= fuel_used
            self.mass -= fuel_used
            acceleration = (self.thrust_force / self.mass) - self.gravity
        else:
            acceleration = -self.gravity

        # --- Euler integration ---
        self.v += acceleration * self.dt
        self.z += self.v * self.dt

        self.steps += 1

        # --- reward shaping (every step) ---
        # Nudge 1: penalize high speed — encourages the agent to slow down
        # Nudge 2: penalize high altitude — encourages descending
        reward = -abs(self.v) * 0.01 - self.z * 0.01

        # --- check termination ---
        terminated = False
        truncated = False

        if self.z <= 0:
            self.z = 0.0
            terminated = True
            if abs(self.v) < self.safe_velocity:
                reward = 100.0   # soft landing!
            else:
                reward = -100.0 * (abs(self.v) / self.safe_velocity)  # harder crash = worse

        elif self.z > self.max_altitude:
            terminated = True
            reward = -100.0

        elif self.steps >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
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
