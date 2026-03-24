"""Reward helpers for the rocket landing reference task."""

from __future__ import annotations


def compute_shaping_reward(
    altitude: float,
    velocity: float,
    safe_velocity: float,
    max_altitude: float,
) -> float:
    """Dense shaping reward used during descent."""

    clipped_altitude = min(max(altitude, 0.0), max_altitude)
    altitude_ratio = clipped_altitude / max_altitude
    target_speed = (0.5 * safe_velocity) + (15.0 * altitude_ratio**0.5)
    target_velocity = -target_speed
    velocity_error = abs(velocity - target_velocity)
    rising_penalty = max(velocity, 0.0)
    descent_speed = max(-velocity, 0.0)
    excess_landing_speed = max(descent_speed - safe_velocity, 0.0)
    flare_window = max(1.0 - (clipped_altitude / 120.0), 0.0)

    return (
        -0.03 * velocity_error
        - 0.002 * clipped_altitude
        - 0.02 * rising_penalty
        - 0.05 * flare_window * excess_landing_speed
    )


def compute_terminal_reward(
    altitude: float,
    velocity: float,
    safe_velocity: float,
    max_altitude: float,
) -> float | None:
    """Return a terminal reward override when an episode ends."""

    if altitude <= 0:
        if abs(velocity) < safe_velocity:
            return 100.0
        return -100.0 * (abs(velocity) / safe_velocity)

    if altitude > max_altitude:
        return -100.0

    return None
