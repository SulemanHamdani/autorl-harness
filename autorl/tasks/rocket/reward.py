"""Reward helpers for the rocket landing reference task."""

from __future__ import annotations


def compute_shaping_reward(altitude: float, velocity: float) -> float:
    """Dense shaping reward used during descent."""

    return -abs(velocity) * 0.01 - altitude * 0.01


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
