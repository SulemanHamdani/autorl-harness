"""Rocket landing reference task package."""

from tasks.rocket import task as _task_registration
from tasks.rocket.env import RocketLandingEnv
from tasks.rocket.task import ROCKET_TASK, RocketTaskAdapter

__all__ = ["ROCKET_TASK", "RocketLandingEnv", "RocketTaskAdapter"]
