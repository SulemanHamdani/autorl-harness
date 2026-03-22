"""Compatibility shim for the rocket task environment."""

from bootstrap import bootstrap_autorl_paths

bootstrap_autorl_paths()

from tasks.rocket.env import RocketLandingEnv

__all__ = ["RocketLandingEnv"]
