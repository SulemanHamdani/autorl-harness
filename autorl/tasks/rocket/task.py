"""Rocket reference task adapter."""

from __future__ import annotations

from autorl.registry import TaskAdapter, register_task
from autorl.schemas import GuardrailSpec, JsonDict, MetricMap, TaskSpec
from tasks.rocket.env import RocketLandingEnv


ROCKET_TASK = {
    "task_id": "rocket",
    "description": "1D rocket landing reference task.",
    "editable_files": [
        "autorl/tasks/rocket/spec.yaml",
        "autorl/tasks/rocket/reward.py",
    ],
    "primary_score": "soft_landing_rate",
    "backend": "sb3",
    "supported_algorithms": ["ppo"],
    "default_budget": {
        "total_timesteps": 500_000,
        "eval_freq": 5_000,
        "n_eval_episodes": 10,
    },
    "allowed_interventions": (
        "hyperparameter_tuning",
        "reward_shaping",
        "initial_state_adjustment",
    ),
    "guardrails": (
        GuardrailSpec(
            name="crash_rate",
            mode="max",
            threshold=1.0,
            description="Crash rate should not exceed 100%.",
        ),
        GuardrailSpec(
            name="timeout_rate",
            mode="max",
            threshold=1.0,
            description="Timeout rate should not exceed 100%.",
        ),
        GuardrailSpec(
            name="invalid_run",
            mode="boolean",
            threshold=False,
            description="Invalid runs are always vetoed.",
        ),
    ),
    "success_criteria": {
        "soft_landing_rate": 0.95,
    },
}


class RocketTaskAdapter(TaskAdapter):
    """Task adapter for the 1D rocket landing environment."""

    @property
    def module_name(self) -> str:
        return "tasks.rocket"

    def get_task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id=ROCKET_TASK["task_id"],
            description=ROCKET_TASK["description"],
            backend=ROCKET_TASK["backend"],
            supported_algorithms=tuple(ROCKET_TASK["supported_algorithms"]),
            default_algorithm=ROCKET_TASK["supported_algorithms"][0],
            default_budget=dict(ROCKET_TASK["default_budget"]),
            editable_files=tuple(ROCKET_TASK["editable_files"]),
            primary_score_name=ROCKET_TASK["primary_score"],
            guardrails=tuple(ROCKET_TASK["guardrails"]),
            allowed_interventions=tuple(ROCKET_TASK["allowed_interventions"]),
            success_criteria=dict(ROCKET_TASK["success_criteria"]),
        )

    def build_train_env(self, **kwargs):
        return RocketLandingEnv(**kwargs)

    def build_eval_env(self, **kwargs):
        return RocketLandingEnv(**kwargs)

    def get_default_experiment_config(self) -> JsonDict:
        spec = self.get_task_spec()
        return {
            "algorithm": spec.default_algorithm,
            "budget": dict(spec.default_budget),
            "seed_set": [0],
            "editable_files": list(spec.editable_files),
        }

    def get_allowed_interventions(self) -> tuple[str, ...]:
        return self.get_task_spec().allowed_interventions

    def get_primary_score_definition(self) -> tuple[str, tuple[GuardrailSpec, ...]]:
        spec = self.get_task_spec()
        return spec.primary_score_name, spec.guardrails

    def compute_summary_metrics(self, raw_metrics: MetricMap) -> MetricMap:
        summary = dict(raw_metrics)
        summary.setdefault("soft_landing_rate", 0.0)
        summary.setdefault("crash_rate", 0.0)
        summary.setdefault("timeout_rate", 0.0)
        summary.setdefault("invalid_run", False)
        return summary

    def get_success_criteria(self) -> JsonDict:
        return dict(self.get_task_spec().success_criteria)


register_task(RocketTaskAdapter())
