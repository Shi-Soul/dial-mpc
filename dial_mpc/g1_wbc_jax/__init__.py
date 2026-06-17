"""JAX migration path for the G1 WBC retargeting MPC."""

from dial_mpc.g1_wbc_jax.constants import ACTION_DIM, OBS_DIM, POLICY_DT, QPOS_DIM


_LAZY_EXPORTS = {
    "save_command_npz": ("dial_mpc.g1_wbc_jax.artifacts", "save_command_npz"),
    "save_rollout_npz": ("dial_mpc.g1_wbc_jax.artifacts", "save_rollout_npz"),
    "G1CommandBatch": ("dial_mpc.g1_wbc_jax.motion", "G1CommandBatch"),
    "G1Motion": ("dial_mpc.g1_wbc_jax.motion", "G1Motion"),
    "load_motion": ("dial_mpc.g1_wbc_jax.motion", "load_motion"),
    "make_optimize_window_with_context": (
        "dial_mpc.g1_wbc_jax.mpc",
        "make_optimize_window_with_context",
    ),
    "RolloutScoreContext": ("dial_mpc.g1_wbc_jax.planner", "RolloutScoreContext"),
    "make_wbc_mjx_rollout_score_fn": (
        "dial_mpc.g1_wbc_jax.planner",
        "make_wbc_mjx_rollout_score_fn",
    ),
    "make_wbc_rollout_score_fn": (
        "dial_mpc.g1_wbc_jax.planner",
        "make_wbc_rollout_score_fn",
    ),
    "WbcActorParams": ("dial_mpc.g1_wbc_jax.policy", "WbcActorParams"),
    "actor_forward": ("dial_mpc.g1_wbc_jax.policy", "actor_forward"),
    "load_torch_actor": ("dial_mpc.g1_wbc_jax.policy", "load_torch_actor"),
    "G1WbcRolloutConfig": ("dial_mpc.g1_wbc_jax.rollout", "G1WbcRolloutConfig"),
    "command_batch_from_qpos_trajectory": (
        "dial_mpc.g1_wbc_jax.rollout",
        "command_batch_from_qpos_trajectory",
    ),
    "mjx_command_batch_from_qpos_trajectory": (
        "dial_mpc.g1_wbc_jax.rollout",
        "mjx_command_batch_from_qpos_trajectory",
    ),
    "make_mjx_open_loop_rollout": (
        "dial_mpc.g1_wbc_jax.rollout",
        "make_mjx_open_loop_rollout",
    ),
    "make_mjx_policy_rollout": (
        "dial_mpc.g1_wbc_jax.rollout",
        "make_mjx_policy_rollout",
    ),
    "make_open_loop_rollout": ("dial_mpc.g1_wbc_jax.rollout", "make_open_loop_rollout"),
    "make_policy_rollout": ("dial_mpc.g1_wbc_jax.rollout", "make_policy_rollout"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "ACTION_DIM",
    "OBS_DIM",
    "POLICY_DT",
    "QPOS_DIM",
    "save_command_npz",
    "save_rollout_npz",
    "G1CommandBatch",
    "G1Motion",
    "load_motion",
    "make_optimize_window_with_context",
    "RolloutScoreContext",
    "make_wbc_mjx_rollout_score_fn",
    "make_wbc_rollout_score_fn",
    "WbcActorParams",
    "actor_forward",
    "load_torch_actor",
    "G1WbcRolloutConfig",
    "command_batch_from_qpos_trajectory",
    "mjx_command_batch_from_qpos_trajectory",
    "make_mjx_open_loop_rollout",
    "make_mjx_policy_rollout",
    "make_open_loop_rollout",
    "make_policy_rollout",
]
