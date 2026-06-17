"""Glue code that connects G1 WBC policy rollouts to DIAL-style MPC."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import mujoco

if TYPE_CHECKING:
    from brax.base import System

from dial_mpc.g1_wbc_jax.metrics import compute_rollout_scores, score_from_terms
from dial_mpc.g1_wbc_jax.motion import G1Motion
from dial_mpc.g1_wbc_jax.obs import WbcObsState
from dial_mpc.g1_wbc_jax.policy import WbcActorParams
from dial_mpc.g1_wbc_jax.rollout import G1WbcRolloutConfig, make_mjx_policy_rollout, make_policy_rollout


class RolloutScoreContext(NamedTuple):
    initial_qpos: jnp.ndarray
    initial_qvel: jnp.ndarray
    initial_last_action: jnp.ndarray
    initial_obs_state: WbcObsState
    ref_start: jnp.ndarray


def make_wbc_rollout_score_fn(
    sys: System,
    actor: WbcActorParams,
    motion: G1Motion,
    rollout_config: G1WbcRolloutConfig | None = None,
    *,
    mode: str = "g1_wbc_joint_global",
    reward_weights: dict[str, float] | None = None,
):
    """Return ``score_fn(candidate_qpos, context)`` for the context-aware MPC core."""

    rollout_fn = make_policy_rollout(sys, actor, motion, rollout_config)

    def score_fn(candidate_qpos: jnp.ndarray, context: RolloutScoreContext) -> jnp.ndarray:
        rollout = rollout_fn(
            candidate_qpos,
            context.initial_qpos,
            context.initial_qvel,
            context.initial_last_action,
            context.initial_obs_state,
            context.ref_start,
        ).trace
        _, terms = compute_rollout_scores(motion, rollout)
        return score_from_terms(terms, mode=mode, reward_weights=reward_weights)

    return score_fn


def make_wbc_mjx_rollout_score_fn(
    model: mujoco.MjModel,
    actor: WbcActorParams,
    motion: G1Motion,
    rollout_config: G1WbcRolloutConfig | None = None,
    *,
    mode: str = "g1_wbc_joint_global",
    reward_weights: dict[str, float] | None = None,
):
    """Return a DIAL score function backed by native MuJoCo MJX rollouts."""

    rollout_fn = make_mjx_policy_rollout(model, actor, motion, rollout_config)

    def score_fn(candidate_qpos: jnp.ndarray, context: RolloutScoreContext) -> jnp.ndarray:
        rollout = rollout_fn(
            candidate_qpos,
            context.initial_qpos,
            context.initial_qvel,
            context.initial_last_action,
            context.initial_obs_state,
            context.ref_start,
        ).trace
        _, terms = compute_rollout_scores(motion, rollout)
        return score_from_terms(terms, mode=mode, reward_weights=reward_weights)

    return score_fn
