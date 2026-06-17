"""DIAL-style receding-horizon MPC core for G1 WBC commands.

This module owns the JAX sampling/update logic. It deliberately depends on a
JAX ``score_fn(candidate_qpos)`` callback so the MJX rollout backend can be
implemented and optimized independently while preserving one MPC implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from dial_mpc.g1_wbc_jax.constants import COMMAND_DELTA_DIM, POLICY_DT, QPOS_DIM
from dial_mpc.g1_wbc_jax.math import (
    axis_angle_from_quat,
    normalize,
    quat_from_axis_angle,
    quat_mul,
)

ScoreFn = Callable[[jnp.ndarray], jnp.ndarray]
ContextScoreFn = Callable[[jnp.ndarray, object], jnp.ndarray]


@dataclass(frozen=True)
class G1WbcDialMpcConfig:
    num_samples: int = 64
    planning_horizon_steps: int = 30
    control_steps: int = 10
    node_count: int = 4
    n_diffuse: int = 1
    n_diffuse_init: int = 4
    temp_sample: float = 0.7
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    root_pos_sigma: float = 0.08
    root_rot_sigma: float = 0.18
    joint_sigma: float = 0.28
    min_root_pos_sigma: float = 0.002
    min_root_rot_sigma: float = 0.004
    min_joint_sigma: float = 0.008
    command_reg_weight: float = 0.0
    command_smooth_weight: float = 0.0
    freeze_first_frame: bool = True


class WindowResult(NamedTuple):
    plan_nodes: jnp.ndarray
    best_qpos: jnp.ndarray
    best_delta: jnp.ndarray
    scores: jnp.ndarray
    best_score: jnp.ndarray
    mean_score: jnp.ndarray


def initial_plan(config: G1WbcDialMpcConfig) -> jnp.ndarray:
    return jnp.zeros((config.node_count, COMMAND_DELTA_DIM), dtype=jnp.float32)


def shift_plan(config: G1WbcDialMpcConfig, plan_nodes: jnp.ndarray, execute_steps: int | None = None) -> jnp.ndarray:
    execute = config.control_steps if execute_steps is None else int(execute_steps)
    horizon_delta = expand_delta_to_horizon(plan_nodes[:, None, :], config.planning_horizon_steps)[:, 0]
    shifted = jnp.roll(horizon_delta, -execute, axis=0)
    shifted = shifted.at[-execute:].set(0.0)
    idx = _node_indices(config.planning_horizon_steps, config.node_count)
    nodes = shifted[idx]
    if config.freeze_first_frame:
        nodes = nodes.at[0].set(0.0)
    return nodes


def make_optimize_window(config: G1WbcDialMpcConfig, score_fn: ScoreFn, *, init: bool = False):
    """Build a JIT-compiled window optimizer around a pure JAX score function.

    ``score_fn`` receives candidate qpos trajectories shaped ``(H, N, QPOS_DIM)``
    and returns one score per candidate shaped ``(N,)``.
    """

    n_diffuse = config.n_diffuse_init if init else config.n_diffuse
    noise_schedule = _noise_schedule(config, n_diffuse)

    @jax.jit
    def optimize_window(
        rng: jnp.ndarray,
        base_qpos: jnp.ndarray,
        plan_nodes: jnp.ndarray,
        joint_low: jnp.ndarray,
        joint_high: jnp.ndarray,
    ) -> tuple[jnp.ndarray, WindowResult]:
        def reverse_once(carry, noise_scale):
            rng_i, mean_nodes, best_score, best_qpos, best_delta = carry
            rng_i, sample_rng = jax.random.split(rng_i)
            eps = jax.random.normal(
                sample_rng,
                (config.num_samples, config.node_count, COMMAND_DELTA_DIM),
                dtype=mean_nodes.dtype,
            )
            candidates = mean_nodes[None, :, :] + eps * noise_scale[None, :, :]
            candidates = candidates.at[:, 0, :].set(0.0) if config.freeze_first_frame else candidates
            candidates = jnp.concatenate([candidates, mean_nodes[None, :, :]], axis=0)
            delta = expand_delta_to_horizon(jnp.swapaxes(candidates, 0, 1), config.planning_horizon_steps)
            qpos = apply_delta_to_qpos(base_qpos, delta, joint_low=joint_low, joint_high=joint_high)
            scores = score_fn(qpos) - command_regularization(
                delta,
                config.command_reg_weight,
                config.command_smooth_weight,
            )
            candidate_best_idx = jnp.argmax(scores)
            candidate_best_score = scores[candidate_best_idx]
            candidate_best_qpos = qpos[:, candidate_best_idx]
            candidate_best_delta = delta[:, candidate_best_idx]
            update_best = candidate_best_score > best_score
            best_score = jnp.where(update_best, candidate_best_score, best_score)
            best_qpos = jnp.where(update_best, candidate_best_qpos, best_qpos)
            best_delta = jnp.where(update_best, candidate_best_delta, best_delta)

            baseline = scores[-1]
            normalized = (scores - baseline) / (jnp.std(scores) + 1.0e-6)
            weights = jax.nn.softmax(normalized / max(float(config.temp_sample), 1.0e-6))
            mean_nodes = jnp.einsum("n,nkd->kd", weights, candidates)
            mean_nodes = mean_nodes.at[0].set(0.0) if config.freeze_first_frame else mean_nodes
            return (rng_i, mean_nodes, best_score, best_qpos, best_delta), scores

        horizon = int(base_qpos.shape[0])
        zero_qpos = jnp.broadcast_to(base_qpos[:, None, :], (horizon, 1, QPOS_DIM))[:, 0]
        zero_delta = jnp.zeros((horizon, COMMAND_DELTA_DIM), dtype=base_qpos.dtype)
        carry0 = (
            rng,
            plan_nodes,
            jnp.asarray(-jnp.inf, dtype=base_qpos.dtype),
            zero_qpos,
            zero_delta,
        )
        (rng, plan_nodes, best_score, best_qpos, best_delta), score_history = jax.lax.scan(
            reverse_once,
            carry0,
            noise_schedule,
        )
        final_scores = score_history[-1]
        return rng, WindowResult(
            plan_nodes=plan_nodes,
            best_qpos=best_qpos,
            best_delta=best_delta,
            scores=final_scores,
            best_score=best_score,
            mean_score=jnp.mean(final_scores),
        )

    return optimize_window


def make_optimize_window_with_context(
    config: G1WbcDialMpcConfig,
    score_fn: ContextScoreFn,
    *,
    init: bool = False,
):
    """Build a JIT-compiled optimizer whose score function also receives state context."""

    n_diffuse = config.n_diffuse_init if init else config.n_diffuse
    noise_schedule = _noise_schedule(config, n_diffuse)

    @jax.jit
    def optimize_window(
        rng: jnp.ndarray,
        base_qpos: jnp.ndarray,
        plan_nodes: jnp.ndarray,
        joint_low: jnp.ndarray,
        joint_high: jnp.ndarray,
        score_context,
    ) -> tuple[jnp.ndarray, WindowResult]:
        def reverse_once(carry, noise_scale):
            rng_i, mean_nodes, best_score, best_qpos, best_delta = carry
            rng_i, sample_rng = jax.random.split(rng_i)
            eps = jax.random.normal(
                sample_rng,
                (config.num_samples, config.node_count, COMMAND_DELTA_DIM),
                dtype=mean_nodes.dtype,
            )
            candidates = mean_nodes[None, :, :] + eps * noise_scale[None, :, :]
            candidates = candidates.at[:, 0, :].set(0.0) if config.freeze_first_frame else candidates
            candidates = jnp.concatenate([candidates, mean_nodes[None, :, :]], axis=0)
            delta = expand_delta_to_horizon(jnp.swapaxes(candidates, 0, 1), config.planning_horizon_steps)
            qpos = apply_delta_to_qpos(base_qpos, delta, joint_low=joint_low, joint_high=joint_high)
            scores = score_fn(qpos, score_context) - command_regularization(
                delta,
                config.command_reg_weight,
                config.command_smooth_weight,
            )
            candidate_best_idx = jnp.argmax(scores)
            candidate_best_score = scores[candidate_best_idx]
            candidate_best_qpos = qpos[:, candidate_best_idx]
            candidate_best_delta = delta[:, candidate_best_idx]
            update_best = candidate_best_score > best_score
            best_score = jnp.where(update_best, candidate_best_score, best_score)
            best_qpos = jnp.where(update_best, candidate_best_qpos, best_qpos)
            best_delta = jnp.where(update_best, candidate_best_delta, best_delta)

            baseline = scores[-1]
            normalized = (scores - baseline) / (jnp.std(scores) + 1.0e-6)
            weights = jax.nn.softmax(normalized / max(float(config.temp_sample), 1.0e-6))
            mean_nodes = jnp.einsum("n,nkd->kd", weights, candidates)
            mean_nodes = mean_nodes.at[0].set(0.0) if config.freeze_first_frame else mean_nodes
            return (rng_i, mean_nodes, best_score, best_qpos, best_delta), scores

        horizon = int(base_qpos.shape[0])
        zero_qpos = jnp.broadcast_to(base_qpos[:, None, :], (horizon, 1, QPOS_DIM))[:, 0]
        zero_delta = jnp.zeros((horizon, COMMAND_DELTA_DIM), dtype=base_qpos.dtype)
        carry0 = (
            rng,
            plan_nodes,
            jnp.asarray(-jnp.inf, dtype=base_qpos.dtype),
            zero_qpos,
            zero_delta,
        )
        (rng, plan_nodes, best_score, best_qpos, best_delta), score_history = jax.lax.scan(
            reverse_once,
            carry0,
            noise_schedule,
        )
        final_scores = score_history[-1]
        return rng, WindowResult(
            plan_nodes=plan_nodes,
            best_qpos=best_qpos,
            best_delta=best_delta,
            scores=final_scores,
            best_score=best_score,
            mean_score=jnp.mean(final_scores),
        )

    return optimize_window


def expand_delta_to_horizon(delta_nodes: jnp.ndarray, horizon: int) -> jnp.ndarray:
    """Expand ``(K, N, 35)`` command delta nodes to ``(H, N, 35)``."""

    if delta_nodes.shape[0] == horizon:
        return delta_nodes
    if delta_nodes.shape[0] <= 1:
        return jnp.broadcast_to(delta_nodes, (horizon,) + delta_nodes.shape[1:])
    root_pos = _linear_interpolate_time_major(delta_nodes[..., :3], horizon)
    root_rot = _slerp_axis_angle_delta(delta_nodes[..., 3:6], horizon)
    joint = _linear_interpolate_time_major(delta_nodes[..., 6:], horizon)
    out = jnp.concatenate([root_pos, root_rot, joint], axis=-1)
    return out.at[0].set(0.0)


def apply_delta_to_qpos(
    base_qpos: jnp.ndarray,
    delta: jnp.ndarray,
    *,
    joint_low: jnp.ndarray,
    joint_high: jnp.ndarray,
) -> jnp.ndarray:
    qpos = jnp.broadcast_to(base_qpos[:, None, :], (delta.shape[0], delta.shape[1], QPOS_DIM))
    root_pos = qpos[..., :3] + delta[..., :3]
    delta_quat = quat_from_axis_angle(delta[..., 3:6])
    root_quat = normalize(quat_mul(delta_quat, qpos[..., 3:7]))
    joints = jnp.clip(qpos[..., 7:] + delta[..., 6:], joint_low, joint_high)
    return jnp.concatenate([root_pos, root_quat, joints], axis=-1)


def command_regularization(delta: jnp.ndarray, reg_weight: float, smooth_weight: float) -> jnp.ndarray:
    reg = jnp.mean(delta * delta, axis=(0, 2))
    if delta.shape[0] > 1:
        smooth = jnp.mean(jnp.diff(delta, axis=0) ** 2, axis=(0, 2)) / (POLICY_DT**2)
    else:
        smooth = jnp.zeros(delta.shape[1], dtype=delta.dtype)
    return float(reg_weight) * reg + float(smooth_weight) * smooth


def _noise_schedule(config: G1WbcDialMpcConfig, n_diffuse: int) -> jnp.ndarray:
    base = _base_sigma(config)
    horizon_scale = config.horizon_diffuse_factor ** jnp.arange(config.node_count - 1, -1, -1)
    node_sigma = base[None, :] * horizon_scale[:, None]
    traj_scale = config.traj_diffuse_factor ** jnp.arange(n_diffuse)
    schedule = node_sigma[None, :, :] * traj_scale[:, None, None]
    min_sigma = _min_sigma(config)
    schedule = jnp.maximum(schedule, min_sigma[None, None, :])
    if config.freeze_first_frame:
        schedule = schedule.at[:, 0, :].set(0.0)
    return schedule


def _base_sigma(config: G1WbcDialMpcConfig) -> jnp.ndarray:
    return jnp.concatenate(
        [
            jnp.full((3,), config.root_pos_sigma, dtype=jnp.float32),
            jnp.full((3,), config.root_rot_sigma, dtype=jnp.float32),
            jnp.full((COMMAND_DELTA_DIM - 6,), config.joint_sigma, dtype=jnp.float32),
        ]
    )


def _min_sigma(config: G1WbcDialMpcConfig) -> jnp.ndarray:
    return jnp.concatenate(
        [
            jnp.full((3,), config.min_root_pos_sigma, dtype=jnp.float32),
            jnp.full((3,), config.min_root_rot_sigma, dtype=jnp.float32),
            jnp.full((COMMAND_DELTA_DIM - 6,), config.min_joint_sigma, dtype=jnp.float32),
        ]
    )


def _linear_interpolate_time_major(value: jnp.ndarray, target_steps: int) -> jnp.ndarray:
    if value.shape[0] == target_steps:
        return value
    positions = jnp.linspace(0.0, value.shape[0] - 1, target_steps)
    left = jnp.floor(positions).astype(jnp.int32)
    right = jnp.clip(left + 1, 0, value.shape[0] - 1)
    blend = (positions - left.astype(value.dtype)).reshape(target_steps, 1, 1)
    return value[left] * (1.0 - blend) + value[right] * blend


def _slerp_axis_angle_delta(delta_axis_angle: jnp.ndarray, target_steps: int) -> jnp.ndarray:
    if delta_axis_angle.shape[0] == target_steps:
        return delta_axis_angle
    if delta_axis_angle.shape[0] <= 1:
        return jnp.broadcast_to(delta_axis_angle, (target_steps,) + delta_axis_angle.shape[1:])
    knot_quat = quat_from_axis_angle(delta_axis_angle)
    positions = jnp.linspace(0.0, knot_quat.shape[0] - 1, target_steps)
    left = jnp.floor(positions).astype(jnp.int32)
    right = jnp.clip(left + 1, 0, knot_quat.shape[0] - 1)
    blend = (positions - left.astype(knot_quat.dtype)).reshape(target_steps, 1, 1)
    q = _quat_slerp(knot_quat[left], knot_quat[right], blend)
    return axis_angle_from_quat(q)


def _quat_slerp(q0: jnp.ndarray, q1: jnp.ndarray, blend: jnp.ndarray) -> jnp.ndarray:
    dot = jnp.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = jnp.where(dot < 0.0, -q1, q1)
    dot = jnp.clip(jnp.abs(dot), 0.0, 1.0)
    small = dot > 0.9995
    theta_0 = jnp.arccos(dot)
    sin_theta_0 = jnp.clip(jnp.sin(theta_0), min=1.0e-8)
    theta = theta_0 * blend
    s0 = jnp.sin(theta_0 - theta) / sin_theta_0
    s1 = jnp.sin(theta) / sin_theta_0
    out = s0 * q0 + s1 * q1
    lerp = q0 + blend * (q1 - q0)
    return normalize(jnp.where(small, lerp, out))


def _node_indices(horizon: int, node_count: int) -> jnp.ndarray:
    return jnp.round(jnp.linspace(0, horizon - 1, node_count)).astype(jnp.int32)
