"""JAX rollout score terms for G1 WBC MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp

from dial_mpc.g1_wbc_jax.constants import (
    ANCHOR_BODY_NAME,
    HAND_EE_BODY_NAMES,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    TASK_EE_BODY_NAMES,
)
from dial_mpc.g1_wbc_jax.math import quat_error_magnitude, subtract_frame_transforms
from dial_mpc.g1_wbc_jax.motion import G1Motion


class RolloutTrace(NamedTuple):
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    body_pos_w: jnp.ndarray
    body_quat_w: jnp.ndarray
    body_lin_vel_w: jnp.ndarray
    body_ang_vel_w: jnp.ndarray
    actions: jnp.ndarray
    controls: jnp.ndarray
    contact_indicator: jnp.ndarray
    contact_force: jnp.ndarray
    ref_indices: jnp.ndarray
    floor_contact_indicator: jnp.ndarray
    floor_contact_force: jnp.ndarray
    dt: float = POLICY_DT


@dataclass(frozen=True)
class MetricThresholds:
    root_pos_mean: float = 0.25
    root_rot_mean: float = 0.6
    ee_global_pos_mean: float = 0.25
    ee_local_pos_mean: float = 0.20
    contact_mismatch_rate: float = 0.35


REWARD_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "g1_wbc_joint_global": {
        "bad_floor_contact": 45.0,
        "bad_floor_force_excess": 10.0,
        "contact_switch": 12.0,
        "contact_force_delta": 2.5,
        "contact_false_positive": 1.5,
        "contact_false_negative": 0.4,
        "control_delta": 1.8,
        "action_delta": 0.6,
        "joint_acc": 0.006,
        "joint_jerk": 0.0012,
        "body_global_pos_error": 4.0,
        "body_global_rot_error": 0.8,
        "ee_global_pos_error": 1.5,
        "ee_global_rot_error": 0.3,
    },
    "g1_wbc_joint": {
        "bad_floor_contact": 35.0,
        "bad_floor_force_excess": 8.0,
        "contact_switch": 10.0,
        "contact_force_delta": 2.0,
        "contact_false_positive": 0.8,
        "contact_false_negative": 0.3,
        "control_delta": 1.6,
        "action_delta": 0.5,
        "joint_acc": 0.006,
        "joint_jerk": 0.0012,
        "body_local_pos_error": 26.0,
        "body_local_rot_error": 3.0,
        "joint_pos_error": 2.1,
        "ee_local_pos_error": 6.0,
        "ee_local_rot_error": 1.2,
        "body_global_pos_error": 0.8,
        "body_global_rot_error": 0.2,
        "ee_global_pos_error": 0.3,
    },
    "g1_wbc_ee": {
        "bad_floor_contact": 35.0,
        "bad_floor_force_excess": 8.0,
        "contact_switch": 10.0,
        "contact_force_delta": 2.0,
        "contact_false_positive": 0.5,
        "contact_false_negative": 0.2,
        "control_delta": 2.0,
        "action_delta": 0.6,
        "joint_acc": 0.006,
        "joint_jerk": 0.0015,
        "hand_global_pos_error": 35.0,
        "hand_global_rot_error": 3.0,
        "hand_local_pos_error": 8.0,
        "hand_local_rot_error": 1.5,
        "ee_global_pos_error": 2.0,
        "ee_global_rot_error": 0.4,
        "body_global_pos_error": 0.8,
        "body_local_pos_error": 0.8,
    },
}


def compute_rollout_metrics(
    motion: G1Motion,
    rollout: RolloutTrace,
    *,
    thresholds: MetricThresholds = MetricThresholds(),
) -> dict[str, float | bool]:
    ref_idx = rollout.ref_indices
    ref_qpos = motion.qpos()[ref_idx]
    ref_qvel = motion.qvel()[ref_idx]
    ref_body_pos = motion.body_pos_w[ref_idx]
    ref_body_quat = motion.body_quat_w[ref_idx]
    ref_contact = motion.contact[ref_idx]

    root_pos_err = jnp.linalg.norm(rollout.qpos[..., :3] - ref_qpos[..., :3], axis=-1)
    root_rot_err = quat_error_magnitude(rollout.qpos[..., 3:7], ref_qpos[..., 3:7])
    joint_pos_err = jnp.linalg.norm(rollout.qpos[..., 7:] - ref_qpos[..., 7:], axis=-1)
    joint_vel_err = jnp.linalg.norm(rollout.qvel[..., 6:] - ref_qvel[..., 6:], axis=-1)

    body_pos_err = jnp.linalg.norm(rollout.body_pos_w - ref_body_pos, axis=-1)
    body_rot_err = quat_error_magnitude(rollout.body_quat_w, ref_body_quat)

    ee_indices = _body_indices(TASK_EE_BODY_NAMES)
    hand_indices = _body_indices(HAND_EE_BODY_NAMES)
    local_body_indices = _body_indices(tuple(name for name in MUJOCO_BODY_NAMES if name != ANCHOR_BODY_NAME))
    ee_pos_err = jnp.take(body_pos_err, ee_indices, axis=-1)
    ee_rot_err = jnp.take(body_rot_err, ee_indices, axis=-1)
    ee_local_pos_err, ee_local_rot_err = _local_body_errors(
        rollout.body_pos_w, rollout.body_quat_w, ref_body_pos, ref_body_quat, ee_indices
    )
    hand_pos_err = jnp.take(body_pos_err, hand_indices, axis=-1)
    hand_rot_err = jnp.take(body_rot_err, hand_indices, axis=-1)
    hand_local_pos_err, hand_local_rot_err = _local_body_errors(
        rollout.body_pos_w, rollout.body_quat_w, ref_body_pos, ref_body_quat, hand_indices
    )
    body_local_pos_err, body_local_rot_err = _local_body_errors(
        rollout.body_pos_w, rollout.body_quat_w, ref_body_pos, ref_body_quat, local_body_indices
    )

    sim_contact_eval = rollout.contact_indicator[1:]
    ref_contact_eval = ref_contact[1:]
    contact_err = jnp.abs(sim_contact_eval - ref_contact_eval)
    false_positive = ((sim_contact_eval > 0.5) & (ref_contact_eval <= 0.5)).astype(jnp.float32)
    false_negative = ((sim_contact_eval <= 0.5) & (ref_contact_eval > 0.5)).astype(jnp.float32)
    contact_switch = _switch_rate(sim_contact_eval)
    ref_contact_switch = _switch_rate(ref_contact_eval)
    contact_force = rollout.contact_force[1:]
    contact_force_excess = _contact_force_excess(contact_force)
    contact_force_delta = _diff_norm(contact_force) / _contact_force_scale()
    bad_floor_contact = rollout.floor_contact_indicator[1:, :, 2:]
    bad_floor_force = rollout.floor_contact_force[1:, :, 2:]
    bad_floor_force_excess = bad_floor_force / _contact_force_scale()
    action_delta = _diff_norm(rollout.actions)
    ctrl_delta = _diff_norm(rollout.controls)
    dt = jnp.maximum(jnp.asarray(rollout.dt, dtype=rollout.qvel.dtype), 1.0e-6)
    joint_acc = _diff_norm(rollout.qvel[..., 6:]) / dt
    joint_jerk = _diff_norm(jnp.diff(rollout.qvel[..., 6:], axis=0)) / dt

    metrics: dict[str, float | bool] = {
        "num_steps": float(rollout.actions.shape[0]),
        "root_pos_error_mean": _scalar_mean(root_pos_err),
        "root_pos_error_max": _scalar_max(root_pos_err),
        "root_rot_error_mean": _scalar_mean(root_rot_err),
        "joint_pos_error_mean": _scalar_mean(joint_pos_err),
        "joint_vel_error_mean": _scalar_mean(joint_vel_err),
        "body_global_pos_error_mean": _scalar_mean(body_pos_err),
        "body_global_rot_error_mean": _scalar_mean(body_rot_err),
        "ee_global_pos_error_mean": _scalar_mean(ee_pos_err),
        "ee_global_rot_error_mean": _scalar_mean(ee_rot_err),
        "ee_local_pos_error_mean": _scalar_mean(ee_local_pos_err),
        "ee_local_rot_error_mean": _scalar_mean(ee_local_rot_err),
        "hand_global_pos_error_mean": _scalar_mean(hand_pos_err),
        "hand_global_rot_error_mean": _scalar_mean(hand_rot_err),
        "hand_local_pos_error_mean": _scalar_mean(hand_local_pos_err),
        "hand_local_rot_error_mean": _scalar_mean(hand_local_rot_err),
        "body_local_pos_error_mean": _scalar_mean(body_local_pos_err),
        "body_local_rot_error_mean": _scalar_mean(body_local_rot_err),
        "contact_mismatch_rate": _scalar_mean(contact_err),
        "contact_false_positive_rate": _scalar_mean(false_positive),
        "contact_false_negative_rate": _scalar_mean(false_negative),
        "contact_switch_rate": _scalar_mean(contact_switch),
        "reference_contact_switch_rate": _scalar_mean(ref_contact_switch),
        "contact_force_active_mean": _active_force_mean(contact_force, sim_contact_eval),
        "contact_force_peak": _scalar_max(contact_force),
        "contact_force_excess_mean": _scalar_mean(contact_force_excess),
        "contact_force_delta_mean": _scalar_mean(contact_force_delta),
        "bad_floor_contact_rate": _scalar_mean(bad_floor_contact),
        "bad_floor_force_mean": _scalar_mean(bad_floor_force),
        "bad_floor_force_excess_mean": _scalar_mean(bad_floor_force_excess),
        "action_delta_mean": _scalar_mean(action_delta),
        "control_delta_mean": _scalar_mean(ctrl_delta),
        "joint_acc_mean": _scalar_mean(joint_acc),
        "joint_jerk_mean": _scalar_mean(joint_jerk),
    }
    score = -(
        4.0 * float(metrics["contact_mismatch_rate"])
        + 2.0 * float(metrics["contact_switch_rate"])
        + 3.0 * float(metrics["ee_global_pos_error_mean"])
        + 2.0 * float(metrics["ee_local_pos_error_mean"])
        + 1.5 * float(metrics["root_pos_error_mean"])
        + 0.5 * float(metrics["root_rot_error_mean"])
        + 0.25 * float(metrics["joint_pos_error_mean"])
        + 0.05 * float(metrics["control_delta_mean"])
    )
    metrics["score"] = score
    metrics["success"] = (
        float(metrics["root_pos_error_mean"]) < thresholds.root_pos_mean
        and float(metrics["root_rot_error_mean"]) < thresholds.root_rot_mean
        and float(metrics["ee_global_pos_error_mean"]) < thresholds.ee_global_pos_mean
        and float(metrics["ee_local_pos_error_mean"]) < thresholds.ee_local_pos_mean
        and float(metrics["contact_mismatch_rate"]) < thresholds.contact_mismatch_rate
    )
    return metrics


def compute_rollout_scores(motion: G1Motion, rollout: RolloutTrace) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    ref_idx = rollout.ref_indices
    ref_qpos = motion.qpos()[ref_idx]
    ref_body_pos = motion.body_pos_w[ref_idx]
    ref_body_quat = motion.body_quat_w[ref_idx]
    ref_contact = motion.contact[ref_idx]

    root_pos_err = jnp.linalg.norm(rollout.qpos[..., :3] - ref_qpos[..., :3], axis=-1)
    root_rot_err = quat_error_magnitude(rollout.qpos[..., 3:7], ref_qpos[..., 3:7])
    joint_pos_err = jnp.linalg.norm(rollout.qpos[..., 7:] - ref_qpos[..., 7:], axis=-1)

    body_pos_err = jnp.linalg.norm(rollout.body_pos_w - ref_body_pos, axis=-1)
    body_rot_err = quat_error_magnitude(rollout.body_quat_w, ref_body_quat)
    ee_indices = _body_indices(TASK_EE_BODY_NAMES)
    hand_indices = _body_indices(HAND_EE_BODY_NAMES)
    local_body_indices = _body_indices(tuple(name for name in MUJOCO_BODY_NAMES if name != ANCHOR_BODY_NAME))
    ee_pos_err = jnp.take(body_pos_err, ee_indices, axis=-1)
    ee_rot_err = jnp.take(body_rot_err, ee_indices, axis=-1)
    ee_local_pos_err, ee_local_rot_err = _local_body_errors(
        rollout.body_pos_w, rollout.body_quat_w, ref_body_pos, ref_body_quat, ee_indices
    )
    hand_pos_err = jnp.take(body_pos_err, hand_indices, axis=-1)
    hand_rot_err = jnp.take(body_rot_err, hand_indices, axis=-1)
    hand_local_pos_err, hand_local_rot_err = _local_body_errors(
        rollout.body_pos_w, rollout.body_quat_w, ref_body_pos, ref_body_quat, hand_indices
    )
    body_local_pos_err, body_local_rot_err = _local_body_errors(
        rollout.body_pos_w, rollout.body_quat_w, ref_body_pos, ref_body_quat, local_body_indices
    )

    sim_contact_eval = rollout.contact_indicator[1:]
    ref_contact_eval = ref_contact[1:]
    contact_err = jnp.abs(sim_contact_eval - ref_contact_eval)
    false_positive = ((sim_contact_eval > 0.5) & (ref_contact_eval <= 0.5)).astype(jnp.float32)
    false_negative = ((sim_contact_eval <= 0.5) & (ref_contact_eval > 0.5)).astype(jnp.float32)
    contact_switch = _switch_rate(sim_contact_eval)
    contact_force = rollout.contact_force[1:]
    contact_force_excess = _contact_force_excess(contact_force)
    contact_force_delta = _diff_norm(contact_force) / _contact_force_scale()
    bad_floor_contact = rollout.floor_contact_indicator[1:, :, 2:]
    bad_floor_force = rollout.floor_contact_force[1:, :, 2:]
    bad_floor_force_excess = bad_floor_force / _contact_force_scale()
    action_delta = _diff_norm(rollout.actions)
    ctrl_delta = _diff_norm(rollout.controls)
    dt = jnp.maximum(jnp.asarray(rollout.dt, dtype=rollout.qvel.dtype), 1.0e-6)
    joint_acc = _diff_norm(rollout.qvel[..., 6:]) / dt
    if rollout.qvel.shape[0] > 2:
        joint_jerk = _diff_norm(jnp.diff(rollout.qvel[..., 6:], axis=0)) / dt
    else:
        joint_jerk = jnp.zeros_like(joint_acc)

    terms = {
        "root_pos_error": _per_env_mean(root_pos_err),
        "root_rot_error": _per_env_mean(root_rot_err),
        "joint_pos_error": _per_env_mean(joint_pos_err),
        "body_global_pos_error": _per_env_mean(body_pos_err),
        "body_global_rot_error": _per_env_mean(body_rot_err),
        "ee_global_pos_error": _per_env_mean(ee_pos_err),
        "ee_global_rot_error": _per_env_mean(ee_rot_err),
        "ee_local_pos_error": _per_env_mean(ee_local_pos_err),
        "ee_local_rot_error": _per_env_mean(ee_local_rot_err),
        "hand_global_pos_error": _per_env_mean(hand_pos_err),
        "hand_global_rot_error": _per_env_mean(hand_rot_err),
        "hand_local_pos_error": _per_env_mean(hand_local_pos_err),
        "hand_local_rot_error": _per_env_mean(hand_local_rot_err),
        "body_local_pos_error": _per_env_mean(body_local_pos_err),
        "body_local_rot_error": _per_env_mean(body_local_rot_err),
        "contact_mismatch": _per_env_mean(contact_err),
        "contact_false_positive": _per_env_mean(false_positive),
        "contact_false_negative": _per_env_mean(false_negative),
        "contact_switch": _per_env_mean(contact_switch),
        "contact_force_excess": _per_env_mean(contact_force_excess),
        "contact_force_delta": _per_env_mean(contact_force_delta),
        "bad_floor_contact": _per_env_mean(bad_floor_contact),
        "bad_floor_force_excess": _per_env_mean(bad_floor_force_excess),
        "action_delta": _per_env_mean(action_delta),
        "control_delta": _per_env_mean(ctrl_delta),
        "joint_acc": _per_env_mean(joint_acc),
        "joint_jerk": _per_env_mean(joint_jerk),
    }
    score = -(
        4.0 * terms["contact_mismatch"]
        + 2.0 * terms["contact_switch"]
        + 3.0 * terms["ee_global_pos_error"]
        + 2.0 * terms["ee_local_pos_error"]
        + 1.5 * terms["root_pos_error"]
        + 0.5 * terms["root_rot_error"]
        + 0.25 * terms["joint_pos_error"]
        + 0.05 * terms["control_delta"]
    )
    return score, terms


def score_from_terms(
    terms: dict[str, jnp.ndarray],
    *,
    mode: str = "g1_wbc_joint_global",
    reward_weights: dict[str, float] | None = None,
) -> jnp.ndarray:
    weights = reward_weights or REWARD_WEIGHT_PRESETS[mode]
    score = None
    for name, weight in weights.items():
        if name not in terms:
            continue
        value = float(weight) * terms[name]
        score = value if score is None else score + value
    if score is None:
        raise ValueError("No reward terms matched the configured weights.")
    return -score


def _body_indices(names: tuple[str, ...]) -> jnp.ndarray:
    return jnp.asarray([MUJOCO_BODY_NAMES.index(name) for name in names], dtype=jnp.int32)


def _local_body_errors(
    body_pos: jnp.ndarray,
    body_quat: jnp.ndarray,
    ref_body_pos: jnp.ndarray,
    ref_body_quat: jnp.ndarray,
    body_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    anchor_idx = MUJOCO_BODY_NAMES.index(ANCHOR_BODY_NAME)
    pos_b, quat_b = _body_pose_in_anchor(body_pos, body_quat, anchor_idx, body_indices)
    ref_pos_b, ref_quat_b = _body_pose_in_anchor(ref_body_pos, ref_body_quat, anchor_idx, body_indices)
    return jnp.linalg.norm(pos_b - ref_pos_b, axis=-1), quat_error_magnitude(quat_b, ref_quat_b)


def _body_pose_in_anchor(
    body_pos: jnp.ndarray,
    body_quat: jnp.ndarray,
    anchor_idx: int,
    body_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    target_pos = jnp.take(body_pos, body_indices, axis=-2)
    target_quat = jnp.take(body_quat, body_indices, axis=-2)
    anchor_pos = jnp.broadcast_to(body_pos[..., anchor_idx : anchor_idx + 1, :], target_pos.shape)
    anchor_quat = jnp.broadcast_to(body_quat[..., anchor_idx : anchor_idx + 1, :], target_quat.shape)
    return subtract_frame_transforms(anchor_pos, anchor_quat, target_pos, target_quat)


def _diff_norm(value: jnp.ndarray) -> jnp.ndarray:
    if value.shape[0] <= 1:
        return jnp.zeros(value.shape[1:-1], dtype=value.dtype)
    return jnp.linalg.norm(jnp.diff(value, axis=0), axis=-1)


def _switch_rate(contact: jnp.ndarray) -> jnp.ndarray:
    if contact.shape[0] <= 1:
        return jnp.zeros(contact.shape[1:], dtype=contact.dtype)
    return (jnp.abs(jnp.diff((contact > 0.5).astype(jnp.float32), axis=0)) > 0.5).astype(jnp.float32)


def _contact_force_scale() -> float:
    return 300.0


def _contact_force_excess(force: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(force - _contact_force_scale(), 0.0) / _contact_force_scale()


def _per_env_mean(value: jnp.ndarray) -> jnp.ndarray:
    if value.size == 0:
        env_count = value.shape[1] if value.ndim >= 2 else 0
        return jnp.zeros((env_count,), dtype=jnp.float32)
    if value.ndim < 2:
        return jnp.nan_to_num(value.astype(jnp.float32))
    env_count = value.shape[1]
    return jnp.nan_to_num(value.astype(jnp.float32)).reshape(value.shape[0], env_count, -1).mean(axis=(0, 2))


def _scalar_mean(value: jnp.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(jnp.nan_to_num(value.astype(jnp.float32)).mean())


def _scalar_max(value: jnp.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(jnp.nan_to_num(value.astype(jnp.float32)).max())


def _active_force_mean(force: jnp.ndarray, contact_indicator: jnp.ndarray) -> float:
    active = contact_indicator > 0.5
    if force.size == 0:
        return 0.0
    total = jnp.where(active, force, 0.0).sum()
    count = jnp.maximum(active.astype(jnp.float32).sum(), 1.0)
    return float(total / count)
