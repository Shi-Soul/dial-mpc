"""Pure JAX observation construction for the G1 WBC actor."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from dial_mpc.g1_wbc_jax.constants import (
    ACTION_DIM,
    COMMAND_BODY_INDICES,
    LIMB_EE_INDICES_IN_COMMAND,
    OBS_DIM,
    OBS_HISTORY_LENGTH,
    TRACKING_ANCHOR_INDEX_IN_COMMAND,
)
from dial_mpc.g1_wbc_jax.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms
from dial_mpc.g1_wbc_jax.motion import G1CommandBatch, G1Motion


class RobotState(NamedTuple):
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    body_pos_w: jnp.ndarray
    body_quat_w: jnp.ndarray
    body_lin_vel_w: jnp.ndarray
    body_ang_vel_w: jnp.ndarray
    base_ang_vel_b: jnp.ndarray | None = None


class HistoryState(NamedTuple):
    buffer: jnp.ndarray
    pointer: jnp.ndarray
    num_pushes: jnp.ndarray


class WbcObsState(NamedTuple):
    ref_limb_ee_pose_b: HistoryState
    robot_limb_ee_pose_b: HistoryState
    projected_gravity: HistoryState
    base_ang_vel: HistoryState
    joint_pos: HistoryState
    joint_vel: HistoryState
    actions: HistoryState


def init_obs_state(num_envs: int, dtype=jnp.float32) -> WbcObsState:
    return WbcObsState(
        ref_limb_ee_pose_b=_empty_history(num_envs, 36, dtype),
        robot_limb_ee_pose_b=_empty_history(num_envs, 36, dtype),
        projected_gravity=_empty_history(num_envs, 3, dtype),
        base_ang_vel=_empty_history(num_envs, 3, dtype),
        joint_pos=_empty_history(num_envs, ACTION_DIM, dtype),
        joint_vel=_empty_history(num_envs, ACTION_DIM, dtype),
        actions=_empty_history(num_envs, ACTION_DIM, dtype),
    )


def compute_obs(
    obs_state: WbcObsState,
    motion: G1Motion | G1CommandBatch,
    robot: RobotState,
    ref_indices: jnp.ndarray,
    last_action: jnp.ndarray,
    default_joint_pos: jnp.ndarray,
) -> tuple[jnp.ndarray, WbcObsState]:
    ref = _ref_fields(motion, ref_indices)
    command = jnp.concatenate([ref["joint_pos"], ref["joint_vel"]], axis=-1)
    ref_limb = _limb_pose_in_anchor_frame(ref["body_pos_w"], ref["body_quat_w"])

    command_body_indices = jnp.asarray(COMMAND_BODY_INDICES, dtype=jnp.int32)
    robot_body_pos = jnp.take(robot.body_pos_w, command_body_indices, axis=1)
    robot_body_quat = jnp.take(robot.body_quat_w, command_body_indices, axis=1)
    robot_limb = _limb_pose_in_anchor_frame(robot_body_pos, robot_body_quat)

    root_quat = robot.qpos[:, 3:7]
    gravity_w = jnp.broadcast_to(jnp.array([0.0, 0.0, -1.0], dtype=robot.qpos.dtype), (robot.qpos.shape[0], 3))
    projected_gravity = quat_apply_inverse(root_quat, gravity_w)
    if robot.base_ang_vel_b is None:
        base_ang_vel_b = quat_apply_inverse(root_quat, robot.body_ang_vel_w[:, 0])
    else:
        base_ang_vel_b = robot.base_ang_vel_b
    joint_pos_rel = robot.qpos[:, 7:] - default_joint_pos.reshape(1, ACTION_DIM)
    joint_vel_rel = robot.qvel[:, 6:]
    motion_ref_ang_vel = ref["body_ang_vel_w"][:, TRACKING_ANCHOR_INDEX_IN_COMMAND]

    ref_hist, ref_limb_flat = append_history(obs_state.ref_limb_ee_pose_b, ref_limb)
    robot_hist, robot_limb_flat = append_history(obs_state.robot_limb_ee_pose_b, robot_limb)
    grav_hist, projected_gravity_flat = append_history(obs_state.projected_gravity, projected_gravity)
    base_hist, base_ang_vel_flat = append_history(obs_state.base_ang_vel, base_ang_vel_b)
    joint_pos_hist, joint_pos_flat = append_history(obs_state.joint_pos, joint_pos_rel)
    joint_vel_hist, joint_vel_flat = append_history(obs_state.joint_vel, joint_vel_rel)
    action_hist, action_flat = append_history(obs_state.actions, last_action)

    next_state = WbcObsState(
        ref_limb_ee_pose_b=ref_hist,
        robot_limb_ee_pose_b=robot_hist,
        projected_gravity=grav_hist,
        base_ang_vel=base_hist,
        joint_pos=joint_pos_hist,
        joint_vel=joint_vel_hist,
        actions=action_hist,
    )
    obs = jnp.concatenate(
        [
            command,
            ref_limb_flat,
            motion_ref_ang_vel,
            robot_limb_flat,
            projected_gravity_flat,
            base_ang_vel_flat,
            joint_pos_flat,
            joint_vel_flat,
            action_flat,
        ],
        axis=-1,
    )
    if obs.shape[-1] != OBS_DIM:
        raise ValueError(f"Expected obs dim {OBS_DIM}, got {obs.shape[-1]}.")
    return obs, next_state


def append_history(state: HistoryState, value: jnp.ndarray) -> tuple[HistoryState, jnp.ndarray]:
    pointer = (state.pointer + 1) % OBS_HISTORY_LENGTH
    buffer = state.buffer.at[pointer].set(value)
    first = state.num_pushes == 0
    backfill = jnp.broadcast_to(value[None, :, :], buffer.shape)
    buffer = jnp.where(first[None, :, None], backfill, buffer)
    num_pushes = state.num_pushes + 1
    next_state = HistoryState(buffer=buffer, pointer=pointer, num_pushes=num_pushes)
    idx = (jnp.arange(OBS_HISTORY_LENGTH) + pointer + 1) % OBS_HISTORY_LENGTH
    flat = buffer[idx].transpose(1, 0, 2).reshape(value.shape[0], -1)
    return next_state, flat


def _empty_history(num_envs: int, dim: int, dtype) -> HistoryState:
    return HistoryState(
        buffer=jnp.zeros((OBS_HISTORY_LENGTH, num_envs, dim), dtype=dtype),
        pointer=jnp.array(-1, dtype=jnp.int32),
        num_pushes=jnp.zeros((num_envs,), dtype=jnp.int32),
    )


def _ref_fields(motion: G1Motion | G1CommandBatch, ref_indices: jnp.ndarray) -> dict[str, jnp.ndarray]:
    ref_indices = jnp.clip(ref_indices, 0, motion.num_frames - 1)
    cmd_body_idx = jnp.asarray(COMMAND_BODY_INDICES, dtype=jnp.int32)
    if isinstance(motion, G1CommandBatch):
        env_ids = jnp.arange(ref_indices.shape[0], dtype=jnp.int32)
        return {
            "joint_pos": motion.joint_pos[ref_indices, env_ids],
            "joint_vel": motion.joint_vel[ref_indices, env_ids],
            "body_pos_w": jnp.take(motion.body_pos_w[ref_indices, env_ids], cmd_body_idx, axis=1),
            "body_quat_w": jnp.take(motion.body_quat_w[ref_indices, env_ids], cmd_body_idx, axis=1),
            "body_ang_vel_w": jnp.take(motion.body_ang_vel_w[ref_indices, env_ids], cmd_body_idx, axis=1),
        }
    return {
        "joint_pos": motion.joint_pos[ref_indices],
        "joint_vel": motion.joint_vel[ref_indices],
        "body_pos_w": jnp.take(motion.body_pos_w[ref_indices], cmd_body_idx, axis=1),
        "body_quat_w": jnp.take(motion.body_quat_w[ref_indices], cmd_body_idx, axis=1),
        "body_ang_vel_w": jnp.take(motion.body_ang_vel_w[ref_indices], cmd_body_idx, axis=1),
    }


def _limb_pose_in_anchor_frame(body_pos_w: jnp.ndarray, body_quat_w: jnp.ndarray) -> jnp.ndarray:
    limb_indices = jnp.asarray(LIMB_EE_INDICES_IN_COMMAND, dtype=jnp.int32)
    limb_pos_w = jnp.take(body_pos_w, limb_indices, axis=1)
    limb_quat_w = jnp.take(body_quat_w, limb_indices, axis=1)
    anchor_pos_w = jnp.broadcast_to(body_pos_w[:, :1], limb_pos_w.shape)
    anchor_quat_w = jnp.broadcast_to(body_quat_w[:, :1], limb_quat_w.shape)
    pos_b, quat_b = subtract_frame_transforms(anchor_pos_w, anchor_quat_w, limb_pos_w, limb_quat_w)
    rot6d = matrix_from_quat(quat_b)[..., :2].reshape(body_pos_w.shape[0], len(LIMB_EE_INDICES_IN_COMMAND), 6)
    return jnp.concatenate([pos_b, rot6d], axis=-1).reshape(body_pos_w.shape[0], -1)
