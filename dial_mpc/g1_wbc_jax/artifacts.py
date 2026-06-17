"""Save JAX G1 WBC outputs in the legacy benchmark artifact schema."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from dial_mpc.g1_wbc_jax.metrics import RolloutTrace
from dial_mpc.g1_wbc_jax.motion import G1CommandBatch


def save_rollout_npz(path: str | Path, trace: RolloutTrace) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        qpos=_np(trace.qpos),
        qvel=_np(trace.qvel),
        body_pos_w=_np(trace.body_pos_w),
        body_quat_w=_np(trace.body_quat_w),
        body_lin_vel_w=_np(trace.body_lin_vel_w),
        body_ang_vel_w=_np(trace.body_ang_vel_w),
        actions=_np(trace.actions),
        controls=_np(trace.controls),
        contact_indicator=_np(trace.contact_indicator),
        contact_force=_np(trace.contact_force),
        ref_indices=_np(trace.ref_indices),
        dt=np.asarray(trace.dt, dtype=np.float32),
        floor_contact_indicator=_np(trace.floor_contact_indicator),
        floor_contact_force=_np(trace.floor_contact_force),
    )


def save_command_npz(
    path: str | Path,
    command: G1CommandBatch,
    *,
    candidate_scores=None,
    selected_env: int = 0,
) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    qpos = _np(command.qpos_trajectory)
    np.savez(
        out,
        refined_qpos=qpos[:, int(selected_env)],
        candidate_scores=np.asarray([] if candidate_scores is None else _np(candidate_scores), dtype=np.float32),
        command_joint_pos=_np(command.joint_pos),
        command_joint_vel=_np(command.joint_vel),
        command_body_pos_w=_np(command.body_pos_w),
        command_body_quat_w=_np(command.body_quat_w),
        command_qpos_trajectory=qpos,
        command_qvel_trajectory=_np(command.qvel_trajectory),
    )


def _np(value) -> np.ndarray:
    return np.asarray(jax.device_get(value))
