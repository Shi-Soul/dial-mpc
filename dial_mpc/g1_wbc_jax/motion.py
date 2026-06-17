"""Motion loading for the JAX G1 WBC migration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import jax
import numpy as np

from dial_mpc.g1_wbc_jax.constants import (
    ACTION_DIM,
    ISAACLAB_TO_MUJOCO_BODY_REINDEX,
    ISAACLAB_TO_MUJOCO_JOINT_REINDEX,
    LEFT_FOOT_BODY_NAME,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
    RIGHT_FOOT_BODY_NAME,
)
from dial_mpc.g1_wbc_jax.math import (
    finite_difference_root_velocity,
    qvel_from_qpos_trajectory,
    world_velocity_to_qvel,
)

MotionType = Literal["auto", "mujoco", "isaaclab"]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class G1Motion:
    path: Path
    motion_type: Literal["mujoco", "isaaclab"]
    fps: float
    joint_pos: jnp.ndarray
    joint_vel: jnp.ndarray
    body_pos_w: jnp.ndarray
    body_quat_w: jnp.ndarray
    body_lin_vel_w: jnp.ndarray
    body_ang_vel_w: jnp.ndarray
    contact: jnp.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def body_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(MUJOCO_BODY_NAMES)}

    def qpos(self) -> jnp.ndarray:
        return jnp.concatenate(
            [self.body_pos_w[:, 0], self.body_quat_w[:, 0], self.joint_pos],
            axis=-1,
        )

    def qvel(self) -> jnp.ndarray:
        root_world_vel = jnp.concatenate(
            [self.body_lin_vel_w[:, 0], self.body_ang_vel_w[:, 0]],
            axis=-1,
        )
        root_qvel = world_velocity_to_qvel(self.qpos()[:, :7], root_world_vel)
        return jnp.concatenate([root_qvel, self.joint_vel], axis=-1)

    def tree_flatten(self):
        children = (
            self.joint_pos,
            self.joint_vel,
            self.body_pos_w,
            self.body_quat_w,
            self.body_lin_vel_w,
            self.body_ang_vel_w,
            self.contact,
        )
        aux = (self.path, self.motion_type, self.fps)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        path, motion_type, fps = aux_data
        joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, contact = children
        return cls(
            path=path,
            motion_type=motion_type,
            fps=fps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            contact=contact,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class G1CommandBatch:
    path: Path
    motion_type: Literal["mujoco", "isaaclab"]
    fps: float
    joint_pos: jnp.ndarray
    joint_vel: jnp.ndarray
    body_pos_w: jnp.ndarray
    body_quat_w: jnp.ndarray
    body_lin_vel_w: jnp.ndarray
    body_ang_vel_w: jnp.ndarray
    qpos_trajectory: jnp.ndarray
    qvel_trajectory: jnp.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.joint_pos.shape[1])

    @property
    def body_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(MUJOCO_BODY_NAMES)}

    def tree_flatten(self):
        children = (
            self.joint_pos,
            self.joint_vel,
            self.body_pos_w,
            self.body_quat_w,
            self.body_lin_vel_w,
            self.body_ang_vel_w,
            self.qpos_trajectory,
            self.qvel_trajectory,
        )
        aux = (self.path, self.motion_type, self.fps)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        path, motion_type, fps = aux_data
        (
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
            qpos_trajectory,
            qvel_trajectory,
        ) = children
        return cls(
            path=path,
            motion_type=motion_type,
            fps=fps,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            qpos_trajectory=qpos_trajectory,
            qvel_trajectory=qvel_trajectory,
        )


def detect_motion_type(path: Path, raw: np.lib.npyio.NpzFile) -> Literal["mujoco", "isaaclab"]:
    if "motion_type" in raw.files:
        value = raw["motion_type"]
        text = str(value.item() if value.shape == () else value.tolist()).lower()
        if "mujoco" in text:
            return "mujoco"
        if "isaac" in text:
            return "isaaclab"
    name = path.name.lower()
    if "mujoco" in name:
        return "mujoco"
    if "isaac" in name or "isaaclab" in name:
        return "isaaclab"
    return "isaaclab"


def load_motion(
    motion_path: str | Path,
    *,
    motion_type: MotionType = "auto",
    target_dt: float = POLICY_DT,
) -> G1Motion:
    path = Path(motion_path).expanduser().resolve()
    raw = np.load(path)
    resolved_type = detect_motion_type(path, raw) if motion_type == "auto" else motion_type
    if resolved_type == "auto":
        raise ValueError("motion_type must resolve to 'mujoco' or 'isaaclab'.")

    required = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing = [key for key in required if key not in raw.files]
    if missing:
        raise ValueError(f"Motion file {path} is missing keys: {missing}")

    fps = float(raw["fps"].item()) if "fps" in raw.files else 1.0 / target_dt
    joint_pos = np.asarray(raw["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(raw["joint_vel"], dtype=np.float32)
    body_pos_w = np.asarray(raw["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(raw["body_quat_w"], dtype=np.float32)
    body_lin_vel_w = np.asarray(raw["body_lin_vel_w"], dtype=np.float32)
    body_ang_vel_w = np.asarray(raw["body_ang_vel_w"], dtype=np.float32)

    if resolved_type == "isaaclab":
        joint_pos = joint_pos[:, ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
        joint_vel = joint_vel[:, ISAACLAB_TO_MUJOCO_JOINT_REINDEX]
        body_pos_w = body_pos_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_quat_w = body_quat_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_lin_vel_w = body_lin_vel_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]
        body_ang_vel_w = body_ang_vel_w[:, ISAACLAB_TO_MUJOCO_BODY_REINDEX]

    motion = G1Motion(
        path=path,
        motion_type=resolved_type,
        fps=fps,
        joint_pos=jnp.asarray(joint_pos),
        joint_vel=jnp.asarray(joint_vel),
        body_pos_w=jnp.asarray(body_pos_w),
        body_quat_w=jnp.asarray(body_quat_w),
        body_lin_vel_w=jnp.asarray(body_lin_vel_w),
        body_ang_vel_w=jnp.asarray(body_ang_vel_w),
        contact=jnp.empty((joint_pos.shape[0], 2), dtype=jnp.float32),
    )
    motion = resample_motion(motion, target_dt=target_dt)
    contact = estimate_foot_contacts(motion)
    out = G1Motion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=1.0 / target_dt,
        joint_pos=motion.joint_pos,
        joint_vel=motion.joint_vel,
        body_pos_w=motion.body_pos_w,
        body_quat_w=motion.body_quat_w,
        body_lin_vel_w=motion.body_lin_vel_w,
        body_ang_vel_w=motion.body_ang_vel_w,
        contact=contact,
    )
    validate_motion_dims(out)
    return out


def resample_motion(motion: G1Motion, *, target_dt: float = POLICY_DT) -> G1Motion:
    src_dt = 1.0 / float(motion.fps)
    if abs(src_dt - target_dt) < 1.0e-7:
        return motion
    joint_pos = _resample_linear_np(np.asarray(motion.joint_pos), src_dt, target_dt)
    joint_vel = _resample_linear_np(np.asarray(motion.joint_vel), src_dt, target_dt)
    body_pos_w = _resample_linear_np(np.asarray(motion.body_pos_w), src_dt, target_dt)
    body_quat_w = _resample_quat_np(np.asarray(motion.body_quat_w), src_dt, target_dt)
    body_lin_vel_w = _resample_linear_np(np.asarray(motion.body_lin_vel_w), src_dt, target_dt)
    body_ang_vel_w = _resample_linear_np(np.asarray(motion.body_ang_vel_w), src_dt, target_dt)

    joint_pos_j = jnp.asarray(joint_pos)
    body_pos_j = jnp.asarray(body_pos_w)
    body_quat_j = jnp.asarray(body_quat_w)
    if np.allclose(joint_vel, 0.0) and joint_pos.shape[0] > 1:
        joint_vel = np.zeros_like(joint_pos)
        joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / target_dt
        joint_vel[-1] = joint_vel[-2]
    root_vel = finite_difference_root_velocity(
        jnp.concatenate([body_pos_j[:, 0], body_quat_j[:, 0]], axis=-1),
        target_dt,
    )
    if np.allclose(body_lin_vel_w, 0.0):
        body_lin_vel_w = body_lin_vel_w.copy()
        body_lin_vel_w[:, 0] = np.asarray(root_vel[:, :3])
    if np.allclose(body_ang_vel_w, 0.0):
        body_ang_vel_w = body_ang_vel_w.copy()
        body_ang_vel_w[:, 0] = np.asarray(root_vel[:, 3:])

    return G1Motion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=1.0 / target_dt,
        joint_pos=joint_pos_j,
        joint_vel=jnp.asarray(joint_vel),
        body_pos_w=body_pos_j,
        body_quat_w=body_quat_j,
        body_lin_vel_w=jnp.asarray(body_lin_vel_w),
        body_ang_vel_w=jnp.asarray(body_ang_vel_w),
        contact=jnp.empty((joint_pos.shape[0], 2), dtype=jnp.float32),
    )


def estimate_foot_contacts(
    motion: G1Motion,
    *,
    height_threshold: float = 0.055,
    speed_threshold: float = 0.35,
) -> jnp.ndarray:
    body_index = motion.body_index
    foot_ids = [body_index[LEFT_FOOT_BODY_NAME], body_index[RIGHT_FOOT_BODY_NAME]]
    pos = motion.body_pos_w[:, foot_ids]
    vel = motion.body_lin_vel_w[:, foot_ids]
    floor_z = jnp.quantile(pos[..., 2].reshape(-1), 0.02)
    height = pos[..., 2] - floor_z
    speed = jnp.linalg.norm(vel, axis=-1)
    contact = ((height < height_threshold) & (speed < speed_threshold)).astype(jnp.float32)
    if contact.shape[0] >= 3:
        prev = contact[:-2]
        cur = contact[1:-1]
        nxt = contact[2:]
        filtered = jnp.where(prev == nxt, prev, cur)
        contact = jnp.concatenate([contact[:1], filtered, contact[-1:]], axis=0)
    return contact


def validate_motion_dims(motion: G1Motion) -> None:
    if motion.joint_pos.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} joint positions, got {motion.joint_pos.shape}.")
    if motion.body_pos_w.shape[1] != len(MUJOCO_BODY_NAMES):
        raise ValueError(
            f"Expected {len(MUJOCO_BODY_NAMES)} bodies, got {motion.body_pos_w.shape}."
        )
    if motion.qpos().shape[-1] != QPOS_DIM or motion.qvel().shape[-1] != QVEL_DIM:
        raise ValueError("Motion qpos/qvel dimensions do not match G1 29dof model.")


def qvel_from_motion_qpos(qpos: jnp.ndarray, dt: float = POLICY_DT) -> jnp.ndarray:
    return qvel_from_qpos_trajectory(qpos, dt)


def _resample_linear_np(x: np.ndarray, src_dt: float, target_dt: float) -> np.ndarray:
    if x.shape[0] <= 1 or abs(src_dt - target_dt) < 1.0e-7:
        return x.astype(np.float32, copy=False)
    duration = (x.shape[0] - 1) * src_dt
    out_len = int(np.floor(duration / target_dt + 1.0e-6)) + 1
    t = np.arange(out_len, dtype=np.float32) * target_dt
    u = np.clip(t / src_dt, 0.0, x.shape[0] - 1)
    i0 = np.floor(u).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.shape[0] - 1)
    a = (u - i0.astype(np.float32)).reshape((-1,) + (1,) * (x.ndim - 1))
    return (x[i0] * (1.0 - a) + x[i1] * a).astype(np.float32)


def _resample_quat_np(x: np.ndarray, src_dt: float, target_dt: float) -> np.ndarray:
    if x.shape[0] <= 1 or abs(src_dt - target_dt) < 1.0e-7:
        return _normalize_np(x)
    duration = (x.shape[0] - 1) * src_dt
    out_len = int(np.floor(duration / target_dt + 1.0e-6)) + 1
    t = np.arange(out_len, dtype=np.float32) * target_dt
    u = np.clip(t / src_dt, 0.0, x.shape[0] - 1)
    i0 = np.floor(u).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.shape[0] - 1)
    a = (u - i0.astype(np.float32)).reshape((-1,) + (1,) * (x.ndim - 1))
    return _slerp_np(x[i0], x[i1], a).astype(np.float32)


def _slerp_np(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    small = dot > 0.9995
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.clip(np.sin(theta_0), 1.0e-8, None)
    theta = theta_0 * alpha
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    out = s0 * q0 + s1 * q1
    lerp = q0 + alpha * (q1 - q0)
    return _normalize_np(np.where(small, lerp, out))


def _normalize_np(x: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), eps, None)
