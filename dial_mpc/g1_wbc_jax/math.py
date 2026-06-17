"""JAX math helpers for G1 WBC migration."""

from __future__ import annotations

import jax.numpy as jnp


def normalize(x: jnp.ndarray, eps: float = 1.0e-9) -> jnp.ndarray:
    return x / jnp.clip(jnp.linalg.norm(x, axis=-1, keepdims=True), min=eps)


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_inv(q: jnp.ndarray, eps: float = 1.0e-9) -> jnp.ndarray:
    return quat_conjugate(q) / jnp.clip(jnp.sum(q * q, axis=-1, keepdims=True), min=eps)


def quat_mul(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    q1, q2 = jnp.broadcast_arrays(q1, q2)
    w1, x1, y1, z1 = jnp.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = jnp.moveaxis(q2, -1, 0)
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_apply(quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    quat, vec = _broadcast_quat_vec(quat, vec)
    xyz = quat[..., 1:]
    t = 2.0 * jnp.cross(xyz, vec, axis=-1)
    return vec + quat[..., :1] * t + jnp.cross(xyz, t, axis=-1)


def quat_apply_inverse(quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    quat, vec = _broadcast_quat_vec(quat, vec)
    xyz = quat[..., 1:]
    t = 2.0 * jnp.cross(xyz, vec, axis=-1)
    return vec - quat[..., :1] * t + jnp.cross(xyz, t, axis=-1)


def matrix_from_quat(q: jnp.ndarray) -> jnp.ndarray:
    r, i, j, k = jnp.moveaxis(q, -1, 0)
    two_s = 2.0 / jnp.clip(jnp.sum(q * q, axis=-1), min=1.0e-9)
    out = jnp.stack(
        [
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return out.reshape(q.shape[:-1] + (3, 3))


def subtract_frame_transforms(
    t01: jnp.ndarray,
    q01: jnp.ndarray,
    t02: jnp.ndarray | None = None,
    q02: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    t12 = quat_apply(q10, t02 - t01) if t02 is not None else quat_apply(q10, -t01)
    return t12, q12


def axis_angle_from_quat(quat: jnp.ndarray, eps: float = 1.0e-6) -> jnp.ndarray:
    quat = jnp.where(quat[..., :1] < 0.0, -quat, quat)
    mag = jnp.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = jnp.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    denom = jnp.where(
        jnp.abs(angle) > eps,
        jnp.sin(half_angle) / angle,
        0.5 - angle * angle / 48.0,
    )
    return quat[..., 1:4] / denom[..., None]


def quat_from_axis_angle(axis_angle: jnp.ndarray, eps: float = 1.0e-8) -> jnp.ndarray:
    angle = jnp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / jnp.clip(angle, min=eps)
    half_angle = 0.5 * angle
    quat = jnp.concatenate([jnp.cos(half_angle), axis * jnp.sin(half_angle)], axis=-1)
    identity = jnp.zeros_like(quat).at[..., 0].set(1.0)
    return jnp.where(angle <= eps, identity, normalize(quat))


def quat_error_magnitude(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(axis_angle_from_quat(quat_mul(q1, quat_conjugate(q2))), axis=-1)


def world_velocity_to_qvel(qpos: jnp.ndarray, world_vel: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(
        [world_vel[..., :3], quat_apply_inverse(qpos[..., 3:7], world_vel[..., 3:6])],
        axis=-1,
    )


def qvel_to_world_velocity(qpos: jnp.ndarray, qvel: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(
        [qvel[..., :3], quat_apply(qpos[..., 3:7], qvel[..., 3:6])],
        axis=-1,
    )


def finite_difference_root_velocity(qpos: jnp.ndarray, dt: float) -> jnp.ndarray:
    lin = jnp.zeros(qpos.shape[:-1] + (3,), dtype=qpos.dtype)
    ang = jnp.zeros_like(lin)
    if qpos.shape[0] <= 1:
        return jnp.concatenate([lin, ang], axis=-1)
    lin_next = (qpos[1:, :3] - qpos[:-1, :3]) / dt
    dq = quat_mul(qpos[1:, 3:7], quat_inv(qpos[:-1, 3:7]))
    ang_next = axis_angle_from_quat(dq) / dt
    lin = lin.at[:-1].set(lin_next).at[-1].set(lin_next[-1])
    ang = ang.at[:-1].set(ang_next).at[-1].set(ang_next[-1])
    return jnp.concatenate([lin, ang], axis=-1)


def qvel_from_qpos_trajectory(qpos: jnp.ndarray, dt: float) -> jnp.ndarray:
    squeeze = qpos.ndim == 2
    qpos_batched = qpos[:, None, :] if squeeze else qpos
    qvel = jnp.zeros(qpos_batched.shape[:-1] + (qpos_batched.shape[-1] - 1,), qpos.dtype)
    if qpos_batched.shape[0] <= 1:
        return qvel[:, 0] if squeeze else qvel

    lin_vel = jnp.zeros(qvel.shape[:-1] + (3,), qpos.dtype)
    ang_vel_w = jnp.zeros_like(lin_vel)
    joint_vel = jnp.zeros(qpos_batched.shape[:-1] + (qpos_batched.shape[-1] - 7,), qpos.dtype)
    lin_delta = (qpos_batched[1:, :, :3] - qpos_batched[:-1, :, :3]) / dt
    delta_quat = quat_mul(qpos_batched[1:, :, 3:7], quat_inv(qpos_batched[:-1, :, 3:7]))
    ang_delta = axis_angle_from_quat(delta_quat) / dt
    joint_delta = (qpos_batched[1:, :, 7:] - qpos_batched[:-1, :, 7:]) / dt
    lin_vel = lin_vel.at[:-1].set(lin_delta).at[-1].set(lin_delta[-1])
    ang_vel_w = ang_vel_w.at[:-1].set(ang_delta).at[-1].set(ang_delta[-1])
    joint_vel = joint_vel.at[:-1].set(joint_delta).at[-1].set(joint_delta[-1])
    root_world_vel = jnp.concatenate([lin_vel, ang_vel_w], axis=-1)
    root_qvel = world_velocity_to_qvel(qpos_batched[..., :7], root_world_vel)
    qvel = qvel.at[..., :6].set(root_qvel).at[..., 6:].set(joint_vel)
    return qvel[:, 0] if squeeze else qvel


def _broadcast_quat_vec(quat: jnp.ndarray, vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    leading = jnp.broadcast_shapes(quat.shape[:-1], vec.shape[:-1])
    return (
        jnp.broadcast_to(quat, leading + (quat.shape[-1],)),
        jnp.broadcast_to(vec, leading + (vec.shape[-1],)),
    )
