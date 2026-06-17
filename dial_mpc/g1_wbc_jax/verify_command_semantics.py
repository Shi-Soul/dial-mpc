"""Compare JAX command-batch construction with saved SPIDER command fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dial_mpc.g1_wbc_jax.model import build_wbc_mj_model, build_wbc_system
from dial_mpc.g1_wbc_jax.motion import G1Motion
from dial_mpc.g1_wbc_jax.rollout import command_batch_from_qpos_trajectory, mjx_command_batch_from_qpos_trajectory


FIELDS = (
    ("joint_pos", "command_joint_pos"),
    ("joint_vel", "command_joint_vel"),
    ("body_pos_w", "command_body_pos_w"),
    ("body_quat_w", "command_body_quat_w"),
    ("qpos_trajectory", "command_qpos_trajectory"),
    ("qvel_trajectory", "command_qvel_trajectory"),
)


def _motion_from_command_npz(path: Path, raw: np.lib.npyio.NpzFile) -> G1Motion:
    frame_count = raw["command_joint_pos"].shape[0]
    body_shape = raw["command_body_pos_w"][:, 0].shape
    return G1Motion(
        path=path,
        motion_type="mujoco",
        fps=50.0,
        joint_pos=jnp.asarray(raw["command_joint_pos"][:, 0], dtype=jnp.float32),
        joint_vel=jnp.asarray(raw["command_joint_vel"][:, 0], dtype=jnp.float32),
        body_pos_w=jnp.asarray(raw["command_body_pos_w"][:, 0], dtype=jnp.float32),
        body_quat_w=jnp.asarray(raw["command_body_quat_w"][:, 0], dtype=jnp.float32),
        body_lin_vel_w=jnp.zeros(body_shape, dtype=jnp.float32),
        body_ang_vel_w=jnp.zeros(body_shape, dtype=jnp.float32),
        contact=jnp.zeros((frame_count, 2), dtype=jnp.float32),
    )


def _compare(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        f"{name}_max_abs": float(np.max(np.abs(diff))),
        f"{name}_rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def run(command_npz: str | Path, seconds: float | None, *, backend: str = "mjx") -> dict[str, float]:
    path = Path(command_npz).expanduser().resolve()
    raw = np.load(path)
    dt = 0.02
    total_frames = raw["command_qpos_trajectory"].shape[0]
    frame_count = total_frames
    if seconds is not None:
        frame_count = min(frame_count, int(round(seconds / dt)) + 1)
    compute_frames = min(total_frames, frame_count + 1)

    motion = _motion_from_command_npz(path, raw)
    qpos = jnp.asarray(raw["command_qpos_trajectory"][:compute_frames], dtype=jnp.float32)
    if backend == "mjx":
        model = build_wbc_mj_model()
        command = mjx_command_batch_from_qpos_trajectory(model, motion, qpos)
    elif backend == "brax":
        sys = build_wbc_system()
        command = command_batch_from_qpos_trajectory(sys, motion, qpos)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    command.body_pos_w.block_until_ready()

    stats: dict[str, float] = {"backend": backend}
    for attr, key in FIELDS:
        actual = np.asarray(getattr(command, attr)[:frame_count])
        expected = np.asarray(raw[key][:frame_count])
        stats.update(_compare(attr, actual, expected))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-npz", required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--backend", choices=("mjx", "brax"), default="mjx")
    args = parser.parse_args()

    stats = run(args.command_npz, args.seconds, backend=args.backend)
    print(
        " ".join(
            f"{key}={value}" if isinstance(value, str) else f"{key}={value:.6e}"
            for key, value in stats.items()
        )
    )


if __name__ == "__main__":
    main()
