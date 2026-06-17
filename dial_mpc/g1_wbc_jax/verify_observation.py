"""Check JAX WBC observation construction against the SPIDER Torch builder."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from dial_mpc.g1_wbc_jax.constants import ACTION_DIM
from dial_mpc.g1_wbc_jax.model import default_joint_pos
from dial_mpc.g1_wbc_jax.motion import G1Motion, load_motion
from dial_mpc.g1_wbc_jax.obs import RobotState, compute_obs, init_obs_state
from dial_mpc.g1_wbc_jax.policy import actor_forward, load_torch_actor


def run(
    motion_path: str | Path,
    *,
    rollout_npz: str | Path | None = None,
    steps: int = 32,
    seed: int = 0,
    spider_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, float]:
    if spider_path is not None:
        sys.path.insert(0, str(Path(spider_path).expanduser().resolve()))

    import torch
    from spider.tasks.g1_wbc.motion import load_motion as load_spider_motion
    from spider.tasks.g1_wbc.obs import G1WbcObservationBuilder
    from spider.tasks.g1_wbc.obs import RobotState as SpiderRobotState
    from spider.tasks.g1_wbc.policy import load_wbc_actor

    motion_path = Path(motion_path).expanduser().resolve()
    jax_motion = load_motion(motion_path)
    spider_motion = load_spider_motion(motion_path)
    max_frames = _available_frames(jax_motion, rollout_npz)
    frame_count = min(int(steps), max_frames)
    if frame_count < 1:
        raise ValueError("Need at least one frame for observation verification.")

    rng = np.random.default_rng(seed)
    last_actions = rng.normal(size=(frame_count, 1, ACTION_DIM)).astype(np.float32)
    jax_obs_state = init_obs_state(1)
    jax_default = default_joint_pos()
    torch_default = torch.from_numpy(np.asarray(jax_default, dtype=np.float32).copy())
    spider_builder = G1WbcObservationBuilder(
        motion=spider_motion,
        num_envs=1,
        default_joint_pos=torch_default,
        device="cpu",
    )
    jax_actor = load_torch_actor(checkpoint) if checkpoint is not None else None
    torch_actor = load_wbc_actor(checkpoint, device="cpu") if checkpoint is not None else None

    robot_arrays = _robot_arrays(jax_motion, rollout_npz, frame_count)
    diffs = []
    torch_obs_values = []
    jax_obs_values = []
    action_diffs = []
    torch_action_values = []
    jax_action_values = []
    for frame_idx in range(frame_count):
        last_action_np = last_actions[frame_idx]
        spider_obs = spider_builder.compute(
            SpiderRobotState(
                qpos=torch.from_numpy(robot_arrays["qpos"][frame_idx]),
                qvel=torch.from_numpy(robot_arrays["qvel"][frame_idx]),
                body_pos_w=torch.from_numpy(robot_arrays["body_pos_w"][frame_idx]),
                body_quat_w=torch.from_numpy(robot_arrays["body_quat_w"][frame_idx]),
                body_lin_vel_w=torch.from_numpy(robot_arrays["body_lin_vel_w"][frame_idx]),
                body_ang_vel_w=torch.from_numpy(robot_arrays["body_ang_vel_w"][frame_idx]),
                base_ang_vel_b=None,
            ),
            torch.tensor([frame_idx], dtype=torch.long),
            torch.from_numpy(last_action_np),
        )
        jax_obs, jax_obs_state = compute_obs(
            jax_obs_state,
            jax_motion,
            RobotState(
                qpos=jnp.asarray(robot_arrays["qpos"][frame_idx], dtype=jnp.float32),
                qvel=jnp.asarray(robot_arrays["qvel"][frame_idx], dtype=jnp.float32),
                body_pos_w=jnp.asarray(robot_arrays["body_pos_w"][frame_idx], dtype=jnp.float32),
                body_quat_w=jnp.asarray(robot_arrays["body_quat_w"][frame_idx], dtype=jnp.float32),
                body_lin_vel_w=jnp.asarray(robot_arrays["body_lin_vel_w"][frame_idx], dtype=jnp.float32),
                body_ang_vel_w=jnp.asarray(robot_arrays["body_ang_vel_w"][frame_idx], dtype=jnp.float32),
                base_ang_vel_b=None,
            ),
            jnp.asarray([frame_idx], dtype=jnp.int32),
            jnp.asarray(last_action_np, dtype=jnp.float32),
            jax_default,
        )
        jax_obs_np = np.asarray(jax.device_get(jax_obs))
        torch_obs_np = spider_obs.detach().cpu().numpy()
        diffs.append(jax_obs_np - torch_obs_np)
        torch_obs_values.append(torch_obs_np)
        jax_obs_values.append(jax_obs_np)
        if jax_actor is not None and torch_actor is not None:
            torch_action_np = torch_actor(spider_obs).detach().cpu().numpy()
            jax_action_np = np.asarray(jax.device_get(actor_forward(jax_actor, jax_obs)))
            action_diffs.append(jax_action_np - torch_action_np)
            torch_action_values.append(torch_action_np)
            jax_action_values.append(jax_action_np)

    diff = np.concatenate(diffs, axis=0)
    torch_obs_all = np.concatenate(torch_obs_values, axis=0)
    jax_obs_all = np.concatenate(jax_obs_values, axis=0)
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(torch_obs_all), 1.0e-6)
    stats = {
        "frames": float(frame_count),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel_diff.max()),
        "mean_rel": float(rel_diff.mean()),
        "torch_mean": float(torch_obs_all.mean()),
        "jax_mean": float(jax_obs_all.mean()),
        "torch_std": float(torch_obs_all.std()),
        "jax_std": float(jax_obs_all.std()),
    }
    if action_diffs:
        action_diff = np.concatenate(action_diffs, axis=0)
        torch_action_all = np.concatenate(torch_action_values, axis=0)
        jax_action_all = np.concatenate(jax_action_values, axis=0)
        action_abs_diff = np.abs(action_diff)
        action_rel_diff = action_abs_diff / np.maximum(np.abs(torch_action_all), 1.0e-6)
        stats.update(
            {
                "action_max_abs": float(action_abs_diff.max()),
                "action_mean_abs": float(action_abs_diff.mean()),
                "action_max_rel": float(action_rel_diff.max()),
                "action_mean_rel": float(action_rel_diff.mean()),
                "torch_action_mean": float(torch_action_all.mean()),
                "jax_action_mean": float(jax_action_all.mean()),
                "torch_action_std": float(torch_action_all.std()),
                "jax_action_std": float(jax_action_all.std()),
            }
        )
    return stats


def _available_frames(motion: G1Motion, rollout_npz: str | Path | None) -> int:
    if rollout_npz is None:
        return motion.num_frames
    raw = np.load(Path(rollout_npz).expanduser())
    return min(motion.num_frames, int(raw["qpos"].shape[0]))


def _robot_arrays(
    motion: G1Motion,
    rollout_npz: str | Path | None,
    frame_count: int,
) -> dict[str, np.ndarray]:
    if rollout_npz is not None:
        raw = np.load(Path(rollout_npz).expanduser())
        return {
            "qpos": np.asarray(raw["qpos"][:frame_count, 0], dtype=np.float32)[:, None, :],
            "qvel": np.asarray(raw["qvel"][:frame_count, 0], dtype=np.float32)[:, None, :],
            "body_pos_w": np.asarray(raw["body_pos_w"][:frame_count, 0], dtype=np.float32)[:, None, :],
            "body_quat_w": np.asarray(raw["body_quat_w"][:frame_count, 0], dtype=np.float32)[:, None, :],
            "body_lin_vel_w": np.asarray(raw["body_lin_vel_w"][:frame_count, 0], dtype=np.float32)[:, None, :],
            "body_ang_vel_w": np.asarray(raw["body_ang_vel_w"][:frame_count, 0], dtype=np.float32)[:, None, :],
        }
    return {
        "qpos": np.asarray(motion.qpos()[:frame_count], dtype=np.float32)[:, None, :],
        "qvel": np.asarray(motion.qvel()[:frame_count], dtype=np.float32)[:, None, :],
        "body_pos_w": np.asarray(motion.body_pos_w[:frame_count], dtype=np.float32)[:, None, :],
        "body_quat_w": np.asarray(motion.body_quat_w[:frame_count], dtype=np.float32)[:, None, :],
        "body_lin_vel_w": np.asarray(motion.body_lin_vel_w[:frame_count], dtype=np.float32)[:, None, :],
        "body_ang_vel_w": np.asarray(motion.body_ang_vel_w[:frame_count], dtype=np.float32)[:, None, :],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--rollout-npz", default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spider-path", default="spider-wbc-framework-integrated")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    stats = run(
        args.motion,
        rollout_npz=args.rollout_npz,
        steps=args.steps,
        seed=args.seed,
        spider_path=args.spider_path,
        checkpoint=args.checkpoint,
    )
    print(" ".join(f"{key}={value:.6e}" for key, value in stats.items()))


if __name__ == "__main__":
    main()
