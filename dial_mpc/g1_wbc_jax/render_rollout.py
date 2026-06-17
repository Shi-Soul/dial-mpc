"""Render a saved G1 WBC rollout NPZ to an MP4 video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

from dial_mpc.g1_wbc_jax.model import build_wbc_mj_model


def main() -> None:
    args = _parse_args()
    rollout_path = Path(args.rollout).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    raw = np.load(rollout_path)
    qpos = raw["qpos"]
    if qpos.ndim == 3:
        qpos = qpos[:, int(args.env_index)]
    if qpos.ndim != 2:
        raise ValueError(f"Expected qpos shape (T, nq) or (T, N, nq), got {qpos.shape}.")

    qvel = raw["qvel"] if "qvel" in raw.files else None
    if qvel is not None and qvel.ndim == 3:
        qvel = qvel[:, int(args.env_index)]

    dt = float(np.asarray(raw["dt"]).item()) if "dt" in raw.files else 0.02
    fps = args.fps if args.fps is not None else max(1, round(1.0 / (dt * args.stride)))
    model = build_wbc_mj_model(args.model_path)
    data = mujoco.MjData(model)
    camera = _resolve_camera(model, args.camera)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    indices = range(0, qpos.shape[0], int(args.stride))
    if args.max_frames is not None:
        indices = list(indices)[: int(args.max_frames)]
    with mujoco.Renderer(model, width=args.width, height=args.height) as renderer:
        for idx in indices:
            data.qpos[:] = qpos[idx]
            if qvel is not None:
                data.qvel[:] = qvel[idx]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    if not frames:
        raise ValueError(f"No frames rendered from {rollout_path}.")
    imageio.mimsave(output_path, frames, fps=fps)
    print(str(output_path))


def _resolve_camera(model: mujoco.MjModel, camera: str | None):
    if camera is None:
        return None
    if camera == "free":
        return None
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, idx)
            for idx in range(model.ncam)
        ]
        raise ValueError(f"Unknown camera {camera!r}. Available cameras: {available}.")
    return camera


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
