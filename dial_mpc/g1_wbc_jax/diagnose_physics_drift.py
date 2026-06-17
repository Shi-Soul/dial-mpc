"""Per-step physics drift diagnostics for the JAX G1 WBC migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from dial_mpc.g1_wbc_jax.constants import ACTION_DIM
from dial_mpc.g1_wbc_jax.model import build_wbc_mj_model, build_wbc_system
from dial_mpc.g1_wbc_jax.rollout import G1WbcRolloutConfig, make_mjx_open_loop_rollout, make_open_loop_rollout
from dial_mpc.g1_wbc_jax.verify_physics_alignment import STATE_FIELDS, _mujoco_open_loop


def main() -> None:
    args = _parse_args()
    payload = diagnose(
        args.rollout_npz,
        seconds=args.seconds,
        max_steps=args.max_steps,
        backend=args.backend,
        fields=tuple(args.field),
        threshold=args.threshold,
        sample_every=args.sample_every,
        integrator=args.integrator,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


def diagnose(
    rollout_npz: str | Path,
    *,
    seconds: float,
    max_steps: int | None,
    backend: str,
    fields: tuple[str, ...],
    threshold: float,
    sample_every: int,
    integrator: str,
) -> dict[str, object]:
    path = Path(rollout_npz).expanduser().resolve()
    raw = np.load(path)
    dt = float(raw["dt"].item()) if "dt" in raw.files else 0.02
    steps = min(int(round(seconds / dt)), raw["controls"].shape[0])
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    if steps < 1:
        raise ValueError("Need at least one control step for physics drift diagnosis.")

    selected_fields = fields or STATE_FIELDS
    actual_by_field = _simulate_backend(raw, steps, backend, dt, integrator)
    per_field = {}
    sampled_rows = []
    for field in selected_fields:
        if field not in raw.files:
            continue
        if field not in actual_by_field:
            continue
        expected = np.asarray(raw[field][: steps + 1])
        actual = np.asarray(actual_by_field[field])
        if actual.shape != expected.shape:
            raise ValueError(f"{field} shape mismatch: {actual.shape} vs {expected.shape}")
        diff = actual.astype(np.float64) - expected.astype(np.float64)
        abs_diff = np.abs(diff)
        axes = tuple(range(1, abs_diff.ndim))
        per_step_max = abs_diff.max(axis=axes)
        per_step_rmse = np.sqrt(np.mean(diff * diff, axis=axes))
        crossing = np.flatnonzero(per_step_max > threshold)
        per_field[field] = {
            "max_abs": float(per_step_max.max()),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "final_max_abs": float(per_step_max[-1]),
            "first_step_above_threshold": int(crossing[0]) if crossing.size else None,
            "first_time_s_above_threshold": float(crossing[0] * dt) if crossing.size else None,
        }
        for step in range(0, steps + 1, max(1, int(sample_every))):
            sampled_rows.append(
                {
                    "field": field,
                    "step": int(step),
                    "time_s": float(step * dt),
                    "max_abs": float(per_step_max[step]),
                    "rmse": float(per_step_rmse[step]),
                }
            )
        if steps % max(1, int(sample_every)) != 0:
            sampled_rows.append(
                {
                    "field": field,
                    "step": int(steps),
                    "time_s": float(steps * dt),
                    "max_abs": float(per_step_max[-1]),
                    "rmse": float(per_step_rmse[-1]),
                }
            )

    return {
        "backend": backend,
        "rollout_npz": str(path),
        "seconds": float(steps * dt),
        "steps": int(steps),
        "dt": dt,
        "integrator": integrator,
        "threshold": float(threshold),
        "fields": per_field,
        "samples": sampled_rows,
    }


def _simulate_backend(raw, steps: int, backend: str, dt: float, integrator: str) -> dict[str, np.ndarray]:
    controls = np.asarray(raw["controls"][:steps], dtype=np.float64)
    if controls.ndim == 3:
        if controls.shape[1] != 1:
            raise ValueError("Only single-env saved rollouts are supported for CPU MuJoCo diagnostics.")
        controls_single = controls[:, 0]
    elif controls.ndim == 2:
        controls_single = controls
    else:
        raise ValueError(f"Unsupported controls shape: {controls.shape}")
    if controls_single.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected controls dim {ACTION_DIM}, got {controls_single.shape}.")

    qpos0 = np.asarray(raw["qpos"][0, 0], dtype=np.float64)
    qvel0 = np.asarray(raw["qvel"][0, 0], dtype=np.float64)
    jax_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    cfg = G1WbcRolloutConfig(max_steps=steps, dt=dt)
    if backend == "mujoco":
        model = build_wbc_mj_model()
        _apply_integrator(model, integrator)
        return _mujoco_open_loop(model, controls_single, qpos0, qvel0)
    if backend == "mjx":
        if integrator == "implicit":
            raise ValueError("MJX does not support MuJoCo's mjINT_IMPLICIT integrator.")
        model = build_wbc_mj_model()
        _apply_integrator(model, integrator)
        fn = make_mjx_open_loop_rollout(model, cfg)
        trace = fn(
            jnp.asarray(controls_single, dtype=jax_dtype),
            jnp.asarray(qpos0, dtype=jax_dtype),
            jnp.asarray(qvel0, dtype=jax_dtype),
        )
        trace.qpos.block_until_ready()
    elif backend == "brax":
        if integrator != "default":
            raise ValueError("--integrator is only supported for mujoco and mjx backends.")
        sys = build_wbc_system()
        fn = make_open_loop_rollout(sys, cfg)
        trace = fn(
            jnp.asarray(controls_single, dtype=jax_dtype),
            jnp.asarray(qpos0, dtype=jax_dtype),
            jnp.asarray(qvel0, dtype=jax_dtype),
        )
        trace.qpos.block_until_ready()
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return {field: np.asarray(getattr(trace, field)) for field in STATE_FIELDS}


def _apply_integrator(model: mujoco.MjModel, integrator: str) -> None:
    if integrator == "default":
        return
    mapping = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    if integrator not in mapping:
        raise ValueError(f"Unsupported integrator: {integrator}")
    model.opt.integrator = mapping[integrator]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-npz", required=True)
    parser.add_argument("--backend", choices=("mjx", "brax", "mujoco"), default="mjx")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--field", action="append", choices=STATE_FIELDS, default=[])
    parser.add_argument("--threshold", type=float, default=1.0e-3)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--integrator", choices=("default", "euler", "implicit", "implicitfast"), default="default")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
