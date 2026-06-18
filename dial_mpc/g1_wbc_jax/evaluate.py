"""CLI for evaluating the all-JAX G1 WBC migration in legacy benchmark format."""

from __future__ import annotations

import argparse
import json
import sys as py_sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dial_mpc.g1_wbc_jax.artifacts import save_command_npz, save_rollout_npz
from dial_mpc.g1_wbc_jax.constants import ACTION_DIM, POLICY_DT
from dial_mpc.g1_wbc_jax.metrics import RolloutTrace, compute_rollout_metrics
from dial_mpc.g1_wbc_jax.model import build_wbc_mj_model, build_wbc_system, joint_limits
from dial_mpc.g1_wbc_jax.motion import G1Motion, load_motion
from dial_mpc.g1_wbc_jax.mpc import (
    G1WbcDialMpcConfig,
    initial_plan,
    make_optimize_window_with_context,
    make_optimize_window_with_context_host_chunked,
    shift_plan,
)
from dial_mpc.g1_wbc_jax.objective_config import (
    DEFAULT_OBJECTIVE_WEIGHTS_JSON,
    EVALUATE_METHOD_CHOICES,
)
from dial_mpc.g1_wbc_jax.obs import init_obs_state
from dial_mpc.g1_wbc_jax.planner import (
    RolloutScoreContext,
    make_wbc_mjx_rollout_score_fn,
    make_wbc_rollout_score_fn,
)
from dial_mpc.g1_wbc_jax.policy import load_torch_actor, resolve_checkpoint_path
from dial_mpc.g1_wbc_jax.rollout import (
    G1WbcRolloutConfig,
    command_batch_from_qpos_trajectory,
    make_mjx_policy_rollout,
    make_policy_rollout,
    mjx_command_batch_from_qpos_trajectory,
)
from dial_mpc.g1_wbc_jax.sim_config import (
    SIM_PRESET_CHOICES,
    apply_sim_preset,
    sim_decimation,
    simulation_payload,
)


def main() -> None:
    args = _parse_args()
    motion = _load_reference_motion(args)
    total_steps = _resolve_steps(motion, args.max_steps)
    mj_model = build_wbc_mj_model(args.model_path)
    apply_sim_preset(mj_model, args.sim_preset)
    sim_decimation_value = sim_decimation(args.sim_preset)
    sys = build_wbc_system(args.model_path) if args.backend == "brax" else None
    if args.sim_preset != "default" and args.backend != "mjx":
        raise ValueError("--sim-preset currently only applies to the MJX backend.")
    actor = load_torch_actor(args.checkpoint)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    if args.method == "no_mpc":
        trace, command_qpos, candidate_scores, mpc_payload = _run_no_mpc(
            args, sys, mj_model, actor, motion, total_steps, sim_decimation_value
        )
    else:
        trace, command_qpos, candidate_scores, mpc_payload = _run_dial_mpc(
            args, sys, mj_model, actor, motion, total_steps, sim_decimation_value
        )

    metrics = compute_rollout_metrics(motion, trace)
    payload = {
        "method": args.method,
        "motion": str(Path(args.motion).expanduser().resolve()) if args.motion else None,
        "reference_rollout": str(Path(args.reference_rollout_npz).expanduser().resolve())
        if args.reference_rollout_npz
        else None,
        "motion_type": motion.motion_type,
        "checkpoint": str(checkpoint_path),
        "device": str(jax.devices()[0]),
        "num_envs": 1,
        "max_steps": total_steps,
        "ref_offset": args.ref_offset,
        "metrics": metrics,
        "simulation": simulation_payload(mj_model, sim_decimation_value, args.sim_preset),
    }
    if mpc_payload is not None:
        payload["mpc"] = mpc_payload

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.save_rollout:
            save_rollout_npz(output_dir / "rollout.npz", trace)
            command = (
                mjx_command_batch_from_qpos_trajectory(mj_model, motion, command_qpos)
                if args.backend == "mjx"
                else command_batch_from_qpos_trajectory(sys, motion, command_qpos)
            )
            save_command_npz(output_dir / "mpc_command.npz", command, candidate_scores=candidate_scores)


def _run_no_mpc(args, sys, mj_model, actor, motion: G1Motion, total_steps: int, sim_decimation: int):
    command_qpos = motion.qpos()[: total_steps + 1]
    rollout_config = G1WbcRolloutConfig(
        max_steps=total_steps,
        ref_offset=args.ref_offset,
        decimation=sim_decimation,
    )
    rollout_fn = (
        make_mjx_policy_rollout(mj_model, actor, motion, rollout_config)
        if args.backend == "mjx"
        else make_policy_rollout(sys, actor, motion, rollout_config)
    )
    out = rollout_fn(command_qpos, motion.qpos()[0], motion.qvel()[0])
    out.trace.qpos.block_until_ready()
    return out.trace, command_qpos[:, None, :], None, None


def _run_dial_mpc(args, sys, mj_model, actor, motion: G1Motion, total_steps: int, sim_decimation: int):
    mpc_config = G1WbcDialMpcConfig(
        num_samples=args.mpc_samples,
        planning_horizon_steps=args.mpc_horizon,
        control_steps=args.mpc_control_steps,
        node_count=args.mpc_nodes,
        n_diffuse=args.mpc_diffuse,
        n_diffuse_init=args.mpc_diffuse_init,
        temp_sample=args.mpc_temperature,
        horizon_diffuse_factor=args.mpc_horizon_diffuse_factor,
        traj_diffuse_factor=args.mpc_traj_diffuse_factor,
        root_pos_sigma=args.mpc_root_pos_sigma,
        root_rot_sigma=args.mpc_root_rot_sigma,
        joint_sigma=args.mpc_joint_sigma,
        min_root_pos_sigma=args.mpc_min_root_pos_sigma,
        min_root_rot_sigma=args.mpc_min_root_rot_sigma,
        min_joint_sigma=args.mpc_min_joint_sigma,
        command_reg_weight=args.mpc_command_reg_weight,
        command_smooth_weight=args.mpc_command_smooth_weight,
        score_batch_size=args.mpc_score_batch_size,
    )
    if mpc_config.planning_horizon_steps <= mpc_config.control_steps:
        raise ValueError("planning horizon must be larger than control steps for receding-horizon execution.")

    reward_weights = _load_reward_weights(args.reward_weights_json, args.method)
    rollout_config = G1WbcRolloutConfig(
        max_steps=mpc_config.planning_horizon_steps,
        ref_offset=args.ref_offset,
        decimation=sim_decimation,
    )
    if args.backend == "mjx":
        score_fn = make_wbc_mjx_rollout_score_fn(
            mj_model,
            actor,
            motion,
            rollout_config,
            mode=args.method,
            reward_weights=reward_weights,
        )
    else:
        score_fn = make_wbc_rollout_score_fn(
            sys,
            actor,
            motion,
            rollout_config,
            mode=args.method,
            reward_weights=reward_weights,
        )
    optimizer_factory = (
        make_optimize_window_with_context_host_chunked
        if args.mpc_host_chunked
        else make_optimize_window_with_context
    )
    optimize_init = optimizer_factory(mpc_config, score_fn, init=True)
    optimize = optimizer_factory(mpc_config, score_fn, init=False)
    execute_fn = (
        make_mjx_policy_rollout(
            mj_model,
            actor,
            motion,
            G1WbcRolloutConfig(
                max_steps=mpc_config.control_steps,
                ref_offset=args.ref_offset,
                decimation=sim_decimation,
            ),
        )
        if args.backend == "mjx"
        else make_policy_rollout(
            sys,
            actor,
            motion,
            G1WbcRolloutConfig(
                max_steps=mpc_config.control_steps,
                ref_offset=args.ref_offset,
                decimation=sim_decimation,
            ),
        )
    )
    joint_low, joint_high = joint_limits(mj_model if args.backend == "mjx" else sys.mj_model)
    rng = jax.random.PRNGKey(args.seed)
    plan_nodes = initial_plan(mpc_config)
    obs_state = init_obs_state(1)
    last_action = jnp.zeros((1, ACTION_DIM), dtype=jnp.float32)
    current_qpos = motion.qpos()[0]
    current_qvel = motion.qvel()[0]
    qpos_ref = motion.qpos()
    traces: list[RolloutTrace] = []
    command_frames = [qpos_ref[0]]
    last_scores = None
    history = []

    start_time = time.perf_counter()
    for start in range(0, total_steps, mpc_config.control_steps):
        execute_steps = min(mpc_config.control_steps, total_steps - start)
        if execute_steps <= 0:
            break
        base_qpos = _window_qpos(qpos_ref, start, mpc_config.planning_horizon_steps)
        ctx = RolloutScoreContext(
            initial_qpos=current_qpos,
            initial_qvel=current_qvel,
            initial_last_action=last_action,
            initial_obs_state=obs_state,
            ref_start=jnp.array(start, dtype=jnp.int32),
        )
        t0 = time.perf_counter()
        opt_fn = optimize_init if start == 0 else optimize
        rng, result = opt_fn(rng, base_qpos, plan_nodes, joint_low, joint_high, ctx)
        result.best_qpos.block_until_ready()
        solve_time = time.perf_counter() - t0
        print(
            "MPC_PROGRESS "
            f"start={int(start)} total_steps={int(total_steps)} "
            f"execute_steps={int(execute_steps)} solve_time_s={solve_time:.6f} "
            f"best_score={float(result.best_score):.6f} mean_score={float(result.mean_score):.6f}",
            file=py_sys.stderr,
            flush=True,
        )
        last_scores = result.scores
        exec_out = execute_fn(
            result.best_qpos,
            current_qpos,
            current_qvel,
            last_action,
            obs_state,
            jnp.array(start, dtype=jnp.int32),
        )
        exec_out.trace.qpos.block_until_ready()
        trace = _truncate_trace(exec_out.trace, execute_steps)
        traces.append(trace)
        command_frames.extend(result.best_qpos[1 : execute_steps + 1])
        current_qpos = trace.qpos[-1, 0]
        current_qvel = trace.qvel[-1, 0]
        last_action = exec_out.final_last_action
        obs_state = exec_out.final_obs_state
        plan_nodes = shift_plan(mpc_config, result.plan_nodes, execute_steps)
        history.append(
            {
                "start": int(start),
                "execute_steps": int(execute_steps),
                "best_score": float(result.best_score),
                "mean_score": float(result.mean_score),
                "solve_time_s": solve_time,
            }
        )

    trace = _concat_traces(traces)
    command_qpos = jnp.stack(command_frames, axis=0)[:, None, :]
    total_wall_time = time.perf_counter() - start_time
    solve_times = [float(item["solve_time_s"]) for item in history]
    runtime_solve_times = solve_times[1:]
    steady_runtime_solve_times = runtime_solve_times[1:] if len(runtime_solve_times) > 1 else runtime_solve_times
    sum_solve_time = sum(solve_times)
    mpc_payload = {
        "algorithm": "dial_mpc_receding_horizon",
        "num_samples": mpc_config.num_samples,
        "planning_horizon_steps": mpc_config.planning_horizon_steps,
        "control_steps": mpc_config.control_steps,
        "node_count": mpc_config.node_count,
        "n_diffuse": mpc_config.n_diffuse,
        "n_diffuse_init": mpc_config.n_diffuse_init,
        "score_batch_size": mpc_config.score_batch_size,
        "host_chunked": bool(args.mpc_host_chunked),
        "sim_decimation": sim_decimation,
        "temperature": mpc_config.temp_sample,
        "reward_weight_source": (
            str(Path(args.reward_weights_json).expanduser().resolve()) if args.reward_weights_json else None
        ),
        "reward_weights": reward_weights,
        "history": history,
        "num_windows": len(history),
        "total_solve_time_s": sum_solve_time,
        "mean_window_time_s": sum_solve_time / max(len(history), 1),
        "total_wall_time_s": total_wall_time,
        "sum_window_solve_time_s": sum_solve_time,
        "mean_window_solve_time_s": sum_solve_time / max(len(history), 1),
        "mean_runtime_solve_time_s": (
            sum(runtime_solve_times) / len(runtime_solve_times) if runtime_solve_times else None
        ),
        "mean_steady_runtime_solve_time_s": (
            sum(steady_runtime_solve_times) / len(steady_runtime_solve_times)
            if steady_runtime_solve_times
            else None
        ),
        "first_window_compile_s": solve_times[0] if solve_times else None,
        "first_runtime_window_compile_s": runtime_solve_times[0] if runtime_solve_times else None,
        "final_scores_mean": float(jnp.mean(last_scores)) if last_scores is not None else None,
        "final_scores_max": float(jnp.max(last_scores)) if last_scores is not None else None,
    }
    return trace, command_qpos, last_scores, mpc_payload


def _load_reference_motion(args) -> G1Motion:
    if args.motion is not None:
        return load_motion(args.motion, motion_type=args.motion_type)
    if args.reference_rollout_npz is None:
        raise ValueError("Either --motion or --reference-rollout-npz is required.")
    return _motion_from_rollout_npz(Path(args.reference_rollout_npz).expanduser().resolve())


def _motion_from_rollout_npz(path: Path) -> G1Motion:
    raw = np.load(path)
    dt = float(raw["dt"].item()) if "dt" in raw.files else POLICY_DT
    qpos = jnp.asarray(raw["qpos"][:, 0], dtype=jnp.float32)
    qvel = jnp.asarray(raw["qvel"][:, 0], dtype=jnp.float32)
    return G1Motion(
        path=path,
        motion_type="mujoco",
        fps=1.0 / dt,
        joint_pos=qpos[:, 7:],
        joint_vel=qvel[:, 6:],
        body_pos_w=jnp.asarray(raw["body_pos_w"][:, 0], dtype=jnp.float32),
        body_quat_w=jnp.asarray(raw["body_quat_w"][:, 0], dtype=jnp.float32),
        body_lin_vel_w=jnp.asarray(raw["body_lin_vel_w"][:, 0], dtype=jnp.float32),
        body_ang_vel_w=jnp.asarray(raw["body_ang_vel_w"][:, 0], dtype=jnp.float32),
        contact=jnp.asarray(raw["contact_indicator"][:, 0], dtype=jnp.float32),
    )


def _resolve_steps(motion: G1Motion, max_steps: int | None) -> int:
    available = max(motion.num_frames - 1, 1)
    return available if max_steps is None else min(int(max_steps), available)


def _load_reward_weights(path: str | None, method: str) -> dict[str, float] | None:
    if path is None:
        return None
    raw = json.loads(Path(path).expanduser().read_text())
    if method in raw and isinstance(raw[method], dict):
        raw = raw[method]
    if not isinstance(raw, dict):
        raise ValueError(f"Reward weight file must contain a JSON object: {path}")
    return {str(key): float(value) for key, value in raw.items()}


def _window_qpos(qpos: jnp.ndarray, start: int, horizon: int) -> jnp.ndarray:
    idx = jnp.clip(jnp.arange(start, start + horizon), 0, qpos.shape[0] - 1)
    return qpos[idx]


def _truncate_trace(trace: RolloutTrace, steps: int) -> RolloutTrace:
    return RolloutTrace(
        qpos=trace.qpos[: steps + 1],
        qvel=trace.qvel[: steps + 1],
        body_pos_w=trace.body_pos_w[: steps + 1],
        body_quat_w=trace.body_quat_w[: steps + 1],
        body_lin_vel_w=trace.body_lin_vel_w[: steps + 1],
        body_ang_vel_w=trace.body_ang_vel_w[: steps + 1],
        actions=trace.actions[:steps],
        controls=trace.controls[:steps],
        contact_indicator=trace.contact_indicator[: steps + 1],
        contact_force=trace.contact_force[: steps + 1],
        ref_indices=trace.ref_indices[: steps + 1],
        floor_contact_indicator=trace.floor_contact_indicator[: steps + 1],
        floor_contact_force=trace.floor_contact_force[: steps + 1],
        dt=trace.dt,
    )


def _concat_traces(traces: list[RolloutTrace]) -> RolloutTrace:
    if not traces:
        raise ValueError("No rollout traces were produced.")

    def cat_state(name: str):
        chunks = [getattr(traces[0], name)]
        chunks.extend(getattr(trace, name)[1:] for trace in traces[1:])
        return jnp.concatenate(chunks, axis=0)

    return RolloutTrace(
        qpos=cat_state("qpos"),
        qvel=cat_state("qvel"),
        body_pos_w=cat_state("body_pos_w"),
        body_quat_w=cat_state("body_quat_w"),
        body_lin_vel_w=cat_state("body_lin_vel_w"),
        body_ang_vel_w=cat_state("body_ang_vel_w"),
        actions=jnp.concatenate([trace.actions for trace in traces], axis=0),
        controls=jnp.concatenate([trace.controls for trace in traces], axis=0),
        contact_indicator=cat_state("contact_indicator"),
        contact_force=cat_state("contact_force"),
        ref_indices=cat_state("ref_indices"),
        floor_contact_indicator=cat_state("floor_contact_indicator"),
        floor_contact_force=cat_state("floor_contact_force"),
        dt=traces[0].dt,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", default=None)
    parser.add_argument("--motion-type", default="auto", choices=("auto", "mujoco", "isaaclab"))
    parser.add_argument("--reference-rollout-npz", default=None)
    parser.add_argument("--checkpoint", default="bc")
    parser.add_argument(
        "--method",
        default="g1_wbc_joint_global",
        choices=EVALUATE_METHOD_CHOICES,
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--backend", default="mjx", choices=("mjx", "brax"))
    parser.add_argument("--sim-preset", default="default", choices=SIM_PRESET_CHOICES)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ref-offset", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-rollout", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-weights-json", default=str(DEFAULT_OBJECTIVE_WEIGHTS_JSON))
    parser.add_argument("--mpc-samples", type=int, default=64)
    parser.add_argument("--mpc-horizon", type=int, default=30)
    parser.add_argument("--mpc-control-steps", type=int, default=10)
    parser.add_argument("--mpc-nodes", type=int, default=4)
    parser.add_argument("--mpc-diffuse", type=int, default=1)
    parser.add_argument("--mpc-diffuse-init", type=int, default=4)
    parser.add_argument("--mpc-temperature", type=float, default=0.7)
    parser.add_argument("--mpc-horizon-diffuse-factor", type=float, default=0.9)
    parser.add_argument("--mpc-traj-diffuse-factor", type=float, default=0.5)
    parser.add_argument("--mpc-root-pos-sigma", type=float, default=0.08)
    parser.add_argument("--mpc-root-rot-sigma", type=float, default=0.18)
    parser.add_argument("--mpc-joint-sigma", type=float, default=0.28)
    parser.add_argument("--mpc-min-root-pos-sigma", type=float, default=0.002)
    parser.add_argument("--mpc-min-root-rot-sigma", type=float, default=0.004)
    parser.add_argument("--mpc-min-joint-sigma", type=float, default=0.008)
    parser.add_argument("--mpc-command-reg-weight", type=float, default=0.0)
    parser.add_argument("--mpc-command-smooth-weight", type=float, default=0.0)
    parser.add_argument(
        "--mpc-score-batch-size",
        type=int,
        default=0,
        help="Score candidate trajectories in fixed-size chunks; 0 scores the full sample batch at once.",
    )
    parser.add_argument(
        "--mpc-host-chunked",
        action="store_true",
        help="Use host-loop DIAL-MPC updates with JIT-compiled score chunks for large correctness runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
