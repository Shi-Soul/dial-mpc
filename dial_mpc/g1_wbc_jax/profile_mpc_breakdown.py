"""Profile coarse G1 WBC DIAL-MPC solve-time components."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco

with contextlib.redirect_stdout(io.StringIO()):
    import mujoco.mjx as mjx

from dial_mpc.g1_wbc_jax.constants import ACTION_DIM, POLICY_DT, QPOS_DIM, QVEL_DIM
from dial_mpc.g1_wbc_jax.metrics import compute_rollout_scores, score_from_terms
from dial_mpc.g1_wbc_jax.model import build_wbc_mj_model, joint_limits
from dial_mpc.g1_wbc_jax.motion import load_motion
from dial_mpc.g1_wbc_jax.mpc import G1WbcDialMpcConfig, initial_plan, make_optimize_window_with_context
from dial_mpc.g1_wbc_jax.obs import init_obs_state
from dial_mpc.g1_wbc_jax.planner import RolloutScoreContext, make_wbc_mjx_rollout_score_fn
from dial_mpc.g1_wbc_jax.policy import load_torch_actor
from dial_mpc.g1_wbc_jax.rollout import (
    G1WbcRolloutConfig,
    contact_groups,
    extract_robot_state,
    make_mjx_open_loop_rollout,
    make_mjx_policy_rollout,
)
from dial_mpc.g1_wbc_jax.math import qvel_from_qpos_trajectory


class _ModelView(NamedTuple):
    mj_model: mujoco.MjModel
    nu: int


def main() -> None:
    args = _parse_args()
    payload = profile(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


def profile(args: argparse.Namespace) -> dict[str, object]:
    model = build_wbc_mj_model(args.model_path)
    actor = load_torch_actor(args.checkpoint)
    motion = load_motion(args.motion, motion_type=args.motion_type)
    reward_weights = _load_reward_weights(args.reward_weights_json, args.method)

    mpc_config = G1WbcDialMpcConfig(
        num_samples=args.mpc_samples,
        planning_horizon_steps=args.mpc_horizon,
        control_steps=args.mpc_control_steps,
        node_count=args.mpc_nodes,
        n_diffuse=args.mpc_diffuse,
        n_diffuse_init=args.mpc_diffuse_init,
        temp_sample=args.mpc_temperature,
        root_pos_sigma=args.mpc_root_pos_sigma,
        root_rot_sigma=args.mpc_root_rot_sigma,
        joint_sigma=args.mpc_joint_sigma,
        min_root_pos_sigma=args.mpc_min_root_pos_sigma,
        min_root_rot_sigma=args.mpc_min_root_rot_sigma,
        min_joint_sigma=args.mpc_min_joint_sigma,
        command_reg_weight=args.mpc_command_reg_weight,
        command_smooth_weight=args.mpc_command_smooth_weight,
    )
    rollout_config = G1WbcRolloutConfig(max_steps=mpc_config.planning_horizon_steps)
    rollout_fn = make_mjx_policy_rollout(model, actor, motion, rollout_config)
    open_loop_fn = make_mjx_open_loop_rollout(model, rollout_config)
    score_fn = make_wbc_mjx_rollout_score_fn(
        model,
        actor,
        motion,
        rollout_config,
        mode=args.method,
        reward_weights=reward_weights,
    )
    optimize = make_optimize_window_with_context(mpc_config, score_fn, init=False)

    horizon = mpc_config.planning_horizon_steps
    num_candidates = mpc_config.num_samples + 1
    base_qpos = motion.qpos()[:horizon]
    candidate_qpos = jnp.broadcast_to(base_qpos[:, None, :], (horizon, num_candidates, QPOS_DIM))
    initial_qpos = motion.qpos()[0]
    initial_qvel = motion.qvel()[0]
    last_action = jnp.zeros((1, ACTION_DIM), dtype=jnp.float32)
    obs_state = init_obs_state(1)
    ref_start = jnp.array(0, dtype=jnp.int32)
    context = RolloutScoreContext(initial_qpos, initial_qvel, last_action, obs_state, ref_start)
    joint_low, joint_high = joint_limits(model)
    plan_nodes = initial_plan(mpc_config)
    rng = jax.random.PRNGKey(args.seed)
    zero_controls = jnp.zeros((horizon, num_candidates, ACTION_DIM), dtype=jnp.float32)

    command_kinematics = _make_command_kinematics(model)

    @jax.jit
    def score_only(qpos):
        return score_fn(qpos, context)

    @jax.jit
    def rollout_only(qpos):
        trace = rollout_fn(qpos, initial_qpos, initial_qvel, last_action, obs_state, ref_start).trace
        return _score_trace_checksum(trace)

    @jax.jit
    def metrics_only(trace):
        _, terms = compute_rollout_scores(motion, trace)
        return jnp.sum(score_from_terms(terms, mode=args.method, reward_weights=reward_weights))

    @jax.jit
    def open_loop_physics_only(ctrl):
        trace = open_loop_fn(ctrl, initial_qpos, initial_qvel)
        return _score_trace_checksum(trace)

    @jax.jit
    def command_kinematics_only(qpos):
        return command_kinematics(qpos)

    @jax.jit
    def full_optimize_once(rng_in, nodes):
        rng_out, result = optimize(rng_in, base_qpos, nodes, joint_low, joint_high, context)
        return rng_out, result.best_score, jnp.sum(result.scores), jnp.sum(result.plan_nodes)

    trace_sample = rollout_fn(candidate_qpos, initial_qpos, initial_qvel, last_action, obs_state, ref_start).trace
    trace_sample.qpos.block_until_ready()

    timings = {
        "full_optimize": _bench(
            lambda r, n: full_optimize_once(r, n),
            rng,
            plan_nodes,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
        "score_fn": _bench(score_only, candidate_qpos, warmups=args.warmups, repeats=args.repeats),
        "policy_rollout_score_fields": _bench(
            rollout_only,
            candidate_qpos,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
        "metric_score_only": _bench(metrics_only, trace_sample, warmups=args.warmups, repeats=args.repeats),
        "mjx_dynamic_rollout_score_fields": _bench(
            open_loop_physics_only,
            zero_controls,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
        "reference_kinematics": _bench(
            command_kinematics_only,
            candidate_qpos,
            warmups=args.warmups,
            repeats=args.repeats,
        ),
    }

    mean = {name: item["mean_s"] for name, item in timings.items()}
    derived = {
        "mpc_sampling_update_estimate_s": max(mean["full_optimize"] - mean["score_fn"], 0.0),
        "score_metric_estimate_s": mean["metric_score_only"],
        "score_rollout_estimate_s": max(mean["score_fn"] - mean["metric_score_only"], 0.0),
        "policy_obs_trace_residual_estimate_s": max(
            mean["policy_rollout_score_fields"]
            - mean["mjx_dynamic_rollout_score_fields"]
            - mean["reference_kinematics"],
            0.0,
        ),
    }
    denominator = mean["full_optimize"] if mean["full_optimize"] > 0 else 1.0
    fractions = {name: value / denominator for name, value in derived.items()}
    fractions.update(
        {
            "score_fn": mean["score_fn"] / denominator,
            "policy_rollout_score_fields": mean["policy_rollout_score_fields"] / denominator,
            "metric_score_only": mean["metric_score_only"] / denominator,
            "mjx_dynamic_rollout_score_fields": mean["mjx_dynamic_rollout_score_fields"] / denominator,
            "reference_kinematics": mean["reference_kinematics"] / denominator,
        }
    )

    return {
        "device": str(jax.devices()[0]),
        "motion": str(Path(args.motion).expanduser().resolve()),
        "config": {
            "num_samples": mpc_config.num_samples,
            "num_candidates": num_candidates,
            "planning_horizon_steps": mpc_config.planning_horizon_steps,
            "control_steps": mpc_config.control_steps,
            "node_count": mpc_config.node_count,
            "n_diffuse": mpc_config.n_diffuse,
            "decimation": rollout_config.decimation,
            "physics_substeps_per_score": int(num_candidates * horizon * rollout_config.decimation),
        },
        "simulation": {
            "timestep": float(model.opt.timestep),
            "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
            "solver": mujoco.mjtSolver(model.opt.solver).name,
            "cone": mujoco.mjtCone(model.opt.cone).name,
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
        },
        "timings": timings,
        "derived": derived,
        "fractions_of_full_optimize": fractions,
    }


def _make_command_kinematics(model: mujoco.MjModel):
    mjx_model = mjx.put_model(model)
    data_template = mjx.make_data(mjx_model)
    sys_view = _ModelView(mj_model=model, nu=model.nu)

    def command_kinematics(qpos_trajectory: jnp.ndarray):
        qvel_trajectory = qvel_from_qpos_trajectory(qpos_trajectory, dt=POLICY_DT)
        flat_qpos = qpos_trajectory.reshape((-1, QPOS_DIM))
        flat_qvel = qvel_trajectory.reshape((-1, QVEL_DIM))
        flat_ctrl = jnp.zeros((flat_qpos.shape[0], model.nu), dtype=qpos_trajectory.dtype)
        flat_state = jax.vmap(
            lambda q, qd, c: mjx.forward(mjx_model, data_template.replace(qpos=q, qvel=qd, ctrl=c))
        )(flat_qpos, flat_qvel, flat_ctrl)
        robot = extract_robot_state(sys_view, flat_state)
        return (
            jnp.sum(robot.body_pos_w)
            + jnp.sum(robot.body_quat_w)
            + jnp.sum(robot.body_lin_vel_w)
            + jnp.sum(robot.body_ang_vel_w)
            + jnp.sum(qvel_trajectory)
        )

    return command_kinematics


def _trace_checksum(trace) -> jnp.ndarray:
    return (
        jnp.sum(trace.qpos)
        + jnp.sum(trace.qvel)
        + jnp.sum(trace.body_pos_w)
        + jnp.sum(trace.body_quat_w)
        + jnp.sum(trace.body_lin_vel_w)
        + jnp.sum(trace.body_ang_vel_w)
        + jnp.sum(trace.actions)
        + jnp.sum(trace.controls)
        + jnp.sum(trace.contact_indicator)
        + jnp.sum(trace.contact_force)
        + jnp.sum(trace.floor_contact_indicator)
        + jnp.sum(trace.floor_contact_force)
        + jnp.sum(trace.ref_indices)
    )


def _score_trace_checksum(trace) -> jnp.ndarray:
    return (
        jnp.sum(trace.qpos)
        + jnp.sum(trace.body_pos_w)
        + jnp.sum(trace.body_quat_w)
        + jnp.sum(trace.controls)
        + jnp.sum(trace.contact_indicator)
        + jnp.sum(trace.ref_indices)
    )


def _bench(fn, *args, warmups: int, repeats: int) -> dict[str, object]:
    for _ in range(warmups):
        _block(fn(*args))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _block(fn(*args))
        times.append(time.perf_counter() - t0)
    return {
        "mean_s": float(sum(times) / len(times)),
        "min_s": float(min(times)),
        "max_s": float(max(times)),
        "samples_s": [float(t) for t in times],
    }


def _block(value) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _load_reward_weights(path: str | None, method: str) -> dict[str, float] | None:
    if path is None:
        return None
    raw = json.loads(Path(path).expanduser().read_text())
    if method in raw and isinstance(raw[method], dict):
        raw = raw[method]
    return {str(key): float(value) for key, value in raw.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--motion-type", default="auto", choices=("auto", "mujoco", "isaaclab"))
    parser.add_argument("--checkpoint", default="bc")
    parser.add_argument("--method", default="g1_wbc_joint")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--reward-weights-json", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mpc-samples", type=int, default=16)
    parser.add_argument("--mpc-horizon", type=int, default=20)
    parser.add_argument("--mpc-control-steps", type=int, default=10)
    parser.add_argument("--mpc-nodes", type=int, default=4)
    parser.add_argument("--mpc-diffuse", type=int, default=1)
    parser.add_argument("--mpc-diffuse-init", type=int, default=2)
    parser.add_argument("--mpc-temperature", type=float, default=0.7)
    parser.add_argument("--mpc-root-pos-sigma", type=float, default=0.015)
    parser.add_argument("--mpc-root-rot-sigma", type=float, default=0.035)
    parser.add_argument("--mpc-joint-sigma", type=float, default=0.06)
    parser.add_argument("--mpc-min-root-pos-sigma", type=float, default=0.002)
    parser.add_argument("--mpc-min-root-rot-sigma", type=float, default=0.004)
    parser.add_argument("--mpc-min-joint-sigma", type=float, default=0.008)
    parser.add_argument("--mpc-command-reg-weight", type=float, default=0.02)
    parser.add_argument("--mpc-command-smooth-weight", type=float, default=0.00005)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
