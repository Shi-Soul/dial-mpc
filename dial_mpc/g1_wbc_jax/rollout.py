"""JAX/MJX policy rollouts for G1 WBC command MPC."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
with contextlib.redirect_stdout(io.StringIO()):
    import mujoco.mjx as mjx

if TYPE_CHECKING:
    from brax.base import System

from dial_mpc.g1_wbc_jax.constants import (
    ACTION_DIM,
    DECIMATION,
    LEFT_FOOT_BODY_NAME,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
    RIGHT_FOOT_BODY_NAME,
)
from dial_mpc.g1_wbc_jax.math import qvel_from_qpos_trajectory
from dial_mpc.g1_wbc_jax.metrics import RolloutTrace
from dial_mpc.g1_wbc_jax.model import default_joint_pos, joint_actuator_specs
from dial_mpc.g1_wbc_jax.motion import G1CommandBatch, G1Motion
from dial_mpc.g1_wbc_jax.obs import RobotState, WbcObsState, compute_obs, init_obs_state
from dial_mpc.g1_wbc_jax.policy import WbcActorParams, actor_forward


@dataclass(frozen=True)
class G1WbcRolloutConfig:
    max_steps: int | None = None
    ref_offset: int = 0
    decimation: int = DECIMATION
    dt: float = POLICY_DT
    forward_after_step: bool = True


class RolloutOutput(NamedTuple):
    trace: RolloutTrace
    final_pipeline_state: object
    final_last_action: jnp.ndarray
    final_obs_state: WbcObsState


class ContactGroups(NamedTuple):
    floor: jnp.ndarray
    left_foot: jnp.ndarray
    right_foot: jnp.ndarray
    other_robot: jnp.ndarray


def command_batch_from_qpos_trajectory(
    sys: System,
    template_motion: G1Motion,
    qpos_trajectory: jnp.ndarray,
    *,
    preserve_template_first: bool = False,
) -> G1CommandBatch:
    """Convert candidate command qpos trajectories to WBC reference fields."""

    pipeline = _brax_pipeline()
    if qpos_trajectory.ndim == 2:
        qpos_trajectory = qpos_trajectory[:, None, :]
    if qpos_trajectory.ndim != 3 or qpos_trajectory.shape[-1] != QPOS_DIM:
        raise ValueError(
            "Expected qpos trajectory shape (T, N, 36) or (T, 36), "
            f"got {qpos_trajectory.shape}."
        )

    qvel_trajectory = qvel_from_qpos_trajectory(qpos_trajectory, dt=POLICY_DT)
    flat_qpos = qpos_trajectory.reshape((-1, QPOS_DIM))
    flat_qvel = qvel_trajectory.reshape((-1, QVEL_DIM))
    flat_ctrl = jnp.zeros((flat_qpos.shape[0], ACTION_DIM), dtype=qpos_trajectory.dtype)

    def init_one(qpos, qvel, ctrl):
        return pipeline.init(sys, qpos, qvel, ctrl=ctrl)

    flat_state = jax.vmap(init_one)(flat_qpos, flat_qvel, flat_ctrl)
    robot = extract_robot_state(sys, flat_state)
    shape_prefix = qpos_trajectory.shape[:2]
    body_pos_w = robot.body_pos_w.reshape(shape_prefix + robot.body_pos_w.shape[1:])
    body_quat_w = robot.body_quat_w.reshape(shape_prefix + robot.body_quat_w.shape[1:])
    body_lin_vel_w = robot.body_lin_vel_w.reshape(shape_prefix + robot.body_lin_vel_w.shape[1:])
    body_ang_vel_w = robot.body_ang_vel_w.reshape(shape_prefix + robot.body_ang_vel_w.shape[1:])

    joint_pos = qpos_trajectory[..., 7:]
    joint_vel = qvel_trajectory[..., 6:]
    if preserve_template_first:
        frame_count = qpos_trajectory.shape[0]
        joint_pos = joint_pos.at[:, 0].set(template_motion.joint_pos[:frame_count])
        joint_vel = joint_vel.at[:, 0].set(template_motion.joint_vel[:frame_count])
        body_pos_w = body_pos_w.at[:, 0].set(template_motion.body_pos_w[:frame_count])
        body_quat_w = body_quat_w.at[:, 0].set(template_motion.body_quat_w[:frame_count])
        body_lin_vel_w = body_lin_vel_w.at[:, 0].set(template_motion.body_lin_vel_w[:frame_count])
        body_ang_vel_w = body_ang_vel_w.at[:, 0].set(template_motion.body_ang_vel_w[:frame_count])
        qpos_trajectory = qpos_trajectory.at[:, 0].set(template_motion.qpos()[:frame_count])
        qvel_trajectory = qvel_trajectory.at[:, 0].set(template_motion.qvel()[:frame_count])

    return G1CommandBatch(
        path=template_motion.path,
        motion_type=template_motion.motion_type,
        fps=template_motion.fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        qpos_trajectory=qpos_trajectory,
        qvel_trajectory=qvel_trajectory,
    )


def mjx_command_batch_from_qpos_trajectory(
    model: mujoco.MjModel,
    template_motion: G1Motion,
    qpos_trajectory: jnp.ndarray,
    *,
    preserve_template_first: bool = False,
) -> G1CommandBatch:
    """Convert command qpos trajectories to WBC reference fields with native MJX."""

    if qpos_trajectory.ndim == 2:
        qpos_trajectory = qpos_trajectory[:, None, :]
    if qpos_trajectory.ndim != 3 or qpos_trajectory.shape[-1] != QPOS_DIM:
        raise ValueError(
            "Expected qpos trajectory shape (T, N, 36) or (T, 36), "
            f"got {qpos_trajectory.shape}."
        )

    mjx_model = mjx.put_model(model)
    data_template = mjx.make_data(mjx_model)

    class _ModelView(NamedTuple):
        mj_model: mujoco.MjModel
        nu: int

    sys_view = _ModelView(mj_model=model, nu=model.nu)

    qvel_trajectory = qvel_from_qpos_trajectory(qpos_trajectory, dt=POLICY_DT)
    flat_qpos = qpos_trajectory.reshape((-1, QPOS_DIM))
    flat_qvel = qvel_trajectory.reshape((-1, QVEL_DIM))
    flat_ctrl = jnp.zeros((flat_qpos.shape[0], model.nu), dtype=qpos_trajectory.dtype)
    flat_state = jax.vmap(
        lambda q, qd, c: mjx.forward(mjx_model, data_template.replace(qpos=q, qvel=qd, ctrl=c))
    )(flat_qpos, flat_qvel, flat_ctrl)
    robot = extract_robot_state(sys_view, flat_state)
    shape_prefix = qpos_trajectory.shape[:2]
    body_pos_w = robot.body_pos_w.reshape(shape_prefix + robot.body_pos_w.shape[1:])
    body_quat_w = robot.body_quat_w.reshape(shape_prefix + robot.body_quat_w.shape[1:])
    body_lin_vel_w = robot.body_lin_vel_w.reshape(shape_prefix + robot.body_lin_vel_w.shape[1:])
    body_ang_vel_w = robot.body_ang_vel_w.reshape(shape_prefix + robot.body_ang_vel_w.shape[1:])

    joint_pos = qpos_trajectory[..., 7:]
    joint_vel = qvel_trajectory[..., 6:]
    if preserve_template_first:
        frame_count = qpos_trajectory.shape[0]
        joint_pos = joint_pos.at[:, 0].set(template_motion.joint_pos[:frame_count])
        joint_vel = joint_vel.at[:, 0].set(template_motion.joint_vel[:frame_count])
        body_pos_w = body_pos_w.at[:, 0].set(template_motion.body_pos_w[:frame_count])
        body_quat_w = body_quat_w.at[:, 0].set(template_motion.body_quat_w[:frame_count])
        body_lin_vel_w = body_lin_vel_w.at[:, 0].set(template_motion.body_lin_vel_w[:frame_count])
        body_ang_vel_w = body_ang_vel_w.at[:, 0].set(template_motion.body_ang_vel_w[:frame_count])
        qpos_trajectory = qpos_trajectory.at[:, 0].set(template_motion.qpos()[:frame_count])
        qvel_trajectory = qvel_trajectory.at[:, 0].set(template_motion.qvel()[:frame_count])

    return G1CommandBatch(
        path=template_motion.path,
        motion_type=template_motion.motion_type,
        fps=template_motion.fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        qpos_trajectory=qpos_trajectory,
        qvel_trajectory=qvel_trajectory,
    )


def make_policy_rollout(
    sys: System,
    actor: WbcActorParams,
    template_motion: G1Motion,
    config: G1WbcRolloutConfig | None = None,
):
    """Create a JIT-compiled policy rollout over batched command trajectories."""

    pipeline = _brax_pipeline()
    cfg = config or G1WbcRolloutConfig()
    groups = contact_groups(sys.mj_model)
    joint_default = default_joint_pos()
    action_scale = joint_actuator_specs()["action_scale"]
    actuator_ids = actuator_ids_by_joint(sys.mj_model)

    @jax.jit
    def run(
        command_qpos: jnp.ndarray,
        initial_qpos: jnp.ndarray,
        initial_qvel: jnp.ndarray,
        initial_last_action: jnp.ndarray | None = None,
        initial_obs_state: WbcObsState | None = None,
        ref_start: jnp.ndarray = jnp.array(0, dtype=jnp.int32),
    ) -> RolloutOutput:
        command = command_batch_from_qpos_trajectory(sys, template_motion, command_qpos)
        num_envs = command.qpos_trajectory.shape[1]
        total_steps = command.num_frames if cfg.max_steps is None else min(cfg.max_steps, command.num_frames)

        qpos0 = _batch_vector(initial_qpos, num_envs, QPOS_DIM)
        qvel0 = _batch_vector(initial_qvel, num_envs, QVEL_DIM)
        last_action0 = (
            jnp.zeros((num_envs, ACTION_DIM), dtype=command_qpos.dtype)
            if initial_last_action is None
            else _batch_vector(initial_last_action, num_envs, ACTION_DIM)
        )
        obs_state0 = (
            init_obs_state(num_envs, dtype=command_qpos.dtype)
            if initial_obs_state is None
            else expand_obs_state(initial_obs_state, num_envs)
        )

        ctrl0 = jnp.zeros((num_envs, ACTION_DIM), dtype=command_qpos.dtype)
        state0 = jax.vmap(lambda q, qd, c: pipeline.init(sys, q, qd, ctrl=c))(qpos0, qvel0, ctrl0)
        trace0 = _trace_state(sys, groups, state0)

        def step(carry, step_idx):
            state, obs_state, last_action = carry
            robot = extract_robot_state(sys, state)
            local_ref = jnp.full(
                (num_envs,),
                jnp.clip(step_idx + int(cfg.ref_offset), 0, command.num_frames - 1),
                dtype=jnp.int32,
            )
            obs, obs_state = compute_obs(obs_state, command, robot, local_ref, last_action, joint_default)
            action = actor_forward(actor, obs)
            ctrl = action * action_scale.reshape(1, ACTION_DIM) + joint_default.reshape(1, ACTION_DIM)
            model_ctrl = joint_order_to_model_ctrl(ctrl, actuator_ids, sys.nu)

            def physics_step(s, _):
                s = jax.vmap(lambda one_state, one_ctrl: pipeline.step(sys, one_state, one_ctrl))(s, model_ctrl)
                return s, None

            state, _ = jax.lax.scan(physics_step, state, None, length=int(cfg.decimation))
            if cfg.forward_after_step:
                state = jax.vmap(lambda q, qd, c: pipeline.init(sys, q, qd, ctrl=c))(
                    state.qpos,
                    state.qvel,
                    model_ctrl,
                )
            traced = _trace_state(sys, groups, state)
            abs_ref = local_ref + ref_start.astype(jnp.int32)
            return (state, obs_state, action), (traced, action, ctrl, abs_ref)

        steps = jnp.arange(total_steps, dtype=jnp.int32)
        (final_state, final_obs_state, final_action), (traced, actions, controls, step_ref_indices) = jax.lax.scan(
            step,
            (state0, obs_state0, last_action0),
            steps,
        )
        trace = _stack_trace(trace0, traced, actions, controls, step_ref_indices, ref_start, cfg.dt)
        return RolloutOutput(trace, final_state, final_action, final_obs_state)

    return run


def make_open_loop_rollout(
    sys: System,
    config: G1WbcRolloutConfig | None = None,
):
    """Create a JIT-compiled MJX rollout driven by saved joint targets."""

    pipeline = _brax_pipeline()
    cfg = config or G1WbcRolloutConfig()
    groups = contact_groups(sys.mj_model)
    actuator_ids = actuator_ids_by_joint(sys.mj_model)

    @jax.jit
    def run(
        controls: jnp.ndarray,
        initial_qpos: jnp.ndarray,
        initial_qvel: jnp.ndarray,
    ) -> RolloutTrace:
        if controls.ndim == 2:
            controls_batched = controls[:, None, :]
        else:
            controls_batched = controls
        if controls_batched.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected controls dim {ACTION_DIM}, got {controls_batched.shape}.")
        num_envs = controls_batched.shape[1]
        total_steps = controls_batched.shape[0] if cfg.max_steps is None else min(cfg.max_steps, controls_batched.shape[0])

        qpos0 = _batch_vector(initial_qpos, num_envs, QPOS_DIM)
        qvel0 = _batch_vector(initial_qvel, num_envs, QVEL_DIM)
        ctrl0 = jnp.zeros((num_envs, ACTION_DIM), dtype=controls_batched.dtype)
        state0 = jax.vmap(lambda q, qd, c: pipeline.init(sys, q, qd, ctrl=c))(qpos0, qvel0, ctrl0)
        trace0 = _trace_state(sys, groups, state0)

        def step(state, ctrl):
            model_ctrl = joint_order_to_model_ctrl(ctrl, actuator_ids, sys.nu)

            def physics_step(s, _):
                s = jax.vmap(lambda one_state, one_ctrl: pipeline.step(sys, one_state, one_ctrl))(s, model_ctrl)
                return s, None

            state, _ = jax.lax.scan(physics_step, state, None, length=int(cfg.decimation))
            if cfg.forward_after_step:
                state = jax.vmap(lambda q, qd, c: pipeline.init(sys, q, qd, ctrl=c))(
                    state.qpos,
                    state.qvel,
                    model_ctrl,
                )
            return state, _trace_state(sys, groups, state)

        _, traced = jax.lax.scan(step, state0, controls_batched[:total_steps])
        actions = jnp.zeros_like(controls_batched[:total_steps])
        ref_indices = jnp.zeros((total_steps, num_envs), dtype=jnp.int32)
        return _stack_trace(trace0, traced, actions, controls_batched[:total_steps], ref_indices, jnp.array(0, dtype=jnp.int32), cfg.dt)

    return run


def make_mjx_open_loop_rollout(
    model: mujoco.MjModel,
    config: G1WbcRolloutConfig | None = None,
):
    """Create a JIT-compiled native MuJoCo MJX rollout driven by joint targets."""

    cfg = config or G1WbcRolloutConfig()
    mjx_model = mjx.put_model(model)
    data_template = mjx.make_data(mjx_model)
    groups = contact_groups(model)
    actuator_ids = actuator_ids_by_joint(model)

    class _ModelView(NamedTuple):
        mj_model: mujoco.MjModel
        nu: int

    sys_view = _ModelView(mj_model=model, nu=model.nu)

    @jax.jit
    def run(
        controls: jnp.ndarray,
        initial_qpos: jnp.ndarray,
        initial_qvel: jnp.ndarray,
    ) -> RolloutTrace:
        if controls.ndim == 2:
            controls_batched = controls[:, None, :]
        else:
            controls_batched = controls
        if controls_batched.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected controls dim {ACTION_DIM}, got {controls_batched.shape}.")
        num_envs = controls_batched.shape[1]
        total_steps = controls_batched.shape[0] if cfg.max_steps is None else min(cfg.max_steps, controls_batched.shape[0])

        qpos0 = _batch_vector(initial_qpos, num_envs, QPOS_DIM)
        qvel0 = _batch_vector(initial_qvel, num_envs, QVEL_DIM)
        ctrl0 = jnp.zeros((num_envs, ACTION_DIM), dtype=controls_batched.dtype)
        model_ctrl0 = joint_order_to_model_ctrl(ctrl0, actuator_ids, model.nu)
        state0 = jax.vmap(
            lambda q, qd, c: mjx.forward(mjx_model, data_template.replace(qpos=q, qvel=qd, ctrl=c))
        )(qpos0, qvel0, model_ctrl0)
        trace0 = _trace_state(sys_view, groups, state0)

        def step(state, ctrl):
            model_ctrl = joint_order_to_model_ctrl(ctrl, actuator_ids, model.nu)

            def physics_step(s, _):
                s = jax.vmap(lambda one_state, one_ctrl: mjx.step(mjx_model, one_state.replace(ctrl=one_ctrl)))(
                    s,
                    model_ctrl,
                )
                return s, None

            state, _ = jax.lax.scan(physics_step, state, None, length=int(cfg.decimation))
            if cfg.forward_after_step:
                state = jax.vmap(lambda one_state, one_ctrl: mjx.forward(mjx_model, one_state.replace(ctrl=one_ctrl)))(
                    state,
                    model_ctrl,
                )
            return state, _trace_state(sys_view, groups, state)

        _, traced = jax.lax.scan(step, state0, controls_batched[:total_steps])
        actions = jnp.zeros_like(controls_batched[:total_steps])
        ref_indices = jnp.zeros((total_steps, num_envs), dtype=jnp.int32)
        return _stack_trace(
            trace0,
            traced,
            actions,
            controls_batched[:total_steps],
            ref_indices,
            jnp.array(0, dtype=jnp.int32),
            cfg.dt,
        )

    return run


def make_mjx_policy_rollout(
    model: mujoco.MjModel,
    actor: WbcActorParams,
    template_motion: G1Motion,
    config: G1WbcRolloutConfig | None = None,
):
    """Create a JIT-compiled native MuJoCo MJX policy rollout."""

    cfg = config or G1WbcRolloutConfig()
    mjx_model = mjx.put_model(model)
    data_template = mjx.make_data(mjx_model)
    groups = contact_groups(model)
    joint_default = default_joint_pos()
    action_scale = joint_actuator_specs()["action_scale"]
    actuator_ids = actuator_ids_by_joint(model)

    class _ModelView(NamedTuple):
        mj_model: mujoco.MjModel
        nu: int

    sys_view = _ModelView(mj_model=model, nu=model.nu)

    def command_from_qpos_trajectory(qpos_trajectory: jnp.ndarray) -> G1CommandBatch:
        if qpos_trajectory.ndim == 2:
            qpos_trajectory = qpos_trajectory[:, None, :]
        if qpos_trajectory.ndim != 3 or qpos_trajectory.shape[-1] != QPOS_DIM:
            raise ValueError(
                "Expected qpos trajectory shape (T, N, 36) or (T, 36), "
                f"got {qpos_trajectory.shape}."
            )

        qvel_trajectory = qvel_from_qpos_trajectory(qpos_trajectory, dt=POLICY_DT)
        flat_qpos = qpos_trajectory.reshape((-1, QPOS_DIM))
        flat_qvel = qvel_trajectory.reshape((-1, QVEL_DIM))
        flat_ctrl = jnp.zeros((flat_qpos.shape[0], model.nu), dtype=qpos_trajectory.dtype)
        flat_state = jax.vmap(
            lambda q, qd, c: mjx.forward(mjx_model, data_template.replace(qpos=q, qvel=qd, ctrl=c))
        )(flat_qpos, flat_qvel, flat_ctrl)
        robot = extract_robot_state(sys_view, flat_state)
        shape_prefix = qpos_trajectory.shape[:2]
        body_pos_w = robot.body_pos_w.reshape(shape_prefix + robot.body_pos_w.shape[1:])
        body_quat_w = robot.body_quat_w.reshape(shape_prefix + robot.body_quat_w.shape[1:])
        body_lin_vel_w = robot.body_lin_vel_w.reshape(shape_prefix + robot.body_lin_vel_w.shape[1:])
        body_ang_vel_w = robot.body_ang_vel_w.reshape(shape_prefix + robot.body_ang_vel_w.shape[1:])

        return G1CommandBatch(
            path=template_motion.path,
            motion_type=template_motion.motion_type,
            fps=template_motion.fps,
            joint_pos=qpos_trajectory[..., 7:],
            joint_vel=qvel_trajectory[..., 6:],
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            qpos_trajectory=qpos_trajectory,
            qvel_trajectory=qvel_trajectory,
        )

    @jax.jit
    def run(
        command_qpos: jnp.ndarray,
        initial_qpos: jnp.ndarray,
        initial_qvel: jnp.ndarray,
        initial_last_action: jnp.ndarray | None = None,
        initial_obs_state: WbcObsState | None = None,
        ref_start: jnp.ndarray = jnp.array(0, dtype=jnp.int32),
    ) -> RolloutOutput:
        command = command_from_qpos_trajectory(command_qpos)
        num_envs = command.qpos_trajectory.shape[1]
        total_steps = command.num_frames if cfg.max_steps is None else min(cfg.max_steps, command.num_frames)

        qpos0 = _batch_vector(initial_qpos, num_envs, QPOS_DIM)
        qvel0 = _batch_vector(initial_qvel, num_envs, QVEL_DIM)
        last_action0 = (
            jnp.zeros((num_envs, ACTION_DIM), dtype=command_qpos.dtype)
            if initial_last_action is None
            else _batch_vector(initial_last_action, num_envs, ACTION_DIM)
        )
        obs_state0 = (
            init_obs_state(num_envs, dtype=command_qpos.dtype)
            if initial_obs_state is None
            else expand_obs_state(initial_obs_state, num_envs)
        )

        ctrl0 = jnp.zeros((num_envs, ACTION_DIM), dtype=command_qpos.dtype)
        model_ctrl0 = joint_order_to_model_ctrl(ctrl0, actuator_ids, model.nu)
        state0 = jax.vmap(
            lambda q, qd, c: mjx.forward(mjx_model, data_template.replace(qpos=q, qvel=qd, ctrl=c))
        )(qpos0, qvel0, model_ctrl0)
        trace0 = _trace_state(sys_view, groups, state0)

        def step(carry, step_idx):
            state, obs_state, last_action = carry
            robot = extract_robot_state(sys_view, state)
            local_ref = jnp.full(
                (num_envs,),
                jnp.clip(step_idx + int(cfg.ref_offset), 0, command.num_frames - 1),
                dtype=jnp.int32,
            )
            obs, obs_state = compute_obs(obs_state, command, robot, local_ref, last_action, joint_default)
            action = actor_forward(actor, obs)
            ctrl = action * action_scale.reshape(1, ACTION_DIM) + joint_default.reshape(1, ACTION_DIM)
            model_ctrl = joint_order_to_model_ctrl(ctrl, actuator_ids, model.nu)

            def physics_step(s, _):
                s = jax.vmap(lambda one_state, one_ctrl: mjx.step(mjx_model, one_state.replace(ctrl=one_ctrl)))(
                    s,
                    model_ctrl,
                )
                return s, None

            state, _ = jax.lax.scan(physics_step, state, None, length=int(cfg.decimation))
            if cfg.forward_after_step:
                state = jax.vmap(lambda one_state, one_ctrl: mjx.forward(mjx_model, one_state.replace(ctrl=one_ctrl)))(
                    state,
                    model_ctrl,
                )
            traced = _trace_state(sys_view, groups, state)
            abs_ref = local_ref + ref_start.astype(jnp.int32)
            return (state, obs_state, action), (traced, action, ctrl, abs_ref)

        steps = jnp.arange(total_steps, dtype=jnp.int32)
        (final_state, final_obs_state, final_action), (traced, actions, controls, step_ref_indices) = jax.lax.scan(
            step,
            (state0, obs_state0, last_action0),
            steps,
        )
        trace = _stack_trace(trace0, traced, actions, controls, step_ref_indices, ref_start, cfg.dt)
        return RolloutOutput(trace, final_state, final_action, final_obs_state)

    return run


def extract_robot_state(sys: System, state) -> RobotState:
    body_ids = _robot_body_ids(sys.mj_model)
    body_pos = jnp.take(state.xpos, body_ids, axis=-2)
    body_quat = jnp.take(state.xquat, body_ids, axis=-2)
    cvel = jnp.take(state.cvel, body_ids, axis=-2)
    root_body_id = _body_id(sys.mj_model, "pelvis")
    root_com = jnp.take(state.subtree_com, jnp.asarray(root_body_id, dtype=jnp.int32), axis=-2)
    while root_com.ndim < body_pos.ndim:
        root_com = jnp.expand_dims(root_com, axis=-2)
    lin_vel_c = cvel[..., 3:6]
    ang_vel_w = cvel[..., 0:3]
    lin_vel_w = lin_vel_c - jnp.cross(ang_vel_w, root_com - body_pos, axis=-1)
    imu_slice = _sensor_slice(sys.mj_model, "imu_ang_vel")
    base_ang_vel_b = None if imu_slice is None else state.sensordata[..., imu_slice]
    return RobotState(
        qpos=state.qpos,
        qvel=state.qvel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=lin_vel_w,
        body_ang_vel_w=ang_vel_w,
        base_ang_vel_b=base_ang_vel_b,
    )


def contact_groups(model: mujoco.MjModel) -> ContactGroups:
    left = _body_geom_ids(model, LEFT_FOOT_BODY_NAME)
    right = _body_geom_ids(model, RIGHT_FOOT_BODY_NAME)
    foot = set(left.tolist()) | set(right.tolist())
    other = [
        geom_id
        for geom_id in range(model.ngeom)
        if geom_id not in foot and _geom_is_robot_collision(model, geom_id)
    ]
    floor = _geom_id(model, "terrain")
    if floor < 0:
        floor = _geom_id(model, "floor")
    return ContactGroups(
        floor=jnp.asarray(floor, dtype=jnp.int32),
        left_foot=jnp.asarray(left, dtype=jnp.int32),
        right_foot=jnp.asarray(right, dtype=jnp.int32),
        other_robot=jnp.asarray(other, dtype=jnp.int32),
    )


def actuator_ids_by_joint(model: mujoco.MjModel) -> jnp.ndarray:
    joint_name_to_actuator: dict[str, int] = {}
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if 0 <= joint_id < model.njnt:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                joint_name_to_actuator[joint_name.removeprefix("robot/")] = int(act_id)
    ids = []
    for joint_name in MUJOCO_JOINT_NAMES:
        if joint_name not in joint_name_to_actuator:
            raise ValueError(f"G1 model is missing actuator for joint {joint_name}")
        ids.append(joint_name_to_actuator[joint_name])
    return jnp.asarray(ids, dtype=jnp.int32)


def joint_order_to_model_ctrl(ctrl: jnp.ndarray, actuator_ids: jnp.ndarray, model_nu: int) -> jnp.ndarray:
    out = jnp.zeros(ctrl.shape[:-1] + (model_nu,), dtype=ctrl.dtype)
    return out.at[..., actuator_ids].set(ctrl)


def expand_obs_state(state: WbcObsState, num_envs: int) -> WbcObsState:
    return WbcObsState(
        ref_limb_ee_pose_b=_expand_history(state.ref_limb_ee_pose_b, num_envs),
        robot_limb_ee_pose_b=_expand_history(state.robot_limb_ee_pose_b, num_envs),
        projected_gravity=_expand_history(state.projected_gravity, num_envs),
        base_ang_vel=_expand_history(state.base_ang_vel, num_envs),
        joint_pos=_expand_history(state.joint_pos, num_envs),
        joint_vel=_expand_history(state.joint_vel, num_envs),
        actions=_expand_history(state.actions, num_envs),
    )


def _trace_state(sys: System, groups: ContactGroups, state) -> tuple[jnp.ndarray, ...]:
    robot = extract_robot_state(sys, state)
    floor_indicator, floor_force = floor_contact(state, groups)
    return (
        robot.qpos,
        robot.qvel,
        robot.body_pos_w,
        robot.body_quat_w,
        robot.body_lin_vel_w,
        robot.body_ang_vel_w,
        floor_indicator[:, :2],
        floor_force[:, :2],
        floor_indicator,
        floor_force,
    )


def _stack_trace(
    trace0: tuple[jnp.ndarray, ...],
    traced: tuple[jnp.ndarray, ...],
    actions: jnp.ndarray,
    controls: jnp.ndarray,
    step_ref_indices: jnp.ndarray,
    ref_start: jnp.ndarray,
    dt: float,
) -> RolloutTrace:
    stacked = tuple(jnp.concatenate([first[None], rest], axis=0) for first, rest in zip(trace0, traced))
    num_envs = actions.shape[1]
    initial_ref = jnp.full((1, num_envs), ref_start.astype(jnp.int32), dtype=jnp.int32)
    ref_indices = jnp.concatenate([initial_ref, step_ref_indices], axis=0)
    return RolloutTrace(
        qpos=stacked[0],
        qvel=stacked[1],
        body_pos_w=stacked[2],
        body_quat_w=stacked[3],
        body_lin_vel_w=stacked[4],
        body_ang_vel_w=stacked[5],
        actions=actions,
        controls=controls,
        contact_indicator=stacked[6],
        contact_force=stacked[7],
        ref_indices=ref_indices,
        floor_contact_indicator=stacked[8],
        floor_contact_force=stacked[9],
        dt=dt,
    )


def floor_contact(state, groups: ContactGroups) -> tuple[jnp.ndarray, jnp.ndarray]:
    geom = state.contact.geom
    dist = state.contact.dist
    margin = state.contact.includemargin
    active = (dist <= margin + 1.0e-5) & (geom[..., 0] >= 0) & (geom[..., 1] >= 0)
    has_floor = (geom[..., 0] == groups.floor) | (geom[..., 1] == groups.floor)
    groups_tuple = (groups.left_foot, groups.right_foot, groups.other_robot)
    indicators = []
    forces = []
    for geom_group in groups_tuple:
        in_group = _pair_has_any(geom, geom_group)
        mask = active & has_floor & in_group
        indicators.append(jnp.any(mask, axis=-1).astype(jnp.float32))
        forces.append(_normal_force_sum(state, mask))
    return jnp.stack(indicators, axis=-1), jnp.stack(forces, axis=-1)


def _normal_force_sum(state, mask: jnp.ndarray) -> jnp.ndarray:
    address = jnp.asarray(state.contact.efc_address)
    if address.ndim == 1 and mask.ndim == 2:
        address = jnp.broadcast_to(address[None, :], mask.shape)
    address = jnp.clip(address, 0, state.efc_force.shape[-1] - 1)
    force = jnp.take_along_axis(state.efc_force, address, axis=-1)
    return jnp.where(mask, jnp.maximum(force, 0.0), 0.0).sum(axis=-1)


def _pair_has_any(geom_pair: jnp.ndarray, geom_ids: jnp.ndarray) -> jnp.ndarray:
    if geom_ids.size == 0:
        return jnp.zeros(geom_pair.shape[:-1], dtype=bool)
    lhs = geom_pair[..., 0, None] == geom_ids
    rhs = geom_pair[..., 1, None] == geom_ids
    return jnp.any(lhs | rhs, axis=-1)


def _expand_history(state, num_envs: int):
    if state.buffer.shape[1] == num_envs:
        return state
    if state.buffer.shape[1] != 1:
        raise ValueError(f"Cannot expand history batch {state.buffer.shape[1]} to {num_envs}.")
    return type(state)(
        buffer=jnp.broadcast_to(state.buffer, (state.buffer.shape[0], num_envs, state.buffer.shape[2])),
        pointer=state.pointer,
        num_pushes=jnp.broadcast_to(state.num_pushes, (num_envs,)),
    )


def _batch_vector(value: jnp.ndarray, num_envs: int, dim: int) -> jnp.ndarray:
    if value.ndim == 1:
        if value.shape[0] != dim:
            raise ValueError(f"Expected vector dim {dim}, got {value.shape}.")
        return jnp.broadcast_to(value[None, :], (num_envs, dim))
    if value.shape == (1, dim):
        return jnp.broadcast_to(value, (num_envs, dim))
    if value.shape != (num_envs, dim):
        raise ValueError(f"Expected {(num_envs, dim)}, got {value.shape}.")
    return value


def _robot_body_ids(model: mujoco.MjModel) -> jnp.ndarray:
    return jnp.asarray([_body_id(model, name) for name in MUJOCO_BODY_NAMES], dtype=jnp.int32)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        return int(body_id)
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{name}"))


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id >= 0:
        return int(geom_id)
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}"))


def _body_geom_ids(model: mujoco.MjModel, body_name: str) -> jnp.ndarray:
    body_id = _body_id(model, body_name)
    return jnp.asarray(
        [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id],
        dtype=jnp.int32,
    )


def _geom_is_robot_collision(model: mujoco.MjModel, geom_id: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""
    if name in ("terrain", "floor"):
        return False
    return name.removeprefix("robot/").endswith("_collision")


def _sensor_slice(model: mujoco.MjModel, sensor_name: str) -> slice | None:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id < 0:
        return None
    start = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])
    return slice(start, start + dim)


def _brax_pipeline():
    with contextlib.redirect_stdout(io.StringIO()):
        import brax.mjx.pipeline as pipeline

    return pipeline
