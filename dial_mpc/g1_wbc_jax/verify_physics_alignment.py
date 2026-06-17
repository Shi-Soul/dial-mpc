"""Compare MJX open-loop physics against saved SPIDER/MJWarp rollout traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np

from dial_mpc.g1_wbc_jax.constants import (
    ACTION_DIM,
    DECIMATION,
    LEFT_FOOT_BODY_NAME,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
    RIGHT_FOOT_BODY_NAME,
)
from dial_mpc.g1_wbc_jax.model import build_wbc_mj_model, build_wbc_system
from dial_mpc.g1_wbc_jax.rollout import (
    G1WbcRolloutConfig,
    make_mjx_open_loop_rollout,
    make_open_loop_rollout,
)


STATE_FIELDS = (
    "qpos",
    "qvel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "contact_indicator",
    "contact_force",
    "floor_contact_indicator",
    "floor_contact_force",
)


def _compare(name: str, actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    if actual.shape != expected.shape:
        raise ValueError(f"{name} shape mismatch: {actual.shape} vs {expected.shape}")
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    abs_diff = np.abs(diff)
    final_diff = np.abs(diff[-1])
    return {
        f"{name}_max_abs": float(abs_diff.max()),
        f"{name}_rmse": float(np.sqrt(np.mean(diff * diff))),
        f"{name}_final_max_abs": float(final_diff.max()),
    }


def run(
    rollout_npz: str | Path,
    seconds: float,
    max_steps: int | None,
    *,
    backend: str = "mjx",
) -> dict[str, float]:
    path = Path(rollout_npz).expanduser().resolve()
    raw = np.load(path)
    dt = float(raw["dt"].item()) if "dt" in raw.files else 0.02
    steps = min(int(round(seconds / dt)), raw["controls"].shape[0])
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    if steps < 1:
        raise ValueError("Need at least one control step for physics alignment.")

    cfg = G1WbcRolloutConfig(max_steps=steps, dt=dt)
    if backend == "brax":
        sys = build_wbc_system()
        fn = make_open_loop_rollout(sys, cfg)
        trace = fn(
            jnp.asarray(raw["controls"][:steps], dtype=jnp.float32),
            jnp.asarray(raw["qpos"][0, 0], dtype=jnp.float32),
            jnp.asarray(raw["qvel"][0, 0], dtype=jnp.float32),
        )
        trace.qpos.block_until_ready()
        actual_by_field = {field: np.asarray(getattr(trace, field)) for field in STATE_FIELDS}
    elif backend == "mjx":
        model = build_wbc_mj_model()
        fn = make_mjx_open_loop_rollout(model, cfg)
        trace = fn(
            jnp.asarray(raw["controls"][:steps], dtype=jnp.float32),
            jnp.asarray(raw["qpos"][0, 0], dtype=jnp.float32),
            jnp.asarray(raw["qvel"][0, 0], dtype=jnp.float32),
        )
        trace.qpos.block_until_ready()
        actual_by_field = {field: np.asarray(getattr(trace, field)) for field in STATE_FIELDS}
    elif backend == "mujoco":
        model = build_wbc_mj_model()
        actual_by_field = _mujoco_open_loop(
            model,
            np.asarray(raw["controls"][:steps, 0], dtype=np.float64),
            np.asarray(raw["qpos"][0, 0], dtype=np.float64),
            np.asarray(raw["qvel"][0, 0], dtype=np.float64),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    stats: dict[str, float] = {"backend": backend, "seconds": steps * dt, "steps": float(steps)}
    for field in STATE_FIELDS:
        if field not in raw.files:
            continue
        actual = actual_by_field[field]
        expected = np.asarray(raw[field][: steps + 1])
        stats.update(_compare(field, actual, expected))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-npz", required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--backend", choices=("mjx", "brax", "mujoco"), default="mjx")
    args = parser.parse_args()

    stats = run(args.rollout_npz, args.seconds, args.max_steps, backend=args.backend)
    print(
        " ".join(
            f"{key}={value}" if isinstance(value, str) else f"{key}={value:.6e}"
            for key, value in stats.items()
        )
    )


def _mujoco_open_loop(
    model: mujoco.MjModel,
    controls: np.ndarray,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
) -> dict[str, np.ndarray]:
    if controls.ndim != 2 or controls.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected controls shape (T, {ACTION_DIM}), got {controls.shape}.")

    data = mujoco.MjData(model)
    actuator_ids = _actuator_ids_by_joint(model)
    body_ids = _body_ids(model)
    groups = _floor_contact_groups(model)

    data.qpos[:] = initial_qpos
    data.qvel[:] = initial_qvel
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    qpos, qvel = [], []
    body_pos, body_quat, body_lin_vel, body_ang_vel = [], [], [], []
    contact_indicator, contact_force = [], []
    floor_contact_indicator, floor_contact_force = [], []

    def append_state() -> None:
        state = _extract_mujoco_state(model, data, body_ids)
        floor_indicator, floor_force = _mujoco_floor_contact(model, data, groups)
        qpos.append(state["qpos"])
        qvel.append(state["qvel"])
        body_pos.append(state["body_pos_w"])
        body_quat.append(state["body_quat_w"])
        body_lin_vel.append(state["body_lin_vel_w"])
        body_ang_vel.append(state["body_ang_vel_w"])
        floor_contact_indicator.append(floor_indicator)
        floor_contact_force.append(floor_force)
        contact_indicator.append(floor_indicator[:2])
        contact_force.append(floor_force[:2])

    append_state()
    for ctrl in controls:
        data.ctrl[:] = _joint_order_to_model_ctrl(ctrl, actuator_ids, model.nu)
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        append_state()

    return {
        "qpos": _batched(np.stack(qpos)),
        "qvel": _batched(np.stack(qvel)),
        "body_pos_w": _batched(np.stack(body_pos)),
        "body_quat_w": _batched(np.stack(body_quat)),
        "body_lin_vel_w": _batched(np.stack(body_lin_vel)),
        "body_ang_vel_w": _batched(np.stack(body_ang_vel)),
        "contact_indicator": _batched(np.stack(contact_indicator)),
        "contact_force": _batched(np.stack(contact_force)),
        "floor_contact_indicator": _batched(np.stack(floor_contact_indicator)),
        "floor_contact_force": _batched(np.stack(floor_contact_force)),
    }


def _batched(value: np.ndarray) -> np.ndarray:
    return value[:, None]


def _extract_mujoco_state(model: mujoco.MjModel, data: mujoco.MjData, body_ids: np.ndarray) -> dict[str, np.ndarray]:
    xpos = np.asarray(data.xpos[body_ids], dtype=np.float64).copy()
    xquat = np.asarray(data.xquat[body_ids], dtype=np.float64).copy()
    cvel = np.asarray(data.cvel[body_ids], dtype=np.float64).copy()
    root_body_id = _body_id(model, "pelvis")
    root_com = np.asarray(data.subtree_com[root_body_id], dtype=np.float64).copy()
    lin_vel_c = cvel[..., 3:6]
    ang_vel_w = cvel[..., 0:3]
    lin_vel_w = lin_vel_c - np.cross(ang_vel_w, root_com[None, :] - xpos)
    return {
        "qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
        "qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
        "body_pos_w": xpos,
        "body_quat_w": xquat,
        "body_lin_vel_w": lin_vel_w,
        "body_ang_vel_w": ang_vel_w,
    }


def _mujoco_floor_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    groups: tuple[int, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    floor_id, left_foot, right_foot, other_robot = groups
    indicator = np.zeros((3,), dtype=np.float64)
    force = np.zeros((3,), dtype=np.float64)
    if floor_id < 0:
        return indicator, force

    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom0, geom1 = int(contact.geom[0]), int(contact.geom[1])
        if geom0 < 0 or geom1 < 0 or contact.dist > contact.includemargin + 1.0e-5:
            continue
        if geom0 != floor_id and geom1 != floor_id:
            continue
        for idx, geom_group in enumerate((left_foot, right_foot, other_robot)):
            if geom0 not in geom_group and geom1 not in geom_group:
                continue
            indicator[idx] = 1.0
            address = int(contact.efc_address)
            if 0 <= address < data.efc_force.shape[0]:
                force[idx] += max(float(data.efc_force[address]), 0.0)
    return indicator, force


def _floor_contact_groups(model: mujoco.MjModel) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    left = _body_geom_ids(model, LEFT_FOOT_BODY_NAME)
    right = _body_geom_ids(model, RIGHT_FOOT_BODY_NAME)
    foot = set(left.tolist()) | set(right.tolist())
    other = np.asarray(
        [
            geom_id
            for geom_id in range(model.ngeom)
            if geom_id not in foot and _geom_is_robot_collision(model, geom_id)
        ],
        dtype=np.int32,
    )
    floor = _geom_id(model, "terrain")
    if floor < 0:
        floor = _geom_id(model, "floor")
    return floor, left, right, other


def _body_ids(model: mujoco.MjModel) -> np.ndarray:
    return np.asarray([_body_id(model, name) for name in MUJOCO_BODY_NAMES], dtype=np.int32)


def _body_geom_ids(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    body_id = _body_id(model, body_name)
    return np.asarray(
        [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id],
        dtype=np.int32,
    )


def _actuator_ids_by_joint(model: mujoco.MjModel) -> np.ndarray:
    joint_to_actuator = {}
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if 0 <= joint_id < model.njnt:
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                joint_to_actuator[joint_name.removeprefix("robot/")] = int(act_id)
    return np.asarray([joint_to_actuator[name] for name in MUJOCO_JOINT_NAMES], dtype=np.int32)


def _joint_order_to_model_ctrl(ctrl: np.ndarray, actuator_ids: np.ndarray, model_nu: int) -> np.ndarray:
    model_ctrl = np.zeros((model_nu,), dtype=np.float64)
    model_ctrl[actuator_ids] = ctrl
    return model_ctrl


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


def _geom_is_robot_collision(model: mujoco.MjModel, geom_id: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""
    if name in ("terrain", "floor"):
        return False
    return name.removeprefix("robot/").endswith("_collision")


if __name__ == "__main__":
    main()
