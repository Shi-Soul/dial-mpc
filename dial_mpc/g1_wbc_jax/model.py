"""MuJoCo/MJX model construction for the JAX G1 WBC rollout."""

from __future__ import annotations

import re
from contextlib import redirect_stdout
import io
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import mujoco

if TYPE_CHECKING:
    from brax.base import System

from dial_mpc.g1_wbc_jax.constants import (
    ACTION_DIM,
    ACTUATOR_GROUPS,
    KNEES_BENT_JOINT_POS,
    MUJOCO_JOINT_NAMES,
    PHYSICS_DT,
    default_attached_g1_model_path,
    default_g1_model_path,
)


def build_wbc_mj_model(model_path: str | Path | None = None) -> mujoco.MjModel:
    """Build the WXY G1 model with WBC-compatible actuators and terrain."""

    attached_path = default_attached_g1_model_path()
    can_attach = hasattr(mujoco.MjSpec, "attach")
    if model_path is None and not can_attach and attached_path.exists():
        model = mujoco.MjModel.from_xml_path(str(attached_path))
        configure_wbc_model(model)
        return model

    explicit_path = Path(model_path).expanduser().resolve() if model_path is not None else None
    if explicit_path is not None and explicit_path == attached_path.resolve():
        model = mujoco.MjModel.from_xml_path(str(explicit_path))
        configure_wbc_model(model)
        return model

    wxy_path = explicit_path if explicit_path is not None else default_g1_model_path()
    mesh_dir = str(wxy_path.parent / "meshes")
    xml_text = wxy_path.read_text()
    xml_text = xml_text.replace('meshdir="meshes"', f'meshdir="{mesh_dir}"')
    robot_spec = mujoco.MjSpec.from_string(xml_text)

    if can_attach:
        spec = mujoco.MjSpec()
        spec.compiler.degree = False
        if hasattr(spec.compiler, "meshdir"):
            spec.compiler.meshdir = mesh_dir
        _add_terrain(spec)
        spec.worldbody.add_site(
            name="env_origin_0",
            pos=(0.0, 0.0, 0.0),
            size=(0.3, 0.3, 0.3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            rgba=(0.2, 0.6, 0.2, 0.3),
            group=4,
        )
        spec.attach(robot_spec, prefix="robot/", frame=spec.worldbody.add_frame(name="robot_frame"))
    else:
        spec = robot_spec
        _add_terrain(spec)
    _configure_collision_spec(spec)
    _add_actuators(spec)
    _add_init_keyframe(spec)
    # tracking_bfm also adds two self-collision contact sensors.  The native
    # MJX JAX backend does not implement their subtree matching semantics, so
    # the JAX model keeps only sensors used by the WBC observation path.

    model = spec.compile()
    configure_wbc_model(model)
    return model


def build_wbc_system(model_path: str | Path | None = None) -> "System":
    with redirect_stdout(io.StringIO()):
        from brax.io import mjcf

    return mjcf.load_model(build_wbc_mj_model(model_path))


def default_joint_pos() -> jnp.ndarray:
    values = jnp.zeros((ACTION_DIM,), dtype=jnp.float32)
    for joint_name, value in KNEES_BENT_JOINT_POS.items():
        values = values.at[MUJOCO_JOINT_NAMES.index(joint_name)].set(float(value))
    return values


def joint_actuator_specs() -> dict[str, jnp.ndarray]:
    kp, kd, effort, armature, action_scale = [], [], [], [], []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_kp, joint_kd, joint_effort, joint_armature = match_actuator_group(joint_name)
        kp.append(joint_kp)
        kd.append(joint_kd)
        effort.append(joint_effort)
        armature.append(joint_armature)
        action_scale.append(joint_effort / (4.0 * joint_kp))
    return {
        "kp": jnp.asarray(kp, dtype=jnp.float32),
        "kd": jnp.asarray(kd, dtype=jnp.float32),
        "effort": jnp.asarray(effort, dtype=jnp.float32),
        "armature": jnp.asarray(armature, dtype=jnp.float32),
        "action_scale": jnp.asarray(action_scale, dtype=jnp.float32),
    }


def joint_limits(model: mujoco.MjModel) -> tuple[jnp.ndarray, jnp.ndarray]:
    low, high = [], []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = _resolve_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"G1 model is missing joint {joint_name}")
        if int(model.jnt_limited[joint_id]):
            low.append(float(model.jnt_range[joint_id, 0]))
            high.append(float(model.jnt_range[joint_id, 1]))
        else:
            low.append(-float("inf"))
            high.append(float("inf"))
    return jnp.asarray(low, dtype=jnp.float32), jnp.asarray(high, dtype=jnp.float32)


def match_actuator_group(joint_name: str) -> tuple[float, float, float, float]:
    matches: list[tuple[float, float, float, float]] = []
    for patterns, kp, kd, effort, armature in ACTUATOR_GROUPS:
        if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
            matches.append((float(kp), float(kd), float(effort), float(armature)))
    if len(matches) != 1:
        raise ValueError(f"Expected one actuator group for {joint_name}, got {len(matches)}")
    return matches[0]


def configure_wbc_model(model: mujoco.MjModel) -> None:
    model.opt.timestep = float(PHYSICS_DT)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    if hasattr(model.opt, "ccd_iterations"):
        model.opt.ccd_iterations = 50
    model.opt.tolerance = 1.0e-8
    model.opt.ls_tolerance = 1.0e-2
    model.opt.disableflags = 0
    model.opt.enableflags = 0

    for joint_name in MUJOCO_JOINT_NAMES:
        kp, kd, effort, armature = match_actuator_group(joint_name)
        joint_id = _resolve_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _find_actuator_by_joint(model, joint_name)
        if joint_id < 0 or actuator_id < 0:
            raise ValueError(f"G1 model is missing joint/actuator {joint_name}")
        dof_id = int(model.jnt_dofadr[joint_id])
        model.dof_armature[dof_id] = armature
        model.dof_damping[dof_id] = 0.0
        model.dof_frictionloss[dof_id] = 0.0
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = (-effort, effort)
        model.actuator_ctrllimited[actuator_id] = 0
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)


def _add_terrain(spec: mujoco.MjSpec) -> None:
    if any((geom.name or "") in ("terrain", "floor") for geom in spec.geoms):
        return
    terrain_body = spec.worldbody.add_body(name="terrain")
    terrain_body.add_geom(
        name="terrain",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.01),
    )


def _configure_collision_spec(spec: mujoco.MjSpec) -> None:
    foot_pattern = re.compile(r"^(?:robot/)?(left|right)_foot[1-7]_collision$")
    for geom in spec.geoms:
        name = geom.name or ""
        if name in ("terrain", "floor"):
            geom.contype = 1
            geom.conaffinity = 1
            geom.condim = 3
            continue
        if not re.fullmatch(r".*_collision", name):
            geom.contype = 0
            geom.conaffinity = 0
            continue
        geom.contype = 1
        geom.conaffinity = 1
        geom.condim = 1
        geom.priority = 0
        if foot_pattern.fullmatch(name):
            geom.condim = 3
            geom.priority = 1
            geom.friction[0] = 0.6


def _add_actuators(spec: mujoco.MjSpec) -> None:
    existing = {actuator.name for actuator in spec.actuators}
    for joint_name in _actuator_joint_names():
        prefixed = f"robot/{joint_name}"
        if prefixed in existing or joint_name in existing:
            continue
        target = prefixed if _find_spec_joint(spec, prefixed) is not None else joint_name
        kp, kd, effort, armature = match_actuator_group(joint_name)
        joint = _find_spec_joint(spec, target)
        if joint is None:
            raise ValueError(f"G1 spec is missing joint {joint_name}")
        joint.armature = float(armature)
        _set_spec_scalar(joint, "damping", 0.0)
        _set_spec_scalar(joint, "frictionloss", 0.0)
        actuator = spec.add_actuator(name=target, target=target)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False
        actuator.forcelimited = True
        actuator.forcerange[0] = -float(effort)
        actuator.forcerange[1] = float(effort)
        delta = float(effort) / float(kp)
        actuator.ctrlrange[0] = float(joint.range[0]) - delta
        actuator.ctrlrange[1] = float(joint.range[1]) + delta
        actuator.gainprm[0] = float(kp)
        actuator.biasprm[1] = -float(kp)
        actuator.biasprm[2] = -float(kd)


def _add_init_keyframe(spec: mujoco.MjSpec) -> None:
    if any((key.name or "") == "init_state" for key in spec.keys):
        return
    qpos = [0.0, 0.0, 0.76, 1.0, 0.0, 0.0, 0.0]
    qpos.extend(float(KNEES_BENT_JOINT_POS.get(name, 0.0)) for name in MUJOCO_JOINT_NAMES)
    ctrl = []
    for actuator in spec.actuators:
        target = actuator.target or ""
        ctrl.append(float(KNEES_BENT_JOINT_POS.get(target.removeprefix("robot/"), 0.0)))
    spec.add_key(name="init_state", qpos=qpos, ctrl=ctrl)


def _add_self_collision_sensors(spec: mujoco.MjSpec) -> None:
    if not hasattr(mujoco.mjtSensor, "mjSENS_CONTACT"):
        return
    existing = {sensor.name for sensor in spec.sensors}
    pelvis = "robot/pelvis" if _find_spec_body(spec, "robot/pelvis") is not None else "pelvis"
    for name, data_bits in (
        ("self_collision_pelvis_found", 1 << 0),
        ("self_collision_pelvis_force", 1 << 1),
    ):
        if name in existing:
            continue
        spec.add_sensor(
            name=name,
            type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_XBODY,
            objname=pelvis,
            reftype=mujoco.mjtObj.mjOBJ_XBODY,
            refname=pelvis,
            intprm=(data_bits, 0, 1),
        )


def _actuator_joint_names() -> tuple[str, ...]:
    names: list[str] = []
    for patterns, *_ in ACTUATOR_GROUPS:
        for joint_name in MUJOCO_JOINT_NAMES:
            if joint_name in names:
                continue
            if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
                names.append(joint_name)
    if set(names) != set(MUJOCO_JOINT_NAMES):
        missing = sorted(set(MUJOCO_JOINT_NAMES) - set(names))
        extra = sorted(set(names) - set(MUJOCO_JOINT_NAMES))
        raise ValueError(f"Invalid G1 actuator groups; missing={missing}, extra={extra}")
    return tuple(names)


def _find_actuator_by_joint(model: mujoco.MjModel, joint_name: str) -> int:
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if not (0 <= joint_id < model.njnt):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name in (joint_name, f"robot/{joint_name}"):
            return int(act_id)
    return -1


def _resolve_name_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id >= 0:
        return int(obj_id)
    return int(mujoco.mj_name2id(model, objtype, f"robot/{name}"))


def _find_spec_joint(spec: mujoco.MjSpec, name: str):
    for joint in spec.joints:
        if (joint.name or "") == name:
            return joint
    return None


def _find_spec_body(spec: mujoco.MjSpec, name: str):
    for body in spec.bodies:
        if (body.name or "") == name:
            return body
    return None


def _set_spec_scalar(obj, name: str, value: float) -> None:
    current = getattr(obj, name)
    try:
        current[0] = value
    except TypeError:
        setattr(obj, name, value)
