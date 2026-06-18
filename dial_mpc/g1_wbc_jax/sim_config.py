"""Simulation option presets for G1 WBC MJX experiments."""

from __future__ import annotations

import mujoco


SIM_PRESET_CHOICES = (
    "default",
    "go2",
    "g1_decim2",
    "g1_lowiter",
    "g1_decim2_lowiter",
)


def apply_sim_preset(model: mujoco.MjModel, preset: str) -> None:
    if preset == "default":
        return
    if preset == "go2":
        _set_common(model, timestep=0.02, iterations=2, ls_iterations=5, ccd_iterations=35)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        model.opt.impratio = 1.0
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        return
    if preset == "g1_decim2":
        _set_common(model, timestep=0.01, iterations=10, ls_iterations=20, ccd_iterations=50)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        return
    if preset == "g1_lowiter":
        _set_common(model, timestep=0.005, iterations=2, ls_iterations=5, ccd_iterations=50)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        return
    if preset == "g1_decim2_lowiter":
        _set_common(model, timestep=0.01, iterations=2, ls_iterations=5, ccd_iterations=50)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        return
    raise ValueError(f"Unsupported simulation preset: {preset}")


def sim_decimation(preset: str) -> int:
    if preset in ("go2",):
        return 1
    if preset in ("g1_decim2", "g1_decim2_lowiter"):
        return 2
    return 4


def simulation_payload(model: mujoco.MjModel, decimation: int, preset: str) -> dict[str, object]:
    return {
        "preset": preset,
        "timestep": float(model.opt.timestep),
        "policy_dt": float(model.opt.timestep * decimation),
        "decimation": int(decimation),
        "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
        "solver": mujoco.mjtSolver(model.opt.solver).name,
        "cone": mujoco.mjtCone(model.opt.cone).name,
        "iterations": int(model.opt.iterations),
        "ls_iterations": int(model.opt.ls_iterations),
        "ccd_iterations": int(model.opt.ccd_iterations) if hasattr(model.opt, "ccd_iterations") else None,
        "tolerance": float(model.opt.tolerance),
        "ls_tolerance": float(model.opt.ls_tolerance),
        "impratio": float(model.opt.impratio),
        "disableflags": int(model.opt.disableflags),
        "enableflags": int(model.opt.enableflags),
    }


def _set_common(
    model: mujoco.MjModel,
    *,
    timestep: float,
    iterations: int,
    ls_iterations: int,
    ccd_iterations: int,
) -> None:
    model.opt.timestep = float(timestep)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.iterations = int(iterations)
    model.opt.ls_iterations = int(ls_iterations)
    if hasattr(model.opt, "ccd_iterations"):
        model.opt.ccd_iterations = int(ccd_iterations)
    model.opt.tolerance = 1.0e-8
    model.opt.ls_tolerance = 1.0e-2
