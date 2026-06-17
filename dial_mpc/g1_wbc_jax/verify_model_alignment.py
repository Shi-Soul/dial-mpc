"""Compare the dial-mpc G1 WBC MuJoCo model against a SPIDER reference model."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mujoco
import numpy as np

from dial_mpc.g1_wbc_jax.model import build_wbc_mj_model


MODEL_FIELDS = (
    "jnt_type",
    "jnt_qposadr",
    "jnt_dofadr",
    "jnt_range",
    "dof_armature",
    "dof_damping",
    "dof_frictionloss",
    "actuator_trnid",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
    "actuator_ctrllimited",
    "actuator_forcelimited",
    "body_parentid",
    "body_pos",
    "body_quat",
    "body_mass",
    "geom_bodyid",
    "geom_type",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_friction",
    "geom_priority",
)


def run(reference_model: str | Path, *, model_path: str | Path | None = None) -> dict[str, float | bool]:
    ours = build_wbc_mj_model(model_path)
    reference = _load_reference_model(reference_model)
    stats: dict[str, float | bool] = {
        "ours_nq": float(ours.nq),
        "reference_nq": float(reference.nq),
        "ours_nv": float(ours.nv),
        "reference_nv": float(reference.nv),
        "ours_nu": float(ours.nu),
        "reference_nu": float(reference.nu),
        "ours_nbody": float(ours.nbody),
        "reference_nbody": float(reference.nbody),
        "ours_ngeom": float(ours.ngeom),
        "reference_ngeom": float(reference.ngeom),
        "ours_nsensor": float(ours.nsensor),
        "reference_nsensor": float(reference.nsensor),
    }
    for field in MODEL_FIELDS:
        stats.update(_compare_field(field, getattr(ours, field), getattr(reference, field)))
    for label, objtype, count in (
        ("body", mujoco.mjtObj.mjOBJ_BODY, ours.nbody),
        ("joint", mujoco.mjtObj.mjOBJ_JOINT, ours.njnt),
        ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, ours.nu),
        ("geom", mujoco.mjtObj.mjOBJ_GEOM, ours.ngeom),
    ):
        stats[f"{label}_names_equal"] = _names(ours, objtype, count) == _names(reference, objtype, count)
    stats["dynamics_layout_equal"] = all(
        bool(stats.get(f"{field}_shape_equal", False)) and float(stats.get(f"{field}_max_abs", 1.0)) == 0.0
        for field in MODEL_FIELDS
    )
    return stats


def _load_reference_model(path: str | Path) -> mujoco.MjModel:
    path = Path(path).expanduser().resolve()
    if path.suffix == ".pkl":
        with path.open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, mujoco.MjModel):
            raise TypeError(f"Expected pickled mujoco.MjModel in {path}, got {type(model)!r}.")
        return model
    return mujoco.MjModel.from_xml_path(str(path))


def _compare_field(name: str, actual, expected) -> dict[str, float | bool]:
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    shape_equal = actual_np.shape == expected_np.shape
    max_abs = float("nan")
    rmse = float("nan")
    if shape_equal:
        diff = actual_np.astype(np.float64) - expected_np.astype(np.float64)
        max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
        rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    return {
        f"{name}_shape_equal": shape_equal,
        f"{name}_max_abs": max_abs,
        f"{name}_rmse": rmse,
    }


def _names(model: mujoco.MjModel, objtype: mujoco.mjtObj, count: int) -> list[str | None]:
    return [mujoco.mj_id2name(model, objtype, i) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-model",
        default="spider-wbc-framework-integrated/spider/assets/robots/unitree_g1/tbfm_model.pkl",
    )
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    stats = run(args.reference_model, model_path=args.model_path)
    print(
        " ".join(
            f"{key}={value}" if isinstance(value, bool) else f"{key}={value:.6e}"
            for key, value in stats.items()
        )
    )


if __name__ == "__main__":
    main()
