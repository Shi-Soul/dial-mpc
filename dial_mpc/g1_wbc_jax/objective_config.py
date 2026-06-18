"""Objective defaults for G1 WBC MPC experiments."""

from __future__ import annotations

from pathlib import Path


DEFAULT_OBJECTIVE_WEIGHTS_JSON = Path(__file__).resolve().parent / "configs" / "v14_objective_weights.json"
PRIMARY_OBJECTIVE_METHODS = ("g1_wbc_joint_global", "g1_wbc_ee")
DIAGNOSTIC_OBJECTIVE_METHODS = ("g1_wbc_joint",)
MPC_OBJECTIVE_METHOD_CHOICES = PRIMARY_OBJECTIVE_METHODS + DIAGNOSTIC_OBJECTIVE_METHODS
EVALUATE_METHOD_CHOICES = ("no_mpc",) + MPC_OBJECTIVE_METHOD_CHOICES
