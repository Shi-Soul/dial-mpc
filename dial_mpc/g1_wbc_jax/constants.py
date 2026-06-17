"""G1 WBC constants used by the all-JAX migration.

The ordering mirrors the existing SPIDER G1 WBC task, but this module is
self-contained so JAX rollout code does not import the Torch implementation.
"""

from __future__ import annotations

from pathlib import Path


MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

ISAACLAB_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MUJOCO_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

ISAACLAB_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

ISAACLAB_TO_MUJOCO_JOINT_REINDEX: tuple[int, ...] = tuple(
    ISAACLAB_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES
)
ISAACLAB_TO_MUJOCO_BODY_REINDEX: tuple[int, ...] = tuple(
    ISAACLAB_BODY_NAMES.index(name) for name in MUJOCO_BODY_NAMES
)

COMMAND_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

ANCHOR_BODY_NAME = "pelvis"
TRACKING_ANCHOR_BODY_NAME = "torso_link"
LIMB_EE_BODY_NAMES: tuple[str, ...] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
HAND_EE_BODY_NAMES: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
TASK_EE_BODY_NAMES: tuple[str, ...] = (*LIMB_EE_BODY_NAMES, "torso_link")
LEFT_FOOT_BODY_NAME = "left_ankle_roll_link"
RIGHT_FOOT_BODY_NAME = "right_ankle_roll_link"

PHYSICS_DT = 0.005
DECIMATION = 4
POLICY_DT = PHYSICS_DT * DECIMATION
ACTION_DIM = 29
OBS_DIM = 886
OBS_HISTORY_LENGTH = 5
ROOT_QPOS_DIM = 7
ROOT_QVEL_DIM = 6
QPOS_DIM = ROOT_QPOS_DIM + ACTION_DIM
QVEL_DIM = ROOT_QVEL_DIM + ACTION_DIM
COMMAND_DELTA_DIM = QPOS_DIM - 1

COMMAND_BODY_INDICES: tuple[int, ...] = tuple(
    MUJOCO_BODY_NAMES.index(name) for name in COMMAND_BODY_NAMES
)
LIMB_EE_INDICES_IN_COMMAND: tuple[int, ...] = tuple(
    COMMAND_BODY_NAMES.index(name) for name in LIMB_EE_BODY_NAMES
)
ANCHOR_INDEX_IN_COMMAND = COMMAND_BODY_NAMES.index(ANCHOR_BODY_NAME)
TRACKING_ANCHOR_INDEX_IN_COMMAND = COMMAND_BODY_NAMES.index(TRACKING_ANCHOR_BODY_NAME)


def default_g1_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "spider"
        / "spider"
        / "assets"
        / "robots"
        / "unitree_g1"
        / "wxy_g1.xml"
    )


def default_attached_g1_model_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "tbfm_g1_attached_nosensors.xml"


def default_checkpoint_dirs() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[3] / "wxy"
    return {
        "bc": root / "0608_ckpt_bc",
        "bcrl": root / "0608_ckpt_bcrl",
    }


KNEES_BENT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def _reflected_inertia(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    return (
        rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
        + rotor_inertia[1] * gear_ratio[2] ** 2
        + rotor_inertia[2]
    )


_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_DAMPING_RATIO = 2.0
_ARMATURE_5020 = _reflected_inertia(
    (0.139e-4, 0.017e-4, 0.169e-4),
    (1.0, 1.0 + 46.0 / 18.0, 1.0 + 56.0 / 16.0),
)
_ARMATURE_7520_14 = _reflected_inertia(
    (0.489e-4, 0.098e-4, 0.533e-4),
    (1.0, 4.5, 1.0 + 48.0 / 22.0),
)
_ARMATURE_7520_22 = _reflected_inertia(
    (0.489e-4, 0.109e-4, 0.738e-4),
    (1.0, 4.5, 5.0),
)
_ARMATURE_4010 = _reflected_inertia((0.068e-4, 0.0, 0.0), (1.0, 5.0, 5.0))


def _kp(armature: float) -> float:
    return armature * _NATURAL_FREQ**2


def _kd(armature: float) -> float:
    return 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ


ACTUATOR_GROUPS: tuple[tuple[tuple[str, ...], float, float, float, float], ...] = (
    (
        (
            ".*_elbow_joint",
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_wrist_roll_joint",
        ),
        _kp(_ARMATURE_5020),
        _kd(_ARMATURE_5020),
        25.0,
        _ARMATURE_5020,
    ),
    (
        (".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
        _kp(_ARMATURE_7520_14),
        _kd(_ARMATURE_7520_14),
        88.0,
        _ARMATURE_7520_14,
    ),
    (
        (".*_hip_roll_joint", ".*_knee_joint"),
        _kp(_ARMATURE_7520_22),
        _kd(_ARMATURE_7520_22),
        139.0,
        _ARMATURE_7520_22,
    ),
    (
        (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
        _kp(_ARMATURE_4010),
        _kd(_ARMATURE_4010),
        5.0,
        _ARMATURE_4010,
    ),
    (
        ("waist_pitch_joint", "waist_roll_joint"),
        _kp(_ARMATURE_5020) * 2.0,
        _kd(_ARMATURE_5020) * 2.0,
        50.0,
        _ARMATURE_5020 * 2.0,
    ),
    (
        (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        _kp(_ARMATURE_5020) * 2.0,
        _kd(_ARMATURE_5020) * 2.0,
        50.0,
        _ARMATURE_5020 * 2.0,
    ),
)
