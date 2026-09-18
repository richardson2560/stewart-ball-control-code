"""Immutable contracts crossing SBC subsystem boundaries."""

from dataclasses import dataclass
from enum import Enum
import numpy as np


def _readonly_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({length},).")
    array.flags.writeable = False
    return array


def _readonly_rotation(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError("platform_rot must be a finite 3x3 matrix.")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-7) or np.linalg.det(array) < 0.0:
        raise ValueError("platform_rot must belong to SO(3).")
    array.flags.writeable = False
    return array


class SupervisorMode(Enum):
    NORMAL_ZERO_SLACK = 0
    RELAXED_DEGRADED = 1
    ONE_CYCLE_FALLBACK = 2
    YAW_UNWIND = 3
    FAILSAFE_HOLD = 4
    MODEL_EXIT = 5


class ActuatorMode(Enum):
    CSP = "cyclic_synchronous_position"
    CSV = "cyclic_synchronous_velocity"


class PlatformStatus(Enum):
    DISCONNECTED = 0
    INITIALIZING = 1
    OPERATIONAL = 2
    FAULT = 3
    SHUTDOWN = 4


@dataclass(frozen=True)
class RawSensorPacket:
    timestamp: float
    leg_positions: np.ndarray
    leg_velocities: np.ndarray
    tactile_pos_raw: np.ndarray
    tactile_force_estimate: float
    tactile_active: bool
    bus_healthy: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "leg_positions", _readonly_vector(self.leg_positions, 6, "leg_positions"))
        object.__setattr__(self, "leg_velocities", _readonly_vector(self.leg_velocities, 6, "leg_velocities"))
        object.__setattr__(self, "tactile_pos_raw", _readonly_vector(self.tactile_pos_raw, 2, "tactile_pos_raw"))


@dataclass(frozen=True)
class StateEstimatePacket:
    timestamp: float
    platform_pos: np.ndarray
    platform_rot: np.ndarray
    platform_twist: np.ndarray
    platform_accel_src: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    yaw_rate: float
    contact_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform_pos", _readonly_vector(self.platform_pos, 3, "platform_pos"))
        object.__setattr__(self, "platform_rot", _readonly_rotation(self.platform_rot))
        object.__setattr__(self, "platform_twist", _readonly_vector(self.platform_twist, 6, "platform_twist"))
        object.__setattr__(self, "platform_accel_src", _readonly_vector(self.platform_accel_src, 3, "platform_accel_src"))
        object.__setattr__(self, "ball_pos", _readonly_vector(self.ball_pos, 2, "ball_pos"))
        object.__setattr__(self, "ball_vel", _readonly_vector(self.ball_vel, 2, "ball_vel"))


@dataclass(frozen=True)
class VirtualControlPacket:
    u_0: np.ndarray
    B_des: np.ndarray
    supervisor_mode: SupervisorMode

    def __post_init__(self) -> None:
        object.__setattr__(self, "u_0", _readonly_vector(self.u_0, 2, "u_0"))
        object.__setattr__(self, "B_des", _readonly_vector(self.B_des, 2, "B_des"))


@dataclass(frozen=True)
class ActuatorCommandPacket:
    timestamp: float
    mode: ActuatorMode
    q_send: np.ndarray
    dot_q_send: np.ndarray
    u_q_applied: np.ndarray
    slack_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_send", _readonly_vector(self.q_send, 6, "q_send"))
        object.__setattr__(self, "dot_q_send", _readonly_vector(self.dot_q_send, 6, "dot_q_send"))
        object.__setattr__(self, "u_q_applied", _readonly_vector(self.u_q_applied, 6, "u_q_applied"))

