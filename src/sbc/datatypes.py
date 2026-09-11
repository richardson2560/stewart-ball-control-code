# src/sbc/datatypes.py

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np


class SupervisorMode(Enum):
    """Execution state according to SBC v30 Hybrid Execution Contract."""
    NORMAL_ZERO_SLACK = auto()
    RELAXED_DEGRADED = auto()
    ONE_CYCLE_FALLBACK = auto()
    YAW_UNWIND = auto()
    FAILSAFE_HOLD = auto()
    MODEL_EXIT = auto()


class ActuatorMode(Enum):
    """CiA 402 drive profile operation mode."""
    CSP = "cyclic_synchronous_position"
    CSV = "cyclic_synchronous_velocity"


class PlatformStatus(Enum):
    """Physical or simulated communication interface state."""
    DISCONNECTED = auto()
    INITIALIZING = auto()
    OPERATIONAL = auto()
    FAULT = auto()
    SHUTDOWN = auto()


@dataclass(frozen=True, slots=True)
class RawSensorPacket:
    """
    Immutable raw data packet acquired directly from the HAL at each control step.
    Contains zero estimation or filtering logic.
    """
    timestamp: float                    # Monotonic clock time [s]
    leg_positions: np.ndarray           # Raw measured joint positions q_meas [m], shape (6,)
    leg_velocities: np.ndarray          # Raw measured joint velocities dot_q_meas [m/s], shape (6,)
    tactile_pos_raw: np.ndarray         # Raw surface contact coordinate / CoP in P [m], shape (2,)
    tactile_force_estimate: float       # Normal load reaction estimate N [N]
    tactile_active: bool                # Detection flag: True if load exceeds threshold N >= N_min
    bus_healthy: bool                   # Hardware communication integrity flag


@dataclass(frozen=True, slots=True)
class StateEstimatePacket:
    """
    State packet processed by perception and observation layers.
    Supplied to controllers, kinematics, and safety SOCP.
    """
    timestamp: float
    platform_pos: np.ndarray            # Origin translation T_p in inertial frame I [m], shape (3,)
    platform_rot: np.ndarray            # Rotation matrix R: P -> I, shape (3, 3)
    platform_twist: np.ndarray          # Spatial twist [v_p; Omega] expressed in P, shape (6,)
    platform_accel_src: np.ndarray      # Analytic platform angular acceleration alpha_src in P, shape (3,)
    ball_pos: np.ndarray                # Ball contact position rho in P (CoP compensated) [m], shape (2,)
    ball_vel: np.ndarray                # Ball relative velocity v_rel in P (Hermite-GP filtered) [m/s], shape (2,)
    yaw_rate: float                     # Conditioned platform body yaw rate Omega_z [rad/s]
    contact_valid: bool                 # Admissible rolling regime (N >= N_min and friction within cone)


@dataclass(frozen=True, slots=True)
class VirtualControlPacket:
    """
    Output of the decoupled ball tracking controller.
    Represents virtual planar command before physical allocation.
    """
    u_0: np.ndarray                     # Desired relative acceleration [m/s^2], shape (2,)
    B_des: np.ndarray                   # Desired platform tangential driving acceleration [m/s^2], shape (2,)
    supervisor_mode: SupervisorMode     # Active supervisor mode driving this command


@dataclass(frozen=True, slots=True)
class ActuatorCommandPacket:
    """
    Mode-qualified actuator output command synthesized for the drive bus.
    """
    timestamp: float
    mode: ActuatorMode
    q_send: np.ndarray                  # Leg position targets with lead compensation [m], shape (6,)
    dot_q_send: np.ndarray              # Leg velocity targets [m/s], shape (6,)
    u_q_applied: np.ndarray             # Accepted leg accelerations from safety SOCP [m/s^2], shape (6,)
    slack_value: float                  # Optimization slack (0.0 in strict mode, >0 in relaxed)


@dataclass(frozen=True, slots=True)
class TelemetryPacket:
    """
    Aggregated non-blocking payload for logging and GUI streaming.
    """
    timestamp: float
    raw_sensor: RawSensorPacket
    state: StateEstimatePacket
    control: VirtualControlPacket
    actuator: ActuatorCommandPacket
    cycle_duration: float               # Execution time of the control cycle [s]
    ref_pos: np.ndarray                 # Reference position rho_d [m], shape (2,)
    ref_vel: np.ndarray                 # Reference velocity dot_rho_d [m/s], shape (2,)
    ref_acc: np.ndarray                 # Reference acceleration ddot_rho_d [m/s^2], shape (2,)