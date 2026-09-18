# src/sbc/interfaces/base_backend.py

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from sbc.datatypes import RawSensorPacket, ActuatorCommandPacket, PlatformStatus


class BasePlatformBackend(ABC):
    """
    Formal Hardware Abstraction Layer (HAL) interface.
    Decouples real-time controller execution from the physical platform,
    multibody simulators (CoppeliaSim, MuJoCo), and native ODE engines.
    """

    def __init__(self, cycle_time: float = 0.002) -> None:
        self._dt: float = cycle_time
        self._status: PlatformStatus = PlatformStatus.DISCONNECTED

    @property
    def cycle_time(self) -> float:
        """Nominal control period Ts [s]. Default: 0.002 s (500 Hz)."""
        return self._dt

    @property
    def status(self) -> PlatformStatus:
        """Current operational status of the interface."""
        return self._status

    @property
    def tactile_position_kind(self) -> str:
        """Raw tactile coordinate semantics: pressure_center or geometric_projection.

        Defaults to a physical pressure-center sensor. Integrators must bypass
        CoP-bias subtraction for a geometric projection supplied by a simulator.
        """
        return "pressure_center"

    @property
    def ideal_kinematic_csp(self) -> bool:
        """Whether CSP writes realize ideal sampled position endpoints."""
        return False

    @property
    def kinematic_closure_tolerance(self) -> float:
        """Admissible actuator/pose closure tube for runtime diagnostics."""
        return 2e-3

    def read_platform_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return measured ``(position_I, rotation_P_to_I)`` when available.

        Hardware backends without a direct pose estimator may return ``None``;
        their integration layer must then supply a certified FK/observer.
        """
        return None

    def read_platform_motion(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return measured pose and 6D twist in the base frame if available."""
        return None

    def read_joint_limits(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return hard actuator coordinate limits when exposed by the backend."""
        return None

    @abstractmethod
    def connect(self) -> bool:
        """
        Establishes communication with the physical bus or simulation client.
        Must configure synchronous stepping mode if targeting a simulator.
        
        Returns:
            True if connection and handshaking succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def read_sensors(self) -> RawSensorPacket:
        """
        Polls leg encoders and plate tactile surface.
        Must be strictly non-blocking or bounded within the execution deadline.
        
        Returns:
            RawSensorPacket containing unfiltered, unlagged measurements.
        """
        pass

    @abstractmethod
    def write_actuators(self, command: ActuatorCommandPacket) -> bool:
        """
        Transmits position/velocity targets to the 6 Stewart platform legs.
        In simulation, updates joint target positions or steps kinematics.
        In hardware, transmits EtherCAT cyclic process data objects (PDOs).

        Args:
            command: Verified actuator packet from the actuation layer.

        Returns:
            True if transmission was acknowledged, False on communication fault.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Safely halts platform motion, locks brake systems, and closes communication.
        """
        pass
