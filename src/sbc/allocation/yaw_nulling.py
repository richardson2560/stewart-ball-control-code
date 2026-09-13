# src/sbc/allocation/yaw_nulling.py

import numpy as np
from typing import Tuple


class YawNullingAllocator:
    """
    Active body-yaw suppression and Berry phase / geometric holonomy monitor.
    Enforces Omega_z = 0 via analytical coordinate rate: dot_psi_0 = (tan(phi) / cos(theta)) * dot_theta.
    Monitors unwrapped yaw accumulation psi(t) and triggers YAW_UNWIND when approaching limits.
    """

    def __init__(
        self,
        yaw_limit: float = np.radians(15.0),
        unwind_recovery_rate: float = np.radians(10.0),
        unwind_exit_margin: float = np.radians(3.0)
    ) -> None:
        """
        Args:
            yaw_limit: Maximum permissible continuous yaw displacement before unwind [rad].
            unwind_recovery_rate: Restoring yaw velocity target Omega_z,u during unwind [rad/s].
            unwind_exit_margin: Deadband threshold to return to strict Omega_z = 0 mode [rad].
        """
        self._psi_max: float = yaw_limit
        self._omega_u_max: float = unwind_recovery_rate
        self._psi_exit: float = unwind_exit_margin

        self._psi_accumulated: float = 0.0
        self._is_unwinding: bool = False

    @property
    def accumulated_yaw(self) -> float:
        """Continuous unwrapped Euler yaw coordinate psi [rad]."""
        return self._psi_accumulated

    @property
    def is_unwinding(self) -> bool:
        """True if supervisor is actively relaxing Omega_z=0 to unwind geometric phase."""
        return self._is_unwinding

    def reset(self, initial_psi: float = 0.0) -> None:
        """Resets accumulated yaw angle to reference origin."""
        self._psi_accumulated = initial_psi
        self._is_unwinding = False

    def compute_yaw_velocity(
        self,
        phi: float,
        theta: float,
        dot_theta: float,
        dt: float
    ) -> Tuple[float, bool]:
        """
        Calculates the required Euler yaw rate dot_psi and updates holonomy state.

        Args:
            phi: Current desired roll angle [rad].
            theta: Current desired pitch angle [rad].
            dot_theta: Derivative of pitch angle [rad/s].
            dt: Control sampling period [s].

        Returns:
            dot_psi_cmd: Synthesized Euler yaw rate command [rad/s].
            unwind_active: True if system is in unwind contingency.
        """
        cos_theta = np.cos(theta)
        cos_phi = np.cos(phi)

        # Protect against Euler representation singularity (pitch -> 90 deg)
        denom = max(1e-4, cos_phi * cos_theta)

        # Holonomy state machine check
        abs_psi = abs(self._psi_accumulated)
        if not self._is_unwinding:
            if abs_psi >= self._psi_max:
                self._is_unwinding = True
        else:
            if abs_psi <= self._psi_exit:
                self._is_unwinding = False

        if not self._is_unwinding:
            # Nominal mode: Force Omega_z = 0 identically (Eq 9.20)
            dot_psi_cmd = (np.tan(phi) / max(1e-4, cos_theta)) * dot_theta
        else:
            # Contingency unwind mode: Relax Omega_z = 0 to drive psi back to zero
            omega_z_des = -np.sign(self._psi_accumulated) * self._omega_u_max
            # Solve for dot_psi in: Omega_z = -sin(phi)*dot_theta + cos(phi)*cos(theta)*dot_psi
            dot_psi_cmd = (omega_z_des + np.sin(phi) * dot_theta) / denom

        # Integrate continuous unwrapped yaw
        self._psi_accumulated += dot_psi_cmd * dt

        return dot_psi_cmd, self._is_unwinding