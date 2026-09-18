"""Integration and CSP lead compensation for admitted leg acceleration."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from sbc.kinematics.forward_6d import LocalForwardKinematics, ForwardKinematicsResult


@dataclass(frozen=True)
class IntegratedJointCommand:
    q_safe: np.ndarray
    q_dot_safe: np.ndarray
    q_send: np.ndarray
    closure: Optional[ForwardKinematicsResult]


class SafeJointTrajectoryIntegrator:
    def __init__(
        self,
        cycle_time: float,
        q_min: np.ndarray,
        q_max: np.ndarray,
        servo_time_constants: Optional[np.ndarray] = None,
        tracking_frequency: float = 18.0,
        tracking_damping: float = 1.0,
    ) -> None:
        self._dt = float(cycle_time)
        self._q_min = np.asarray(q_min, dtype=np.float64)
        self._q_max = np.asarray(q_max, dtype=np.float64)
        self._tau = np.zeros(6) if servo_time_constants is None else np.asarray(servo_time_constants, dtype=np.float64)
        self._tracking_kp = float(tracking_frequency) ** 2
        self._tracking_kd = 2.0 * float(tracking_damping) * float(tracking_frequency)
        if any(x.shape != (6,) for x in (self._q_min, self._q_max, self._tau)):
            raise ValueError("Joint bounds and servo constants must be six-vectors.")
        if tracking_frequency <= 0.0 or tracking_damping <= 0.0:
            raise ValueError("Joint tracking dynamics must be strictly positive.")
        # Semi-implicit Euler error transition for
        # e_ddot + kd*e_dot + kp*e = 0.  Reject gains whose sampled poles do
        # not lie strictly inside the unit disk.
        transition = np.array([
            [
                1.0 - self._dt**2 * self._tracking_kp,
                self._dt * (1.0 - self._dt * self._tracking_kd),
            ],
            [
                -self._dt * self._tracking_kp,
                1.0 - self._dt * self._tracking_kd,
            ],
        ])
        if float(np.max(np.abs(np.linalg.eigvals(transition)))) >= 1.0:
            raise ValueError(
                "Joint tracking gains are not Schur-stable at this cycle time."
            )

    def tracking_acceleration(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_des: np.ndarray,
        q_dot_des: np.ndarray,
        q_ddot_ff: np.ndarray,
    ) -> np.ndarray:
        """Close the actuator-coordinate pose loop before safety projection.

        The nominal error dynamics are
        ``e_ddot + kd e_dot + kp e = 0``.  The SOCP remains the final authority
        and may modify this acceleration to preserve hard constraints.
        """
        vectors = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (q, q_dot, q_des, q_dot_des, q_ddot_ff)
        )
        if not all(value.shape == (6,) and np.all(np.isfinite(value)) for value in vectors):
            raise ValueError("Joint tracking inputs must be finite six-vectors.")
        q, q_dot, q_des, q_dot_des, q_ddot_ff = vectors
        return (
            q_ddot_ff
            + self._tracking_kd * (q_dot_des - q_dot)
            + self._tracking_kp * (q_des - q)
        )

    def integrate(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        admitted_acceleration: np.ndarray,
        forward_solver: Optional[LocalForwardKinematics] = None,
        seed_translation: Optional[np.ndarray] = None,
        seed_rotation: Optional[np.ndarray] = None,
    ) -> IntegratedJointCommand:
        q = np.asarray(q, dtype=np.float64)
        q_dot = np.asarray(q_dot, dtype=np.float64)
        acceleration = np.asarray(admitted_acceleration, dtype=np.float64)
        if not all(np.all(np.isfinite(x)) and x.shape == (6,) for x in (q, q_dot, acceleration)):
            raise ValueError("Joint integration inputs must be finite six-vectors.")
        q_dot_safe = q_dot + self._dt * acceleration
        q_safe = q + self._dt * q_dot + 0.5 * self._dt**2 * acceleration
        violation = np.maximum(self._q_min - q_safe, q_safe - self._q_max)
        if np.max(violation) > 1e-10:
            index = int(np.argmax(violation))
            raise RuntimeError(
                "Admitted acceleration crosses a hard stroke limit: "
                f"leg={index + 1}, q_next={q_safe[index]:+.9f}, "
                f"interval=[{self._q_min[index]:+.9f}, "
                f"{self._q_max[index]:+.9f}], "
                f"excess={violation[index]:.3e}."
            )
        # Remove roundoff only; a material violation is never clipped.
        q_safe = np.minimum(np.maximum(q_safe, self._q_min), self._q_max)

        closure = None
        if forward_solver is not None:
            if seed_translation is None or seed_rotation is None:
                raise ValueError("A pose seed is required for forward-closure validation.")
            closure = forward_solver.solve(q_safe, seed_translation, seed_rotation)
            if not closure.valid:
                raise RuntimeError("Integrated q_safe left the certified FK branch or closure tube.")
        q_send = q_safe + self._tau * q_dot_safe
        return IntegratedJointCommand(q_safe, q_dot_safe, q_send, closure)
