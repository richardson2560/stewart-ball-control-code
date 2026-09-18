# src/sbc/models/reference_model.py

import numpy as np
from typing import Tuple, Optional


class HurwitzReferenceModel:
    """
    Second-order Hurwitz acceleration reference generator matching SBC v30 Section 7.2.
    Produces smooth, dynamically feasible trajectories (rho_d, dot_rho_d, ddot_rho_d)
    from arbitrary external commands, projecting candidate accelerations onto a 
    physically realizable Euclidean acceleration ball to protect the friction cone.
    """

    def __init__(
        self,
        omega_n: float = 8.0,
        zeta: float = 1.0,
        max_acceleration: float = 1.50,
        cycle_time: float = 0.002
    ) -> None:
        """
        Args:
            omega_n: Natural frequency omega_n [rad/s].
            zeta: Damping ratio zeta (1.0 = critical damping, non-oscillatory).
            max_acceleration: Maximum feasible acceleration norm a_max [m/s^2].
            cycle_time: Sampling period Ts [s].
        """
        self._dt: float = cycle_time
        self._a_max: float = max_acceleration

        # Hurwitz gains: K_p,r = omega_n^2 * I_2, K_v,r = 2 * zeta * omega_n * I_2
        self._kp: float = omega_n ** 2
        self._kv: float = 2.0 * zeta * omega_n

        # Internal reference states in platform frame P
        self._rho_d = np.zeros(2, dtype=np.float64)
        self._dot_rho_d = np.zeros(2, dtype=np.float64)
        self._ddot_rho_d = np.zeros(2, dtype=np.float64)

        # Verification of discrete Schur stability gate (Eq 7.7)
        schur_value = (self._dt * omega_n) ** 2 + 4.0 * zeta * self._dt * omega_n
        assert schur_value < 4.0, f"Schur stability condition failed: {schur_value:.4f} >= 4.0"

    @property
    def position(self) -> np.ndarray:
        return self._rho_d.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._dot_rho_d.copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self._ddot_rho_d.copy()

    def reset(self, initial_pos: np.ndarray, initial_vel: Optional[np.ndarray] = None) -> None:
        """Resets the reference generator state to match initial boundary conditions."""
        initial_pos = np.asarray(initial_pos, dtype=np.float64)
        if initial_pos.shape != (2,) or not np.all(np.isfinite(initial_pos)):
            raise ValueError("initial_pos must be a finite vector with shape (2,).")
        if initial_vel is not None:
            initial_vel = np.asarray(initial_vel, dtype=np.float64)
            if initial_vel.shape != (2,) or not np.all(np.isfinite(initial_vel)):
                raise ValueError("initial_vel must be a finite vector with shape (2,).")

        # Boundary packets are intentionally read-only.  The reference model
        # owns mutable state, so it must never retain the packet's buffer.
        self._rho_d = np.array(initial_pos, dtype=np.float64, copy=True)
        self._dot_rho_d = (
            np.zeros(2, dtype=np.float64)
            if initial_vel is None
            else np.array(initial_vel, dtype=np.float64, copy=True)
        )
        self._ddot_rho_d = np.zeros(2, dtype=np.float64)

    def update(
        self,
        rho_cmd: np.ndarray,
        v_cmd_ff: Optional[np.ndarray] = None,
        a_cmd_ff: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advances the reference state by one discrete step Ts.

        Args:
            rho_cmd: Commanded planar target coordinate [m], shape (2,).
            v_cmd_ff: Optional feedforward velocity [m/s], shape (2,).
            a_cmd_ff: Optional feedforward acceleration [m/s^2], shape (2,).

        Returns:
            rho_d: Feasible target position [m], shape (2,)
            dot_rho_d: Feasible target velocity [m/s], shape (2,)
            ddot_rho_d: Feasible target acceleration [m/s^2], shape (2,)
        """
        v_ff = np.zeros(2, dtype=np.float64) if v_cmd_ff is None else v_cmd_ff
        a_ff = np.zeros(2, dtype=np.float64) if a_cmd_ff is None else a_cmd_ff

        # Candidate unconstrained acceleration (Eq 7.4)
        e_r = rho_cmd - self._rho_d
        dot_e_r = v_ff - self._dot_rho_d
        a_candidate = a_ff + self._kv * dot_e_r + self._kp * e_r

        # Radial Euclidean projection onto feasible acceleration ball (Eq 7.5)
        a_norm = float(np.linalg.norm(a_candidate))
        if a_norm > self._a_max:
            a_lim = (self._a_max / a_norm) * a_candidate
        else:
            a_lim = a_candidate.copy()

        self._ddot_rho_d = a_lim

        # Semi-implicit Euler integration (Eq 7.6)
        self._dot_rho_d += self._dt * self._ddot_rho_d
        self._rho_d += self._dt * self._dot_rho_d

        return self._rho_d.copy(), self._dot_rho_d.copy(), self._ddot_rho_d.copy()
