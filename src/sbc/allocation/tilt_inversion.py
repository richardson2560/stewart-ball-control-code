# src/sbc/allocation/tilt_inversion.py

import numpy as np
from typing import Tuple, Optional


class TiltInverter:
    """
    Exact geometric gravity tilt inversion matching SBC v30 Prop 9.2 and Prop 9.3.
    Maps tangential driving target B_des to continuous roll (phi_d) and pitch (theta_d)
    angles with strict analytical domain guards and one-step predictive dynamic compensation.
    """

    def __init__(
        self,
        gravity: float = 9.81,
        max_roll: float = np.radians(15.0),
        max_pitch: float = np.radians(15.0),
        ball_radius: float = 0.025
    ) -> None:
        """
        Args:
            gravity: Acceleration due to gravity g [m/s^2].
            max_roll: Safe angular limit for roll phi_max [rad].
            max_pitch: Safe angular limit for pitch theta_max [rad].
            ball_radius: Sphere radius r [m].
        """
        self._g: float = gravity
        self._phi_max: float = max_roll
        self._theta_max: float = max_pitch
        self._r: float = ball_radius

        # Cached previous state for one-step predictive dynamic compensation d_B (Prop 9.2)
        self._prev_A_p = np.zeros(2, dtype=np.float64)
        self._prev_alpha = np.zeros(3, dtype=np.float64)
        self._prev_omega = np.zeros(3, dtype=np.float64)
        self._prev_rho = np.zeros(2, dtype=np.float64)
        self._has_history: bool = False

    def reset(self) -> None:
        """Clears predictive acceleration history."""
        self._prev_A_p.fill(0.0)
        self._prev_alpha.fill(0.0)
        self._prev_omega.fill(0.0)
        self._prev_rho.fill(0.0)
        self._has_history = False

    def compute_predictive_compensation(
        self,
        A_p_xy: np.ndarray,
        alpha: np.ndarray,
        omega: np.ndarray,
        rho: np.ndarray
    ) -> np.ndarray:
        """
        Evaluates the dynamic platform acceleration terms d_B at t_{k-1} (Eq 9.16).
        d_B = - A_p,parallel - (alpha x r_bp)_parallel - (Omega x (Omega x r_bp))_parallel
        """
        # Sphere center offset in platform frame P: r_bp = [x, y, r]^T
        r_bp = np.array([rho[0], rho[1], self._r], dtype=np.float64)

        # Euler acceleration: alpha x r_bp
        euler_term = np.cross(alpha, r_bp)[:2]

        # Centripetal acceleration: Omega x (Omega x r_bp)
        centripetal_term = np.cross(omega, np.cross(omega, r_bp))[:2]

        d_B = - A_p_xy - euler_term - centripetal_term
        return d_B

    def invert(
        self,
        B_des: np.ndarray,
        A_p_xy: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None
    ) -> Tuple[float, float, bool]:
        """
        Computes desired platform tilt angles (phi_d, theta_d) from virtual command B_des.

        Args:
            B_des: Tangential acceleration command in P [m/s^2], shape (2,).
            A_p_xy: Current platform linear acceleration [m/s^2], shape (2,).
            alpha: Current platform angular acceleration [rad/s^2], shape (3,).
            omega: Current platform angular velocity [rad/s], shape (3,).
            rho: Current ball contact position in P [m], shape (2,).

        Returns:
            phi_d: Target roll angle [rad].
            theta_d: Target pitch angle [rad].
            is_valid: False if command exceeded boundary and required safe saturation.
        """
        # Apply one-step predictive compensation using history if available
        if self._has_history:
            d_B_hat = self.compute_predictive_compensation(
                self._prev_A_p, self._prev_alpha, self._prev_omega, self._prev_rho
            )
            b_g = B_des - d_B_hat
        else:
            b_g = B_des.copy()

        # Update cache for the next cycle
        if A_p_xy is not None and alpha is not None and omega is not None and rho is not None:
            self._prev_A_p = A_p_xy.copy()
            self._prev_alpha = alpha.copy()
            self._prev_omega = omega.copy()
            self._prev_rho = rho.copy()
            self._has_history = True

        # Numerical domain validation (Eq 9.14)
        is_valid = True
        limit_bx = self._g * np.sin(self._theta_max)
        
        # Step 1: Invert Pitch theta_d = arcsin(b_g,x / g)
        b_gx = float(b_g[0])
        if abs(b_gx) >= limit_bx:
            b_gx = np.sign(b_gx) * (limit_bx - 1e-6)
            is_valid = False

        sin_theta = np.clip(b_gx / self._g, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_d = float(np.arcsin(sin_theta))
        cos_theta = np.sqrt(max(1e-7, 1.0 - sin_theta**2))

        # Step 2: Invert Roll phi_d = - arcsin(b_g,y / (g * cos(theta_d)))
        limit_by = self._g * cos_theta * np.sin(self._phi_max)
        b_gy = float(b_g[1])
        if abs(b_gy) >= limit_by:
            b_gy = np.sign(b_gy) * (limit_by - 1e-6)
            is_valid = False

        sin_phi = np.clip(-b_gy / (self._g * cos_theta), -1.0 + 1e-7, 1.0 - 1e-7)
        phi_d = float(np.arcsin(sin_phi))

        return phi_d, theta_d, is_valid