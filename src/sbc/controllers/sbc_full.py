# src/sbc/controllers/sbc_full.py

import numpy as np
from typing import Optional

from sbc.controllers.base import BaseController
from sbc.datatypes import StateEstimatePacket, VirtualControlPacket, SupervisorMode


class SBCFullController(BaseController):
    """
    Direct platform-frame fixed-parameter robust tracking controller matching SBC v30 Theorem 7.1.
    Evaluates tracking feedback and nominal dynamic compensation directly in platform frame P.
    Guarantees Input-to-State Stability (ISS) against ball inertia ratio mismatch,
    unmeasured normal spin, rolling resistance, and sensor differentiation residuals.
    """

    def __init__(
        self,
        kp: float = 6.5,
        kd: float = 2.8,
        ball_mass: float = 0.065449846949792,
        ball_radius: float = 0.025,
        inertia_ratio_lambda0: float = 5.0 / 7.0,
        rolling_resistance_coeff: float = 0.0015,
        gravity: float = 9.81
    ) -> None:
        """
        Args:
            kp: Proportional tracking gain [s^-2].
            kd: Derivative tracking gain [s^-1].
            ball_mass: Sphere mass m [kg].
            ball_radius: Sphere radius r [m].
            inertia_ratio_lambda0: Offline calibrated structural inertia ratio lambda_0 = 1 / (1 + kappa_I).
            rolling_resistance_coeff: Calibrated rolling resistance coefficient c_rr.
            gravity: Acceleration due to gravity g [m/s^2].
        """
        self._Kp = np.diag([kp, kp]).astype(np.float64)
        self._Kd = np.diag([kd, kd]).astype(np.float64)
        self._m: float = ball_mass
        self._r: float = ball_radius
        self._lambda_0: float = inertia_ratio_lambda0
        self._c_rr: float = rolling_resistance_coeff
        self._g: float = gravity

        # Planar rotation generator J in so(2)
        self._J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
        self._eps_v: float = 1e-3

    def reset(self) -> None:
        pass

    def compute_control(
        self,
        state: StateEstimatePacket,
        rho_d: np.ndarray,
        dot_rho_d: np.ndarray,
        ddot_rho_d: np.ndarray
    ) -> VirtualControlPacket:
        """
        Computes the desired tangential platform drive acceleration B_des (Eq 7.9).

        Args:
            state: Filtered perception state packet.
            rho_d: Feasible reference position in P [m], shape (2,).
            dot_rho_d: Feasible reference velocity in P [m/s], shape (2,).
            ddot_rho_d: Feasible reference acceleration in P [m/s^2], shape (2,).

        Returns:
            VirtualControlPacket containing virtual acceleration u_0 and physical drive target B_des.
        """
        # Tracking error coordinates in platform frame P
        e_rho = state.ball_pos - rho_d
        e_v = state.ball_vel - dot_rho_d

        # 1. Virtual ball acceleration command u_0 (Eq 7.7)
        u_0 = ddot_rho_d - self._Kp @ e_rho - self._Kd @ e_v

        # 2. Implemented known acceleration terms a_k (Eq 7.8)
        # alpha_src is the analytic platform angular acceleration from previous command
        # c_alpha = (alpha x n)_parallel = [alpha_y, -alpha_x]^T
        c_alpha = np.array([state.platform_accel_src[1], -state.platform_accel_src[0]], dtype=np.float64)
        euler_accel = - self._r * c_alpha

        # Coriolis coupling: Omega_z * J * v_rel
        coriolis_accel = - state.yaw_rate * (self._J @ state.ball_vel)

        a_k = euler_accel + coriolis_accel

        # 3. Work-equivalent rolling resistance force estimate f_rr (Eq 2.2)
        v_norm = float(np.linalg.norm(state.ball_vel))
        v_unit = state.ball_vel / np.sqrt(v_norm**2 + self._eps_v**2)
        # Normal force approximation under nominal tilt
        N_est = self._m * self._g
        f_rr_est = - self._c_rr * N_est * v_unit

        # 4. Synthesize desired platform tangential drive B_des (Eq 7.9)
        # B_des = (u_0 - a_k) / lambda_0 + Omega_z * J * v_rel - (1/m) * f_rr
        B_des = (u_0 - a_k) / self._lambda_0 - coriolis_accel - (1.0 / self._m) * f_rr_est

        return VirtualControlPacket(
            u_0=u_0,
            B_des=B_des,
            supervisor_mode=SupervisorMode.NORMAL_ZERO_SLACK
        )