# src/sbc/allocation/redundancy.py

import numpy as np
from typing import Tuple, Optional
from sbc.kinematics.platform import StewartKinematics


class RedundancyResolver:
    """
    Kinematic redundancy resolution matching SBC v30 Section 9.1.3 and Section 9.1.4.
    Enforces the 3-DOF augmented orientation task [dot_phi_d, dot_theta_d, dot_psi_0]^T
    via weighted pseudoinverse A_K_mu_dagger and projects the mid-stroke leg recentering
    gradient nabla V_center strictly into the null space ker(A).
    """

    def __init__(
        self,
        leg_stiffness: Optional[np.ndarray] = None,
        q_min: Optional[np.ndarray] = None,
        q_max: Optional[np.ndarray] = None,
        k_center: float = 1.5,
        damping_mu_0: float = 1e-4,
        singularity_threshold: float = 1e-3,
        **kwargs
    ) -> None:
        """
        Args:
            leg_stiffness: Diagonal weighting matrix K [N/m], shape (6,). Default: unity.
            q_min: Minimum mechanical actuator strokes [m], shape (6,).
            q_max: Maximum mechanical actuator strokes [m], shape (6,).
            k_center: Null-space recentering gradient gain matching Eq (9.38).
            damping_mu_0: Base Tikhonov regularization damping factor.
            singularity_threshold: Singular value threshold sigma_sing to activate damping.
        """
        self._K = np.ones(6, dtype=np.float64) if leg_stiffness is None else np.ascontiguousarray(leg_stiffness, dtype=np.float64)
        self._K_inv = 1.0 / self._K

        self._q_min = np.full(6, -0.05, dtype=np.float64) if q_min is None else np.ascontiguousarray(q_min, dtype=np.float64)
        self._q_max = np.full(6, 0.05, dtype=np.float64) if q_max is None else np.ascontiguousarray(q_max, dtype=np.float64)
        self._q_mid = 0.5 * (self._q_min + self._q_max)
        self._stroke_span_sq = (self._q_max - self._q_min) ** 2

        # Support both k_center (canonical) and k_leg_center (legacy)
        self._k_center: float = kwargs.get("k_leg_center", k_center)
        self._mu_0: float = damping_mu_0
        self._sigma_sing: float = singularity_threshold

        self._prev_dot_q_cmd = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        """Resets differentiator cache."""
        self._prev_dot_q_cmd.fill(0.0)

    @staticmethod
    def _compute_euler_kinematic_matrix(phi: float, theta: float) -> np.ndarray:
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)

        E = np.array([
            [1.0, 0.0, -s_t],
            [0.0, c_p, s_p * c_t],
            [0.0, -s_p, c_p * c_t]
        ], dtype=np.float64)
        return E

    def resolve(
        self,
        kinematics: StewartKinematics,
        phi_d: float,
        theta_d: float,
        psi_d: float,
        dot_phi_d: float,
        dot_theta_d: float,
        dot_psi_0: float,
        q_meas: np.ndarray,
        dt: float,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synthesizes leg positions, velocities, and accelerations matching Eq (9.35 - 9.42).

        Args:
            kinematics: StewartKinematics instance for scene geometry.
            phi_d, theta_d, psi_d: Desired orientation angles [rad].
            dot_phi_d, dot_theta_d, dot_psi_0: Desired angular rates [rad/s].
            q_meas: Current measured leg positions [m], shape (6,).
            dt: Control period Ts [s].

        Returns:
            q_cmd: Safe desired leg lengths [m], shape (6,)
            dot_q_cmd: Desired leg velocities [m/s], shape (6,)
            ddot_q_cmd: Desired leg accelerations [m/s^2], shape (6,)
        """
        # 1. Platform-centered geometric inverse kinematics for position target
        q_cmd_geom, _, _ = kinematics.inverse_kinematics(phi_d, theta_d, psi_d, kinematics.T_p_nominal)

        # 2. Build augmented task matrix A in R^(3x6) (Eq 9.21 - 9.22)
        R = kinematics.get_rotation_matrix(phi_d, theta_d, psi_d)
        E = self._compute_euler_kinematic_matrix(phi_d, theta_d)
        RE = R @ E

        T_twist = np.zeros((6, 6), dtype=np.float64)
        T_twist[:3, :3] = np.eye(3)
        T_twist[3:, 3:] = RE

        J_inv = kinematics.compute_inverse_jacobian(phi_d, theta_d, psi_d, kinematics.T_p_nominal)
        G = J_inv @ T_twist
        J_pose = np.linalg.inv(G)

        # A corresponds to the rows of Euler angle rates: [dot_phi, dot_theta, dot_psi]
        A = J_pose[3:6, :]  # shape (3, 6)
        x_dot_0 = np.array([dot_phi_d, dot_theta_d, dot_psi_0], dtype=np.float64)

        # 3. Regularized weighted pseudoinverse A_dagger (Eq 9.35)
        S_A = (A * self._K_inv[np.newaxis, :]) @ A.T  # shape (3, 3)
        sigma_min = float(np.sqrt(max(1e-12, np.linalg.eigvalsh(S_A)[0])))
        damping_ratio = np.clip(sigma_min / self._sigma_sing, 0.0, 1.0)
        mu_damp = float(self._mu_0 * (1.0 - damping_ratio) ** 2)

        S_A_reg = S_A + mu_damp * np.eye(3)
        S_A_inv = np.linalg.inv(S_A_reg)
        A_dagger = (self._K_inv[:, np.newaxis] * A.T) @ S_A_inv  # shape (6, 3)

        # 4. Mid-stroke recentering potential gradient (Design Rule 9.2, Eq 9.38)
        grad_V = (q_meas - self._q_mid) / self._stroke_span_sq
        delta_dot_q_null = - self._k_center * (self._K_inv * grad_V)

        # 5. Project strictly into null space: P_0 * delta_q_null (Eq 9.39)
        dot_q_task = A_dagger @ x_dot_0
        dot_q_null = delta_dot_q_null - A_dagger @ (A @ delta_dot_q_null)
        dot_q_cmd = dot_q_task + dot_q_null

        # 6. Combined position command with null-space stroke adjustment
        q_cmd = q_cmd_geom + dot_q_null * dt

        # Analytical acceleration
        ddot_q_cmd = (dot_q_cmd - self._prev_dot_q_cmd) / dt
        self._prev_dot_q_cmd = dot_q_cmd.copy()

        return q_cmd, dot_q_cmd, ddot_q_cmd

    # Backwards-compatible alias for resolve_with_authority
    resolve_with_authority = resolve