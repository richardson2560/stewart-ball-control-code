# src/sbc/kinematics/platform.py

import numpy as np
from typing import Tuple, Optional


class StewartKinematics:
    """
    Kinematic model of the 6-DOF Stewart-Gough parallel platform.
    Computes analytical inverse kinematics, orientation rotation matrices (ZYX),
    and the Plucker-coordinate inverse velocity Jacobian J_inv matching SBC v30 Eq (2.10).
    """

    def __init__(
        self,
        base_anchors: np.ndarray,
        platform_anchors: np.ndarray,
        leg_offsets: np.ndarray,
        nominal_height: float = 0.40,
        initial_translation: Optional[np.ndarray] = None
    ) -> None:
        """
        Args:
            base_anchors: Coordinates b_i of base joints in inertial frame I, shape (6, 3).
            platform_anchors: Coordinates a_i of platform joints in platform frame P, shape (6, 3).
            leg_offsets: Length offset l_offset_i between joint center distance and actuator position, shape (6,).
            nominal_height: Default platform height z_p [m].
            initial_translation: Initial origin translation vector T_p in I, shape (3,).
        """
        assert base_anchors.shape == (6, 3), "base_anchors must have shape (6, 3)"
        assert platform_anchors.shape == (6, 3), "platform_anchors must have shape (6, 3)"
        assert leg_offsets.shape == (6,), "leg_offsets must have shape (6,)"

        self.b = np.array(base_anchors, dtype=np.float64, copy=True, order="C")
        self.a = np.array(platform_anchors, dtype=np.float64, copy=True, order="C")
        self.offsets = np.array(leg_offsets, dtype=np.float64, copy=True, order="C")
        
        if initial_translation is not None:
            self.T_p_nominal = np.array(initial_translation, dtype=np.float64, copy=True, order="C")
        else:
            self.T_p_nominal = np.array([0.0, 0.0, nominal_height], dtype=np.float64)

    @staticmethod
    def get_rotation_matrix(phi: float, theta: float, psi: float = 0.0) -> np.ndarray:
        """
        Computes platform rotation matrix R mapping platform frame P to inertial frame I.
        Standard ZYX Tait-Bryan Euler angle convention: R = R_z(psi) * R_y(theta) * R_x(phi).
        """
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_y, s_y = np.cos(psi), np.sin(psi)

        R = np.array([
            [c_y * c_t, c_y * s_t * s_p - s_y * c_p, c_y * s_t * c_p + s_y * s_p],
            [s_y * c_t, s_y * s_t * s_p + c_y * c_p, s_y * s_t * c_p - c_y * s_p],
            [-s_t,      c_t * s_p,                   c_t * c_p]
        ], dtype=np.float64)
        return R

    @staticmethod
    def rotation_matrix_to_zyx(rotation: np.ndarray) -> Tuple[float, float, float]:
        """Extract nonsingular ZYX angles from a measured rotation matrix."""
        R = np.asarray(rotation, dtype=np.float64)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            raise ValueError("rotation must be a finite 3x3 matrix.")
        theta = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        cos_theta = float(np.cos(theta))
        if abs(cos_theta) < 1e-8:
            raise ValueError("ZYX attitude is too close to gimbal lock.")
        phi = float(np.arctan2(R[2, 1], R[2, 2]))
        psi = float(np.arctan2(R[1, 0], R[0, 0]))
        return phi, theta, psi

    def inverse_kinematics(
        self,
        phi: float,
        theta: float,
        psi: float = 0.0,
        T_p: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates leg actuator positions and unit leg directions from desired pose.

        Args:
            phi: Desired roll angle [rad].
            theta: Desired pitch angle [rad].
            psi: Desired yaw angle [rad].
            T_p: Desired platform translation origin in I [m], shape (3,).

        Returns:
            q: Motor positions q_i = ||l_i|| - offset_i [m], shape (6,)
            L: Actual geometric lengths ||l_i|| [m], shape (6,)
            s: Unit direction vectors s_i along leg axes in I, shape (6, 3)
        """
        if T_p is None:
            T_p = self.T_p_nominal

        R = self.get_rotation_matrix(phi, theta, psi)
        
        # Rotated platform anchor points in inertial frame: R * a_i
        Ra = self.a @ R.T  # shape (6, 3)
        
        # Leg vectors: l_i = T_p + R * a_i - b_i
        l_vecs = T_p + Ra - self.b  # shape (6, 3)
        
        # Euclidean leg lengths
        L = np.linalg.norm(l_vecs, axis=1)  # shape (6,)
        
        # Unit direction vectors s_i
        s = l_vecs / L[:, np.newaxis]  # shape (6, 3)
        
        # Actuator motor coordinates q
        q = L - self.offsets
        
        return q, L, s

    def compute_inverse_jacobian(
        self,
        phi: float,
        theta: float,
        psi: float = 0.0,
        T_p: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Assembles the 6x6 inverse velocity Jacobian matrix J_inv in inertial frame coordinates.
        Satisfies: dot_q = J_inv * xi_Op, where xi_Op = [v_Op; omega_p] in I.
        Each row is the Plucker actuation screw: S_i = [s_i^T, (R * a_i x s_i)^T].
        """
        if T_p is None:
            T_p = self.T_p_nominal

        R = self.get_rotation_matrix(phi, theta, psi)
        Ra = self.a @ R.T  # shape (6, 3)
        l_vecs = T_p + Ra - self.b
        L = np.linalg.norm(l_vecs, axis=1)
        s = l_vecs / L[:, np.newaxis]

        # Moment component of actuation screw: h_i = R * a_i x s_i
        h = np.cross(Ra, s)  # shape (6, 3)

        # Assemble J_inv: rows are [s_i, h_i]
        J_inv = np.hstack([s, h])  # shape (6, 6)
        return J_inv

    def compute_inverse_jacobian_dot(
        self,
        phi: float,
        theta: float,
        psi: float,
        T_p: np.ndarray,
        linear_velocity_I: np.ndarray,
        angular_velocity_I: np.ndarray
    ) -> np.ndarray:
        """Analytical time derivative of ``J_inv`` along a platform motion.

        All vector arguments are expressed in the inertial frame.  This avoids
        the high-speed error produced by treating ``J_inv`` as constant in
        ``q_ddot = J_inv xi_dot + J_inv_dot xi``.
        """
        T_p = np.ascontiguousarray(T_p, dtype=np.float64)
        v_I = np.ascontiguousarray(linear_velocity_I, dtype=np.float64)
        omega_I = np.ascontiguousarray(angular_velocity_I, dtype=np.float64)
        R = self.get_rotation_matrix(phi, theta, psi)
        Ra = self.a @ R.T
        l_vecs = T_p + Ra - self.b
        lengths = np.linalg.norm(l_vecs, axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("Degenerate Stewart leg length while evaluating J_inv_dot.")

        s = l_vecs / lengths[:, np.newaxis]
        Ra_dot = np.cross(np.broadcast_to(omega_I, Ra.shape), Ra)
        l_dot = v_I + Ra_dot
        length_dot = np.einsum("ij,ij->i", s, l_dot)
        s_dot = (l_dot - length_dot[:, np.newaxis] * s) / lengths[:, np.newaxis]
        h_dot = np.cross(Ra_dot, s) + np.cross(Ra, s_dot)
        return np.hstack([s_dot, h_dot])

    @staticmethod
    def zyx_body_angular_velocity(
        phi: float,
        theta: float,
        euler_rates: np.ndarray
    ) -> np.ndarray:
        """Map ZYX Euler rates ``[phi_dot, theta_dot, psi_dot]`` to body omega."""
        phi_dot, theta_dot, psi_dot = np.asarray(euler_rates, dtype=np.float64)
        s_phi, c_phi = np.sin(phi), np.cos(phi)
        s_theta, c_theta = np.sin(theta), np.cos(theta)
        E = np.array([
            [1.0, 0.0, -s_theta],
            [0.0, c_phi, s_phi * c_theta],
            [0.0, -s_phi, c_phi * c_theta],
        ], dtype=np.float64)
        return E @ np.array([phi_dot, theta_dot, psi_dot], dtype=np.float64)

    @staticmethod
    def zyx_body_angular_acceleration(
        phi: float,
        theta: float,
        euler_rates: np.ndarray,
        euler_accelerations: np.ndarray
    ) -> np.ndarray:
        """Return ``alpha = d(omega_body)/dt`` for a ZYX trajectory."""
        phi_dot, theta_dot, psi_dot = np.asarray(euler_rates, dtype=np.float64)
        s_phi, c_phi = np.sin(phi), np.cos(phi)
        s_theta, c_theta = np.sin(theta), np.cos(theta)
        E = np.array([
            [1.0, 0.0, -s_theta],
            [0.0, c_phi, s_phi * c_theta],
            [0.0, -s_phi, c_phi * c_theta],
        ], dtype=np.float64)
        E_dot = np.array([
            [0.0, 0.0, -c_theta * theta_dot],
            [0.0, -s_phi * phi_dot,
             c_phi * phi_dot * c_theta - s_phi * s_theta * theta_dot],
            [0.0, -c_phi * phi_dot,
             -s_phi * phi_dot * c_theta - c_phi * s_theta * theta_dot],
        ], dtype=np.float64)
        rates = np.array([phi_dot, theta_dot, psi_dot], dtype=np.float64)
        accels = np.asarray(euler_accelerations, dtype=np.float64)
        return E @ accels + E_dot @ rates

    def actuator_trajectory(
        self,
        phi: float,
        theta: float,
        psi: float,
        T_p: np.ndarray,
        linear_velocity_I: np.ndarray,
        angular_velocity_I: np.ndarray,
        linear_acceleration_I: np.ndarray,
        angular_acceleration_I: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Derivative-consistent ``(q, q_dot, q_ddot)`` for one SE(3) state."""
        q, _, _ = self.inverse_kinematics(phi, theta, psi, T_p)
        J_inv = self.compute_inverse_jacobian(phi, theta, psi, T_p)
        J_inv_dot = self.compute_inverse_jacobian_dot(
            phi, theta, psi, T_p, linear_velocity_I, angular_velocity_I
        )
        xi = np.hstack([linear_velocity_I, angular_velocity_I])
        xi_dot = np.hstack([linear_acceleration_I, angular_acceleration_I])
        q_dot = J_inv @ xi
        q_ddot = J_inv @ xi_dot + J_inv_dot @ xi
        return q, q_dot, q_ddot
