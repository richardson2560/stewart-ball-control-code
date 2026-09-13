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

        self.b = np.ascontiguousarray(base_anchors, dtype=np.float64)
        self.a = np.ascontiguousarray(platform_anchors, dtype=np.float64)
        self.offsets = np.ascontiguousarray(leg_offsets, dtype=np.float64)
        
        if initial_translation is not None:
            self.T_p_nominal = np.ascontiguousarray(initial_translation, dtype=np.float64)
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