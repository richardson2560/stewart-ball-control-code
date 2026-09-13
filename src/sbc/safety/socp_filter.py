# src/sbc/safety/socp_filter.py

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
import clarabel

from sbc.kinematics.platform import StewartKinematics
from sbc.datatypes import StateEstimatePacket


class ClarabelSafetyFilter:
    """
    Heterogeneous Second-Order Cone Program (SOCP) safety filter matching SBC v30 Chapter 11.
    Guarantees unilateral contact (N >= N_min), Coulomb adherence, plate containment,
    and mechanical leg stroke preservation.
    """

    def __init__(
        self,
        ball_mass: float = 0.06545,
        ball_radius: float = 0.025,
        friction_coeff: float = 0.35,
        friction_interior_ratio: float = 0.80,
        min_normal_force: float = 0.10,
        plate_radius: float = 0.45,
        q_min: Optional[np.ndarray] = None,
        q_max: Optional[np.ndarray] = None,
        gravity: float = 9.81
    ) -> None:
        self._m: float = ball_mass
        self._r: float = ball_radius
        self._mu: float = friction_coeff * friction_interior_ratio
        self._N_min: float = min_normal_force
        self._R_sq: float = plate_radius ** 2
        self._g: float = gravity
        self._lambda_0: float = 5.0 / 7.0

        # Physical Stewart leg limits matching CoppeliaSim model (-0.5m to +0.5m)
        # Scaled conservatively to +-0.35m (+-350 mm)
        self._q_min = np.full(6, -0.35, dtype=np.float64) if q_min is None else np.ascontiguousarray(q_min, dtype=np.float64)
        self._q_max = np.full(6, 0.35, dtype=np.float64) if q_max is None else np.ascontiguousarray(q_max, dtype=np.float64)

        # Factorized ECBF gains (Eq 11.6): alpha_1 = alpha_2 = 6.0 rad/s
        alpha_val = 6.0
        self._K1 = 2.0 * alpha_val
        self._K0 = alpha_val ** 2

        self._P_sparse = sp.csc_matrix(np.eye(6, dtype=np.float64))

        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self._settings.max_iter = 40
        self._settings.tol_gap_abs = 1e-4
        self._settings.tol_gap_rel = 1e-4

        # Zero acceleration fallback vector (Prop 11.2 - zero-jerk contingency)
        self._zero_u = np.zeros(6, dtype=np.float64)

    def filter_acceleration(
        self,
        u_q_cmd: np.ndarray,
        state: StateEstimatePacket,
        kinematics: StewartKinematics,
        q_meas: np.ndarray,
        dot_q_meas: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        phi = float(np.arctan2(state.platform_rot[2, 1], state.platform_rot[2, 2]))
        theta = float(-np.arcsin(np.clip(state.platform_rot[2, 0], -1.0, 1.0)))
        psi = float(np.arctan2(state.platform_rot[1, 0], state.platform_rot[0, 0]))

        J_inv = kinematics.compute_inverse_jacobian(phi, theta, psi, state.platform_pos)
        J_inv_inv = np.linalg.pinv(J_inv)

        F_t_I = J_inv_inv[:3, :]
        F_t_P = state.platform_rot.T @ F_t_I

        F_w_I = J_inv_inv[3:, :]
        F_w_P = state.platform_rot.T @ F_w_I

        # 1. Normal force affine map
        r_bp = np.array([state.ball_pos[0], state.ball_pos[1], self._r], dtype=np.float64)
        c_N = self._m * (F_t_P[2, :] + r_bp[0] * F_w_P[1, :] - r_bp[1] * F_w_P[0, :])
        g_P = state.platform_rot.T @ np.array([0.0, 0.0, -self._g])
        d_N = self._m * (-g_P[2])

        # 2. Friction force affine map
        C_a_rel = - self._lambda_0 * F_t_P[:2, :] - self._r * np.vstack([F_w_P[1, :], -F_w_P[0, :]])
        d_a_rel = self._lambda_0 * g_P[:2]

        C_f = self._m * (C_a_rel + F_t_P[:2, :])
        d_f = self._m * (d_a_rel - g_P[:2])

        # Assembling constraints
        row_contact = -c_N
        b_contact = d_N - self._N_min

        h3 = self._R_sq - float(np.dot(state.ball_pos, state.ball_pos))
        dot_h3 = - 2.0 * float(np.dot(state.ball_pos, state.ball_vel))
        
        row_domain = 2.0 * (state.ball_pos @ C_a_rel)
        b_domain = self._K1 * dot_h3 + self._K0 * h3 - 2.0 * float(np.dot(state.ball_vel, state.ball_vel)) - 2.0 * float(np.dot(state.ball_pos, d_a_rel))

        A_stroke = np.zeros((6, 6), dtype=np.float64)
        b_stroke = np.zeros(6, dtype=np.float64)
        for i in range(6):
            qi = q_meas[i]
            dqi = dot_q_meas[i]
            h4_i = (self._q_max[i] - qi) * (qi - self._q_min[i])
            dot_h4_i = (self._q_max[i] + self._q_min[i] - 2.0 * qi) * dqi
            
            coeff = self._q_max[i] + self._q_min[i] - 2.0 * qi
            A_stroke[i, i] = -coeff
            b_stroke[i] = - 2.0 * (dqi ** 2) + self._K1 * dot_h4_i + self._K0 * h4_i

        A_lin = np.vstack([row_contact[np.newaxis, :], row_domain[np.newaxis, :], A_stroke])
        b_lin = np.hstack([b_contact, b_domain, b_stroke])

        A_soc = np.vstack([- self._mu * c_N[np.newaxis, :], - C_f])
        b_soc = np.hstack([self._mu * d_N, d_f])

        A_all = sp.csc_matrix(np.vstack([A_lin, A_soc]))
        b_all = np.hstack([b_lin, b_soc])

        cones = [
            clarabel.NonnegativeConeT(8),
            clarabel.SecondOrderConeT(3)
        ]

        # Clamping requested acceleration to physically realizable bounds
        u_q_clamped = np.clip(u_q_cmd, -8.0, 8.0)
        q_cost = - np.ascontiguousarray(u_q_clamped, dtype=np.float64)

        try:
            solver = clarabel.DefaultSolver(self._P_sparse, q_cost, A_all, b_all, cones, self._settings)
            solution = solver.solve()

            if solution.status == clarabel.SolverStatus.Solved:
                u_q_applied = np.array(solution.x, dtype=np.float64)
                return u_q_applied, 0.0, True
            else:
                # Mathematically safe contingency: Zero acceleration (zero jerk, constant velocity coast)
                return self._zero_u.copy(), 0.0, False

        except Exception:
            return self._zero_u.copy(), 0.0, False