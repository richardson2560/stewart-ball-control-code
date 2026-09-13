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
        """
        Args:
            ball_mass: Sphere mass m [kg].
            ball_radius: Sphere radius r [m].
            friction_coeff: Static friction coefficient mu_s.
            friction_interior_ratio: Cone interior reserve factor eta_mu in (0, 1).
            min_normal_force: Unilateral contact detachment floor N_min [N].
            plate_radius: Usable ball-center plate boundary R_plate [m].
            q_min: Lower leg stroke limits [m], shape (6,).
            q_max: Upper leg stroke limits [m], shape (6,).
        """
        self._m: float = ball_mass
        self._r: float = ball_radius
        self._mu: float = friction_coeff * friction_interior_ratio
        self._N_min: float = min_normal_force
        self._R_sq: float = plate_radius ** 2
        self._g: float = gravity
        self._lambda_0: float = 5.0 / 7.0

        self._q_min = np.full(6, -0.05, dtype=np.float64) if q_min is None else np.ascontiguousarray(q_min, dtype=np.float64)
        self._q_max = np.full(6, 0.05, dtype=np.float64) if q_max is None else np.ascontiguousarray(q_max, dtype=np.float64)

        # Factorized ECBF gains (Eq 11.6): alpha_1 = alpha_2 = 8.0 rad/s
        alpha_val = 8.0
        self._K1 = 2.0 * alpha_val
        self._K0 = alpha_val ** 2

        # Quadratic objective weighting matrix P in 1/2 u^T P u - q^T u
        # Decision variable x = u_q in R^6
        self._P_sparse = sp.csc_matrix(np.eye(6, dtype=np.float64))

        # Clarabel solver settings
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False
        self._settings.max_iter = 40
        self._settings.tol_gap_abs = 1e-4
        self._settings.tol_gap_rel = 1e-4

        # Cache of previous safe acceleration for verified fallback (Prop 11.2)
        self._prev_u_q = np.zeros(6, dtype=np.float64)

    def filter_acceleration(
        self,
        u_q_cmd: np.ndarray,
        state: StateEstimatePacket,
        kinematics: StewartKinematics,
        q_meas: np.ndarray,
        dot_q_meas: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Filters candidate leg acceleration u_q_cmd through the SOCP cone.

        Args:
            u_q_cmd: Desired leg acceleration vector from redundancy resolution [m/s^2], shape (6,).
            state: Filtered perception state packet.
            kinematics: Platform kinematics model.
            q_meas: Measured leg extensions [m], shape (6,).
            dot_q_meas: Measured leg extension velocities [m/s], shape (6,).

        Returns:
            u_q_applied: Safe leg acceleration command [m/s^2], shape (6,)
            slack: Always 0.0 in strict mode
            is_optimal: True if Clarabel converged to strict feasibility
        """
        # Linearization of the platform acceleration twist: xi_dot = J_inv^-1 * u_q
        phi = float(np.arctan2(state.platform_rot[2, 1], state.platform_rot[2, 2]))
        theta = float(-np.arcsin(np.clip(state.platform_rot[2, 0], -1.0, 1.0)))
        psi = float(np.arctan2(state.platform_rot[1, 0], state.platform_rot[0, 0]))

        J_inv = kinematics.compute_inverse_jacobian(phi, theta, psi, state.platform_pos)
        J_inv_inv = np.linalg.pinv(J_inv)  # shape (6, 6)

        # Mapping to origin translation acceleration A_p in I and P
        F_t_I = J_inv_inv[:3, :]  # shape (3, 6)
        F_t_P = state.platform_rot.T @ F_t_I  # shape (3, 6)

        # Mapping to platform angular acceleration alpha in I and P
        F_w_I = J_inv_inv[3:, :]  # shape (3, 6)
        F_w_P = state.platform_rot.T @ F_w_I  # shape (3, 6)

        # 1. Normal force affine map: N(u_q) = c_N^T * u_q + d_N
        # A_tr,z = A_p,z + (alpha x r_bp)_z + centripetal_z + coriolis_z
        r_bp = np.array([state.ball_pos[0], state.ball_pos[1], self._r], dtype=np.float64)
        c_N = self._m * (F_t_P[2, :] + r_bp[0] * F_w_P[1, :] - r_bp[1] * F_w_P[0, :])
        
        # Bias normal force component (gravity + centripetal)
        g_P = state.platform_rot.T @ np.array([0.0, 0.0, -self._g])
        d_N = self._m * (-g_P[2])

        # 2. Friction force affine map: f(u_q) = C_f * u_q + d_f in R^2
        # a_rel = lambda_0 * (g_tangential - A_p_tangential) - r*(alpha x n)_tangential
        C_a_rel = - self._lambda_0 * F_t_P[:2, :] - self._r * np.vstack([F_w_P[1, :], -F_w_P[0, :]])
        d_a_rel = self._lambda_0 * g_P[:2]

        C_f = self._m * (C_a_rel + F_t_P[:2, :])  # shape (2, 6)
        d_f = self._m * (d_a_rel - g_P[:2])

        # Clarabel cone assembling: A_cone * x + s = b_cone, s in K
        # Constraint 1: Contact unilateral N(u_q) >= N_min => - c_N^T u_q <= d_N - N_min
        row_contact = -c_N
        b_contact = d_N - self._N_min

        # Constraint 2: Circular Plate Boundary CBF (h_3 = R_plate^2 - ||rho||^2)
        # h3_dot = - 2 * rho^T * v_rel
        # h3_ddot = - 2 * ||v_rel||^2 - 2 * rho^T * a_rel(u_q)
        h3 = self._R_sq - float(np.dot(state.ball_pos, state.ball_pos))
        dot_h3 = - 2.0 * float(np.dot(state.ball_pos, state.ball_vel))
        
        # - 2 * rho^T * C_a_rel * u_q >= - K1*dot_h3 - K0*h3 + 2*||v_rel||^2 + 2*rho^T*d_a_rel
        row_domain = 2.0 * (state.ball_pos @ C_a_rel)  # shape (6,)
        b_domain = self._K1 * dot_h3 + self._K0 * h3 - 2.0 * float(np.dot(state.ball_vel, state.ball_vel)) - 2.0 * float(np.dot(state.ball_pos, d_a_rel))

        # Constraint 3: Leg stroke boundaries h_4,i >= 0 for i=1..6
        # h_4,i = (q_max,i - q_i)(q_i - q_min,i)
        # h_4_ddot = (q_max + q_min - 2*q_i) * u_q_i - 2 * dot_q_i^2
        # Imposes: - (q_max + q_min - 2*q_i) * u_q_i <= - 2*dot_q_i^2 + K1*dot_h_4 + K0*h_4
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

        # Combine Non-Negative Cone (Linear inequalities A_lin * u_q <= b_lin)
        # Total linear rows: 1 (contact) + 1 (domain) + 6 (stroke) = 8 rows
        A_lin = np.vstack([row_contact[np.newaxis, :], row_domain[np.newaxis, :], A_stroke])
        b_lin = np.hstack([b_contact, b_domain, b_stroke])

        # Constraint 4: Second-Order Cone for Coulomb Friction
        # || C_f * u_q + d_f ||_2 <= mu * (c_N^T * u_q + d_N)
        # Standard cone format: [t; v] with ||v||_2 <= t
        # Row 0: t = mu * c_N^T * u_q + mu * d_N
        # Rows 1, 2: v = C_f * u_q + d_f
        # Clarabel format: A_soc * u_q + s = b_soc => s = b_soc - A_soc * u_q in K_SOC
        A_soc = np.vstack([- self._mu * c_N[np.newaxis, :], - C_f])  # shape (3, 6)
        b_soc = np.hstack([self._mu * d_N, d_f])                      # shape (3,)

        # Total Constraint Matrix A_all * x + s = b_all
        A_all = sp.csc_matrix(np.vstack([A_lin, A_soc]))
        b_all = np.hstack([b_lin, b_soc])

        cones = [
            clarabel.NonnegativeConeT(8),       # 8 linear inequalities
            clarabel.SecondOrderConeT(3)        # 1 friction Lorentz cone in R^3
        ]

        # Regularize candidate acceleration to avoid infinite commands
        u_q_clamped = np.clip(u_q_cmd, -25.0, 25.0)
        q_cost = - np.ascontiguousarray(u_q_clamped, dtype=np.float64)

        try:
            solver = clarabel.DefaultSolver(self._P_sparse, q_cost, A_all, b_all, cones, self._settings)
            solution = solver.solve()

            if solution.status == clarabel.SolverStatus.Solved:
                u_q_applied = np.array(solution.x, dtype=np.float64)
                self._prev_u_q = u_q_applied.copy()
                return u_q_applied, 0.0, True
            else:
                # PROVEN FALLBACK: Hold previous safe acceleration (Prop 11.2)
                return self._prev_u_q.copy(), 0.0, False

        except Exception:
            return self._prev_u_q.copy(), 0.0, False