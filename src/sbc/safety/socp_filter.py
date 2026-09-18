# src/sbc/safety/socp_filter.py

import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional
import clarabel

from sbc.kinematics.platform import StewartKinematics
from sbc.datatypes import StateEstimatePacket
from sbc.safety.fallback import OneCycleFallback


class ClarabelSafetyFilter:
    """
    Heterogeneous Second-Order Cone Program (SOCP) safety filter matching SBC v30 Chapter 11.
    Guarantees unilateral contact (N >= N_min), Coulomb adherence, plate containment,
    and mechanical leg stroke preservation.
    """

    def __init__(
        self,
        ball_mass: float = 0.065449846949792,
        ball_radius: float = 0.025,
        friction_coeff: float = 0.35,
        friction_interior_ratio: float = 0.80,
        min_normal_force: float = 0.10,
        plate_radius: float = 0.45,
        q_min: Optional[np.ndarray] = None,
        q_max: Optional[np.ndarray] = None,
        gravity: float = 9.81,
        cycle_time: float = 0.002,
        max_leg_acceleration: float = 8.0,
        rolling_resistance_coeff: float = 0.0015,
        contact_margin: float = 0.0,
        friction_margin: float = 0.0,
        domain_margin: float = 0.0,
        stroke_margins: Optional[np.ndarray] = None,
        admission_tolerance: float = 5e-5,
    ) -> None:
        self._m: float = ball_mass
        self._r: float = ball_radius
        self._mu: float = friction_coeff * friction_interior_ratio
        self._N_min: float = min_normal_force
        self._R_sq: float = plate_radius ** 2
        self._g: float = gravity
        self._lambda_0: float = 5.0 / 7.0
        self._dt: float = float(cycle_time)
        self._u_max: float = float(max_leg_acceleration)
        self._c_rr: float = float(rolling_resistance_coeff)
        self._M_N = float(contact_margin)
        self._M_f = float(friction_margin)
        self._M_3 = float(domain_margin)
        self._M_4 = (
            np.zeros(6, dtype=np.float64)
            if stroke_margins is None
            else np.asarray(stroke_margins, dtype=np.float64)
        )
        self._admission_tol = float(admission_tolerance)
        if self._M_4.shape != (6,) or np.any(self._M_4 < 0.0):
            raise ValueError("stroke_margins must be a nonnegative six-vector.")
        if min(self._M_N, self._M_f, self._M_3, self._admission_tol) < 0.0:
            raise ValueError("Safety margins and admission tolerance must be nonnegative.")

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
        self._settings.max_iter = 60
        self._settings.tol_gap_abs = 1e-6
        self._settings.tol_gap_rel = 1e-6
        if hasattr(self._settings, "tol_feas"):
            self._settings.tol_feas = 1e-7

        self._last_feasible_u = np.zeros(6, dtype=np.float64)
        self._fallback = OneCycleFallback(self._dt, self._u_max)
        self._fallback_admissible = False
        self._last_reason = "not_run"
        self._last_solver_status = "not_run"
        self._last_linear_violation = float("inf")
        self._last_cone_violation = float("inf")
        self._verification_tol = 1e-6

    @property
    def q_min(self) -> np.ndarray:
        return self._q_min.copy()

    @property
    def q_max(self) -> np.ndarray:
        return self._q_max.copy()

    @property
    def fallback_admissible(self) -> bool:
        return self._fallback_admissible

    @property
    def last_reason(self) -> str:
        return self._last_reason

    @property
    def last_solver_status(self) -> str:
        return self._last_solver_status

    @property
    def last_linear_violation(self) -> float:
        return self._last_linear_violation

    @property
    def last_cone_violation(self) -> float:
        return self._last_cone_violation

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        x, y, z = vector
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    def _braking_fallback(
        self, q: np.ndarray, q_dot: np.ndarray, reason: str = "fallback"
    ) -> np.ndarray:
        """One-cycle hold if admitted; otherwise an explicitly uncertified brake."""
        decision = self._fallback.select(
            self._last_feasible_u, q, q_dot, self._q_min, self._q_max
        )
        self._fallback_admissible = decision.one_cycle_admissible
        self._last_reason = f"{reason}:{decision.reason}"
        return decision.acceleration

    def emergency_braking(
        self, q: np.ndarray, q_dot: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Return a bounded brake satisfying the exact one-step stroke box.

        This is an actuator-protection fallback, not a coupled contact/friction
        certificate.  It is used only after nominal SOCP admission has failed.
        """
        q = np.asarray(q, dtype=np.float64)
        q_dot = np.asarray(q_dot, dtype=np.float64)
        if not all(x.shape == (6,) and np.all(np.isfinite(x)) for x in (q, q_dot)):
            return np.zeros(6, dtype=np.float64), False
        lower_step = 2.0 * (
            self._q_min - q - self._dt * q_dot
        ) / self._dt**2
        upper_step = 2.0 * (
            self._q_max - q - self._dt * q_dot
        ) / self._dt**2
        lower = np.maximum(lower_step, -self._u_max)
        upper = np.minimum(upper_step, self._u_max)
        feasible = bool(np.all(lower <= upper))
        brake = np.clip(-8.0 * q_dot, -self._u_max, self._u_max)
        if feasible:
            brake = np.minimum(np.maximum(brake, lower), upper)
        return brake, feasible

    def filter_acceleration(
        self,
        u_q_cmd: np.ndarray,
        state: StateEstimatePacket,
        kinematics: StewartKinematics,
        q_meas: np.ndarray,
        dot_q_meas: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        self._last_linear_violation = float("inf")
        self._last_cone_violation = float("inf")
        u_q_cmd = np.asarray(u_q_cmd, dtype=np.float64)
        q_meas = np.asarray(q_meas, dtype=np.float64)
        dot_q_meas = np.asarray(dot_q_meas, dtype=np.float64)
        if not all(x.shape == (6,) and np.all(np.isfinite(x)) for x in (q_meas, dot_q_meas)):
            self._fallback_admissible = False
            self._last_solver_status = "NotRunInvalidMeasurement"
            self._last_reason = "invalid_joint_measurement"
            return np.zeros(6, dtype=np.float64), 0.0, False
        if (not state.contact_valid or u_q_cmd.shape != (6,)
                or not np.all(np.isfinite(u_q_cmd))):
            reason = "contact_invalid" if not state.contact_valid else "invalid_nominal_command"
            self._last_solver_status = (
                "NotRunContactInvalid"
                if not state.contact_valid
                else "NotRunInvalidCommand"
            )
            return self._braking_fallback(q_meas, dot_q_meas, reason), 0.0, False
        phi = float(np.arctan2(state.platform_rot[2, 1], state.platform_rot[2, 2]))
        theta = float(-np.arcsin(np.clip(state.platform_rot[2, 0], -1.0, 1.0)))
        psi = float(np.arctan2(state.platform_rot[1, 0], state.platform_rot[0, 0]))

        J_inv = kinematics.compute_inverse_jacobian(phi, theta, psi, state.platform_pos)
        if np.linalg.cond(J_inv) > 1e6:
            return self._braking_fallback(q_meas, dot_q_meas, "singular_jacobian"), 0.0, False
        J_inv_inv = np.linalg.inv(J_inv)

        xi_I = np.asarray(state.platform_twist, dtype=np.float64)
        J_inv_dot = kinematics.compute_inverse_jacobian_dot(
            phi, theta, psi, state.platform_pos, xi_I[:3], xi_I[3:]
        )
        # xi_dot = C_xi u_q + d_xi, including the essential J_inv_dot term.
        C_xi = J_inv_inv
        d_xi = -J_inv_inv @ J_inv_dot @ xi_I

        C_A = state.platform_rot.T @ C_xi[:3, :]
        d_A = state.platform_rot.T @ d_xi[:3]

        C_alpha = state.platform_rot.T @ C_xi[3:, :]
        d_alpha = state.platform_rot.T @ d_xi[3:]

        omega_P = state.platform_rot.T @ xi_I[3:]
        v_rel_3 = np.array([state.ball_vel[0], state.ball_vel[1], 0.0])
        rho_3 = np.array([state.ball_pos[0], state.ball_pos[1], 0.0])

        r_bp = np.array([state.ball_pos[0], state.ball_pos[1], self._r], dtype=np.float64)
        g_P = state.platform_rot.T @ np.array([0.0, 0.0, -self._g])
        centripetal = np.cross(omega_P, np.cross(omega_P, r_bp))
        coriolis_transport = 2.0 * np.cross(omega_P, v_rel_3)

        # Complete transport acceleration A_tr = A_p + alpha x r_bp + ...
        C_A_tr = C_A - self._skew(r_bp) @ C_alpha
        d_A_tr = d_A - self._skew(r_bp) @ d_alpha + centripetal + coriolis_transport

        # 1. Normal force affine map.
        c_N = self._m * C_A_tr[2, :]
        d_N = self._m * (d_A_tr[2] - g_P[2])

        # 2. Complete spin-free tangential acceleration affine map.
        C_B = -C_A[:2, :] + (self._skew(rho_3) @ C_alpha)[:2, :]
        d_B = g_P[:2] - d_A[:2] + (self._skew(rho_3) @ d_alpha)[:2] - centripetal[:2]
        n_P = np.array([0.0, 0.0, 1.0])
        C_euler = self._r * (self._skew(n_P) @ C_alpha)[:2, :]
        d_euler = self._r * (self._skew(n_P) @ d_alpha)[:2]

        speed = float(np.linalg.norm(state.ball_vel))
        f_rr = -self._c_rr * max(d_N, self._N_min) * state.ball_vel / np.sqrt(speed**2 + 1e-6)
        yaw_term = -(1.0 + self._lambda_0) * state.yaw_rate * np.array(
            [-state.ball_vel[1], state.ball_vel[0]], dtype=np.float64
        )
        C_a_rel = self._lambda_0 * C_B + C_euler
        d_a_rel = self._lambda_0 * d_B + d_euler + yaw_term + self._lambda_0 * f_rr / self._m

        C_f = self._m * (C_a_rel + C_A_tr[:2, :])
        d_f = self._m * (d_a_rel + d_A_tr[:2] - g_P[:2]) - f_rr

        # Assembling constraints
        row_contact = -c_N
        b_contact = d_N - self._N_min - self._M_N

        h3 = self._R_sq - float(np.dot(state.ball_pos, state.ball_pos))
        dot_h3 = - 2.0 * float(np.dot(state.ball_pos, state.ball_vel))
        
        row_domain = 2.0 * (state.ball_pos @ C_a_rel)
        b_domain = (self._K1 * dot_h3 + self._K0 * h3
                    - 2.0 * float(np.dot(state.ball_vel, state.ball_vel))
                    - 2.0 * float(np.dot(state.ball_pos, d_a_rel))
                    - self._M_3)

        A_stroke = np.zeros((6, 6), dtype=np.float64)
        b_stroke = np.zeros(6, dtype=np.float64)
        for i in range(6):
            qi = q_meas[i]
            dqi = dot_q_meas[i]
            h4_i = (self._q_max[i] - qi) * (qi - self._q_min[i])
            dot_h4_i = (self._q_max[i] + self._q_min[i] - 2.0 * qi) * dqi
            
            coeff = self._q_max[i] + self._q_min[i] - 2.0 * qi
            A_stroke[i, i] = -coeff
            b_stroke[i] = (-2.0 * dqi**2 + self._K1 * dot_h4_i
                           + self._K0 * h4_i - self._M_4[i])

        # The acceleration bound is a hard constraint on the solution, not
        # merely a clamp on the objective target.
        A_bounds = np.vstack([np.eye(6), -np.eye(6)])
        b_bounds = np.full(12, self._u_max, dtype=np.float64)

        # Exact sampled-data stroke condition for the same integration rule
        # used by the CSP command generator:
        # q_next = q + dt*q_dot + 0.5*dt^2*u.
        one_step_upper = 2.0 * (
            self._q_max - q_meas - self._dt * dot_q_meas
        ) / self._dt**2
        one_step_lower = 2.0 * (
            q_meas + self._dt * dot_q_meas - self._q_min
        ) / self._dt**2
        A_step = np.vstack([np.eye(6), -np.eye(6)])
        b_step = np.hstack([one_step_upper, one_step_lower])
        A_lin = np.vstack([
            row_contact[np.newaxis, :], row_domain[np.newaxis, :],
            A_stroke, A_bounds, A_step
        ])
        b_lin = np.hstack([
            b_contact, b_domain, b_stroke, b_bounds, b_step
        ])

        A_soc = np.vstack([- self._mu * c_N[np.newaxis, :], - C_f])
        b_soc = np.hstack([self._mu * d_N - self._M_f, d_f])

        # Solve a numerically tightened problem.  This is essential if an
        # AlmostSolved candidate is to be considered: the final independent
        # check below is performed against the original, untightened set.
        row_scale = np.maximum(
            1.0,
            np.maximum(np.linalg.norm(A_lin, axis=1), np.abs(b_lin)),
        )
        b_lin_solve = b_lin - self._admission_tol * row_scale
        b_soc_solve = b_soc.copy()
        b_soc_solve[0] -= self._admission_tol * max(
            1.0, float(np.linalg.norm(A_soc[0])), abs(float(b_soc[0]))
        )
        A_all = sp.csc_matrix(np.vstack([A_lin, A_soc]))
        b_all = np.hstack([b_lin_solve, b_soc_solve])

        cones = [
            clarabel.NonnegativeConeT(int(A_lin.shape[0])),
            clarabel.SecondOrderConeT(3)
        ]

        # Clamping requested acceleration to physically realizable bounds
        u_q_clamped = np.clip(u_q_cmd, -self._u_max, self._u_max)
        q_cost = - np.ascontiguousarray(u_q_clamped, dtype=np.float64)

        try:
            solver = clarabel.DefaultSolver(self._P_sparse, q_cost, A_all, b_all, cones, self._settings)
            solution = solver.solve()
            self._last_solver_status = str(solution.status)

            admissible_statuses = {clarabel.SolverStatus.Solved}
            almost_solved = getattr(clarabel.SolverStatus, "AlmostSolved", None)
            if almost_solved is not None:
                admissible_statuses.add(almost_solved)
            if solution.status in admissible_statuses:
                u_q_applied = np.array(solution.x, dtype=np.float64)
                # Independent post-solve numerical admission.
                if not np.all(np.isfinite(u_q_applied)):
                    return self._braking_fallback(q_meas, dot_q_meas, "nonfinite_solution"), 0.0, False
                linear_violation = float(np.max(A_lin @ u_q_applied - b_lin))
                self._last_linear_violation = linear_violation
                if linear_violation > self._verification_tol:
                    return self._braking_fallback(q_meas, dot_q_meas, "linear_admission"), 0.0, False
                cone_vec = b_soc - A_soc @ u_q_applied
                cone_violation = float(np.linalg.norm(cone_vec[1:]) - cone_vec[0])
                self._last_cone_violation = cone_violation
                if cone_violation > self._verification_tol:
                    return self._braking_fallback(q_meas, dot_q_meas, "cone_admission"), 0.0, False
                self._last_feasible_u = u_q_applied.copy()
                self._fallback_admissible = False
                self._last_reason = "strict_admitted"
                return u_q_applied, 0.0, True
            else:
                return self._braking_fallback(
                    q_meas, dot_q_meas, f"solver_{solution.status}"
                ), 0.0, False

        except Exception as exc:
            self._last_solver_status = type(exc).__name__
            return self._braking_fallback(
                q_meas, dot_q_meas, f"solver_exception_{type(exc).__name__}"
            ), 0.0, False
