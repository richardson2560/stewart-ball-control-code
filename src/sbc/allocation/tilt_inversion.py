# src/sbc/allocation/tilt_inversion.py

import numpy as np
from typing import Tuple, Optional


class ResidualTranslationAllocator:
    """Gravity-first allocation with translational residual completion.

    For a generated attitude trajectory, the exact nominal identity

        B_des = g_parallel - A_p_parallel
                - (alpha x rho)_parallel
                - (Omega x (Omega x r_bp))_parallel

    is closed by solving algebraically for ``A_p_parallel``.  Bounds are
    deliberate command modifications and the returned ``B_residual`` must be
    included in the ISS realization budget.
    """

    def __init__(
        self,
        cycle_time: float = 0.002,
        max_acceleration: float = 3.0,
        max_jerk: float = 80.0,
        max_velocity: float = 0.35,
        max_offset: float = 0.060,
        centering_start_ratio: float = 0.70,
        centering_kp: float = 8.0,
        centering_kd: float = 4.0,
    ) -> None:
        self._dt = float(cycle_time)
        self._a_max = float(max_acceleration)
        self._j_max = float(max_jerk)
        self._v_max = float(max_velocity)
        self._p_max = float(max_offset)
        self._centering_start = float(centering_start_ratio) * self._p_max
        self._centering_kp = float(centering_kp)
        self._centering_kd = float(centering_kd)
        if min(self._dt, self._a_max, self._j_max, self._v_max, self._p_max) <= 0.0:
            raise ValueError("Translation allocator limits must be strictly positive.")
        if not 0.0 < centering_start_ratio < 1.0:
            raise ValueError("centering_start_ratio must lie strictly inside (0, 1).")
        if min(self._centering_kp, self._centering_kd) <= 0.0:
            raise ValueError("Centering gains must be strictly positive.")
        self.reset()

    def reset(self) -> None:
        self._acceleration_I = np.zeros(3, dtype=np.float64)
        self._velocity_I = np.zeros(3, dtype=np.float64)
        self._offset_I = np.zeros(3, dtype=np.float64)

    def synchronize_measured_state(
        self,
        offset_I: np.ndarray,
        velocity_I: np.ndarray,
        rotation_P_to_I: Optional[np.ndarray] = None,
    ) -> bool:
        """Synchronize the command model with independently measured motion.

        Position and velocity are deliberately *not* clipped: clipping a
        measurement would conceal a genuine envelope exit.  The returned flag
        tells the supervisor whether the measured state is certified.
        """
        offset = np.asarray(offset_I, dtype=np.float64)
        velocity = np.asarray(velocity_I, dtype=np.float64)
        if offset.shape != (3,) or velocity.shape != (3,):
            raise ValueError("Measured translation state must contain two 3-vectors.")
        if not np.all(np.isfinite(np.hstack([offset, velocity]))):
            raise ValueError("Measured translation state must be finite.")
        if rotation_P_to_I is not None:
            R = np.asarray(rotation_P_to_I, dtype=np.float64)
            if R.shape != (3, 3) or not np.all(np.isfinite(R)):
                raise ValueError("Measured rotation must be a finite 3x3 matrix.")
            # The allocator owns tangential translation authority only.  The
            # complementary normal component is regulated separately to the
            # nominal platform pose by the actuator-coordinate pose loop.
            normal_I = R[:, 2]
            projector = np.eye(3) - np.outer(normal_I, normal_I)
            offset = projector @ offset
            velocity = projector @ velocity
        self._offset_I = np.array(offset, dtype=np.float64, copy=True)
        self._velocity_I = np.array(velocity, dtype=np.float64, copy=True)
        return self.within_limits

    @property
    def acceleration_I(self) -> np.ndarray:
        return self._acceleration_I.copy()

    @property
    def velocity_I(self) -> np.ndarray:
        return self._velocity_I.copy()

    @property
    def offset_I(self) -> np.ndarray:
        return self._offset_I.copy()

    @property
    def within_limits(self) -> bool:
        return bool(
            np.linalg.norm(self._offset_I) <= self._p_max + 1e-12
            and np.linalg.norm(self._velocity_I) <= self._v_max + 1e-12
            and np.linalg.norm(self._acceleration_I) <= self._a_max + 1e-12
        )

    @staticmethod
    def required_acceleration(
        B_des: np.ndarray,
        gravity_parallel: np.ndarray,
        alpha_P: np.ndarray,
        omega_P: np.ndarray,
        rho: np.ndarray,
        ball_radius: float
    ) -> np.ndarray:
        """Exact platform-frame translational acceleration that realizes B_des."""
        B_des = np.asarray(B_des, dtype=np.float64)
        gravity_parallel = np.asarray(gravity_parallel, dtype=np.float64)
        alpha_P = np.asarray(alpha_P, dtype=np.float64)
        omega_P = np.asarray(omega_P, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        values = np.hstack([B_des, gravity_parallel, alpha_P, omega_P, rho])
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite residual-translation allocation input.")
        r_bp = np.array([rho[0], rho[1], ball_radius], dtype=np.float64)
        euler_rho = np.cross(alpha_P, np.array([rho[0], rho[1], 0.0]))[:2]
        centripetal = np.cross(omega_P, np.cross(omega_P, r_bp))[:2]
        return gravity_parallel - euler_rho - centripetal - B_des

    @staticmethod
    def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector if norm <= maximum else vector * (maximum / norm)

    def allocate(
        self,
        B_des: np.ndarray,
        rotation_P_to_I: np.ndarray,
        gravity_parallel: np.ndarray,
        alpha_P: np.ndarray,
        omega_P: np.ndarray,
        rho: np.ndarray,
        ball_radius: float,
        commit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Return applied ``A_p,parallel``, B-realization residual, exact flag.

        The state update is semi-implicit and confined to the horizontal
        translational workspace.  When a limit is active, the mismatch is
        explicitly returned rather than silently claiming exact inversion.
        """
        R = np.asarray(rotation_P_to_I, dtype=np.float64)
        required_P = self.required_acceleration(
            B_des, gravity_parallel, alpha_P, omega_P, rho, ball_radius
        )
        required_I = R @ np.array([required_P[0], required_P[1], 0.0])
        normal_I = R[:, 2]
        tangent_projector = np.eye(3) - np.outer(normal_I, normal_I)

        # Preserve full residual authority around the nominal origin, then
        # blend continuously into a damped re-centering field before the hard
        # workspace boundary.  Any sacrificed ball-drive authority remains in
        # B_residual below and is therefore visible to the ISS budget.
        offset_norm = float(np.linalg.norm(self._offset_I))
        blend = float(np.clip(
            (offset_norm - self._centering_start)
            / (self._p_max - self._centering_start),
            0.0,
            1.0,
        ))
        if blend > 0.0:
            center_I = (
                -self._centering_kp * self._offset_I
                -self._centering_kd * self._velocity_I
            )
            required_I = tangent_projector @ (
                (1.0 - blend) * required_I + blend * center_I
            )

        candidate_I = self._limit_norm(required_I, self._a_max)
        delta_a = candidate_I - self._acceleration_I
        candidate_I = self._acceleration_I + self._limit_norm(
            delta_a, self._j_max * self._dt
        )
        candidate_I = tangent_projector @ candidate_I

        velocity_next = self._velocity_I + self._dt * candidate_I
        velocity_next = self._limit_norm(velocity_next, self._v_max)
        candidate_I = (velocity_next - self._velocity_I) / self._dt
        offset_next = self._offset_I + self._dt * velocity_next

        if np.linalg.norm(offset_next) > self._p_max:
            # A bounded centering command is a deliberate degraded allocation.
            center_I = (
                -self._centering_kp * self._offset_I
                -self._centering_kd * self._velocity_I
            )
            center_I = tangent_projector @ center_I
            center_I = self._limit_norm(center_I, self._a_max)
            delta_a = center_I - self._acceleration_I
            candidate_I = self._acceleration_I + self._limit_norm(
                delta_a, self._j_max * self._dt
            )
            candidate_I = tangent_projector @ candidate_I
            velocity_next = self._limit_norm(
                self._velocity_I + self._dt * candidate_I, self._v_max
            )
            offset_next = self._offset_I + self._dt * velocity_next

        if commit:
            self._acceleration_I = candidate_I
            self._velocity_I = velocity_next
            self._offset_I = offset_next

        applied_P_3 = R.T @ candidate_I
        applied_P = applied_P_3[:2]
        # B_actual - B_des = -(A_applied - A_required).
        B_residual = required_P - applied_P
        exact = bool(np.linalg.norm(B_residual) <= 1e-9)
        return applied_P, B_residual, exact

    def commit_realized(self, acceleration_I: np.ndarray) -> bool:
        """Advance an observer-only state when no pose measurement exists.

        This fallback is not used by the Coppelia backend.  It refuses an
        update that would leave the certified position envelope instead of
        committing an invalid state and raising one cycle later.
        """
        acceleration_I = np.asarray(acceleration_I, dtype=np.float64).copy()
        if acceleration_I.shape != (3,) or not np.all(np.isfinite(acceleration_I)):
            raise ValueError("Realized translation acceleration must be a finite 3-vector.")
        acceleration_I = self._limit_norm(acceleration_I, self._a_max)
        velocity_next = self._limit_norm(
            self._velocity_I + self._dt * acceleration_I, self._v_max
        )
        offset_next = self._offset_I + self._dt * velocity_next
        if np.linalg.norm(offset_next) > self._p_max + 1e-12:
            return False
        self._acceleration_I = acceleration_I
        self._velocity_I = velocity_next
        self._offset_I = offset_next
        return self.within_limits


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
        d_B = - A_p,parallel - (alpha x rho)_parallel - (Omega x (Omega x r_bp))_parallel
        """
        # Sphere center offset in platform frame P: r_bp = [x, y, r]^T
        r_bp = np.array([rho[0], rho[1], self._r], dtype=np.float64)

        # B contains alpha x rho.  The separate -r(alpha x n) term belongs
        # to the controller cancellation and must not be counted here again.
        rho_3 = np.array([rho[0], rho[1], 0.0], dtype=np.float64)
        euler_term = np.cross(alpha, rho_3)[:2]

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

        # Strict partial inverse: infeasible and non-finite inputs are rejected.
        if not np.all(np.isfinite(b_g)):
            return float("nan"), float("nan"), False
        limit_bx = self._g * np.sin(self._theta_max)
        
        # Step 1: Invert Pitch theta_d = arcsin(b_g,x / g)
        b_gx = float(b_g[0])
        if abs(b_gx) >= limit_bx:
            return float("nan"), float("nan"), False

        sin_theta = np.clip(b_gx / self._g, -1.0 + 1e-7, 1.0 - 1e-7)
        theta_d = float(np.arcsin(sin_theta))
        cos_theta = np.sqrt(max(1e-7, 1.0 - sin_theta**2))

        # Step 2: Invert Roll phi_d = - arcsin(b_g,y / (g * cos(theta_d)))
        limit_by = self._g * cos_theta * np.sin(self._phi_max)
        b_gy = float(b_g[1])
        if abs(b_gy) >= limit_by:
            return float("nan"), float("nan"), False

        sin_phi = np.clip(-b_gy / (self._g * cos_theta), -1.0 + 1e-7, 1.0 - 1e-7)
        phi_d = float(np.arcsin(sin_phi))

        return phi_d, theta_d, True

    def project_preferred_gravity(self, target: np.ndarray) -> Tuple[float, float, np.ndarray, bool]:
        """Deliberately project a target into the exact gravity-tilt domain.

        Unlike :meth:`invert`, this is an allocator: any rejected component is
        expected to be supplied by ``ResidualTranslationAllocator``.
        """
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            return float("nan"), float("nan"), np.full(2, np.nan), False
        eps = 1e-6
        bx_limit = self._g * np.sin(self._theta_max) - eps
        bx = float(np.clip(target[0], -bx_limit, bx_limit))
        theta = float(np.arcsin(bx / self._g))
        by_limit = self._g * np.cos(theta) * np.sin(self._phi_max) - eps
        by = float(np.clip(target[1], -by_limit, by_limit))
        phi = float(np.arcsin(-by / (self._g * np.cos(theta))))
        projected = np.array([bx, by], dtype=np.float64)
        exact_tilt = bool(np.linalg.norm(projected - target) <= 1e-9)
        return phi, theta, projected, exact_tilt
