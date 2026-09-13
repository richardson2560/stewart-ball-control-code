# src/sbc/perception/tactile.py

import numpy as np


class TactileProcessor:
    """
    Preprocesses raw tactile surface readings.
    Compensates for the Center of Pressure (CoP) displacement induced by
    rolling resistance deformation: delta_p = c_rr(N) * r * (v / ||v||).
    Validates unilateral contact threshold N >= F_act.
    """

    def __init__(
        self,
        ball_radius: float = 0.025,
        rolling_resistance_coeff: float = 0.0015,
        min_activation_force: float = 0.20,
        vel_regularization: float = 1e-3
    ) -> None:
        """
        Args:
            ball_radius: Sphere radius r [m].
            rolling_resistance_coeff: Baseline dimensionless rolling loss c_rr.
            min_activation_force: Minimum detectable normal force F_act = N_min [N].
            vel_regularization: Numerical smoothing velocity epsilon_v [m/s].
        """
        self._r: float = ball_radius
        self._c_rr: float = rolling_resistance_coeff
        self._f_act: float = min_activation_force
        self._eps_v: float = vel_regularization

        # Cache of previous valid contact position for dropout hold
        self._last_valid_pos = np.zeros(2, dtype=np.float64)
        self._contact_active: bool = False

    @property
    def contact_active(self) -> bool:
        """True if the normal force satisfies unilateral contact condition."""
        return self._contact_active

    def compute_cop_bias(self, velocity: np.ndarray, normal_force: float) -> np.ndarray:
        """
        Computes the physical CoP forward displacement vector d_p in platform frame P.
        """
        speed_norm = np.linalg.norm(velocity)
        unit_direction = velocity / np.sqrt(speed_norm**2 + self._eps_v**2)

        # Dynamic rolling resistance coefficient scaling with load (if non-constant)
        c_rr_eff = self._c_rr
        d_p = c_rr_eff * self._r * unit_direction
        return d_p

    def process(
        self,
        tactile_pos_raw: np.ndarray,
        normal_force: float,
        current_vel_est: np.ndarray
    ) -> np.ndarray:
        """
        Validates contact and strips CoP bias to recover the true sphere contact point rho.

        Args:
            tactile_pos_raw: Raw sensor coordinate in P [m], shape (2,).
            normal_force: Estimated normal reaction N [N].
            current_vel_est: Filtered ball relative velocity estimate v_rel [m/s], shape (2,).

        Returns:
            rho: True contact point coordinate on the platform [m], shape (2,).
        """
        self._contact_active = normal_force >= self._f_act

        if not self._contact_active:
            # When contact is lost or force is below detection, retain last known coordinate
            return self._last_valid_pos.copy()

        # Deduct CoP displacement
        cop_bias = self.compute_cop_bias(current_vel_est, normal_force)
        rho = tactile_pos_raw - cop_bias

        self._last_valid_pos = rho.copy()
        return rho