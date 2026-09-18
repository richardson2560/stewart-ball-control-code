# src/sbc/perception/hermite_gp.py

import numpy as np
from typing import Tuple


class HermiteGPFilter:
    """
    Causal, moment-constrained FIR filter derived from Gaussian Process regression.
    Precomputes weights satisfying exact polynomial moment reproduction up to degree 2.
    Executes in O(n_FIR) time per cycle without online matrix factorizations.
    """

    def __init__(self, window_size: int = 9, cycle_time: float = 0.002) -> None:
        """
        Args:
            window_size: Number of causal history samples n_FIR (must be >= 3).
            cycle_time: Sampling period Ts [s].
        """
        assert window_size >= 3, "window_size must be at least 3 for quadratic reproduction"
        self._n: int = window_size
        self._dt: float = cycle_time

        # Circular buffers: position in P (2D) and raw yaw rate (1D)
        self._buf_pos = np.zeros((self._n, 2), dtype=np.float64)
        self._buf_yaw = np.zeros(self._n, dtype=np.float64)
        self._count: int = 0
        self._yaw_count: int = 0
        self._dropout_active: bool = False
        self._reinitialized: bool = False
        self._last_position = np.zeros(2, dtype=np.float64)
        self._last_velocity = np.zeros(2, dtype=np.float64)

        # Weights: w_val for smoothing (d=0), w_der for derivative (d=1)
        self._w_val: np.ndarray = np.zeros(self._n, dtype=np.float64)
        self._w_der: np.ndarray = np.zeros(self._n, dtype=np.float64)
        self._compute_weights()

    def _compute_weights(self) -> None:
        """
        Precomputes the FIR weight vectors w^(0) and w^(1) by solving the 
        moment constraint equality: A_tau * w^(d) = b_d.
        """
        # Relative time offsets tau_j in [- (n-1)*dt, 0], with current endpoint at tau = 0
        tau = np.linspace(-(self._n - 1) * self._dt, 0.0, self._n, dtype=np.float64)

        # Moment constraint matrix for degree 0, 1, and 2: shape (3, n)
        A_tau = np.vstack([
            np.ones(self._n, dtype=np.float64),
            tau,
            tau**2
        ])

        # Target moment vectors b_d: [sum w, sum w*tau, sum w*tau^2]
        b_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # Evaluation at tau = 0
        b_1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # 1st derivative at tau = 0

        # Minimum-norm pseudo-inverse solution: w = A_tau.T * (A_tau * A_tau.T)^(-1) * b
        ATA_inv = np.linalg.inv(A_tau @ A_tau.T)
        self._w_val = A_tau.T @ (ATA_inv @ b_0)
        self._w_der = A_tau.T @ (ATA_inv @ b_1)

    def reset(self) -> None:
        """Clears rolling buffers."""
        self._buf_pos.fill(0.0)
        self._buf_yaw.fill(0.0)
        self._count = 0
        self._yaw_count = 0
        self._dropout_active = False
        self._reinitialized = False
        self._last_position.fill(0.0)
        self._last_velocity.fill(0.0)

    @property
    def dropout_active(self) -> bool:
        return self._dropout_active

    @property
    def reinitialized(self) -> bool:
        return self._reinitialized

    def update(
        self,
        ball_pos_raw: np.ndarray,
        yaw_rate_raw: float,
        measurement_valid: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Pushes new raw observations and computes smoothed states and causal derivatives.

        Args:
            ball_pos_raw: Measured ball contact coordinate in P, shape (2,).
            yaw_rate_raw: Reconstructed platform yaw velocity Omega_z [rad/s].

        Returns:
            pos_smoothed: Filtered ball position in P [m], shape (2,)
            vel_estimated: Estimated relative ball velocity v_rel in P [m/s], shape (2,)
            yaw_smoothed: Filtered body yaw rate Omega_z [rad/s]
            alpha_z_estimated: Estimated body yaw acceleration alpha_z [rad/s^2]
        """
        ball_pos_raw = np.asarray(ball_pos_raw, dtype=np.float64)
        if ball_pos_raw.shape != (2,) or not np.all(np.isfinite(ball_pos_raw)):
            raise ValueError("ball_pos_raw must be a finite 2-vector.")
        if not np.isfinite(yaw_rate_raw):
            raise ValueError("yaw_rate_raw must be finite.")
        self._reinitialized = False

        # Yaw is a platform measurement and remains valid independently of
        # tactile contact, so it has its own sample counter.
        self._buf_yaw[:-1] = self._buf_yaw[1:]
        self._buf_yaw[-1] = yaw_rate_raw
        self._yaw_count += 1
        if self._yaw_count < self._n:
            yaw_smoothed = float(np.mean(self._buf_yaw[-self._yaw_count:]))
            alpha_z_estimated = 0.0
        else:
            yaw_smoothed = float(np.dot(self._w_val, self._buf_yaw))
            alpha_z_estimated = float(np.dot(self._w_der, self._buf_yaw))

        # Missing tactile samples are not repeated into a uniformly sampled
        # FIR window.  Doing so creates a false zero-velocity segment and an
        # impulsive derivative at recontact.
        if not measurement_valid:
            self._dropout_active = True
            return (
                self._last_position.copy(),
                np.zeros(2, dtype=np.float64),
                yaw_smoothed,
                alpha_z_estimated,
            )

        if self._dropout_active:
            # Restart the position warm-up on the new contact epoch.  Velocity
            # stays explicitly untrusted/zero until a uniform window exists.
            self._buf_pos[:] = ball_pos_raw
            self._count = 1
            self._dropout_active = False
            self._reinitialized = True
            self._last_position = ball_pos_raw.copy()
            self._last_velocity.fill(0.0)
            return (
                self._last_position.copy(),
                self._last_velocity.copy(),
                yaw_smoothed,
                alpha_z_estimated,
            )

        # Roll valid tactile data and insert the newest uniformly timed sample.
        self._buf_pos[:-1] = self._buf_pos[1:]
        self._buf_pos[-1] = ball_pos_raw

        self._count += 1

        # While filling buffer on cold-start, extrapolate using available samples
        if self._count < self._n:
            fill = self._count
            weights_val = self._w_val[-fill:] / np.sum(self._w_val[-fill:])
            pos_smoothed = np.sum(self._buf_pos[-fill:] * weights_val[:, np.newaxis], axis=0)
            vel_estimated = np.zeros(2, dtype=np.float64)
            self._last_position = pos_smoothed.copy()
            self._last_velocity = vel_estimated.copy()
            return pos_smoothed, vel_estimated, yaw_smoothed, alpha_z_estimated

        # Dot product evaluations: O(n_FIR) operations
        pos_smoothed = np.dot(self._w_val, self._buf_pos)
        vel_estimated = np.dot(self._w_der, self._buf_pos)

        self._last_position = pos_smoothed.copy()
        self._last_velocity = vel_estimated.copy()
        return pos_smoothed, vel_estimated, yaw_smoothed, alpha_z_estimated
