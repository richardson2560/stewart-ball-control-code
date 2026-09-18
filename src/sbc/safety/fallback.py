"""One-cycle fallback policy with explicit admission checks."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FallbackDecision:
    acceleration: np.ndarray
    one_cycle_admissible: bool
    reason: str


class OneCycleFallback:
    """Reuse the last admitted acceleration only inside a tightened envelope."""

    def __init__(self, cycle_time: float, acceleration_limit: float, stroke_margin: float = 1e-4) -> None:
        self._dt = float(cycle_time)
        self._u_max = float(acceleration_limit)
        self._margin = float(stroke_margin)

    def select(
        self,
        previous_acceleration: np.ndarray,
        q: np.ndarray,
        q_dot: np.ndarray,
        q_min: np.ndarray,
        q_max: np.ndarray,
    ) -> FallbackDecision:
        previous = np.asarray(previous_acceleration, dtype=np.float64)
        finite = previous.shape == (6,) and np.all(np.isfinite(previous))
        bounded = finite and np.max(np.abs(previous)) <= self._u_max
        if bounded:
            q_next = q + self._dt * q_dot + 0.5 * self._dt**2 * previous
            inside = np.all(q_next >= q_min + self._margin) and np.all(q_next <= q_max - self._margin)
            if inside:
                return FallbackDecision(previous.copy(), True, "hold_previous_one_cycle")

        braking = np.clip(-8.0 * q_dot, -self._u_max, self._u_max)
        return FallbackDecision(braking, False, "failsafe_braking_not_certified")

