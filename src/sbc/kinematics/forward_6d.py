"""Local forward kinematics with branch and closure verification."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sbc.kinematics.platform import StewartKinematics


@dataclass(frozen=True)
class ForwardKinematicsResult:
    translation: np.ndarray
    rotation: np.ndarray
    residual_norm: float
    jacobian_condition: float
    valid: bool


class LocalForwardKinematics:
    """Solve only in the neighbourhood of a supplied, continuous branch seed."""

    def __init__(
        self,
        kinematics: StewartKinematics,
        closure_tolerance: float = 2e-6,
        condition_limit: float = 1e5,
        branch_translation_radius: float = 0.03,
        branch_rotation_radius: float = np.radians(8.0),
    ) -> None:
        self._kin = kinematics
        self._tol = float(closure_tolerance)
        self._condition_limit = float(condition_limit)
        self._translation_radius = float(branch_translation_radius)
        self._rotation_radius = float(branch_rotation_radius)

    def solve(self, q_target: np.ndarray, seed_translation: np.ndarray, seed_rotation: np.ndarray) -> ForwardKinematicsResult:
        q_target = np.asarray(q_target, dtype=np.float64)
        seed_translation = np.asarray(seed_translation, dtype=np.float64)
        seed_rotation = np.asarray(seed_rotation, dtype=np.float64)
        if q_target.shape != (6,) or not np.all(np.isfinite(q_target)):
            raise ValueError("q_target must be a finite six-vector.")
        seed_rotvec = Rotation.from_matrix(seed_rotation).as_rotvec()
        x0 = np.hstack([seed_translation, seed_rotvec])
        lower = np.hstack([
            seed_translation - self._translation_radius,
            seed_rotvec - self._rotation_radius,
        ])
        upper = np.hstack([
            seed_translation + self._translation_radius,
            seed_rotvec + self._rotation_radius,
        ])

        def residual(x: np.ndarray) -> np.ndarray:
            rotation = Rotation.from_rotvec(x[3:]).as_matrix()
            # Convert through ZYX only at the existing kinematics interface.
            psi, theta, phi = Rotation.from_matrix(rotation).as_euler("ZYX")
            q, _, _ = self._kin.inverse_kinematics(phi, theta, psi, x[:3])
            return q - q_target

        solution = least_squares(residual, x0, bounds=(lower, upper), method="trf", max_nfev=40)
        rotation = Rotation.from_rotvec(solution.x[3:]).as_matrix()
        psi, theta, phi = Rotation.from_matrix(rotation).as_euler("ZYX")
        jacobian = self._kin.compute_inverse_jacobian(phi, theta, psi, solution.x[:3])
        condition = float(np.linalg.cond(jacobian))
        residual_norm = float(np.linalg.norm(solution.fun, ord=np.inf))
        valid = bool(solution.success and residual_norm <= self._tol and condition <= self._condition_limit)
        return ForwardKinematicsResult(solution.x[:3].copy(), rotation, residual_norm, condition, valid)

