import numpy as np

from sbc.kinematics.platform import StewartKinematics


def _regular_kinematics() -> StewartKinematics:
    base_angles = np.radians([10, 50, 130, 170, 250, 290])
    plate_angles = np.radians([20, 100, 140, 220, 260, 340])
    base = np.c_[0.30 * np.cos(base_angles), 0.30 * np.sin(base_angles), np.zeros(6)]
    plate = np.c_[0.16 * np.cos(plate_angles), 0.16 * np.sin(plate_angles), np.zeros(6)]
    return StewartKinematics(base, plate, np.zeros(6), initial_translation=np.array([0.0, 0.0, 0.40]))


def test_analytic_jacobian_dot_matches_rigid_motion_difference() -> None:
    kin = _regular_kinematics()
    phi, theta, psi = 0.12, -0.09, 0.07
    translation = np.array([0.01, -0.02, 0.40])
    velocity = np.array([0.03, -0.02, 0.01])
    omega = np.array([0.2, -0.1, 0.15])
    rotation = kin.get_rotation_matrix(phi, theta, psi)

    def skew(x: np.ndarray) -> np.ndarray:
        return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])

    def jacobian(R: np.ndarray, T: np.ndarray) -> np.ndarray:
        rotated = kin.a @ R.T
        legs = T + rotated - kin.b
        directions = legs / np.linalg.norm(legs, axis=1)[:, None]
        return np.hstack([directions, np.cross(rotated, directions)])

    eps = 1e-6
    numerical = (
        jacobian((np.eye(3) + eps * skew(omega)) @ rotation, translation + eps * velocity)
        - jacobian((np.eye(3) - eps * skew(omega)) @ rotation, translation - eps * velocity)
    ) / (2.0 * eps)
    analytical = kin.compute_inverse_jacobian_dot(phi, theta, psi, translation, velocity, omega)
    np.testing.assert_allclose(analytical, numerical, atol=1e-7, rtol=1e-7)


def test_actuator_acceleration_contains_jdot_term() -> None:
    kin = _regular_kinematics()
    translation = np.array([0.0, 0.0, 0.40])
    velocity = np.array([0.02, -0.01, 0.0])
    omega = np.array([0.1, -0.05, 0.03])
    _, _, q_ddot = kin.actuator_trajectory(
        0.05, -0.04, 0.02, translation, velocity, omega,
        np.zeros(3), np.zeros(3)
    )
    assert np.linalg.norm(q_ddot) > 0.0

