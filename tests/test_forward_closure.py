import numpy as np

from sbc.kinematics.forward_6d import LocalForwardKinematics
from sbc.kinematics.platform import StewartKinematics


def test_forward_solver_returns_same_local_branch() -> None:
    base_angles = np.radians([10, 50, 130, 170, 250, 290])
    plate_angles = np.radians([20, 100, 140, 220, 260, 340])
    base = np.c_[0.30*np.cos(base_angles), 0.30*np.sin(base_angles), np.zeros(6)]
    plate = np.c_[0.16*np.cos(plate_angles), 0.16*np.sin(plate_angles), np.zeros(6)]
    translation = np.array([0.004, -0.003, 0.405])
    kin = StewartKinematics(base, plate, np.zeros(6), initial_translation=np.array([0.0, 0.0, 0.4]))
    q, _, _ = kin.inverse_kinematics(0.03, -0.02, 0.01, translation)
    seed_R = kin.get_rotation_matrix(0.03, -0.02, 0.01)
    result = LocalForwardKinematics(kin).solve(q, translation, seed_R)
    assert result.valid
    assert result.residual_norm < 1e-8

