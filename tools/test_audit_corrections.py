#!/usr/bin/env python3
"""Unit checks for the physical-parameter and tactile-semantics audit fixes."""

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from sbc.config import load_controller_config
from sbc.perception.tactile import TactileProcessor
from sbc.safety.socp_filter import ClarabelSafetyFilter


BALL_MASS = 0.065449846949792
BALL_RADIUS = 0.025
PLATE_RADIUS = 0.25


def test_shared_physical_configuration() -> None:
    config = load_controller_config()
    ball = config["physics"]["ball"]
    platform = config["physics"]["platform"]

    assert np.isclose(ball["mass"], BALL_MASS)
    assert np.isclose(ball["radius"], BALL_RADIUS)
    assert np.isclose(platform["plate_radius"], PLATE_RADIUS)


def test_geometric_projection_bypasses_cop_compensation() -> None:
    processor = TactileProcessor(
        ball_radius=BALL_RADIUS,
        rolling_resistance_coeff=0.0015,
        min_activation_force=0.10,
    )
    raw_position = np.array([0.10, -0.04], dtype=np.float64)
    velocity = np.array([1.0, 0.0], dtype=np.float64)

    geometric_position = processor.process(
        raw_position, 1.0, velocity, is_geometric=True
    )
    physical_cop_position = processor.process(
        raw_position, 1.0, velocity, is_geometric=False
    )

    assert np.allclose(geometric_position, raw_position)
    assert physical_cop_position[0] < raw_position[0]


def test_safety_filter_uses_scene_plate_radius() -> None:
    safety_filter = ClarabelSafetyFilter(
        ball_mass=BALL_MASS,
        ball_radius=BALL_RADIUS,
    )
    assert np.isclose(safety_filter._R_sq, PLATE_RADIUS**2)


def test_coppelia_configuration_uses_kinematic_csp() -> None:
    config = load_controller_config()
    coppelia = config["coppelia"]

    assert coppelia["joint_command_mode"] == "kinematic"
    assert coppelia["actuator_mode"] == "CSP"
    assert coppelia["scene_name"] == "stewart_platform_ideal.ttt"


def main() -> None:
    test_shared_physical_configuration()
    test_geometric_projection_bypasses_cop_compensation()
    test_safety_filter_uses_scene_plate_radius()
    test_coppelia_configuration_uses_kinematic_csp()
    print("Audit correction tests passed.")


if __name__ == "__main__":
    main()
