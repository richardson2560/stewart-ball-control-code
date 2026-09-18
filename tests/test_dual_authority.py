import numpy as np

from sbc.allocation.tilt_inversion import ResidualTranslationAllocator, TiltInverter


def test_residual_translation_exactly_closes_drive_identity() -> None:
    rng = np.random.default_rng(17)
    for _ in range(300):
        B = rng.normal(size=2)
        gravity = rng.normal(size=2)
        alpha = rng.normal(size=3)
        omega = rng.normal(size=3)
        rho = 0.1 * rng.normal(size=2)
        radius = 0.025
        acceleration = ResidualTranslationAllocator.required_acceleration(
            B, gravity, alpha, omega, rho, radius
        )
        r_bp = np.array([rho[0], rho[1], radius])
        reconstructed = (
            gravity - acceleration
            - np.cross(alpha, np.array([rho[0], rho[1], 0.0]))[:2]
            - np.cross(omega, np.cross(omega, r_bp))[:2]
        )
        np.testing.assert_allclose(reconstructed, B, atol=1e-12, rtol=1e-12)


def test_strict_tilt_inverse_rejects_instead_of_clipping() -> None:
    inverter = TiltInverter(max_roll=np.radians(10), max_pitch=np.radians(10))
    phi, theta, valid = inverter.invert(np.array([10.0, 0.0]))
    assert not valid
    assert np.isnan(phi) and np.isnan(theta)


def test_preferred_gravity_projection_is_inside_exact_domain() -> None:
    inverter = TiltInverter(max_roll=np.radians(10), max_pitch=np.radians(12))
    phi, theta, achieved, tilt_only = inverter.project_preferred_gravity(np.array([10.0, -10.0]))
    assert not tilt_only
    assert abs(phi) < np.radians(10.0)
    assert abs(theta) < np.radians(12.0)
    assert np.all(np.isfinite(achieved))


def test_unlimited_allocator_preserves_tangent_acceleration_under_tilt() -> None:
    allocator = ResidualTranslationAllocator(
        cycle_time=0.002,
        max_acceleration=1e6,
        max_jerk=1e12,
        max_velocity=1e6,
        max_offset=1e6,
    )
    phi, theta, psi = 0.13, -0.08, 0.21
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    rotation = np.array([
        [cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp],
        [sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp],
        [-st, ct*sp, ct*cp],
    ])
    B = np.array([1.2, -0.7])
    gravity = (rotation.T @ np.array([0.0, 0.0, -9.81]))[:2]
    alpha = np.array([0.3, -0.2, 0.1])
    omega = np.array([0.4, 0.2, -0.1])
    rho = np.array([0.03, -0.02])
    applied, residual, exact = allocator.allocate(
        B, rotation, gravity, alpha, omega, rho, 0.025
    )
    assert exact
    np.testing.assert_allclose(residual, np.zeros(2), atol=1e-12)
