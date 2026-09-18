import numpy as np

from sbc.allocation.tilt_inversion import ResidualTranslationAllocator
from sbc.datatypes import (
    ActuatorCommandPacket,
    ActuatorMode,
    PlatformStatus,
    RawSensorPacket,
)
from sbc.interfaces.backends.coppelia import CoppeliaBackend
from sbc.actuation.joint_lead_comp import SafeJointTrajectoryIntegrator
from sbc.kinematics.platform import StewartKinematics
from sbc.models.reference_model import HurwitzReferenceModel
from sbc.controllers.sbc_full import SBCFullController
from sbc.datatypes import StateEstimatePacket
from sbc.allocation.tilt_inversion import TiltInverter
from sbc.perception.hermite_gp import HermiteGPFilter


def test_reference_reset_copies_read_only_packet_memory() -> None:
    packet = RawSensorPacket(
        timestamp=0.0,
        leg_positions=np.zeros(6),
        leg_velocities=np.zeros(6),
        tactile_pos_raw=np.array([0.03, -0.02]),
        tactile_force_estimate=1.0,
        tactile_active=True,
        bus_healthy=True,
    )
    assert not packet.tactile_pos_raw.flags.writeable

    model = HurwitzReferenceModel(cycle_time=0.002)
    model.reset(packet.tactile_pos_raw)
    position, velocity, acceleration = model.update(np.zeros(2))

    assert np.all(np.isfinite(np.hstack([position, velocity, acceleration])))
    assert model._rho_d.flags.writeable
    assert not np.shares_memory(model._rho_d, packet.tactile_pos_raw)


def test_measured_translation_is_not_silently_clipped() -> None:
    allocator = ResidualTranslationAllocator(max_offset=0.060)
    valid = allocator.synchronize_measured_state(
        np.array([0.061, 0.0, 0.0]), np.zeros(3)
    )
    assert not valid
    np.testing.assert_allclose(allocator.offset_I, [0.061, 0.0, 0.0])


def test_normal_displacement_is_not_charged_to_tangential_allocator() -> None:
    allocator = ResidualTranslationAllocator(max_offset=0.060)
    valid = allocator.synchronize_measured_state(
        np.array([0.010, -0.010, -0.0813]),
        np.zeros(3),
        np.eye(3),
    )
    assert valid
    np.testing.assert_allclose(allocator.offset_I, [0.010, -0.010, 0.0])


def test_observer_fallback_refuses_envelope_crossing() -> None:
    allocator = ResidualTranslationAllocator(
        cycle_time=0.01,
        max_acceleration=3.0,
        max_velocity=0.35,
        max_offset=0.060,
    )
    assert allocator.synchronize_measured_state(
        np.array([0.0599, 0.0, 0.0]), np.array([0.02, 0.0, 0.0])
    )
    before = allocator.offset_I
    assert not allocator.commit_realized(np.zeros(3))
    np.testing.assert_allclose(allocator.offset_I, before)


def test_rotation_round_trip_for_measured_platform_pose() -> None:
    expected = np.array([0.13, -0.09, 0.07])
    rotation = StewartKinematics.get_rotation_matrix(*expected)
    recovered = StewartKinematics.rotation_matrix_to_zyx(rotation)
    np.testing.assert_allclose(recovered, expected, atol=1e-12, rtol=0.0)


def test_coppelia_motion_is_reported_in_base_frame() -> None:
    class FakeSim:
        handle_world = -1

        @staticmethod
        def getObjectPosition(handle, relative):
            return [0.01, -0.02, 0.40]

        @staticmethod
        def getObjectMatrix(handle, relative):
            if handle == 10:  # plate relative to base
                return [1.0, 0.0, 0.0, 0.01,
                        0.0, 1.0, 0.0, -0.02,
                        0.0, 0.0, 1.0, 0.40]
            # Base yawed +90 degrees in world.
            return [0.0, -1.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0]

        @staticmethod
        def getObjectVelocity(handle):
            return [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]

    backend = CoppeliaBackend(joint_command_mode="dynamic")
    backend._sim = FakeSim()
    backend._h_plate = 10
    backend._h_base = 20
    backend._status = PlatformStatus.OPERATIONAL
    position, rotation, twist = backend.read_platform_motion()
    np.testing.assert_allclose(position, [0.01, -0.02, 0.40])
    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(twist, [1.0, 0.0, 0.0, 0.0, 0.0, 2.0], atol=1e-12)


def test_kinematic_platform_twist_comes_from_pose_difference() -> None:
    class FakeSim:
        handle_world = -1

        def __init__(self):
            self.position = np.zeros(3)
            self.angle = 0.0

        def getObjectPosition(self, handle, relative):
            return self.position.tolist()

        def getObjectMatrix(self, handle, relative):
            c, s = np.cos(self.angle), np.sin(self.angle)
            return [c, -s, 0.0, self.position[0],
                    s, c, 0.0, self.position[1],
                    0.0, 0.0, 1.0, self.position[2]]

        @staticmethod
        def getObjectVelocity(handle):
            # Deliberately wrong/zero, as can occur after kinematic writes.
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    backend = CoppeliaBackend(cycle_time=0.002, joint_command_mode="kinematic")
    backend._sim = FakeSim()
    backend._h_plate = 10
    backend._h_base = 20
    backend._status = PlatformStatus.OPERATIONAL
    _, _, first_twist = backend.read_platform_motion()
    np.testing.assert_allclose(first_twist, np.zeros(6))

    backend._sim.position = np.array([0.0002, -0.0004, 0.0001])
    backend._sim.angle = 0.002
    _, _, twist = backend.read_platform_motion()
    np.testing.assert_allclose(twist[:3], [0.1, -0.2, 0.05], atol=1e-12)
    np.testing.assert_allclose(twist[3:], [0.0, 0.0, 1.0], atol=1e-10)


def test_coppelia_joint_limits_come_from_scene() -> None:
    class FakeSim:
        @staticmethod
        def getJointInterval(handle):
            return False, [-0.10 + 0.01 * handle, 0.40]

    backend = CoppeliaBackend()
    backend._sim = FakeSim()
    backend._h_motors = [1, 2, 3, 4, 5, 6]
    lower, upper = backend.read_joint_limits()
    np.testing.assert_allclose(lower, [-0.09, -0.08, -0.07, -0.06, -0.05, -0.04])
    np.testing.assert_allclose(upper, lower + 0.40)


def test_scene_closure_tube_accepts_measured_residual_but_not_drift() -> None:
    backend = CoppeliaBackend(joint_command_mode="kinematic")
    assert 72.94e-6 < backend.kinematic_closure_tolerance
    assert 0.6e-3 > backend.kinematic_closure_tolerance
    dynamic = CoppeliaBackend(joint_command_mode="dynamic")
    assert dynamic.kinematic_closure_tolerance == 2e-3


def test_coppelia_actuation_advances_exactly_one_synchronous_step() -> None:
    class FakeSim:
        def __init__(self):
            self.positions = []
            self.position_by_handle = {}
            self.time = 0.0

        def setJointPosition(self, handle, value):
            self.positions.append((handle, value))
            self.position_by_handle[handle] = value

        def getJointPosition(self, handle):
            return self.position_by_handle[handle]

        def getSimulationTime(self):
            return self.time

    class FakeIK:
        def __init__(self):
            self.calls = 0

        def handleGroup(self, environment, group, options):
            self.calls += 1

    class FakeClient:
        def __init__(self, simulator):
            self.steps = 0
            self.simulator = simulator

        def step(self):
            self.steps += 1
            self.simulator.time += 0.002

    backend = CoppeliaBackend(joint_command_mode="kinematic")
    backend._status = PlatformStatus.OPERATIONAL
    backend._sim = FakeSim()
    backend._sim_ik = FakeIK()
    backend._client = FakeClient(backend._sim)
    backend._h_motors = [1, 2, 3, 4, 5, 6]
    backend._ik_env = 10
    backend._ik_group = 20
    command = ActuatorCommandPacket(
        timestamp=0.0,
        mode=ActuatorMode.CSP,
        q_send=np.linspace(0.0, 0.05, 6),
        dot_q_send=np.zeros(6),
        u_q_applied=np.zeros(6),
        slack_value=0.0,
    )
    assert backend.write_actuators(command)
    assert len(backend._sim.positions) == 6
    assert backend._sim_ik.calls == 1
    assert backend._client.steps == 1
    assert backend.last_step_duration == 0.002
    assert backend._has_kinematic_command
    np.testing.assert_allclose(backend._last_kinematic_qdot, command.dot_q_send)


def test_geometric_contact_uses_hysteresis_not_ten_micron_chatter() -> None:
    backend = CoppeliaBackend(ball_radius=0.025)
    assert backend._update_contact_state(0.02432)
    # The supplied failing log toggled at 28.01 mm under a 28 mm threshold.
    assert backend._update_contact_state(0.02801)
    assert not backend._update_contact_state(0.02851)
    # Re-acquisition is deliberately stricter than retention.
    assert not backend._update_contact_state(0.02800)
    assert backend._update_contact_state(0.02749)


def test_joint_pose_tracking_closes_acceleration_level_drift() -> None:
    dt = 0.002
    tracker = SafeJointTrajectoryIntegrator(
        dt,
        -np.ones(6),
        np.ones(6),
        tracking_frequency=18.0,
        tracking_damping=1.0,
    )
    q = np.zeros(6)
    q_dot = np.zeros(6)
    q_des = np.full(6, 0.01)
    for _ in range(int(1.0 / dt)):
        acceleration = tracker.tracking_acceleration(
            q, q_dot, q_des, np.zeros(6), np.zeros(6)
        )
        q = q + dt * q_dot + 0.5 * dt**2 * acceleration
        q_dot = q_dot + dt * acceleration
    np.testing.assert_allclose(q, q_des, atol=2e-7, rtol=0.0)
    np.testing.assert_allclose(q_dot, np.zeros(6), atol=4e-6, rtol=0.0)


def test_joint_tracker_rejects_discrete_unstable_gain() -> None:
    import pytest

    with pytest.raises(ValueError):
        SafeJointTrajectoryIntegrator(
            0.1,
            -np.ones(6),
            np.ones(6),
            tracking_frequency=100.0,
            tracking_damping=1.0,
        )


def test_allocator_generates_inward_reentry_acceleration() -> None:
    allocator = ResidualTranslationAllocator(max_offset=0.060)
    assert not allocator.synchronize_measured_state(
        np.array([0.070, 0.0, 0.0]), np.zeros(3)
    )
    applied, _, _ = allocator.allocate(
        B_des=np.zeros(2),
        rotation_P_to_I=np.eye(3),
        gravity_parallel=np.zeros(2),
        alpha_P=np.zeros(3),
        omega_P=np.zeros(3),
        rho=np.zeros(2),
        ball_radius=0.025,
        commit=False,
    )
    assert applied[0] < 0.0
    assert abs(applied[1]) < 1e-12


def test_smooth_capture_does_not_drive_ball_to_opposite_wall() -> None:
    dt = 0.002
    rho = np.array([0.018, -0.230])
    velocity = np.zeros(2)
    initial_radius = float(np.linalg.norm(rho))
    reference = HurwitzReferenceModel(
        omega_n=7.0, zeta=1.0, max_acceleration=2.5, cycle_time=dt
    )
    reference.reset(rho)
    controller = SBCFullController(kp=6.5, kd=2.8, ball_mass=0.06545)
    inverter = TiltInverter(
        max_roll=np.radians(16.0), max_pitch=np.radians(16.0)
    )

    for step in range(int(3.0 / dt)):
        state = StateEstimatePacket(
            step * dt,
            np.array([0.0, 0.0, 0.4]),
            np.eye(3),
            np.zeros(6),
            np.zeros(3),
            rho,
            velocity,
            0.0,
            True,
        )
        rho_d, dot_rho_d, ddot_rho_d = reference.update(np.zeros(2))
        virtual = controller.compute_control(
            state, rho_d, dot_rho_d, ddot_rho_d
        )
        _, _, achieved_B, _ = inverter.project_preferred_gravity(virtual.B_des)
        acceleration = (5.0 / 7.0) * achieved_B
        velocity = velocity + dt * acceleration
        rho = rho + dt * velocity
        assert np.linalg.norm(rho) <= initial_radius + 1e-6

    assert np.linalg.norm(rho) < 0.01


def test_hermite_dropout_does_not_pollute_uniform_time_window() -> None:
    dt = 0.002
    filt = HermiteGPFilter(window_size=5, cycle_time=dt)
    for index in range(8):
        position, velocity, _, _ = filt.update(
            np.array([index * dt, 0.0]), 0.1 * index, measurement_valid=True
        )
    count_before = filt._count
    held_position = position.copy()

    for index in range(3):
        position, velocity, yaw, _ = filt.update(
            np.array([10.0 + index, 10.0]),
            0.8 + 0.1 * index,
            measurement_valid=False,
        )
        np.testing.assert_allclose(position, held_position)
        np.testing.assert_allclose(velocity, np.zeros(2))
        assert np.isfinite(yaw)
    assert filt._count == count_before
    assert filt.dropout_active

    position, velocity, _, _ = filt.update(
        np.array([0.020, 0.0]), 1.1, measurement_valid=True
    )
    np.testing.assert_allclose(position, [0.020, 0.0])
    np.testing.assert_allclose(velocity, np.zeros(2))
    assert filt.reinitialized
    assert not filt.dropout_active
