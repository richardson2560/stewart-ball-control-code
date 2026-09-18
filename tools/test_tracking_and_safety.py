# tools/test_tracking_and_safety.py

import argparse
import sys
import time
import numpy as np

from sbc.datatypes import ActuatorCommandPacket, ActuatorMode, StateEstimatePacket
from sbc.interfaces.factory import BackendFactory
from sbc.interfaces.headless_launcher import CoppeliaLauncher
from sbc.kinematics.platform import StewartKinematics
from sbc.perception.hermite_gp import HermiteGPFilter
from sbc.perception.tactile import TactileProcessor
from sbc.models.reference_model import HurwitzReferenceModel
from sbc.controllers.sbc_full import SBCFullController
from sbc.allocation.tilt_inversion import TiltInverter, ResidualTranslationAllocator
from sbc.allocation.yaw_nulling import YawNullingAllocator
from sbc.safety.socp_filter import ClarabelSafetyFilter
from sbc.actuation.joint_lead_comp import SafeJointTrajectoryIntegrator


def test_residual_translation_closes_B_identity() -> None:
    """The new allocator must recover B_des to floating-point precision."""
    rng = np.random.default_rng(17)
    for _ in range(200):
        B_des = rng.normal(size=2)
        gravity_parallel = rng.normal(size=2)
        alpha = rng.normal(size=3)
        omega = rng.normal(size=3)
        rho = 0.1 * rng.normal(size=2)
        radius = 0.025
        A_xy = ResidualTranslationAllocator.required_acceleration(
            B_des, gravity_parallel, alpha, omega, rho, radius
        )
        r_bp = np.array([rho[0], rho[1], radius])
        recovered = (
            gravity_parallel - A_xy
            - np.cross(alpha, np.array([rho[0], rho[1], 0.0]))[:2]
            - np.cross(omega, np.cross(omega, r_bp))[:2]
        )
        np.testing.assert_allclose(recovered, B_des, atol=1e-12, rtol=1e-12)


def test_inverse_jacobian_dot_matches_finite_difference() -> None:
    """Independent numerical check of the analytical high-speed J_inv_dot."""
    rng = np.random.default_rng(23)
    base = rng.normal(size=(6, 3))
    anchors = 0.1 * rng.normal(size=(6, 3))
    kin = StewartKinematics(base, anchors, np.zeros(6), initial_translation=np.array([0.0, 0.0, 0.4]))
    phi, theta, psi = 0.12, -0.09, 0.07
    translation = np.array([0.01, -0.02, 0.4])
    velocity = np.array([0.03, -0.02, 0.01])
    omega = np.array([0.2, -0.1, 0.15])
    rotation = kin.get_rotation_matrix(phi, theta, psi)

    def skew(x: np.ndarray) -> np.ndarray:
        return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])

    def jacobian_from_pose(R: np.ndarray, T: np.ndarray) -> np.ndarray:
        rotated = anchors @ R.T
        legs = T + rotated - base
        directions = legs / np.linalg.norm(legs, axis=1)[:, None]
        return np.hstack([directions, np.cross(rotated, directions)])

    eps = 1e-6
    R_plus = (np.eye(3) + eps * skew(omega)) @ rotation
    R_minus = (np.eye(3) - eps * skew(omega)) @ rotation
    numerical = (
        jacobian_from_pose(R_plus, translation + eps * velocity)
        - jacobian_from_pose(R_minus, translation - eps * velocity)
    ) / (2.0 * eps)
    analytical = kin.compute_inverse_jacobian_dot(
        phi, theta, psi, translation, velocity, omega
    )
    np.testing.assert_allclose(analytical, numerical, atol=1e-7, rtol=1e-7)


def main() -> None:
    parser = argparse.ArgumentParser(description="SBC v30 - Verified Rigorous Closed-Loop Tracking.")
    parser.add_argument("--duration", type=float, default=15.0, help="Test duration [s]. Default: 15.0 s.")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s] (500 Hz).")
    parser.add_argument("--radius", type=float, default=0.08, help="Circle radius [m]. Default: 0.08 m.")
    parser.add_argument("--speed", type=float, default=1.2, help="Orbit speed [rad/s]. Default: 1.2 rad/s.")
    parser.add_argument("--max-tilt", type=float, default=14.0, help="Tilt saturation limit [deg]. Default: 14.0 deg.")
    parser.add_argument("--safety", action="store_true", help="Enable Clarabel SOCP Safety Filter.")
    parser.add_argument("--no-auto-launch", action="store_true", help="Connect to running CoppeliaSim.")
    parser.add_argument("--gui", action="store_true", help="Run with visual GUI enabled.")
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
        process = CoppeliaLauncher.start(
            scene_name="stewart_platform_ideal.ttt",
            headless=not args.gui
        )
        time.sleep(2.5)

    BALL_MASS = 0.065449846949792
    BALL_RADIUS = 0.025
    GRAVITY = 9.81

    config = {
        "cycle_time": args.dt,
        "ball_mass": BALL_MASS,
        "ball_radius": BALL_RADIUS,
        "gravity": GRAVITY,
        "coppelia": {
            "host": "localhost",
            "port": 23000,
            "joint_command_mode": "kinematic",
            "actuator_mode": "CSP",
        }
    }
    backend = BackendFactory.create("coppelia", config)

    try:
        backend.connect()
        b_anchors, p_anchors, l_offsets, T_p_init = backend.extract_kinematic_parameters()

        # 1. Subsystems Initialization matching Monograph
        kinematics = StewartKinematics(b_anchors, p_anchors, l_offsets, initial_translation=T_p_init)
        tactile_proc = TactileProcessor(ball_radius=BALL_RADIUS, min_activation_force=0.10)
        filter_gp = HermiteGPFilter(window_size=11, cycle_time=args.dt)
        ref_model = HurwitzReferenceModel(omega_n=5.0, zeta=1.0, max_acceleration=1.5, cycle_time=args.dt)

        controller = SBCFullController(kp=5.5, kd=2.4, ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, gravity=GRAVITY)

        tilt_limit_rad = np.radians(args.max_tilt)
        inverter = TiltInverter(gravity=GRAVITY, max_roll=tilt_limit_rad, max_pitch=tilt_limit_rad, ball_radius=BALL_RADIUS)
        translation_allocator = ResidualTranslationAllocator(cycle_time=args.dt)
        yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))

        joint_limits = backend.read_joint_limits()
        q_limit_min, q_limit_max = (
            (None, None) if joint_limits is None else joint_limits
        )
        safety_filter = ClarabelSafetyFilter(
            ball_mass=BALL_MASS,
            ball_radius=BALL_RADIUS,
            plate_radius=0.25,
            q_min=q_limit_min,
            q_max=q_limit_max,
        ) if args.safety else None
        q_min = (
            np.full(6, -0.35)
            if safety_filter is None and q_limit_min is None
            else (q_limit_min if safety_filter is None else safety_filter.q_min)
        )
        q_max = (
            np.full(6, 0.35)
            if safety_filter is None and q_limit_max is None
            else (q_limit_max if safety_filter is None else safety_filter.q_max)
        )
        joint_integrator = SafeJointTrajectoryIntegrator(args.dt, q_min, q_max)

        init_sensor = backend.read_sensors()
        ref_model.reset(initial_pos=init_sensor.tactile_pos_raw)

        a_tilt_capacity = (5.0 / 7.0) * GRAVITY * np.sin(tilt_limit_rad)
        a_centripetal = (args.speed ** 2) * args.radius
        print(f"[+] Max Tilt: {args.max_tilt:.1f} deg -> Available Acceleration: {a_tilt_capacity:.2f} m/s^2.")
        print(f"[+] Orbit Demand: a_cen = {a_centripetal:.2f} m/s^2 (R = {args.radius * 100:.1f} cm, w = {args.speed:.2f} rad/s).")
        print(f"[+] Safety Filter (Clarabel SOCP): {'ENABLED' if args.safety else 'DISABLED'}.")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 110)
        print(f"{'Time [s]':>8} | {'Track Err [mm]':>14} | {'Ball Pos [m]':>18} | {'Ref Pos [m]':>18} | {'N [N]':>7} | {'Safety':>8}")
        print("=" * 110)

        vel_est = np.zeros(2, dtype=np.float64)
        phi_filt = 0.0
        theta_filt = 0.0
        max_tilt_rate = np.radians(35.0)  # Max 35 deg/s to guarantee bounded angular jerk
        dot_phi_prev = 0.0
        dot_theta_prev = 0.0
        dot_psi_prev = 0.0
        alpha_body_src = np.zeros(3, dtype=np.float64)
        previous_contact_valid = bool(init_sensor.tactile_active)

        for step in range(total_steps):
            sensor_data = backend.read_sensors()
            t = sensor_data.timestamp

            # A. PERCEPTION: Tactile CoP compensation + Hermite-GP Causal Filtering
            rho_clean = tactile_proc.process(
                sensor_data.tactile_pos_raw,
                sensor_data.tactile_force_estimate,
                vel_est,
                is_geometric=backend.tactile_position_kind == "geometric_projection"
            )
            
            motion_measurement = backend.read_platform_motion()
            has_pose_feedback = motion_measurement is not None
            if has_pose_feedback:
                T_p_cur, R_cur, twist_meas = motion_measurement
                phi_meas, theta_meas, psi_meas = kinematics.rotation_matrix_to_zyx(R_cur)
            else:
                T_p_cur = kinematics.T_p_nominal + translation_allocator.offset_I
                phi_meas, theta_meas = phi_filt, theta_filt
                psi_meas = yaw_nuller.accumulated_yaw
                R_cur = kinematics.get_rotation_matrix(phi_meas, theta_meas, psi_meas)
            J_inv_cur = kinematics.compute_inverse_jacobian(
                phi_meas, theta_meas, psi_meas, T_p_cur
            )
            if not has_pose_feedback:
                twist_meas = np.linalg.pinv(J_inv_cur) @ sensor_data.leg_velocities
            q_from_measured_pose, _, _ = kinematics.inverse_kinematics(
                phi_meas, theta_meas, psi_meas, T_p_cur
            )
            closure_error = float(np.max(np.abs(
                q_from_measured_pose - sensor_data.leg_positions
            )))
            closure_tolerance = float(getattr(
                backend, "kinematic_closure_tolerance", 2e-3
            ))
            if closure_error > closure_tolerance:
                raise RuntimeError(
                    "Measured Stewart closure is inconsistent with the actuator "
                    f"model (error={closure_error:.3e} m)."
                )
            measured_translation_valid = True
            if has_pose_feedback:
                measured_translation_valid = translation_allocator.synchronize_measured_state(
                    T_p_cur - kinematics.T_p_nominal,
                    twist_meas[:3],
                    R_cur,
                )
            yaw_rate_raw = float((R_cur.T @ twist_meas[3:])[2])

            rho_hat, vel_est, yaw_hat, alpha_z_hat = filter_gp.update(
                rho_clean,
                yaw_rate_raw,
                measurement_valid=tactile_proc.contact_active,
            )
            if tactile_proc.contact_active and not previous_contact_valid:
                ref_model.reset(rho_hat, np.zeros(2, dtype=np.float64))
                phi_filt, theta_filt = phi_meas, theta_meas
                yaw_nuller.reset(psi_meas)
                dot_phi_prev = dot_theta_prev = dot_psi_prev = 0.0
            previous_contact_valid = bool(tactile_proc.contact_active)

            state = StateEstimatePacket(
                timestamp=t,
                platform_pos=T_p_cur,
                platform_rot=R_cur,
                platform_twist=twist_meas,
                platform_accel_src=alpha_body_src,
                ball_pos=rho_hat,
                ball_vel=vel_est,
                yaw_rate=yaw_hat,
                contact_valid=tactile_proc.contact_active
            )

            # B. TRAJECTORY GENERATION (Hurwitz feasible reference)
            # Phase 1 (t < 2.5s): Soft capture to center [0, 0]
            # Phase 2 (t >= 2.5s): Circular orbit
            if t < 2.5:
                target_pt = np.zeros(2, dtype=np.float64)
            else:
                t_orbit = t - 2.5
                ramp = np.clip(t_orbit / 1.5, 0.0, 1.0)
                cur_r = args.radius * ramp
                target_pt = np.array([
                    cur_r * np.cos(args.speed * t_orbit),
                    cur_r * np.sin(args.speed * t_orbit)
                ], dtype=np.float64)

            rho_d, dot_rho_d, ddot_rho_d = ref_model.update(target_pt)

            # C. ROBUST CONTROLLER (Theorem 7.1)
            u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)

            # D. EXACT TILT INVERSION + SMOOTH RATE LIMITING (Sec 9.1.1)
            phi_raw, theta_raw, _, tilt_only = inverter.project_preferred_gravity(u_virt.B_des)
            if not np.isfinite(phi_raw + theta_raw):
                raise RuntimeError("Non-finite preferred-gravity allocation.")

            tau_tilt = 0.05
            dot_phi = np.clip((phi_raw - phi_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            dot_theta = np.clip((theta_raw - theta_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)

            phi_filt += dot_phi * args.dt
            theta_filt += dot_theta * args.dt
            ddot_phi = (dot_phi - dot_phi_prev) / args.dt
            ddot_theta = (dot_theta - dot_theta_prev) / args.dt
            dot_phi_prev, dot_theta_prev = dot_phi, dot_theta

            # E. ACTIVE YAW SUPPRESSION (Proposition 9.4: Omega_z = 0)
            dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_filt, theta_filt, dot_theta, args.dt)
            psi_filt = yaw_nuller.accumulated_yaw
            ddot_psi = (dot_psi_0 - dot_psi_prev) / args.dt
            dot_psi_prev = dot_psi_0

            euler_rates = np.array([dot_phi, dot_theta, dot_psi_0])
            euler_accels = np.array([ddot_phi, ddot_theta, ddot_psi])
            omega_P = kinematics.zyx_body_angular_velocity(phi_filt, theta_filt, euler_rates)
            alpha_P = kinematics.zyx_body_angular_acceleration(
                phi_filt, theta_filt, euler_rates, euler_accels
            )
            alpha_body_src = alpha_P.copy()
            R_cmd = kinematics.get_rotation_matrix(phi_filt, theta_filt, psi_filt)
            g_P = R_cmd.T @ np.array([0.0, 0.0, -GRAVITY])
            A_p_xy, B_allocation_residual, translation_exact = translation_allocator.allocate(
                u_virt.B_des, R_cmd, g_P[:2], alpha_P, omega_P,
                rho_hat, BALL_RADIUS,
                commit=has_pose_feedback and measured_translation_valid
            )

            # F. DERIVATIVE-CONSISTENT SE(3) ACTUATOR TRAJECTORY
            T_p_cmd = (
                kinematics.T_p_nominal + translation_allocator.offset_I
                if has_pose_feedback and measured_translation_valid
                else T_p_cur
            )
            v_p_I = translation_allocator.velocity_I
            a_p_I = R_cmd @ np.array([A_p_xy[0], A_p_xy[1], 0.0])
            omega_I = R_cmd @ omega_P
            alpha_I = R_cmd @ alpha_P
            q_geom, dot_q_nom, u_q_feedforward = kinematics.actuator_trajectory(
                phi_filt, theta_filt, psi_filt, T_p_cmd,
                v_p_I, omega_I, a_p_I, alpha_I
            )
            u_q_nom = joint_integrator.tracking_acceleration(
                sensor_data.leg_positions,
                sensor_data.leg_velocities,
                q_geom,
                dot_q_nom,
                u_q_feedforward,
            )

            # G. CLARABEL SAFETY SOCP SUPERVISION (Chapter 11)
            safety_status = "BYPASS"
            if safety_filter is not None:
                u_q_applied, slack_value, is_safe = safety_filter.filter_acceleration(
                    u_q_cmd=u_q_nom,
                    state=state,
                    kinematics=kinematics,
                    q_meas=sensor_data.leg_positions,
                    dot_q_meas=sensor_data.leg_velocities
                )
                safety_status = "OK" if is_safe else "FLBK"
            else:
                u_q_applied = u_q_nom
                slack_value = 0.0

            if not state.contact_valid:
                u_q_applied = np.clip(-8.0 * sensor_data.leg_velocities, -8.0, 8.0)
                safety_status = "CONTACT"

            if not has_pose_feedback:
                J_dot_cur = kinematics.compute_inverse_jacobian_dot(
                    phi_meas, theta_meas, psi_meas, T_p_cur,
                    twist_meas[:3], twist_meas[3:]
                )
                xi_dot_applied = np.linalg.pinv(J_inv_cur) @ (
                    u_q_applied - J_dot_cur @ twist_meas
                )
                if not translation_allocator.commit_realized(xi_dot_applied[:3]):
                    u_q_applied = np.clip(-8.0 * sensor_data.leg_velocities, -8.0, 8.0)
                    safety_status = "TRANS"

            # H. ACTUATION: integrate the acceleration actually admitted.
            joint_command = joint_integrator.integrate(
                sensor_data.leg_positions, sensor_data.leg_velocities, u_q_applied
            )
            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=joint_command.q_send,
                dot_q_send=joint_command.q_dot_safe,
                u_q_applied=u_q_applied,
                slack_value=slack_value
            )
            if not backend.write_actuators(cmd_packet):
                raise RuntimeError("Actuator command was not acknowledged.")

            # I. TELEMETRY
            if step % print_interval == 0:
                err_mm = np.linalg.norm(rho_hat - rho_d) * 1000.0
                ball_str = f"[{rho_hat[0]:+.3f}, {rho_hat[1]:+.3f}]"
                ref_str = f"[{rho_d[0]:+.3f}, {rho_d[1]:+.3f}]"

                print(
                    f"{t:8.3f} | {err_mm:14.2f} | {ball_str:>18} | "
                    f"{ref_str:>18} | {sensor_data.tactile_force_estimate:7.3f} | {safety_status:>8}"
                    f" | dB={np.linalg.norm(B_allocation_residual):.3f}"
                )

        print("=" * 110)
        print("[+] Test completed.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as exc:
        print(f"\n[!] Error: {exc}")
        sys.exit(1)
    finally:
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()
