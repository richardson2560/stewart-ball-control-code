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
from sbc.allocation.tilt_inversion import TiltInverter
from sbc.allocation.yaw_nulling import YawNullingAllocator
from sbc.allocation.redundancy import RedundancyResolver
from sbc.safety.socp_filter import ClarabelSafetyFilter


def main() -> None:
    parser = argparse.ArgumentParser(description="SBC v30 - Full Closed-Loop Tracking & Safety Validation.")
    parser.add_argument("--duration", type=float, default=15.0, help="Test duration [s].")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s] (500 Hz).")
    parser.add_argument("--radius", type=float, default=0.10, help="Orbit circle radius [m]. Default: 0.10 m.")
    parser.add_argument("--speed", type=float, default=1.8, help="Orbit angular speed [rad/s]. Default: 1.8 rad/s.")
    parser.add_argument("--max-tilt", type=float, default=16.0, help="Maximum allowable tilt [deg]. Default: 16.0 deg.")
    parser.add_argument("--safety", action="store_true", help="Enable Clarabel SOCP Safety Filter.")
    parser.add_argument("--no-auto-launch", action="store_true", help="Connect to running CoppeliaSim.")
    parser.add_argument("--gui", action="store_true", help="Run with visual GUI enabled.")
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
        process = CoppeliaLauncher.start(scene_name="stewart_platform.ttt", headless=not args.gui)
        time.sleep(2.5)

    BALL_MASS = 0.06545
    BALL_RADIUS = 0.025
    GRAVITY = 9.81

    config = {
        "cycle_time": args.dt,
        "ball_mass": BALL_MASS,
        "ball_radius": BALL_RADIUS,
        "gravity": GRAVITY,
        "coppelia": {"host": "localhost", "port": 23000}
    }
    backend = BackendFactory.create("coppelia", config)

    try:
        backend.connect()
        b_anchors, p_anchors, l_offsets, T_p_init = backend.extract_kinematic_parameters()

        # 1. Initialize Subsystems
        kinematics = StewartKinematics(b_anchors, p_anchors, l_offsets, initial_translation=T_p_init)
        tactile_proc = TactileProcessor(ball_radius=BALL_RADIUS, min_activation_force=0.10)
        filter_gp = HermiteGPFilter(window_size=11, cycle_time=args.dt)
        ref_model = HurwitzReferenceModel(omega_n=5.5, zeta=1.0, max_acceleration=2.2, cycle_time=args.dt)

        controller = SBCFullController(kp=6.0, kd=2.8, ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, gravity=GRAVITY)

        tilt_limit_rad = np.radians(args.max_tilt)
        inverter = TiltInverter(gravity=GRAVITY, max_roll=tilt_limit_rad, max_pitch=tilt_limit_rad, ball_radius=BALL_RADIUS)
        yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))
        redundancy = RedundancyResolver(k_center=1.5)

        safety_filter = ClarabelSafetyFilter(ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, plate_radius=0.45) if args.safety else None

        init_sensor = backend.read_sensors()
        ref_model.reset(initial_pos=init_sensor.tactile_pos_raw)

        # Theoretical acceleration capacity check (Corollary 13.1)
        a_tilt_capacity = (5.0 / 7.0) * GRAVITY * np.sin(tilt_limit_rad)
        a_centripetal_demand = (args.speed ** 2) * args.radius
        print(f"[+] Max tilt: {args.max_tilt:.1f} deg -> Available acceleration: {a_tilt_capacity:.2f} m/s^2.")
        print(f"[+] Trajectory demand: a_cen = {a_centripetal_demand:.2f} m/s^2 (Orbit R = {args.radius * 100:.1f} cm, w = {args.speed:.2f} rad/s).")
        if a_centripetal_demand > a_tilt_capacity:
            print("[!] WARNING: Trajectory acceleration exceeds maximum tilt capacity. Tracking will saturate.")
        else:
            print("[+] Feasible regime: Acceleration demand is within tilt authority.")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 115)
        print(f"{'Time [s]':>8} | {'Track Error [mm]':>16} | {'Ball Pos [m]':>18} | {'Ref Pos [m]':>18} | {'Tilt (R/P) [deg]':>18} | {'Safety':>8} | {'Null [mm]':>10}")
        print("=" * 115)

        vel_est = np.zeros(2, dtype=np.float64)
        phi_filt = 0.0
        theta_filt = 0.0
        max_tilt_rate = np.radians(45.0)

        for step in range(total_steps):
            sensor_data = backend.read_sensors()
            t = sensor_data.timestamp

            # A. PERCEPTION
            rho_clean = tactile_proc.process(sensor_data.tactile_pos_raw, sensor_data.tactile_force_estimate, vel_est)
            J_inv_cur = kinematics.compute_inverse_jacobian(phi_filt, theta_filt, yaw_nuller.accumulated_yaw)
            twist_meas = np.linalg.pinv(J_inv_cur) @ sensor_data.leg_velocities
            yaw_rate_raw = float(twist_meas[5])

            rho_hat, vel_est, yaw_hat, alpha_z_hat = filter_gp.update(rho_clean, yaw_rate_raw)

            R_cur = kinematics.get_rotation_matrix(phi_filt, theta_filt, yaw_nuller.accumulated_yaw)
            state = StateEstimatePacket(
                timestamp=t,
                platform_pos=T_p_init,
                platform_rot=R_cur,
                platform_twist=twist_meas,
                platform_accel_src=np.zeros(3),
                ball_pos=rho_hat,
                ball_vel=vel_est,
                yaw_rate=yaw_hat,
                contact_valid=tactile_proc.contact_active
            )

            # B. TRAJECTORY GENERATION
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

            # C. ROBUST TRACKING
            u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)

            # D. EXACT TILT INVERSION & SMOOTHING
            phi_raw, theta_raw, _ = inverter.invert(u_virt.B_des)

            tau_tilt = 0.05
            dot_phi = np.clip((phi_raw - phi_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            dot_theta = np.clip((theta_raw - theta_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)

            phi_filt += dot_phi * args.dt
            theta_filt += dot_theta * args.dt

            # E. ACTIVE YAW SUPPRESSION
            dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_filt, theta_filt, dot_theta, args.dt)
            psi_filt = yaw_nuller.accumulated_yaw

            # F. CANONICAL REDUNDANCY RESOLUTION (Task + Null-Space Recentering)
            q_cmd, dot_q_cmd, u_q_cmd = redundancy.resolve(
                kinematics=kinematics,
                phi_d=phi_filt,
                theta_d=theta_filt,
                psi_d=psi_filt,
                dot_phi_d=dot_phi,
                dot_theta_d=dot_theta,
                dot_psi_0=dot_psi_0,
                q_meas=sensor_data.leg_positions,
                dt=args.dt
            )

            # G. CLARABEL SAFETY SOCP FILTER
            safety_status = "BYPASS"
            if safety_filter is not None:
                u_q_applied, _, is_safe = safety_filter.filter_acceleration(
                    u_q_cmd=u_q_cmd,
                    state=state,
                    kinematics=kinematics,
                    q_meas=sensor_data.leg_positions,
                    dot_q_meas=sensor_data.leg_velocities
                )
                safety_status = "OK" if is_safe else "FLBK"

            # H. ACTUATION
            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_cmd,
                dot_q_send=dot_q_cmd,
                u_q_applied=u_q_cmd,
                slack_value=0.0
            )
            backend.write_actuators(cmd_packet)

            # I. DIAGNOSTICS
            if step % print_interval == 0:
                err_mm = np.linalg.norm(rho_hat - rho_d) * 1000.0
                ball_str = f"[{rho_hat[0]:+.3f}, {rho_hat[1]:+.3f}]"
                ref_str = f"[{rho_d[0]:+.3f}, {rho_d[1]:+.3f}]"
                tilt_str = f"[{np.degrees(phi_filt):+5.2f}, {np.degrees(theta_filt):+5.2f}]"
                null_dev_mm = np.max(np.abs(q_cmd - kinematics.inverse_kinematics(phi_filt, theta_filt, psi_filt, kinematics.T_p_nominal)[0])) * 1000.0

                print(
                    f"{t:8.3f} | {err_mm:16.2f} | {ball_str:>18} | "
                    f"{ref_str:>18} | {tilt_str:>18} | {safety_status:>8} | {null_dev_mm:10.2f}"
                )

        print("=" * 115)
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