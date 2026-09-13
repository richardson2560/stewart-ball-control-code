# tools/test_dual_authority.py

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
    parser = argparse.ArgumentParser(
        description="SBC v30 - Rigorous Dual-Authority Allocation (Tilt DC + Translation AC)."
    )
    parser.add_argument("--duration", type=float, default=12.0, help="Test duration [s].")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s].")
    parser.add_argument("--radius", type=float, default=0.10, help="Circle radius [m]. Default: 0.10 m.")
    parser.add_argument("--speed", type=float, default=2.5, help="Angular speed [rad/s]. Default: 2.5 rad/s.")
    parser.add_argument("--max-tilt", type=float, default=18.0, help="Tilt saturation limit [deg]. Default: 18.0 deg.")
    parser.add_argument("--enable-layer2", action="store_true", default=True, help="Enable Layer 2 translational kick.")
    parser.add_argument("--no-auto-launch", action="store_true", help="Connect to running CoppeliaSim instance.")
    parser.add_argument("--gui", action="store_true", help="Launch with visual GUI enabled.")
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

        kinematics = StewartKinematics(b_anchors, p_anchors, l_offsets, initial_translation=T_p_init)
        tactile_proc = TactileProcessor(ball_radius=BALL_RADIUS, min_activation_force=0.10)
        filter_gp = HermiteGPFilter(window_size=11, cycle_time=args.dt)
        ref_model = HurwitzReferenceModel(omega_n=6.0, zeta=1.0, max_acceleration=2.8, cycle_time=args.dt)
        controller = SBCFullController(kp=7.0, kd=3.0, ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, gravity=GRAVITY)

        tilt_limit_rad = np.radians(args.max_tilt)
        inverter = TiltInverter(gravity=GRAVITY, max_roll=tilt_limit_rad, max_pitch=tilt_limit_rad, ball_radius=BALL_RADIUS)
        yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))
        
        # Redundancy resolver with high-stiffness centering spring to prevent workspace drift
        redundancy = RedundancyResolver(
            k_trans_spring=20.0,
            d_trans_damping=6.0,
            max_trans_accel=1.8,
            max_trans_disp=0.030  # Max +-3 cm translation
        )

        safety_filter = ClarabelSafetyFilter(ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, plate_radius=0.45)

        init_sensor = backend.read_sensors()
        ref_model.reset(initial_pos=init_sensor.tactile_pos_raw)

        a_tilt_capacity = (5.0 / 7.0) * GRAVITY * np.sin(tilt_limit_rad)
        a_centripetal = (args.speed ** 2) * args.radius
        print(f"[+] Tilt Limit: {args.max_tilt:.1f} deg -> Available Tilt Accel: {a_tilt_capacity:.2f} m/s^2.")
        print(f"[+] Trajectory Demand: a_cen = {a_centripetal:.2f} m/s^2 (R = {args.radius * 100:.1f} cm, w = {args.speed:.2f} rad/s).")
        print(f"[+] Layer 2 (Translational Kick): {'ENABLED' if args.enable_layer2 else 'DISABLED'}.")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 118)
        print(f"{'Time [s]':>8} | {'Track Err [mm]':>14} | {'Ball Pos [m]':>18} | {'Tilt (R/P) [deg]':>18} | {'Plat Trans [mm]':>18} | {'Safety':>8}")
        print("=" * 118)

        vel_est = np.zeros(2, dtype=np.float64)
        phi_filt = 0.0
        theta_filt = 0.0
        max_tilt_rate = np.radians(45.0)

        # High-pass filter memory for transient Layer 2 allocation
        prev_B_des = np.zeros(2, dtype=np.float64)

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
                platform_pos=kinematics.T_p_nominal + redundancy.current_translation,
                platform_rot=R_cur,
                platform_twist=twist_meas,
                platform_accel_src=np.zeros(3),
                ball_pos=rho_hat,
                ball_vel=vel_est,
                yaw_rate=yaw_hat,
                contact_valid=tactile_proc.contact_active
            )

            # B. TRAJECTORY GENERATION
            if t < 2.0:
                target_pt = np.zeros(2, dtype=np.float64)
            else:
                t_orbit = t - 2.0
                ramp = np.clip(t_orbit / 1.5, 0.0, 1.0)
                cur_r = args.radius * ramp
                target_pt = np.array([
                    cur_r * np.cos(args.speed * t_orbit),
                    cur_r * np.sin(args.speed * t_orbit)
                ], dtype=np.float64)

            rho_d, dot_rho_d, ddot_rho_d = ref_model.update(target_pt)

            # C. CONTROL: B_des
            u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)

            # D. LAYER 1: Tilt Inversion (Saturated to max_tilt)
            phi_raw, theta_raw, _ = inverter.invert(u_virt.B_des)

            tau_tilt = 0.04
            dot_phi = np.clip((phi_raw - phi_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            dot_theta = np.clip((theta_raw - theta_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)

            phi_filt += dot_phi * args.dt
            theta_filt += dot_theta * args.dt

            # E. LAYER 2: High-pass transient translational kick
            # Detects acceleration demand rate: dB_des/dt
            jerk_B = (u_virt.B_des - prev_B_des) / args.dt
            prev_B_des = u_virt.B_des.copy()

            if args.enable_layer2:
                # Delivers an inertia kick proportional to rapid command changes
                # A_p = - k_kick * jerk_B
                A_p_assist = np.clip(- 0.04 * jerk_B, -1.8, 1.8)
            else:
                A_p_assist = np.zeros(2, dtype=np.float64)

            # F. YAW NULLING
            dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_filt, theta_filt, dot_theta, args.dt)
            psi_filt = yaw_nuller.accumulated_yaw

            # G. DUAL REDUNDANCY RESOLUTION
            q_cmd, dot_q_cmd, u_q_cmd = redundancy.resolve_with_authority(
                kinematics=kinematics,
                phi_d=phi_filt,
                theta_d=theta_filt,
                psi_d=psi_filt,
                dot_phi_d=dot_phi,
                dot_theta_d=dot_theta,
                dot_psi_0=dot_psi_0,
                A_p_assist_xy=A_p_assist,
                q_meas=sensor_data.leg_positions,
                dt=args.dt
            )

            # H. CLARABEL SAFETY FILTER
            u_q_applied, _, is_safe = safety_filter.filter_acceleration(
                u_q_cmd=u_q_cmd,
                state=state,
                kinematics=kinematics,
                q_meas=sensor_data.leg_positions,
                dot_q_meas=sensor_data.leg_velocities
            )

            # I. ACTUATION
            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_cmd,
                dot_q_send=dot_q_cmd,
                u_q_applied=u_q_applied,
                slack_value=0.0
            )
            backend.write_actuators(cmd_packet)

            # J. TELEMETRY
            if step % print_interval == 0:
                err_mm = np.linalg.norm(rho_hat - rho_d) * 1000.0
                ball_str = f"[{rho_hat[0]:+.3f}, {rho_hat[1]:+.3f}]"
                tilt_str = f"[{np.degrees(phi_filt):+5.2f}, {np.degrees(theta_filt):+5.2f}]"
                trans_mm = redundancy.current_translation * 1000.0
                trans_str = f"[{trans_mm[0]:+5.1f}, {trans_mm[1]:+5.1f}]"

                print(
                    f"{t:8.3f} | {err_mm:14.2f} | {ball_str:>18} | "
                    f"{tilt_str:>18} | {trans_str:>18} | {'OK' if is_safe else 'FLBK':>8}"
                )

        print("=" * 118)
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