# tools/test_nominal_stabilization.py

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
from sbc.controllers.sbc_full import SBCFullController
from sbc.allocation.tilt_inversion import TiltInverter
from sbc.allocation.yaw_nulling import YawNullingAllocator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SBC v30 - Pure Nominal Ball Regulation (No Safety Layer, Pure Centered Tilt)."
    )
    parser.add_argument("--duration", type=float, default=12.0, help="Test duration [s]. Default: 12.0 s.")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s]. Default: 0.002 s (500 Hz).")
    parser.add_argument("--no-auto-launch", action="store_true", help="Connect to running CoppeliaSim instance.")
    parser.add_argument("--gui", action="store_true", help="Run with visual GUI enabled.")
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
        process = CoppeliaLauncher.start(scene_name="stewart_platform.ttt", headless=not args.gui)
        time.sleep(2.5)

    # Actual physical parameters confirmed by user
    BALL_MASS = 0.06545        # [kg]
    BALL_RADIUS = 0.025        # [m] (diameter 0.05 m)
    GRAVITY = 9.81             # [m/s^2]
    EXPECTED_WEIGHT = BALL_MASS * GRAVITY  # ~ 0.642 N

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

        # 1. Initialize Kinematics and Perception
        kinematics = StewartKinematics(b_anchors, p_anchors, l_offsets, initial_translation=T_p_init)
        tactile_proc = TactileProcessor(
            ball_radius=BALL_RADIUS,
            rolling_resistance_coeff=0.0015,
            min_activation_force=0.10  # Scaled for 0.065 kg ball (weight is ~0.64 N)
        )
        filter_gp = HermiteGPFilter(window_size=11, cycle_time=args.dt)

        # 2. Control & Inversion: Gentle gains for smooth settling
        controller = SBCFullController(
            kp=4.5,
            kd=2.5,
            ball_mass=BALL_MASS,
            ball_radius=BALL_RADIUS,
            inertia_ratio_lambda0=5.0 / 7.0,
            rolling_resistance_coeff=0.0015,
            gravity=GRAVITY
        )
        inverter = TiltInverter(
            gravity=GRAVITY,
            max_roll=np.radians(8.0),   # Bounded tilt to avoid launching
            max_pitch=np.radians(8.0),
            ball_radius=BALL_RADIUS
        )
        yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))

        print(f"[+] Nominal regulation active for ball mass = {BALL_MASS * 1000:.1f} g.")
        print(f"[+] Nominal normal load N = {EXPECTED_WEIGHT:.3f} N.")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 90)
        print(f"{'Time [s]':>8} | {'Ball Pos [m]':>18} | {'Tilt (R/P) [deg]':>18} | {'Vel [m/s]':>16} | {'N [N]':>7}")
        print("=" * 90)

        vel_est = np.zeros(2, dtype=np.float64)
        phi_filt = 0.0
        theta_filt = 0.0
        max_tilt_rate = np.radians(30.0)  # Max 30 deg/s for smooth motion

        # Target is strictly the origin [0.0, 0.0]
        rho_d = np.zeros(2, dtype=np.float64)
        dot_rho_d = np.zeros(2, dtype=np.float64)
        ddot_rho_d = np.zeros(2, dtype=np.float64)

        for step in range(total_steps):
            # A. SENSE
            sensor_data = backend.read_sensors()
            t = sensor_data.timestamp

            # B. PERCEPTION
            rho_clean = tactile_proc.process(sensor_data.tactile_pos_raw, sensor_data.tactile_force_estimate, vel_est)
            
            # Approximate yaw rate
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

            # C. ROBUST CONTROLLER (Regulation to [0, 0])
            u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)

            # D. EXACT TILT INVERSION & SMOOTH FILTERING
            phi_raw, theta_raw, _ = inverter.invert(u_virt.B_des)

            tau_tilt = 0.06  # 60 ms critically damped smoothing filter
            dot_phi = np.clip((phi_raw - phi_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            dot_theta = np.clip((theta_raw - theta_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)

            phi_filt += dot_phi * args.dt
            theta_filt += dot_theta * args.dt

            # E. ACTIVE YAW SUPPRESSION
            dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_filt, theta_filt, dot_theta, args.dt)
            psi_filt = yaw_nuller.accumulated_yaw

            # F. PURE CENTERED KINEMATICS (T_p fixed at T_p_init -> ZERO translational catapulting)
            q_cmd, _, _ = kinematics.inverse_kinematics(
                phi=phi_filt,
                theta=theta_filt,
                psi=psi_filt,
                T_p=T_p_init
            )

            # G. ACTUATION: Direct position command (zero artificial lead-spikes during settling)
            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_cmd,
                dot_q_send=np.zeros(6, dtype=np.float64),
                u_q_applied=np.zeros(6, dtype=np.float64),
                slack_value=0.0
            )
            backend.write_actuators(cmd_packet)

            # H. DIAGNOSTICS
            if step % print_interval == 0:
                ball_str = f"[{rho_hat[0]:+.3f}, {rho_hat[1]:+.3f}]"
                tilt_str = f"[{np.degrees(phi_filt):+5.2f}, {np.degrees(theta_filt):+5.2f}]"
                vel_str = f"[{vel_est[0]:+.3f}, {vel_est[1]:+.3f}]"

                print(
                    f"{t:8.3f} | {ball_str:>18} | {tilt_str:>18} | "
                    f"{vel_str:>16} | {sensor_data.tactile_force_estimate:7.3f}"
                )

        print("=" * 90)
        print("[+] Nominal stabilization test completed successfully.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as exc:
        print(f"\n[!] Error during nominal test: {exc}")
        sys.exit(1)
    finally:
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()