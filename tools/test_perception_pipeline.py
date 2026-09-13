# tools/test_perception_pipeline.py

import argparse
import sys
import time
import numpy as np

from sbc.datatypes import ActuatorCommandPacket, ActuatorMode
from sbc.interfaces.factory import BackendFactory
from sbc.interfaces.headless_launcher import CoppeliaLauncher
from sbc.kinematics.platform import StewartKinematics
from sbc.perception.hermite_gp import HermiteGPFilter
from sbc.perception.tactile import TactileProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SBC v30 - Tactile Preprocessing & Hermite-GP Filter Live Validation."
    )
    parser.add_argument("--duration", type=float, default=8.0, help="Test duration [s]. Default: 8.0 s.")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s]. Default: 0.002 s (500 Hz).")
    parser.add_argument("--window", type=int, default=11, help="Hermite-GP FIR window size. Default: 11 samples.")
    parser.add_argument("--no-auto-launch", action="store_true", help="Connect to running CoppeliaSim instance.")
    parser.add_argument("--gui", action="store_true", help="Run with visual GUI enabled.")
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
        process = CoppeliaLauncher.start(scene_name="stewart_platform.ttt", headless=not args.gui)
        time.sleep(2.5)

    config = {
        "cycle_time": args.dt,
        "coppelia": {"host": "localhost", "port": 23000}
    }
    backend = BackendFactory.create("coppelia", config)

    try:
        backend.connect()
        b_anchors, p_anchors, l_offsets, T_p_init = backend.extract_kinematic_parameters()
        kinematics = StewartKinematics(b_anchors, p_anchors, l_offsets, initial_translation=T_p_init)

        # Initialize perception pipeline
        tactile_proc = TactileProcessor(ball_radius=0.025, rolling_resistance_coeff=0.0015, min_activation_force=0.20)
        filter_gp = HermiteGPFilter(window_size=args.window, cycle_time=args.dt)

        print(f"[+] Perception pipeline initialized (FIR window: {args.window} taps = {args.window * args.dt * 1000:.1f} ms).")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 96)
        print(f"{'Time [s]':>8} | {'Raw CoP [m]':>18} | {'Comp Rho [m]':>18} | {'Est Vel [m/s]':>18} | {'Alpha_z [r/s2]':>14} | {'Contact':>7}")
        print("=" * 96)

        vel_estimate = np.zeros(2, dtype=np.float64)

        for step in range(total_steps):
            # 1. READ: Raw measurement from HAL
            sensor_data = backend.read_sensors()
            t = sensor_data.timestamp

            # 2. PERCEPTION STAGE 1: Tactile CoP Compensation
            # Uses previous cycle's estimated velocity to break algebraic loop cleanly
            rho_clean = tactile_proc.process(
                sensor_data.tactile_pos_raw,
                sensor_data.tactile_force_estimate,
                vel_estimate
            )

            # Derive raw platform yaw rate from joint speeds via Jacobian for testing
            J_inv = kinematics.compute_inverse_jacobian(phi=0.0, theta=0.0, psi=0.0)
            # Spatial twist approximation in inertial frame: xi = J_inv^+ * dot_q
            twist_approx = np.linalg.pinv(J_inv) @ sensor_data.leg_velocities
            yaw_rate_raw = float(twist_approx[5])

            # 3. PERCEPTION STAGE 2: Hermite-GP Causal FIR Filtering
            rho_hat, vel_estimate, yaw_hat, alpha_z_hat = filter_gp.update(rho_clean, yaw_rate_raw)

            # 4. MOTION STIMULUS: Mild tilt oscillation to roll the ball
            phi_cmd = np.radians(4.0) * np.sin(2.0 * np.pi * 0.4 * t)
            theta_cmd = np.radians(4.0) * np.cos(2.0 * np.pi * 0.4 * t)
            q_cmd, _, _ = kinematics.inverse_kinematics(phi=phi_cmd, theta=theta_cmd, psi=0.0)

            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_cmd,
                dot_q_send=np.zeros(6),
                u_q_applied=np.zeros(6),
                slack_value=0.0
            )
            backend.write_actuators(cmd_packet)

            # 5. DIAGNOSTICS
            if step % print_interval == 0:
                raw_str = f"[{sensor_data.tactile_pos_raw[0]:+.3f}, {sensor_data.tactile_pos_raw[1]:+.3f}]"
                clean_str = f"[{rho_hat[0]:+.3f}, {rho_hat[1]:+.3f}]"
                vel_str = f"[{vel_estimate[0]:+.3f}, {vel_estimate[1]:+.3f}]"
                contact_str = "YES" if tactile_proc.contact_active else "NO"

                print(
                    f"{t:8.3f} | {raw_str:>18} | {clean_str:>18} | "
                    f"{vel_str:>18} | {alpha_z_hat:14.3f} | {contact_str:>7}"
                )

        print("=" * 96)
        print("[+] Perception pipeline test completed successfully.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as exc:
        print(f"\n[!] Error during perception test: {exc}")
        sys.exit(1)
    finally:
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()