# tools/test_platform_motion.py

import argparse
import sys
import time
import numpy as np

from sbc.datatypes import ActuatorCommandPacket, ActuatorMode
from sbc.interfaces.factory import BackendFactory
from sbc.interfaces.headless_launcher import CoppeliaLauncher
from sbc.kinematics.platform import StewartKinematics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SBC v30 - Stewart Platform Motion & Kinematics Closed-Loop Output Test."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Total test duration [s]. Default: 12.0 s."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Nominal sampling period [s]. Default: 0.002 s (500 Hz)."
    )
    parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Connect to an already open CoppeliaSim instance."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch CoppeliaSim with GUI enabled."
    )
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
        print("[+] Launching CoppeliaSim process...")
        process = CoppeliaLauncher.start(scene_name="stewart_platform.ttt", headless=not args.gui)
        time.sleep(2.5)

    config = {
        "cycle_time": args.dt,
        "coppelia": {"host": "localhost", "port": 23000}
    }
    backend = BackendFactory.create("coppelia", config)

    try:
        print("[+] Connecting to CoppeliaSim...")
        backend.connect()

        # Extract real scene geometry to build analytical kinematics
        b_anchors, p_anchors, l_offsets, T_p_init = backend.extract_kinematic_parameters()
        kinematics = StewartKinematics(
            base_anchors=b_anchors,
            platform_anchors=p_anchors,
            leg_offsets=l_offsets,
            initial_translation=T_p_init
        )
        print(f"[+] Kinematics calibrated. Base anchors norm: {np.linalg.norm(b_anchors, axis=1).mean():.3f} m")

        # Trajectory parameters
        max_roll = np.radians(8.0)     # Roll amplitude: 8 degrees
        max_pitch = np.radians(6.0)    # Pitch amplitude: 6 degrees
        max_yaw = np.radians(4.0)      # Yaw amplitude: 4 degrees
        freq_roll = 0.5                # 0.5 Hz
        freq_pitch = 0.35              # 0.35 Hz
        freq_yaw = 0.25                # 0.25 Hz

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 85)
        print(f"{'Time [s]':>8} | {'Roll (deg)':>10} | {'Pitch (deg)':>11} | {'Leg Error [mm]':>14} | {'Ball Pos [m]':>18} | {'N [N]':>7}")
        print("=" * 85)

        for step in range(total_steps):
            # 1. READ: Sensor feedback from platform and tactile surface
            sensor_data = backend.read_sensors()
            t = sensor_data.timestamp

            # 2. GENERATE DESIRED ORIENTATION
            phi_cmd = max_roll * np.sin(2.0 * np.pi * freq_roll * t)
            theta_cmd = max_pitch * np.cos(2.0 * np.pi * freq_pitch * t)
            psi_cmd = max_yaw * np.sin(2.0 * np.pi * freq_yaw * t)

            # 3. KINEMATIC INVERSION: Desired angles -> Leg positions q_cmd
            q_cmd, _, _ = kinematics.inverse_kinematics(phi=phi_cmd, theta=theta_cmd, psi=psi_cmd)

            # 4. WRITE: Transmit actuator packet via HAL
            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_cmd,
                dot_q_send=np.zeros(6, dtype=np.float64),
                u_q_applied=np.zeros(6, dtype=np.float64),
                slack_value=0.0
            )
            success = backend.write_actuators(cmd_packet)
            if not success:
                print("[!] Communication broken.")
                break

            # 5. TELEMETRY & TRACKING VALIDATION
            if step % print_interval == 0:
                # Euclidean tracking error between commanded and actual leg lengths
                leg_error_mm = np.linalg.norm(sensor_data.leg_positions - q_cmd) * 1000.0
                ball_str = f"[{sensor_data.tactile_pos_raw[0]:+.3f}, {sensor_data.tactile_pos_raw[1]:+.3f}]"

                print(
                    f"{t:8.3f} | {np.degrees(phi_cmd):10.2f} | {np.degrees(theta_cmd):11.2f} | "
                    f"{leg_error_mm:14.3f} | {ball_str:>18} | {sensor_data.tactile_force_estimate:7.2f}"
                )

        print("=" * 85)
        print("[+] Platform motion test completed successfully.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as exc:
        print(f"\n[!] Error during test: {exc}")
        sys.exit(1)
    finally:
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()