# tools/test_allocation_pipeline.py

import argparse
import sys
import time
import numpy as np

from sbc.datatypes import ActuatorCommandPacket, ActuatorMode
from sbc.interfaces.factory import BackendFactory
from sbc.interfaces.headless_launcher import CoppeliaLauncher
from sbc.kinematics.platform import StewartKinematics
from sbc.allocation.tilt_inversion import TiltInverter
from sbc.allocation.yaw_nulling import YawNullingAllocator
from sbc.allocation.redundancy import RedundancyResolver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SBC v30 - Kinematic Allocation & Yaw Nulling Pipeline Validation."
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration [s]. Default: 10.0 s.")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s]. Default: 0.002 s (500 Hz).")
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

        # Initialize allocation subsystems
        inverter = TiltInverter(gravity=9.81, max_roll=np.radians(10.0), max_pitch=np.radians(10.0))
        yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))
        redundancy = RedundancyResolver(k_center=3.0)
        redundancy.reset(T_p_init)

        print("[+] Kinematic allocation pipeline initialized.")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 105)
        print(f"{'Time [s]':>8} | {'B_des [m/s2]':>18} | {'Tilt (R/P) [deg]':>18} | {'Omega_z [r/s]':>14} | {'Yaw Acc [deg]':>14} | {'Max Stroke [mm]':>16}")
        print("=" * 105)

        prev_theta_d = 0.0

        for step in range(total_steps):
            sensor_data = backend.read_sensors()
            t = sensor_data.timestamp

            # 1. VIRTUAL INPUT STIMULUS: Rotating acceleration vector B_des
            b_mag = 1.0  # 1.0 m/s^2 driving acceleration
            B_des = np.array([
                b_mag * np.sin(2.0 * np.pi * 0.5 * t),
                b_mag * np.cos(2.0 * np.pi * 0.5 * t)
            ], dtype=np.float64)

            # 2. EXACT TILT INVERSION: B_des -> (phi_d, theta_d)
            phi_d, theta_d, _ = inverter.invert(B_des)
            dot_theta_d = (theta_d - prev_theta_d) / args.dt
            prev_theta_d = theta_d

            # 3. ACTIVE YAW SUPPRESSION: Impose Omega_z = 0 -> dot_psi_0
            dot_psi_0, is_unwinding = yaw_nuller.compute_yaw_velocity(phi_d, theta_d, dot_theta_d, args.dt)
            psi_d = yaw_nuller.accumulated_yaw

            # 4. REDUNDANCY RESOLUTION & NULL-SPACE RECENTERING
            q_cmd, dot_q_cmd, u_q_cmd = redundancy.resolve(
                kinematics=kinematics,
                phi_d=phi_d,
                theta_d=theta_d,
                psi_d=psi_d,
                dot_phi_d=0.0,
                dot_theta_d=dot_theta_d,
                dot_psi_0=dot_psi_0,
                q_meas=sensor_data.leg_positions,
                dt=args.dt
            )

            # 5. WRITE: Send actuator targets via HAL
            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_cmd,
                dot_q_send=dot_q_cmd,
                u_q_applied=u_q_cmd,
                slack_value=0.0
            )
            backend.write_actuators(cmd_packet)

            # 6. TELEMETRY
            if step % print_interval == 0:
                # Estimate current real body yaw rate from kinematics
                J_inv = kinematics.compute_inverse_jacobian(phi_d, theta_d, psi_d)
                twist = np.linalg.pinv(J_inv) @ sensor_data.leg_velocities
                omega_z_real = twist[5]

                b_str = f"[{B_des[0]:+.2f}, {B_des[1]:+.2f}]"
                tilt_str = f"[{np.degrees(phi_d):+5.2f}, {np.degrees(theta_d):+5.2f}]"
                max_dev_mm = np.max(np.abs(sensor_data.leg_positions)) * 1000.0

                print(
                    f"{t:8.3f} | {b_str:>18} | {tilt_str:>18} | "
                    f"{omega_z_real:14.4f} | {np.degrees(psi_d):14.2f} | {max_dev_mm:16.2f}"
                )

        print("=" * 105)
        print("[+] Kinematic allocation test completed successfully.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as exc:
        print(f"\n[!] Error during allocation test: {exc}")
        sys.exit(1)
    finally:
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()