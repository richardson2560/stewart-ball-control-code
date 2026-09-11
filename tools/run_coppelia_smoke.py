# tools/run_coppelia_smoke.py

import argparse
import sys
import time
import numpy as np

from sbc.datatypes import ActuatorCommandPacket, ActuatorMode
from sbc.interfaces.factory import BackendFactory
from sbc.interfaces.headless_launcher import CoppeliaLauncher


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SBC v30 - CoppeliaSim HAL Smoke Test under pure gravity."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Total simulation run time [s]. Default: 5.0 s."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.002,
        help="Nominal sampling period [s]. Default: 0.002 s (500 Hz)."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch CoppeliaSim with visual GUI instead of headless mode."
    )
    parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Do not auto-spawn CoppeliaSim (connects to an already running instance)."
    )
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
        print("[+] Launching CoppeliaSim subprocess...")
        process = CoppeliaLauncher.start(
            scene_name="stewart_platform.ttt",
            headless=not args.gui
        )
        print("[+] Waiting for ZeroMQ remote server initialization...")
        time.sleep(2.5)

    config = {
        "cycle_time": args.dt,
        "ball_mass": 0.110,
        "ball_radius": 0.025,
        "gravity": 9.81,
        "coppelia": {
            "host": "localhost",
            "port": 23000
        }
    }

    backend = BackendFactory.create("coppelia", config)

    try:
        print("[+] Connecting to CoppeliaSim backend...")
        backend.connect()
        print(f"[+] Operational at Ts = {backend.cycle_time * 1000:.1f} ms (500 Hz).")

        # Step 0: Read initial configuration to hold joint positions
        initial_packet = backend.read_sensors()
        held_q = initial_packet.leg_positions.copy()
        print(f"[+] Initial leg lengths [m]: {np.round(held_q, 4)}")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)  # Display telemetry at 10 Hz

        print("\n" + "=" * 80)
        print(f"{'Time [s]':>8} | {'Ball Pos [m]':>18} | {'Normal Load [N]':>15} | {'Contact':>8} | {'Bus':>5}")
        print("=" * 80)

        t_sim = 0.0
        for step in range(total_steps):
            # Read clean physical state
            sensor_data = backend.read_sensors()
            t_sim = sensor_data.timestamp

            # Hold legs stationary (zero acceleration command)
            command = ActuatorCommandPacket(
                timestamp=t_sim,
                mode=ActuatorMode.CSP,
                q_send=held_q,
                dot_q_send=np.zeros(6, dtype=np.float64),
                u_q_applied=np.zeros(6, dtype=np.float64),
                slack_value=0.0
            )

            # Advance simulation by exactly one step dt
            success = backend.write_actuators(command)
            if not success:
                print(f"[!] Communication lost at t = {t_sim:.3f} s.")
                break

            # Diagnostic telemetry at 10 Hz
            if step % print_interval == 0:
                bx, by = sensor_data.tactile_pos_raw[0], sensor_data.tactile_pos_raw[1]
                b_str = f"[{bx:+.4f}, {by:+.4f}]"
                contact_str = "YES" if sensor_data.tactile_active else "NO"
                bus_str = "OK" if sensor_data.bus_healthy else "FAIL"

                print(
                    f"{t_sim:8.3f} | {b_str:>18} | "
                    f"{sensor_data.tactile_force_estimate:15.3f} | "
                    f"{contact_str:>8} | {bus_str:>5}"
                )

        print("=" * 80)
        print("[+] Smoke test finished successfully.")

    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.")
    except Exception as exc:
        print(f"\n[!] Error during execution: {exc}")
        sys.exit(1)
    finally:
        print("[+] Disconnecting backend...")
        backend.disconnect()
        if process is not None:
            print("[+] Terminating CoppeliaSim process...")
            CoppeliaLauncher.stop(process)
            print("[+] Process closed.")


if __name__ == "__main__":
    main()