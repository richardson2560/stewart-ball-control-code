# tools/test_closed_loop.py

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
    parser = argparse.ArgumentParser(description="SBC v30 - Full Closed-Loop Tracking.")
    parser.add_argument("--duration", type=float, default=15.0, help="Test duration [s]. Default: 15.0 s.")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s]. Default: 0.002 s (500 Hz).")
    parser.add_argument("--radius", type=float, default=0.06, help="Circle radius [m]. Default: 0.06 m.")
    parser.add_argument("--speed", type=float, default=0.7, help="Orbit speed [rad/s]. Default: 0.7 rad/s.")
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
        tactile_proc = TactileProcessor(ball_radius=0.025, rolling_resistance_coeff=0.0015, min_activation_force=0.20)
        filter_gp = HermiteGPFilter(window_size=11, cycle_time=args.dt)
        ref_model = HurwitzReferenceModel(omega_n=5.0, zeta=1.0, max_acceleration=1.2, cycle_time=args.dt)
        
        # Softened gains for smooth settling
        controller = SBCFullController(kp=5.5, kd=2.2, ball_mass=0.110, ball_radius=0.025)
        inverter = TiltInverter(gravity=9.81, max_roll=np.radians(10.0), max_pitch=np.radians(10.0))
        yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))
        redundancy = RedundancyResolver(k_center=2.5)
        redundancy.reset(T_p_init)
        
        # Safe plate radius matching CoppeliaSim geometry (R = 0.26 m)
        safety_filter = ClarabelSafetyFilter(ball_mass=0.110, ball_radius=0.025, plate_radius=0.26)

        init_sensor = backend.read_sensors()
        ref_model.reset(initial_pos=init_sensor.tactile_pos_raw)

        print("[+] Closed-loop tracking system fully initialized.")

        total_steps = int(args.duration / args.dt)
        print_interval = int(0.1 / args.dt)

        print("\n" + "=" * 105)
        print(f"{'Time [s]':>8} | {'Track Error [mm]':>16} | {'Ball Pos [m]':>18} | {'Ref Pos [m]':>18} | {'N [N]':>7} | {'SOCP':>6}")
        print("=" * 105)

        vel_est = np.zeros(2, dtype=np.float64)
        phi_filt = 0.0
        theta_filt = 0.0
        dot_phi_filt = 0.0
        dot_theta_filt = 0.0
        max_tilt_rate = np.radians(45.0)  # Max 45 deg/s to prevent kinematic jerk

        for step in range(total_steps):
            sensor_data = backend.read_sensors()
            t = sensor_data.timestamp

            # 1. Perception
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

            # 2. Reference Planning: Centering (t < 3s) -> Circular Orbit (t >= 3s)
            if t < 3.0:
                target_pt = np.zeros(2, dtype=np.float64)
            else:
                t_orbit = t - 3.0
                target_pt = np.array([
                    args.radius * np.cos(args.speed * t_orbit),
                    args.radius * np.sin(args.speed * t_orbit)
                ], dtype=np.float64)

            rho_d, dot_rho_d, ddot_rho_d = ref_model.update(target_pt)

            # 3. Robust ISS Controller
            u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)

            # 4. Tilt Inversion & Rate Limiter
            phi_raw, theta_raw, _ = inverter.invert(u_virt.B_des)
            
            # Smooth 1st-order rate-limited tilt filter: dot_theta = clip((theta_raw - theta_filt)/tau)
            tau_tilt = 0.04  # 40 ms smoothing filter
            dot_phi_target = np.clip((phi_raw - phi_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            dot_theta_target = np.clip((theta_raw - theta_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            
            phi_filt += dot_phi_target * args.dt
            theta_filt += dot_theta_target * args.dt
            dot_phi_filt = dot_phi_target
            dot_theta_filt = dot_theta_target

            # 5. Yaw Suppression & Redundancy
            dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_filt, theta_filt, dot_theta_filt, args.dt)
            psi_d = yaw_nuller.accumulated_yaw

            q_cmd, dot_q_cmd, u_q_cmd = redundancy.resolve(
                kinematics=kinematics,
                phi_d=phi_filt,
                theta_d=theta_filt,
                psi_d=psi_d,
                dot_phi_d=dot_phi_filt,
                dot_theta_d=dot_theta_filt,
                dot_psi_0=dot_psi_0,
                q_meas=sensor_data.leg_positions,
                dt=args.dt
            )

            # 6. Safety Filter with Clarabel
            u_q_safe, slack, is_safe = safety_filter.filter_acceleration(
                u_q_cmd=u_q_cmd,
                state=state,
                kinematics=kinematics,
                q_meas=sensor_data.leg_positions,
                dot_q_meas=sensor_data.leg_velocities
            )

            # 7. Actuation with Lead Feedforward
            tau_leg = 0.015
            q_send = q_cmd + tau_leg * dot_q_cmd

            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_send,
                dot_q_send=dot_q_cmd,
                u_q_applied=u_q_safe,
                slack_value=slack
            )
            backend.write_actuators(cmd_packet)

            # 8. Diagnostics
            if step % print_interval == 0:
                err_mm = np.linalg.norm(rho_hat - rho_d) * 1000.0
                ball_str = f"[{rho_hat[0]:+.3f}, {rho_hat[1]:+.3f}]"
                ref_str = f"[{rho_d[0]:+.3f}, {rho_d[1]:+.3f}]"
                socp_str = "OK" if is_safe else "FLBK"

                print(
                    f"{t:8.3f} | {err_mm:16.2f} | {ball_str:>18} | "
                    f"{ref_str:>18} | {sensor_data.tactile_force_estimate:7.2f} | {socp_str:>6}"
                )

        print("=" * 105)
        print("[+] Closed-loop control test completed successfully.")

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as exc:
        print(f"\n[!] Error during closed-loop test: {exc}")
        sys.exit(1)
    finally:
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()