# tools/run_gui.py

import argparse
import queue
import threading
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
from sbc.safety.socp_filter import ClarabelSafetyFilter
from sbc.gui.dashboard import TelemetryDashboard


class ControllerThread(threading.Thread):
    """
    Dedicated high-speed control loop running at 500 Hz (Ts = 2 ms).
    Pushes non-blocking packets to GUI telemetry queue.
    """

    def __init__(
        self,
        telemetry_queue: queue.Queue,
        command_queue: queue.Queue,
        backend,
        kinematics: StewartKinematics,
        dt: float = 0.002
    ) -> None:
        super().__init__(daemon=True)
        self.t_queue = telemetry_queue
        self.cmd_queue = command_queue
        self.backend = backend
        self.kin = kinematics
        self.dt = dt
        self.running = True

        # Trajectory state
        self.mode = "Circle Orbit"
        self.radius = 0.08
        self.speed = 1.20
        self.target_override: Optional[np.ndarray] = None
        self.orbit_active = True

    def run(self) -> None:
        BALL_MASS = 0.06545
        BALL_RADIUS = 0.025
        GRAVITY = 9.81

        tactile_proc = TactileProcessor(ball_radius=BALL_RADIUS, min_activation_force=0.10)
        filter_gp = HermiteGPFilter(window_size=11, cycle_time=self.dt)
        ref_model = HurwitzReferenceModel(omega_n=5.0, zeta=1.0, max_acceleration=1.6, cycle_time=self.dt)
        controller = SBCFullController(kp=5.5, kd=2.4, ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, gravity=GRAVITY)
        inverter = TiltInverter(gravity=GRAVITY, max_roll=np.radians(15.0), max_pitch=np.radians(15.0), ball_radius=BALL_RADIUS)
        yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))
        safety_filter = ClarabelSafetyFilter(ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, plate_radius=0.45)

        init_sensor = self.backend.read_sensors()
        ref_model.reset(initial_pos=init_sensor.tactile_pos_raw)

        vel_est = np.zeros(2, dtype=np.float64)
        phi_filt = 0.0
        theta_filt = 0.0
        max_tilt_rate = np.radians(35.0)

        step = 0
        t_start = time.perf_counter()

        while self.running:
            t_cycle_start = time.perf_counter()

            # 1. Process asynchronous GUI commands
            try:
                while not self.cmd_queue.empty():
                    cmd = self.cmd_queue.get_nowait()
                    action = cmd.get("action")
                    if action == "center":
                        self.target_override = np.zeros(2, dtype=np.float64)
                        self.orbit_active = False
                    elif action == "orbit":
                        self.target_override = None
                        self.orbit_active = True
                    elif action == "update_params":
                        self.mode = cmd.get("mode", self.mode)
                        self.radius = cmd.get("radius", self.radius)
                        self.speed = cmd.get("speed", self.speed)
            except queue.Empty:
                pass

            # 2. SENSE
            sensor_data = self.backend.read_sensors()
            t = sensor_data.timestamp

            # 3. PERCEPTION
            rho_clean = tactile_proc.process(sensor_data.tactile_pos_raw, sensor_data.tactile_force_estimate, vel_est)
            J_inv_cur = self.kin.compute_inverse_jacobian(phi_filt, theta_filt, yaw_nuller.accumulated_yaw)
            twist_meas = np.linalg.pinv(J_inv_cur) @ sensor_data.leg_velocities
            yaw_rate_raw = float(twist_meas[5])

            rho_hat, vel_est, yaw_hat, alpha_z_hat = filter_gp.update(rho_clean, yaw_rate_raw)

            R_cur = self.kin.get_rotation_matrix(phi_filt, theta_filt, yaw_nuller.accumulated_yaw)
            state = StateEstimatePacket(
                timestamp=t,
                platform_pos=self.kin.T_p_nominal,
                platform_rot=R_cur,
                platform_twist=twist_meas,
                platform_accel_src=np.zeros(3),
                ball_pos=rho_hat,
                ball_vel=vel_est,
                yaw_rate=yaw_hat,
                contact_valid=tactile_proc.contact_active
            )

            # 4. REFERENCE PLANNING
            if not self.orbit_active or t < 2.0:
                target_pt = np.zeros(2, dtype=np.float64)
            else:
                t_orb = t - 2.0
                ramp = np.clip(t_orb / 1.5, 0.0, 1.0)
                cur_r = self.radius * ramp
                if self.mode == "Circle Orbit":
                    target_pt = np.array([cur_r * np.cos(self.speed * t_orb), cur_r * np.sin(self.speed * t_orb)], dtype=np.float64)
                elif self.mode == "Lissajous 8":
                    target_pt = np.array([cur_r * np.sin(self.speed * t_orb), cur_r * np.sin(2.0 * self.speed * t_orb)], dtype=np.float64)
                else:
                    target_pt = np.zeros(2, dtype=np.float64)

            rho_d, dot_rho_d, ddot_rho_d = ref_model.update(target_pt)

            # 5. CONTROL & INVERSION
            u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)
            phi_raw, theta_raw, _ = inverter.invert(u_virt.B_des)

            tau_tilt = 0.05
            dot_phi = np.clip((phi_raw - phi_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            dot_theta = np.clip((theta_raw - theta_filt) / tau_tilt, -max_tilt_rate, max_tilt_rate)
            phi_filt += dot_phi * self.dt
            theta_filt += dot_theta * self.dt

            dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_filt, theta_filt, dot_theta, self.dt)
            psi_filt = yaw_nuller.accumulated_yaw

            # 6. INVERSE KINEMATICS
            q_cmd, _, _ = self.kin.inverse_kinematics(phi_filt, theta_filt, psi_filt, self.kin.T_p_nominal)

            # 7. SAFETY SOCP FILTER
            _, _, is_safe = safety_filter.filter_acceleration(
                u_q_cmd=np.zeros(6),
                state=state,
                kinematics=self.kin,
                q_meas=sensor_data.leg_positions,
                dot_q_meas=sensor_data.leg_velocities
            )

            # 8. ACTUATE
            cmd_packet = ActuatorCommandPacket(
                timestamp=t,
                mode=ActuatorMode.CSP,
                q_send=q_cmd,
                dot_q_send=np.zeros(6),
                u_q_applied=np.zeros(6),
                slack_value=0.0
            )
            self.backend.write_actuators(cmd_packet)

            # 9. TELEMETRY PUSH (At ~30 Hz: every 16 steps of 2 ms)
            if step % 16 == 0:
                telemetry = {
                    "t": t,
                    "ball_pos": rho_hat.copy(),
                    "ref_pos": rho_d.copy(),
                    "tilt": [float(np.degrees(phi_filt)), float(np.degrees(theta_filt))],
                    "normal_force": float(sensor_data.tactile_force_estimate),
                    "safety_ok": is_safe
                }
                if not self.t_queue.full():
                    self.t_queue.put_nowait(telemetry)

            step += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="SBC v30 - High-Speed Cockpit GUI.")
    parser.add_argument("--no-auto-launch", action="store_true", help="Connect to running CoppeliaSim.")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s]. Default: 0.002 s (500 Hz).")
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
        # Launch CoppeliaSim silently in background (headless)
        process = CoppeliaLauncher.start(scene_name="stewart_platform.ttt", headless=True)
        time.sleep(2.5)

    config = {
        "cycle_time": args.dt,
        "ball_mass": 0.06545,
        "ball_radius": 0.025,
        "gravity": 9.81,
        "coppelia": {"host": "localhost", "port": 23000}
    }
    backend = BackendFactory.create("coppelia", config)

    try:
        backend.connect()
        b_anchors, p_anchors, l_offsets, T_p_init = backend.extract_kinematic_parameters()
        kinematics = StewartKinematics(b_anchors, p_anchors, l_offsets, initial_translation=T_p_init)

        # Thread-safe queues
        telemetry_queue = queue.Queue(maxsize=10)
        command_queue = queue.Queue(maxsize=10)

        # Start 500 Hz real-time control thread
        ctrl_thread = ControllerThread(telemetry_queue, command_queue, backend, kinematics, dt=args.dt)
        ctrl_thread.start()

        def on_gui_command(cmd: dict):
            if not command_queue.full():
                command_queue.put_nowait(cmd)

        # Launch Tkinter Dashboard on main thread
        app = TelemetryDashboard(
            telemetry_queue=telemetry_queue,
            command_callback=on_gui_command,
            plate_radius=0.45
        )

        def on_closing():
            ctrl_thread.running = False
            ctrl_thread.join(timeout=1.0)
            backend.disconnect()
            if process is not None:
                CoppeliaLauncher.stop(process)
            app.destroy()

        app.protocol("WM_DELETE_WINDOW", on_closing)
        app.mainloop()

    except Exception as exc:
        print(f"[!] Error: {exc}")
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()