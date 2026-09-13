# tools/run_gui.py

import argparse
import queue
import threading
import time
import traceback
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


class FastAttitudeFilter:
    """
    Second-order Hurwitz attitude filter tuned for high-speed tracking (omega_n = 32 rad/s).
    Minimizes group delay to tau_delay = 2*zeta/omega_n = 62.5 ms, preventing phase resonance in sharp curves.
    """
    def __init__(self, omega_n: float = 32.0, zeta: float = 1.0, dt: float = 0.002):
        self._dt = dt
        self._kp = omega_n ** 2
        self._kv = 2.0 * zeta * omega_n
        self.angles = np.zeros(2, dtype=np.float64)
        self.rates = np.zeros(2, dtype=np.float64)
        self.accels = np.zeros(2, dtype=np.float64)

    def reset(self, phi0: float = 0.0, theta0: float = 0.0):
        self.angles = np.array([phi0, theta0], dtype=np.float64)
        self.rates.fill(0.0)
        self.accels.fill(0.0)

    def update(self, target_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        error = target_angles - self.angles
        self.accels = self._kp * error - self._kv * self.rates
        
        acc_norm = float(np.linalg.norm(self.accels))
        if acc_norm > 25.0:  # Allow higher acceleration for sharp curves
            self.accels = (25.0 / acc_norm) * self.accels

        self.rates += self._dt * self.accels
        self.angles += self._dt * self.rates
        return self.angles.copy(), self.rates.copy(), self.accels.copy()


class ControlThread(threading.Thread):
    def __init__(
        self,
        telemetry_queue: queue.Queue,
        command_queue: queue.Queue,
        backend,
        kinematics: StewartKinematics,
        plate_radius: float = 0.25,
        dt: float = 0.002
    ) -> None:
        super().__init__(daemon=True)
        self.t_queue = telemetry_queue
        self.cmd_queue = command_queue
        self.backend = backend
        self.kin = kinematics
        self.plate_radius = plate_radius
        self.dt = dt
        self.running = True

        self.mode = "Lissajous 2:3"
        self.p1 = 0.08      # Amplitude R
        self.p2 = 0.75      # Base frequency w
        self.orbit_active = True

    def run(self) -> None:
        try:
            BALL_MASS = 0.06545
            BALL_RADIUS = 0.025
            GRAVITY = 9.81

            tactile_proc = TactileProcessor(ball_radius=BALL_RADIUS, min_activation_force=0.10)
            filter_gp = HermiteGPFilter(window_size=11, cycle_time=self.dt)
            
            # Fast tracking reference model (a_max = 2.5 m/s^2)
            ref_model = HurwitzReferenceModel(omega_n=7.0, zeta=1.0, max_acceleration=2.5, cycle_time=self.dt)
            controller = SBCFullController(kp=6.5, kd=2.8, ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, gravity=GRAVITY)
            inverter = TiltInverter(gravity=GRAVITY, max_roll=np.radians(16.0), max_pitch=np.radians(16.0), ball_radius=BALL_RADIUS)
            att_filter = FastAttitudeFilter(omega_n=32.0, zeta=1.0, dt=self.dt)
            yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))
            safety_filter = ClarabelSafetyFilter(ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, plate_radius=self.plate_radius)

            init_sensor = self.backend.read_sensors()
            ref_model.reset(initial_pos=init_sensor.tactile_pos_raw)
            att_filter.reset(0.0, 0.0)

            vel_est = np.zeros(2, dtype=np.float64)
            step = 0

            while self.running:
                # 1. Asynchronous GUI Commands
                try:
                    while not self.cmd_queue.empty():
                        cmd = self.cmd_queue.get_nowait()
                        action = cmd.get("action")
                        if action == "center":
                            self.orbit_active = False
                        elif action == "orbit":
                            self.orbit_active = True
                        elif action == "update_params":
                            self.mode = cmd.get("mode", self.mode)
                            self.p1 = cmd.get("p1", self.p1)
                            self.p2 = cmd.get("p2", self.p2)
                except queue.Empty:
                    pass

                # 2. SENSE
                sensor_data = self.backend.read_sensors()
                t = sensor_data.timestamp

                # 3. PERCEPTION
                rho_clean = tactile_proc.process(sensor_data.tactile_pos_raw, sensor_data.tactile_force_estimate, vel_est)
                
                phi_cur, theta_cur = att_filter.angles[0], att_filter.angles[1]
                psi_cur = yaw_nuller.accumulated_yaw
                J_inv_cur = self.kin.compute_inverse_jacobian(phi_cur, theta_cur, psi_cur, self.kin.T_p_nominal)
                twist_meas = np.linalg.pinv(J_inv_cur) @ sensor_data.leg_velocities
                yaw_rate_raw = float(twist_meas[5])

                rho_hat, vel_est, yaw_hat, alpha_z_hat = filter_gp.update(rho_clean, yaw_rate_raw)

                R_cur = self.kin.get_rotation_matrix(phi_cur, theta_cur, psi_cur)
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

                # 4. TRAJECTORY GENERATION WITH FEEDFORWARD
                if not self.orbit_active or t < 2.0:
                    target_pt = np.zeros(2, dtype=np.float64)
                    v_cmd_ff = np.zeros(2, dtype=np.float64)
                    a_cmd_ff = np.zeros(2, dtype=np.float64)
                else:
                    t_orb = t - 2.0
                    ramp = np.clip(t_orb / 1.5, 0.0, 1.0)
                    R = self.p1 * ramp
                    w = self.p2

                    if self.mode == "Static Point":
                        target_pt = np.array([self.p1, self.p2], dtype=np.float64)
                        v_cmd_ff = np.zeros(2, dtype=np.float64)
                        a_cmd_ff = np.zeros(2, dtype=np.float64)

                    elif "1:2" in self.mode:  # Lissajous 1:2
                        target_pt = np.array([R * np.sin(w * t_orb), R * np.sin(2.0 * w * t_orb)], dtype=np.float64)
                        v_cmd_ff = np.array([w * R * np.cos(w * t_orb), 2.0 * w * R * np.cos(2.0 * w * t_orb)], dtype=np.float64)
                        a_cmd_ff = np.array([-(w**2) * R * np.sin(w * t_orb), -4.0 * (w**2) * R * np.sin(2.0 * w * t_orb)], dtype=np.float64)

                    elif "2:3" in self.mode:  # Lissajous 2:3
                        target_pt = np.array([R * np.sin(2.0 * w * t_orb), R * np.cos(3.0 * w * t_orb)], dtype=np.float64)
                        v_cmd_ff = np.array([2.0 * w * R * np.cos(2.0 * w * t_orb), -3.0 * w * R * np.sin(3.0 * w * t_orb)], dtype=np.float64)
                        a_cmd_ff = np.array([-4.0 * (w**2) * R * np.sin(2.0 * w * t_orb), -9.0 * (w**2) * R * np.cos(3.0 * w * t_orb)], dtype=np.float64)

                    else:  # Circle Orbit
                        target_pt = np.array([R * np.cos(w * t_orb), R * np.sin(w * t_orb)], dtype=np.float64)
                        v_cmd_ff = np.array([-w * R * np.sin(w * t_orb), w * R * np.cos(w * t_orb)], dtype=np.float64)
                        a_cmd_ff = np.array([-(w**2) * R * np.cos(w * t_orb), -(w**2) * R * np.sin(w * t_orb)], dtype=np.float64)

                rho_d, dot_rho_d, ddot_rho_d = ref_model.update(target_pt, v_cmd_ff, a_cmd_ff)

                # 5. CONTROL & FAST TILT INVERSION
                u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)
                phi_raw, theta_raw, _ = inverter.invert(u_virt.B_des)

                # Fast attitude filter
                att_angles, att_rates, _ = att_filter.update(np.array([phi_raw, theta_raw], dtype=np.float64))
                phi_d, theta_d = att_angles[0], att_angles[1]
                dot_phi_d, dot_theta_d = att_rates[0], att_rates[1]

                dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_d, theta_d, dot_theta_d, self.dt)
                psi_d = yaw_nuller.accumulated_yaw

                # 6. INVERSE KINEMATICS
                q_cmd, _, _ = self.kin.inverse_kinematics(phi_d, theta_d, psi_d, self.kin.T_p_nominal)

                # 7. SAFETY SOCP
                _, _, is_safe = safety_filter.filter_acceleration(
                    u_q_cmd=np.zeros(6),
                    state=state,
                    kinematics=self.kin,
                    q_meas=sensor_data.leg_positions,
                    dot_q_meas=sensor_data.leg_velocities
                )

                # 8. ACTUATION
                cmd_packet = ActuatorCommandPacket(
                    timestamp=t,
                    mode=ActuatorMode.CSP,
                    q_send=q_cmd,
                    dot_q_send=np.zeros(6),
                    u_q_applied=np.zeros(6),
                    slack_value=0.0
                )
                self.backend.write_actuators(cmd_packet)

                # 9. TELEMETRY STREAM TO GUI (~30 Hz)
                if step % 16 == 0:
                    # Query real platform center translation and linear velocity from CoppeliaSim
                    try:
                        sim = self.backend._sim
                        plate_pos = sim.getObjectPosition(self.backend._h_plate, self.backend._h_base)
                        lin_vel, _ = sim.getObjectVelocity(self.backend._h_plate)
                        
                        # Translation offset in mm relative to nominal
                        p_trans = [
                            (plate_pos[0] - self.kin.T_p_nominal[0]) * 1000.0,
                            (plate_pos[1] - self.kin.T_p_nominal[1]) * 1000.0,
                            (plate_pos[2] - self.kin.T_p_nominal[2]) * 1000.0
                        ]
                        p_vel = [lin_vel[0] * 1000.0, lin_vel[1] * 1000.0, lin_vel[2] * 1000.0]
                    except Exception:
                        p_trans = [0.0, 0.0, 0.0]
                        p_vel = [0.0, 0.0, 0.0]

                    telemetry = {
                        "t": t,
                        "ball_pos": rho_hat.copy(),
                        "ball_pos_raw": sensor_data.tactile_pos_raw.copy(),
                        "ball_vel": vel_est.copy(),
                        "ref_pos": rho_d.copy(),
                        "tilt": [float(np.degrees(phi_d)), float(np.degrees(theta_d)), float(np.degrees(psi_d))],
                        "normal_force": float(sensor_data.tactile_force_estimate),
                        "safety_ok": is_safe,
                        "platform_trans": p_trans,
                        "platform_vel": p_vel
                    }
                    if not self.t_queue.full():
                        self.t_queue.put_nowait(telemetry)

                step += 1

        except Exception as exc:
            print(f"[!] Critical error in ControlThread: {exc}")
            traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description="SBC v30 - High-Speed Cockpit GUI.")
    parser.add_argument("--no-auto-launch", action="store_true", help="Connect to running CoppeliaSim instance.")
    parser.add_argument("--plate-radius", type=float, default=0.25, help="Platform boundary radius [m]. Default: 0.25 m.")
    parser.add_argument("--dt", type=float, default=0.002, help="Nominal period [s] (500 Hz).")
    args = parser.parse_args()

    process = None
    if not args.no_auto_launch:
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

        telemetry_queue = queue.Queue(maxsize=10)
        command_queue = queue.Queue(maxsize=10)

        ctrl_thread = ControlThread(
            telemetry_queue=telemetry_queue,
            command_queue=command_queue,
            backend=backend,
            kinematics=kinematics,
            plate_radius=args.plate_radius,
            dt=args.dt
        )
        ctrl_thread.start()

        def on_gui_command(cmd: dict):
            if not command_queue.full():
                command_queue.put_nowait(cmd)

        app = TelemetryDashboard(
            telemetry_queue=telemetry_queue,
            command_callback=on_gui_command,
            plate_radius=args.plate_radius
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
        print(f"[!] Error in main: {exc}")
        traceback.print_exc()
        backend.disconnect()
        if process is not None:
            CoppeliaLauncher.stop(process)


if __name__ == "__main__":
    main()