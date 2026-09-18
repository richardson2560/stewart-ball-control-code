# tools/run_gui.py

import argparse
import socket
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
from sbc.allocation.tilt_inversion import TiltInverter, ResidualTranslationAllocator
from sbc.allocation.yaw_nulling import YawNullingAllocator
from sbc.safety.socp_filter import ClarabelSafetyFilter
from sbc.actuation.joint_lead_comp import SafeJointTrajectoryIntegrator
from sbc.core.supervisor import Supervisor, SupervisorInputs
from sbc.datatypes import SupervisorMode
from sbc.gui.dashboard import TelemetryDashboard
from sbc.config import load_controller_config


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
            BALL_MASS = 0.065449846949792
            BALL_RADIUS = 0.025
            GRAVITY = 9.81

            tactile_proc = TactileProcessor(ball_radius=BALL_RADIUS, min_activation_force=0.10)
            filter_gp = HermiteGPFilter(window_size=11, cycle_time=self.dt)
            
            # Fast tracking reference model (a_max = 2.5 m/s^2)
            ref_model = HurwitzReferenceModel(omega_n=7.0, zeta=1.0, max_acceleration=2.5, cycle_time=self.dt)
            controller = SBCFullController(kp=6.5, kd=2.8, ball_mass=BALL_MASS, ball_radius=BALL_RADIUS, gravity=GRAVITY)
            inverter = TiltInverter(gravity=GRAVITY, max_roll=np.radians(16.0), max_pitch=np.radians(16.0), ball_radius=BALL_RADIUS)
            translation_allocator = ResidualTranslationAllocator(
                cycle_time=self.dt, max_acceleration=3.0, max_jerk=80.0,
                max_velocity=0.35, max_offset=0.060
            )
            att_filter = FastAttitudeFilter(omega_n=32.0, zeta=1.0, dt=self.dt)
            yaw_nuller = YawNullingAllocator(yaw_limit=np.radians(15.0))
            joint_limits = self.backend.read_joint_limits()
            q_min, q_max = (None, None) if joint_limits is None else joint_limits
            safety_filter = ClarabelSafetyFilter(
                ball_mass=BALL_MASS,
                ball_radius=BALL_RADIUS,
                plate_radius=self.plate_radius,
                q_min=q_min,
                q_max=q_max,
            )
            joint_integrator = SafeJointTrajectoryIntegrator(
                self.dt, safety_filter.q_min, safety_filter.q_max
            )
            supervisor = Supervisor()

            init_sensor = self.backend.read_sensors()
            ref_model.reset(initial_pos=init_sensor.tactile_pos_raw)
            initial_pose = self.backend.read_platform_pose()
            if initial_pose is None:
                att_filter.reset(0.0, 0.0)
                yaw_nuller.reset(0.0)
            else:
                _, initial_rotation = initial_pose
                phi_initial, theta_initial, psi_initial = self.kin.rotation_matrix_to_zyx(initial_rotation)
                att_filter.reset(phi_initial, theta_initial)
                yaw_nuller.reset(psi_initial)

            vel_est = np.zeros(2, dtype=np.float64)
            alpha_body_src = np.zeros(3, dtype=np.float64)
            previous_dot_psi = 0.0
            step = 0
            capture_complete = False
            capture_hold_steps = 0
            orbit_epoch = None
            previous_safety_signature = None
            previous_contact_valid = bool(init_sensor.tactile_active)

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
                            orbit_epoch = None
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
                rho_clean = tactile_proc.process(
                    sensor_data.tactile_pos_raw,
                    sensor_data.tactile_force_estimate,
                    vel_est,
                    is_geometric=backend.tactile_position_kind == "geometric_projection"
                )
                
                motion_measurement = self.backend.read_platform_motion()
                has_pose_feedback = motion_measurement is not None
                if has_pose_feedback:
                    T_p_cur, R_cur, twist_meas = motion_measurement
                    phi_cur, theta_cur, psi_cur = self.kin.rotation_matrix_to_zyx(R_cur)
                else:
                    phi_cur, theta_cur = att_filter.angles[0], att_filter.angles[1]
                    psi_cur = yaw_nuller.accumulated_yaw
                    T_p_cur = self.kin.T_p_nominal + translation_allocator.offset_I
                    R_cur = self.kin.get_rotation_matrix(phi_cur, theta_cur, psi_cur)
                J_inv_cur = self.kin.compute_inverse_jacobian(phi_cur, theta_cur, psi_cur, T_p_cur)
                if not has_pose_feedback:
                    twist_meas = np.linalg.pinv(J_inv_cur) @ sensor_data.leg_velocities
                q_from_measured_pose, _, _ = self.kin.inverse_kinematics(
                    phi_cur, theta_cur, psi_cur, T_p_cur
                )
                closure_error = float(np.max(np.abs(
                    q_from_measured_pose - sensor_data.leg_positions
                )))
                closure_tolerance = float(getattr(
                    self.backend, "kinematic_closure_tolerance", 2e-3
                ))
                if closure_error > closure_tolerance:
                    raise RuntimeError(
                        "Measured Stewart closure is inconsistent with the "
                        "actuator model; refusing a corrupt feedback loop "
                        f"(max |q_IK(T,R)-q_meas|={closure_error:.3e} m, "
                        f"tolerance={closure_tolerance:.3e} m)."
                    )
                measured_translation_valid = True
                if has_pose_feedback:
                    measured_translation_valid = translation_allocator.synchronize_measured_state(
                        T_p_cur - self.kin.T_p_nominal,
                        twist_meas[:3],
                        R_cur,
                    )
                yaw_rate_raw = float((R_cur.T @ twist_meas[3:])[2])

                rho_hat, vel_est, yaw_hat, alpha_z_hat = filter_gp.update(
                    rho_clean,
                    yaw_rate_raw,
                    measurement_valid=tactile_proc.contact_active,
                )
                contact_reacquired = bool(
                    tactile_proc.contact_active and not previous_contact_valid
                )
                if contact_reacquired:
                    # Bumpless restart of the rolling-mode states.  The
                    # airborne interval is outside the rolling ISS model.
                    ref_model.reset(rho_hat, np.zeros(2, dtype=np.float64))
                    att_filter.reset(phi_cur, theta_cur)
                    yaw_nuller.reset(psi_cur)
                    previous_dot_psi = 0.0
                    capture_complete = False
                    capture_hold_steps = 0
                    orbit_epoch = None
                previous_contact_valid = bool(tactile_proc.contact_active)

                # Do not start the requested orbit on wall-clock time alone.
                # Require a short, settled center capture first; if the ball
                # later returns close to the wall, re-enter capture mode.
                capture_condition = bool(
                    np.linalg.norm(rho_hat) <= 0.020
                    and np.linalg.norm(vel_est) <= 0.050
                )
                if capture_condition:
                    capture_hold_steps += 1
                else:
                    capture_hold_steps = 0
                if capture_hold_steps >= max(1, int(0.25 / self.dt)):
                    if not capture_complete:
                        orbit_epoch = None
                    capture_complete = True
                if np.linalg.norm(rho_hat) >= 0.90 * self.plate_radius:
                    capture_complete = False
                    capture_hold_steps = 0
                    orbit_epoch = None

                state = StateEstimatePacket(
                    timestamp=t,
                    platform_pos=T_p_cur,
                    platform_rot=R_cur,
                    platform_twist=twist_meas,
                    platform_accel_src=alpha_body_src,
                    ball_pos=rho_hat,
                    ball_vel=vel_est,
                    yaw_rate=yaw_hat,
                    contact_valid=tactile_proc.contact_active
                )

                # 4. TRAJECTORY GENERATION WITH FEEDFORWARD
                if not self.orbit_active or not capture_complete:
                    target_pt = np.zeros(2, dtype=np.float64)
                    v_cmd_ff = np.zeros(2, dtype=np.float64)
                    a_cmd_ff = np.zeros(2, dtype=np.float64)
                else:
                    if orbit_epoch is None:
                        orbit_epoch = t
                    t_orb = t - orbit_epoch
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

                # 5. CONTROL AND GRAVITY-FIRST / TRANSLATION-RESIDUAL ALLOCATION
                u_virt = controller.compute_control(state, rho_d, dot_rho_d, ddot_rho_d)
                phi_raw, theta_raw, _, tilt_only = inverter.project_preferred_gravity(u_virt.B_des)
                if not np.isfinite(phi_raw + theta_raw):
                    raise RuntimeError("Non-finite preferred-gravity allocation.")

                # Fast attitude filter
                att_angles, att_rates, att_accels = att_filter.update(np.array([phi_raw, theta_raw], dtype=np.float64))
                phi_d, theta_d = att_angles[0], att_angles[1]
                dot_phi_d, dot_theta_d = att_rates[0], att_rates[1]

                dot_psi_0, _ = yaw_nuller.compute_yaw_velocity(phi_d, theta_d, dot_theta_d, self.dt)
                psi_d = yaw_nuller.accumulated_yaw
                ddot_psi = (dot_psi_0 - previous_dot_psi) / self.dt
                previous_dot_psi = dot_psi_0

                euler_rates = np.array([dot_phi_d, dot_theta_d, dot_psi_0])
                euler_accels = np.array([att_accels[0], att_accels[1], ddot_psi])
                omega_P = self.kin.zyx_body_angular_velocity(phi_d, theta_d, euler_rates)
                alpha_P = self.kin.zyx_body_angular_acceleration(
                    phi_d, theta_d, euler_rates, euler_accels
                )
                alpha_body_src = alpha_P.copy()
                R_cmd = self.kin.get_rotation_matrix(phi_d, theta_d, psi_d)
                g_P = R_cmd.T @ np.array([0.0, 0.0, -GRAVITY])
                A_p_xy, B_allocation_residual, translation_exact = translation_allocator.allocate(
                    B_des=u_virt.B_des,
                    rotation_P_to_I=R_cmd,
                    gravity_parallel=g_P[:2],
                    alpha_P=alpha_P,
                    omega_P=omega_P,
                    rho=rho_hat,
                    ball_radius=BALL_RADIUS,
                    commit=has_pose_feedback and measured_translation_valid
                )

                # 6. DERIVATIVE-CONSISTENT SE(3) COMMAND (includes J_inv_dot)
                T_p_cmd = (
                    self.kin.T_p_nominal + translation_allocator.offset_I
                    if has_pose_feedback and measured_translation_valid
                    else T_p_cur
                )
                v_p_I = translation_allocator.velocity_I
                a_p_I = R_cmd @ np.array([A_p_xy[0], A_p_xy[1], 0.0])
                omega_I = R_cmd @ omega_P
                alpha_I = R_cmd @ alpha_P
                q_geom, dot_q_nom, u_q_feedforward = self.kin.actuator_trajectory(
                    phi_d, theta_d, psi_d, T_p_cmd, v_p_I, omega_I, a_p_I, alpha_I
                )
                u_q_nom = joint_integrator.tracking_acceleration(
                    sensor_data.leg_positions,
                    sensor_data.leg_velocities,
                    q_geom,
                    dot_q_nom,
                    u_q_feedforward,
                )

                # 7. SAFETY SOCP
                u_q_applied, slack_value, is_safe = safety_filter.filter_acceleration(
                    u_q_cmd=u_q_nom,
                    state=state,
                    kinematics=self.kin,
                    q_meas=sensor_data.leg_positions,
                    dot_q_meas=sensor_data.leg_velocities
                )
                model_valid = bool(
                    measured_translation_valid
                    and
                    translation_allocator.within_limits
                    and np.linalg.cond(J_inv_cur) <= 1e6
                    and np.all(np.isfinite(B_allocation_residual))
                )
                supervisor_mode = supervisor.update(SupervisorInputs(
                    bus_healthy=sensor_data.bus_healthy,
                    contact_valid=state.contact_valid,
                    model_valid=model_valid,
                    strict_solution=is_safe,
                    fallback_admissible=safety_filter.fallback_admissible,
                    yaw_unwind_requested=yaw_nuller.is_unwinding,
                ))
                safety_signature = (
                    supervisor_mode.name,
                    safety_filter.last_solver_status,
                    safety_filter.last_reason,
                )
                if safety_signature != previous_safety_signature:
                    stroke_margin = float(np.min(np.hstack([
                        sensor_data.leg_positions - safety_filter.q_min,
                        safety_filter.q_max - sensor_data.leg_positions,
                    ])))
                    ball_height = getattr(
                        self.backend, "last_ball_height_relative", float("nan")
                    )
                    tracking_error = float(np.max(np.abs(
                        q_geom - sensor_data.leg_positions
                    )))
                    translation_norm = float(np.linalg.norm(
                        T_p_cur - self.kin.T_p_nominal
                    ))
                    print(
                        "[SBC safety] "
                        f"mode={safety_signature[0]} "
                        f"solver={safety_signature[1]} "
                        f"reason={safety_signature[2]} "
                        f"ball_h={1000.0 * ball_height:+.2f}mm "
                        f"stroke_margin={1000.0 * stroke_margin:.2f}mm "
                        f"closure={1000.0 * closure_error:.3f}mm "
                        f"q_track={1000.0 * tracking_error:.3f}mm "
                        f"platform_offset={1000.0 * translation_norm:.2f}mm "
                        f"rho={np.linalg.norm(rho_hat):.4f}m "
                        f"speed={np.linalg.norm(vel_est):.4f}m/s "
                        f"perception={'dropout' if filter_gp.dropout_active else 'valid'}"
                    )
                    previous_safety_signature = safety_signature
                if supervisor_mode in {SupervisorMode.FAILSAFE_HOLD, SupervisorMode.MODEL_EXIT}:
                    u_q_applied, emergency_stroke_safe = safety_filter.emergency_braking(
                        sensor_data.leg_positions,
                        sensor_data.leg_velocities,
                    )
                    if not emergency_stroke_safe:
                        raise RuntimeError(
                            "No bounded one-step stroke-safe emergency action exists."
                        )
                    is_safe = False

                # Coppelia supplies platform pose directly.  Never integrate a
                # second, conflicting pose from finite-differenced leg motion.
                # The observer fallback is retained only for backends without
                # a direct pose/FK estimate.
                if not has_pose_feedback:
                    J_dot_cur = self.kin.compute_inverse_jacobian_dot(
                        phi_cur, theta_cur, psi_cur, T_p_cur,
                        twist_meas[:3], twist_meas[3:]
                    )
                    xi_dot_applied = np.linalg.pinv(J_inv_cur) @ (
                        u_q_applied - J_dot_cur @ twist_meas
                    )
                    if not translation_allocator.commit_realized(xi_dot_applied[:3]):
                        u_q_applied = np.clip(
                            -8.0 * sensor_data.leg_velocities, -8.0, 8.0
                        )
                        is_safe = False

                # 8. ACTUATION.  The safety output has actual authority.
                joint_command = joint_integrator.integrate(
                    sensor_data.leg_positions, sensor_data.leg_velocities, u_q_applied
                )
                cmd_packet = ActuatorCommandPacket(
                    timestamp=t,
                    mode=ActuatorMode.CSP,
                    q_send=joint_command.q_send,
                    dot_q_send=joint_command.q_dot_safe,
                    u_q_applied=u_q_applied,
                    slack_value=slack_value
                )
                if not self.backend.write_actuators(cmd_packet):
                    backend_reason = getattr(self.backend, "last_error", "")
                    raise RuntimeError(
                        "Actuator command was not acknowledged"
                        + (f": {backend_reason}" if backend_reason else ".")
                    )

                # 9. TELEMETRY STREAM TO GUI (~30 Hz)
                if step % 16 == 0:
                    p_trans = ((T_p_cur - self.kin.T_p_nominal) * 1000.0).tolist()
                    p_vel = (twist_meas[:3] * 1000.0).tolist()

                    telemetry = {
                        "t": t,
                        "ball_pos": rho_hat.copy(),
                        "ball_pos_raw": sensor_data.tactile_pos_raw.copy(),
                        "ball_vel": vel_est.copy(),
                        "ref_pos": rho_d.copy(),
                        "tilt": [float(np.degrees(phi_cur)), float(np.degrees(theta_cur)), float(np.degrees(psi_cur))],
                        "tilt_command": [float(np.degrees(phi_d)), float(np.degrees(theta_d)), float(np.degrees(psi_d))],
                        "normal_force": float(sensor_data.tactile_force_estimate),
                        "safety_ok": is_safe,
                        "supervisor_mode": supervisor_mode.name,
                        "safety_reason": safety_filter.last_reason,
                        "safety_linear_violation": safety_filter.last_linear_violation,
                        "safety_cone_violation": safety_filter.last_cone_violation,
                        "tilt_only": tilt_only,
                        "translation_exact": translation_exact,
                        "allocation_residual": float(np.linalg.norm(B_allocation_residual)),
                        "capture_complete": capture_complete,
                        "perception_dropout": filter_gp.dropout_active,
                        "perception_reinitialized": filter_gp.reinitialized,
                        "ball_height_relative": getattr(
                            self.backend,
                            "last_ball_height_relative",
                            float("nan"),
                        ),
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
    parser.add_argument("--port", type=int, default=23000, help="Remote API port; use with --no-auto-launch for an existing instance.")
    args = parser.parse_args()

    process = None
    if not 1 <= args.port <= 65535:
        parser.error("Invalid TCP port.")
    if not args.no_auto_launch:
        if args.port != 23000:
            parser.error("For a custom port, launch Coppelia explicitly and use --no-auto-launch.")
        with socket.socket() as probe:
            probe.settimeout(0.5)
            occupied = probe.connect_ex(("localhost", args.port)) == 0
        if occupied:
            raise RuntimeError(
                "Port 23000 is already occupied. No second simulator was launched. "
                "Close the old instance or use --no-auto-launch --port with the "
                "port printed by the intended Coppelia instance."
            )
        controller_config = load_controller_config()
        scene_name = controller_config["coppelia"]["scene_name"]
        process = CoppeliaLauncher.start(scene_name=scene_name, headless=True)
        time.sleep(2.5)

    config = {
        "cycle_time": args.dt,
        "ball_mass":0.065449846949792,
        "ball_radius": 0.025,
        "gravity": 9.81,
        "coppelia": {
            "host": "localhost",
            "port": args.port,
            "joint_command_mode": "kinematic",
            "actuator_mode": "CSP",
            "scene_name": "stewart_platform_ideal.ttt",
        }
    }
    backend = BackendFactory.create("coppelia", config)

    try:
        print(f"[SBC connection] localhost:{args.port}")
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
