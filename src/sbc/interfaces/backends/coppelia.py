# src/sbc/interfaces/backends/coppelia.py

import os
import sys
import time
import numpy as np
from typing import Optional, List, Tuple

from sbc.interfaces.base_backend import BasePlatformBackend
from sbc.datatypes import (
    RawSensorPacket,
    ActuatorCommandPacket,
    PlatformStatus,
    ActuatorMode
)


class CoppeliaBackend(BasePlatformBackend):
    """
    Synchronous stepping CoppeliaSim backend adhering to SBC v30 HAL.
    Communicates via ZeroMQ Remote API without contaminating data with
    runtime filters or controller dependencies.
    """

    def __init__(
        self,
        cycle_time: float = 0.002,
        host: str = "localhost",
        port: int = 23000,
        ball_mass: float = 0.065449846949792,
        ball_radius: float = 0.025,
        gravity: float = 9.81,
        joint_command_mode: str = "dynamic",
        kinematic_closure_tolerance: float = 5e-4,
        scene_config: Optional[dict] = None,
        actuator_mode: str = "CSP",
        shutdown_timeout: float = 2.0,
    ) -> None:
        super().__init__(cycle_time=cycle_time)
        if not np.isfinite(cycle_time) or cycle_time <= 0:
            raise ValueError("cycle_time must be finite and positive")
        self._scene_config = dict(scene_config or {})
        if actuator_mode not in {"CSP", "CSV"}:
            raise ValueError("actuator_mode must be CSP or CSV")
        self._actuator_mode = actuator_mode
        self._shutdown_timeout = float(shutdown_timeout)
        if not np.isfinite(self._shutdown_timeout) or self._shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be finite and positive")
        self._started_by_backend = False
        self._h_surface = -1
        self._h_base_anchors = []
        self._h_platform_anchors = []
        self._host = host
        self._port = port
        self._m = ball_mass
        self._r = ball_radius
        self._g = gravity
        if joint_command_mode not in {"kinematic", "dynamic"}:
            raise ValueError("joint_command_mode must be 'kinematic' or 'dynamic'.")
        self._joint_command_mode = joint_command_mode
        self._kinematic_closure_tolerance = float(kinematic_closure_tolerance)
        if not np.isfinite(self._kinematic_closure_tolerance) or self._kinematic_closure_tolerance <= 0.0:
            raise ValueError("kinematic_closure_tolerance must be positive.")

        # ZeroMQ client and simulator references
        self._client = None
        self._sim = None
        self._sim_ik = None

        # Scene object handles
        self._h_base: int = -1
        self._h_plate: int = -1
        self._h_ball: int = -1
        self._h_motors: List[int] = []
        self._h_tips: List[int] = []
        self._h_targets: List[int] = []
        self._h_ik_tips: List[int] = []
        # Coordinate transform from Coppelia's signed prismatic coordinate to
        # the SBC convention in which positive q lengthens a leg.
        self._motor_length_signs = np.ones(6, dtype=np.float64)
        self._binding_summary = "not_resolved"

        # Kinematics solver handles (internal to CoppeliaSim IK group)
        self._ik_env = None
        self._ik_group = None

        # Previous state cache for finite-difference fallback in simulation
        self._prev_q = np.zeros(6, dtype=np.float64)
        self._has_prev_q = False
        self._prev_platform_position = np.zeros(3, dtype=np.float64)
        self._prev_platform_rotation = np.eye(3, dtype=np.float64)
        self._has_prev_platform_pose = False
        self._last_step_duration = 0.0
        self._last_error = ""
        self._last_ball_height_relative = float("nan")
        self._contact_active = False
        self._last_kinematic_qdot = np.zeros(6, dtype=np.float64)
        self._has_kinematic_command = False

    @property
    def last_step_duration(self) -> float:
        return self._last_step_duration

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def last_ball_height_relative(self) -> float:
        return self._last_ball_height_relative

    @property
    def ideal_kinematic_csp(self) -> bool:
        return self._joint_command_mode == "kinematic"

    @property
    def kinematic_closure_tolerance(self) -> float:
        return self._kinematic_closure_tolerance

    def connect(self) -> bool:
        """Connect to a stopped scene. Never silently convert its mechanics."""
        if self._status == PlatformStatus.OPERATIONAL:
            return True
        self._status = PlatformStatus.INITIALIZING
        self._last_error = ""
        try:
            try:
                from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            except ImportError:
                zmq_path = os.environ.get("COPPELIASIM_ZMQ_PATH")
                if not zmq_path:
                    raise RuntimeError("Install coppeliasim-zmqremoteapi-client")
                if zmq_path not in sys.path:
                    sys.path.append(zmq_path)
                from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self._client = RemoteAPIClient(host=self._host, port=self._port)
            self._sim = self._client.require("sim")
            if self._sim.getSimulationState() != self._sim.simulation_stopped:
                raise RuntimeError("Stop the scene before connecting the backend")
            self._resolve_scene_handles()
            self._validate_scene()
            if self._joint_command_mode == "kinematic":
                self._sim_ik = self._client.require("simIK")
                self._setup_kinematic_solver()
            self._resolve_geometry_markers()
            self.read_joint_limits()
            self._sim.setFloatParam(self._sim.floatparam_simulation_time_step, self._dt)
            # Phase 1 contact reader uses pass 0: require exactly one physics pass.
            if self._joint_command_mode == "dynamic":
                physics_dt = float(self._sim.getFloatParam(self._sim.floatparam_physicstimestep))
                if not np.isclose(physics_dt, self._dt, rtol=1e-6, atol=1e-9):
                    raise RuntimeError(
                        f"Set scene physics timestep to {self._dt} s; got {physics_dt}. "
                        "Phase 1 requires one physics pass per simulation step."
                    )
                for motor in self._h_motors:
                    if self._actuator_mode == "CSP":
                        self._sim.setJointTargetPosition(motor, self._sim.getJointPosition(motor))
                    else:
                        self._sim.setJointTargetVelocity(motor, 0.0)
            self._client.setStepping(True)
            self._sim.startSimulation()
            self._started_by_backend = True
            self._client.step()
            if self._joint_command_mode == "dynamic":
                self._validate_dynamic_enabled()
            self._has_prev_q = False
            self._has_prev_platform_pose = False
            self._status = PlatformStatus.OPERATIONAL
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._cleanup(stop=self._started_by_backend)
            self._status = PlatformStatus.FAULT
            raise RuntimeError(f"Coppelia backend initialization rejected: {exc}") from exc

    def _get_object_handle(self, name: str) -> int:
        """Tries resolving with and without leading slash without throwing raw RPC exceptions."""
        candidates = [name]
        if name.startswith('/'):
            candidates.append(name[1:])
        else:
            candidates.append('/' + name)

        for path in candidates:
            h = self._sim.getObject(path, {'noError': True})
            if h != -1:
                return h

        # Commissioning markers can be nested below the measured platform
        # frame. Resolve only when the terminal alias is globally unique;
        # ambiguity remains a hard error rather than a guessed binding.
        alias = name.rstrip('/').split('/')[-1]
        matches = [
            h for h in self._sim.getObjectsInTree(
                self._sim.handle_scene, self._sim.handle_all, 0
            )
            if self._sim.getObjectAlias(h) == alias
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Object path '{name}' did not resolve and alias '{alias}' is ambiguous."
            )

        scene_name = self._sim.getStringParam(self._sim.stringparam_scene_name)
        raise RuntimeError(
            f"Object '{name}' not found in CoppeliaSim. "
            f"Active scene: '{scene_name}'. "
            f"Verify that 'simulations/coppelia/stewart_platform.ttt' is loaded."
        )

    def _resolve_scene_handles(self) -> None:
        cfg = self._scene_config
        self._h_base = self._get_object_handle(cfg.get("base", "/stewartPlatform"))
        self._h_plate = self._get_object_handle(cfg.get("plate_frame", "/stewartPlatform/tip"))
        self._h_surface = self._get_object_handle(cfg.get("surface", "/platformTable"))
        self._h_ball = self._get_object_handle(cfg.get("ball", "/Sphere"))
        order = [2, 3, 4, 5, 6, 1]  # Keep existing public actuator order.
        paths = cfg.get("motors", [f"/stewartPlatform/motor{i}" for i in order])
        if len(paths) != 6:
            raise ValueError("scene.motors must contain six paths")
        self._h_motors = [self._get_object_handle(x) for x in paths]
        if len(set(self._h_motors)) != 6:
            raise ValueError("Motor handles must be distinct")
        self._motor_length_signs = np.asarray(cfg.get("motor_length_signs", [1]*6), dtype=float)
        if self._motor_length_signs.shape != (6,) or not np.all(np.isin(self._motor_length_signs, [-1, 1])):
            raise ValueError("motor_length_signs must contain six +1/-1 signs")
        tips = cfg.get("closure_tips", [f"/stewartPlatform/downArm{i}Tip" for i in range(1,6)])
        targets = cfg.get("closure_targets", [f"/stewartPlatform/downArm{i}Target" for i in range(1,6)])
        if len(tips) != 5 or len(targets) != 5:
            raise ValueError("Five closure tip/target pairs are required")
        self._h_ik_tips = [self._get_object_handle(x) for x in tips]
        self._h_targets = [self._get_object_handle(x) for x in targets]
        self._binding_summary = "actuator order [2,3,4,5,6,1]; signs require commissioning"

    def _validate_scene(self) -> None:
        sim = self._sim
        errors = []
        dynamic = self._joint_command_mode == "dynamic"
        expected = sim.jointmode_dynamic if dynamic else sim.jointmode_kinematic
        for motor in self._h_motors:
            if sim.getObjectType(motor) != sim.object_joint_type:
                errors.append(f"motor handle {motor} is not a joint")
                continue
            if sim.getJointType(motor) != sim.joint_prismatic_subtype:
                errors.append(f"motor {motor} is not prismatic")
            if sim.getJointMode(motor) != expected:
                errors.append(f"motor {motor} mode != {self._joint_command_mode}")
            if dynamic:
                control = sim.getObjectInt32Param(motor, sim.jointintparam_dynctrlmode)
                wanted = sim.jointdynctrl_position if self._actuator_mode == "CSP" else sim.jointdynctrl_velocity
                if control != wanted:
                    errors.append(f"motor {motor} control incompatible with {self._actuator_mode}")
                force = float(sim.getJointTargetForce(motor))
                if not np.isfinite(force) or abs(force) <= 0:
                    errors.append(f"motor {motor} requires a finite nonzero force limit")
        for tip, target in zip(self._h_ik_tips, self._h_targets):
            # addElementFromScene accepts general scene objects in the
            # kinematic validation model.  A physical dynamics-loop closure,
            # however, must be a linked dummy pair.
            if dynamic:
                if any(sim.getObjectType(h) != sim.object_dummy_type for h in (tip, target)):
                    errors.append(
                        f"closure {tip}/{target}: dynamic mode requires linked DUMMIES"
                    )
                    continue
                if sim.getLinkDummy(tip) != target or sim.getLinkDummy(target) != tip:
                    errors.append(f"closure {tip}/{target} is not reciprocally linked")
                for h in (tip, target):
                    if sim.getObjectInt32Param(h, sim.dummyintparam_link_type) != sim.dummy_linktype_dynamics_loop_closure:
                        errors.append(f"dummy {h} is not a dynamic loop closure")
        if dynamic:
            for name, h in (("surface", self._h_surface), ("ball", self._h_ball)):
                if sim.getObjectType(h) != sim.object_shape_type:
                    errors.append(f"{name} must be a shape")
                    continue
                if sim.getObjectInt32Param(h, sim.shapeintparam_static):
                    errors.append(f"{name} is static")
                if not sim.getObjectInt32Param(h, sim.shapeintparam_respondable):
                    errors.append(f"{name} is not respondable")
            if not sim.getObjectInt32Param(self._h_base, sim.shapeintparam_static):
                errors.append("base must be fixed/static for this backend frame convention")
        if errors:
            raise RuntimeError("Scene preflight failed:\n - " + "\n - ".join(errors))

    def _validate_dynamic_enabled(self) -> None:
        handles = list(self._h_motors) + [self._h_surface, self._h_ball]
        handles += list(self._sim.getObjectsInTree(self._h_base, self._sim.object_joint_type, 0))
        disabled = [h for h in set(handles) if not self._sim.isDynamicallyEnabled(h)]
        if disabled:
            raise RuntimeError(f"Not dynamically enabled after first physics step: {disabled}")
        self._verify_closure()

    def _verify_closure(self) -> None:
        for tip, target in zip(self._h_ik_tips, self._h_targets):
            delta = np.asarray(self._sim.getObjectPosition(tip, target), dtype=float)
            if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) > self._kinematic_closure_tolerance:
                raise RuntimeError(f"Closure {tip}/{target} outside configured tolerance")

    def _resolve_geometry_markers(self) -> None:
        # Markers must be created at physical joint centers. No inferred motor-origin anchors.
        cfg = self._scene_config
        order = [2, 3, 4, 5, 6, 1]
        bp = cfg.get("base_anchors", [f"/stewartPlatform/baseAnchor{i}" for i in order])
        pp = cfg.get("platform_anchors", [f"/stewartPlatform/platformAnchor{i}" for i in order])
        if len(bp) != 6 or len(pp) != 6:
            raise ValueError("Six explicit base/platform anchor markers are required")
        self._h_base_anchors = [self._get_object_handle(x) for x in bp]
        self._h_platform_anchors = [self._get_object_handle(x) for x in pp]
        self._h_tips = self._h_platform_anchors.copy()

    def _setup_kinematic_solver(self) -> None:
        """Sets up closed-loop kinematic constraint resolution inside CoppeliaSim."""
        self._ik_env = self._sim_ik.createEnvironment()
        self._ik_group = self._sim_ik.createGroup(self._ik_env)

        sim_to_ik_map_total = {}
        for i in range(5):
            _, sim_to_ik, _ = self._sim_ik.addElementFromScene(
                self._ik_env,
                self._ik_group,
                self._h_base,
                self._h_ik_tips[i],
                self._h_targets[i],
                self._sim_ik.constraint_position
            )
            sim_to_ik_map_total.update(sim_to_ik)

        for motor in self._h_motors:
            if motor in sim_to_ik_map_total:
                ik_motor = sim_to_ik_map_total[motor]
                self._sim_ik.setJointMode(self._ik_env, ik_motor, self._sim_ik.jointmode_passive)

    def _update_contact_state(self, ball_height_relative: float) -> bool:
        """Classify geometric contact with hysteresis around the ball radius."""
        # Preserve the scene's original 3 mm geometric tolerance, with only a
        # +/-0.5 mm hysteresis band.  This rejects threshold chatter without
        # pretending that a clearly airborne ball remains observable.
        acquire_threshold = self._r + 0.0025
        release_threshold = self._r + 0.0035
        if self._contact_active:
            self._contact_active = ball_height_relative <= release_threshold
        else:
            self._contact_active = ball_height_relative <= acquire_threshold
        return self._contact_active

    def read_sensors(self) -> RawSensorPacket:
        """
        Extracts raw leg states and contact coordinates without applying low-pass filters.
        Tactile surface coordinate is the ball contact point on the top plate.
        """
        if self._status != PlatformStatus.OPERATIONAL:
            raise RuntimeError("Cannot read sensors: CoppeliaBackend is not operational.")

        t = float(self._sim.getSimulationTime())

        # 1. Read prismatic leg positions
        q_meas = np.zeros(6, dtype=np.float64)
        for i, motor in enumerate(self._h_motors):
            q_meas[i] = (
                self._motor_length_signs[i]
                * float(self._sim.getJointPosition(motor))
            )

        # Simulator velocities are interval measurements, not endpoint derivatives.
        dot_q_meas = np.array([
            sign * float(self._sim.getJointVelocity(h))
            for sign, h in zip(self._motor_length_signs, self._h_motors)
        ])

        # 3. Read ball position relative to moving platform plate frame P
        # CoppeliaSim convention: Plate origin is at the center of the surface
        pos_relative = self._sim.getObjectPosition(self._h_ball, self._h_plate)
        tactile_pos_raw = np.array([pos_relative[0], pos_relative[1]], dtype=np.float64)
        ball_height_rel = float(pos_relative[2])
        self._last_ball_height_relative = ball_height_rel

        # The ball remains a dynamic body in the kinematic-platform validation
        # scene.  Use the physics-engine contact in both modes; never synthesize
        # N=mg and present it as measured evidence.
        is_touching, normal_force = self._read_contact_load()
        self._contact_active = is_touching
        if not np.all(np.isfinite(np.r_[q_meas, dot_q_meas, tactile_pos_raw, normal_force])):
            raise RuntimeError("Non-finite sensor sample")

        return RawSensorPacket(
            timestamp=t,
            leg_positions=q_meas,
            leg_velocities=dot_q_meas,
            tactile_pos_raw=tactile_pos_raw,
            tactile_force_estimate=normal_force,
            tactile_active=is_touching,
            bus_healthy=True
        )

    @property
    def tactile_position_kind(self) -> str:
        """Geometric center projection, NOT a pressure-center measurement.

        Integration must bypass CoP-bias subtraction for this sensor source.
        """
        return "geometric_projection"

    def _read_contact_load(self) -> Tuple[bool, float]:
        matrix = np.asarray(self._sim.getObjectMatrix(self._h_plate, self._sim.handle_world)).reshape(3,4)
        normal_world = matrix[:, 2]
        load = 0.0
        touching = False
        # connect() enforces one physics pass; do not sum across substeps.
        for index in range(10000):
            result = self._sim.getContactInfo(0, self._h_ball, index)
            if not result or result[0] is None or len(result[0]) == 0:
                return touching, load
            pair, point, force = result[:3]
            if set(pair) != {self._h_ball, self._h_surface}:
                continue
            f = np.asarray(force, dtype=float)
            if f.shape != (3,) or not np.all(np.isfinite(f)):
                raise RuntimeError("Invalid contact force")
            # Magnitude is independent of pair ordering. Top-surface selection
            # is geometrical, so side and underside collisions are not tactile.
            point_P = matrix[:, :3].T @ (np.asarray(point)-matrix[:, 3])
            if abs(float(point_P[2])) > 0.003:
                continue
            load += abs(float(f @ normal_world))
            touching = True
        raise RuntimeError("Contact enumeration exceeded limit")

    def read_platform_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read the platform pose in the fixed base frame without differentiation."""
        if self._status != PlatformStatus.OPERATIONAL:
            raise RuntimeError("Cannot read platform pose: backend is not operational.")
        position = np.array(
            self._sim.getObjectPosition(self._h_plate, self._h_base),
            dtype=np.float64,
            copy=True,
        )
        matrix_3x4 = np.asarray(
            self._sim.getObjectMatrix(self._h_plate, self._h_base),
            dtype=np.float64,
        ).reshape(3, 4)
        rotation = matrix_3x4[:, :3].copy()

        # Remote-API roundoff can make R very slightly non-orthogonal.  The
        # polar factor is the closest element of SO(3), with reflection guard.
        left, _, right_t = np.linalg.svd(rotation)
        rotation = left @ right_t
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_t
        return position, rotation

    def read_platform_motion(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pose and twist in base I with mode-consistent differentiation.

        Kinematic joint writes move the closed chain outside the dynamics
        integrator, so Coppelia's object velocity can be zero despite a pose
        increment.  In that mode the twist is reconstructed from consecutive
        synchronized poses.  Dynamic mode uses the simulator velocity.
        """
        position, rotation = self.read_platform_pose()
        if self._joint_command_mode == "kinematic":
            if not self._has_prev_platform_pose:
                linear_base = np.zeros(3, dtype=np.float64)
                angular_base = np.zeros(3, dtype=np.float64)
            else:
                linear_base = (
                    position - self._prev_platform_position
                ) / self._dt
                delta_rotation = rotation @ self._prev_platform_rotation.T
                cosine = float(np.clip(
                    (np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0
                ))
                angle = float(np.arccos(cosine))
                skew_vector = np.array([
                    delta_rotation[2, 1] - delta_rotation[1, 2],
                    delta_rotation[0, 2] - delta_rotation[2, 0],
                    delta_rotation[1, 0] - delta_rotation[0, 1],
                ])
                if angle < 1e-8:
                    rotation_vector = 0.5 * skew_vector
                else:
                    rotation_vector = (
                        angle / (2.0 * np.sin(angle))
                    ) * skew_vector
                angular_base = rotation_vector / self._dt
            self._prev_platform_position = position.copy()
            self._prev_platform_rotation = rotation.copy()
            self._has_prev_platform_pose = True
        else:
            linear_world, angular_world = self._sim.getObjectVelocity(self._h_plate | self._sim.handleflag_axis)
            base_matrix = np.asarray(
                self._sim.getObjectMatrix(self._h_base, self._sim.handle_world),
                dtype=np.float64,
            ).reshape(3, 4)
            rotation_base_to_world = base_matrix[:, :3]
            rotation_world_to_base = rotation_base_to_world.T
            linear_base = rotation_world_to_base @ np.asarray(
                linear_world, dtype=np.float64
            )
            angular_base = rotation_world_to_base @ np.asarray(
                angular_world, dtype=np.float64
            )
        twist = np.hstack([linear_base, angular_base])
        if twist.shape != (6,) or not np.all(np.isfinite(twist)):
            raise RuntimeError("CoppeliaSim returned a non-finite platform twist.")
        return position, rotation, twist

    def read_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read the non-cyclic prismatic intervals configured in the scene."""
        lower = np.zeros(6, dtype=np.float64)
        upper = np.zeros(6, dtype=np.float64)
        for index, motor in enumerate(self._h_motors):
            cyclic, interval = self._sim.getJointInterval(motor)
            interval = np.asarray(interval, dtype=np.float64)
            if cyclic or interval.shape != (2,) or interval[1] <= 0.0:
                raise RuntimeError(
                    f"Motor {index + 1} must expose a finite non-cyclic prismatic interval."
                )
            simulator_lower = interval[0]
            simulator_upper = interval[0] + interval[1]
            transformed = self._motor_length_signs[index] * np.array(
                [simulator_lower, simulator_upper], dtype=np.float64
            )
            lower[index] = float(np.min(transformed))
            upper[index] = float(np.max(transformed))
        if not np.all(np.isfinite(np.hstack([lower, upper]))):
            raise RuntimeError("CoppeliaSim returned non-finite joint limits.")
        return lower, upper

    def write_actuators(self, command: ActuatorCommandPacket) -> bool:
        """
        Sets joint targets and executes one synchronous simulation step.
        """
        if self._status != PlatformStatus.OPERATIONAL:
            return False

        try:
            if command.mode not in (ActuatorMode.CSP, ActuatorMode.CSV):
                raise ValueError("Unsupported actuator mode")
            expected = ActuatorMode.CSP if self._actuator_mode == "CSP" else ActuatorMode.CSV
            if command.mode != expected:
                raise ValueError("Command mode differs from commissioned servo mode")
            values = np.asarray(command.q_send if command.mode == ActuatorMode.CSP else command.dot_q_send, dtype=float)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError("Command must contain six finite values")
            if command.mode == ActuatorMode.CSP:
                lower, upper = self.read_joint_limits()
                if np.any(values < lower) or np.any(values > upper):
                    raise ValueError("q_send exceeds physical joint limits")
            elif self._joint_command_mode == "kinematic":
                raise ValueError("Kinematic CSV is not supported")
            time_before = float(self._sim.getSimulationTime())
            # Apply leg targets according to selected mode
            if command.mode == ActuatorMode.CSP:
                for i, motor in enumerate(self._h_motors):
                    if self._joint_command_mode == "dynamic":
                        self._sim.setJointTargetPosition(
                            motor,
                            float(self._motor_length_signs[i] * command.q_send[i]),
                        )
                    else:
                        # Explicit kinematic validation mode; its results must
                        # never be reported as actuator-dynamic evidence.
                        self._sim.setJointPosition(
                            motor,
                            float(self._motor_length_signs[i] * command.q_send[i]),
                        )
            elif command.mode == ActuatorMode.CSV:
                for i, motor in enumerate(self._h_motors):
                    self._sim.setJointTargetVelocity(
                        motor,
                        float(self._motor_length_signs[i] * command.dot_q_send[i]),
                    )

            # Solve closed-chain kinematic constraints
            if self._joint_command_mode == "kinematic":
                result, flags, precision = self._sim_ik.handleGroup(
                    self._ik_env, self._ik_group, {'syncWorlds': True})
                if result != self._sim_ik.result_success:
                    raise RuntimeError(f"IK closure failed: flags={flags}, precision={precision}")

            if command.mode == ActuatorMode.CSP and self._joint_command_mode == "kinematic":
                realized = np.array([
                    self._motor_length_signs[i]
                    * float(self._sim.getJointPosition(motor))
                    for i, motor in enumerate(self._h_motors)
                ])
                error = float(np.max(np.abs(realized - command.q_send)))
                if error > 1e-7:
                    raise RuntimeError(
                        "Coppelia IK modified an ideal CSP endpoint; the "
                        f"acceleration-level contract is invalid (max error={error:.3e} m)."
                    )
                self._last_kinematic_qdot = np.asarray(
                    command.dot_q_send, dtype=np.float64
                ).copy()
                self._has_kinematic_command = True

            # Advance simulation clock by exactly dt
            self._client.step()
            time_after = float(self._sim.getSimulationTime())
            self._last_step_duration = time_after - time_before
            tolerance = max(1e-9, 1e-6 * self._dt)
            if abs(self._last_step_duration - self._dt) > tolerance:
                raise RuntimeError(
                    "Synchronous step mismatch: "
                    f"expected {self._dt:.9g} s, observed "
                    f"{self._last_step_duration:.9g} s."
                )
            self._verify_closure()
            self._last_error = ""
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._status = PlatformStatus.FAULT
            return False

    def extract_kinematic_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts base anchors b_i, platform anchors a_i, leg length offsets,
        and initial platform translation directly from CoppeliaSim scene objects.
        """
        if self._status != PlatformStatus.OPERATIONAL:
            raise RuntimeError("Backend must be connected before extracting parameters.")

        T_p_initial = np.array(self._sim.getObjectPosition(self._h_plate, self._h_base), dtype=np.float64)

        if len(self._h_base_anchors) != 6 or len(self._h_platform_anchors) != 6:
            raise RuntimeError("Explicit geometry markers have not been resolved")
        b_anchors = np.array([self._sim.getObjectPosition(h, self._h_base) for h in self._h_base_anchors])
        p_anchors = np.array([self._sim.getObjectPosition(h, self._h_plate) for h in self._h_platform_anchors])
        endpoints = np.array([self._sim.getObjectPosition(h, self._h_base) for h in self._h_platform_anchors])
        lengths = np.linalg.norm(endpoints-b_anchors, axis=1)
        q = np.array([sign*self._sim.getJointPosition(h) for sign,h in zip(self._motor_length_signs,self._h_motors)])
        if not np.all(np.isfinite(np.r_[b_anchors.ravel(), p_anchors.ravel(), lengths, q])) or np.any(lengths <= 1e-9):
            raise RuntimeError("Invalid anchor geometry")
        return b_anchors, p_anchors, lengths-q, T_p_initial

    def _cleanup(self, stop: bool = True) -> None:
        errors = []
        if self._sim is not None and stop:
            try:
                self._sim.stopSimulation()
                if self._client is not None:
                    self._client.setStepping(False)
                deadline = time.monotonic() + self._shutdown_timeout
                while self._sim.getSimulationState() != self._sim.simulation_stopped:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Timed out waiting for simulation stop")
                    time.sleep(0.01)
            except Exception as exc:
                errors.append(str(exc))
        if self._sim_ik is not None and self._ik_env is not None:
            try:
                self._sim_ik.eraseEnvironment(self._ik_env)
            except Exception as exc:
                errors.append(str(exc))
        self._ik_env = None
        self._ik_group = None
        self._started_by_backend = False
        if errors:
            self._last_error += "; cleanup: " + "; ".join(errors)

    def disconnect(self) -> None:
        """Best-effort stop even after FAULT; timeout bounds polling, not RPC transport."""
        self._cleanup(stop=True)
        self._status = PlatformStatus.SHUTDOWN
