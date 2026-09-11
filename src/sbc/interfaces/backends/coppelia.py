# src/sbc/interfaces/backends/coppelia.py

import os
import sys
import time
import numpy as np
from typing import Optional, List

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
        ball_mass: float = 0.110,
        ball_radius: float = 0.025,
        gravity: float = 9.81
    ) -> None:
        super().__init__(cycle_time=cycle_time)
        self._host = host
        self._port = port
        self._m = ball_mass
        self._r = ball_radius
        self._g = gravity

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

        # Kinematics solver handles (internal to CoppeliaSim IK group)
        self._ik_env = None
        self._ik_group = None

        # Previous state cache for finite-difference fallback in simulation
        self._prev_q = np.zeros(6, dtype=np.float64)
        self._has_prev_q = False

    def connect(self) -> bool:
        """Connects to CoppeliaSim ZeroMQ server and initializes handles."""
        self._status = PlatformStatus.INITIALIZING
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self._client = RemoteAPIClient(host=self._host, port=self._port)
            self._sim = self._client.require('sim')
            self._sim_ik = self._client.require('simIK')
        except ImportError:
            # Handle manual path resolution if coppeliasim_zmqremoteapi_client is local
            zmq_path = os.environ.get("COPPELIASIM_ZMQ_PATH")
            if zmq_path and zmq_path not in sys.path:
                sys.path.append(zmq_path)
                from coppeliasim_zmqremoteapi_client import RemoteAPIClient
                self._client = RemoteAPIClient(host=self._host, port=self._port)
                self._sim = self._client.require('sim')
                self._sim_ik = self._client.require('simIK')
            else:
                self._status = PlatformStatus.FAULT
                raise RuntimeError(
                    "CoppeliaSim ZeroMQ client not found. Install 'coppeliasim-zmqremoteapi-client' "
                    "or set COPPELIASIM_ZMQ_PATH."
               )

        try:
            self._resolve_scene_handles()
            self._setup_kinematic_solver()
            
            # Enable stepping mode for deterministic, clock-synchronized execution
            self._client.setStepping(True)
            self._sim.setFloatParam(self._sim.floatparam_simulation_time_step, self._dt)
            self._sim.startSimulation()
            
            self._status = PlatformStatus.OPERATIONAL
            return True
        except Exception as exc:
            self._status = PlatformStatus.FAULT
            raise RuntimeError(f"Failed to initialize CoppeliaSim backend: {exc}") from exc

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

        scene_name = self._sim.getStringParam(self._sim.stringparam_scene_name)
        raise RuntimeError(
            f"Object '{name}' not found in CoppeliaSim. "
            f"Active scene: '{scene_name}'. "
            f"Verify that 'simulations/coppelia/stewart_platform.ttt' is loaded."
        )

    def _resolve_scene_handles(self) -> None:
        """Retrieves and verifies all required object handles in the scene."""
        self._h_base = self._get_object_handle('/stewartPlatform')
        self._h_plate = self._get_object_handle('/stewartPlatform/tip')
        self._h_ball = self._get_object_handle('/Sphere')

        # Resolve the 6 prismatic motors (Stewart legs)
        # Compatible with canonical Stewart scene hierarchy
        self._h_motors = [self._get_object_handle(f'/stewartPlatform/motor{i}') for i in range(2, 7)] + [self._get_object_handle('/stewartPlatform/motor1')]

        # Resolve platform attachment points
        table_handle = self._get_object_handle('/platformTable')
        master_tip = self._sim.getObjectParent(table_handle)
        self._h_tips = [
            self._get_object_handle(f'/stewartPlatform/downArm{i}Tip') for i in range(1, 6)
        ] + [master_tip]
        self._h_targets = [
            self._get_object_handle(f'/stewartPlatform/downArm{i}Target') for i in range(1, 6)
        ]

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
                self._h_tips[i],
                self._h_targets[i],
                self._sim_ik.constraint_position
            )
            sim_to_ik_map_total.update(sim_to_ik)

        for motor in self._h_motors:
            if motor in sim_to_ik_map_total:
                ik_motor = sim_to_ik_map_total[motor]
                self._sim_ik.setJointMode(self._ik_env, ik_motor, self._sim_ik.jointmode_passive)

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
            q_meas[i] = float(self._sim.getJointPosition(motor))

        # 2. Derive raw leg velocities from joint sensors or stepping delta
        dot_q_meas = np.zeros(6, dtype=np.float64)
        if self._has_prev_q:
            dot_q_meas = (q_meas - self._prev_q) / self._dt
        self._prev_q = q_meas.copy()
        self._has_prev_q = True

        # 3. Read ball position relative to moving platform plate frame P
        # CoppeliaSim convention: Plate origin is at the center of the surface
        pos_relative = self._sim.getObjectPosition(self._h_ball, self._h_plate)
        tactile_pos_raw = np.array([pos_relative[0], pos_relative[1]], dtype=np.float64)
        ball_height_rel = float(pos_relative[2])

        # 4. Assess normal contact load and detachment
        # If ball z-coordinate exceeds sphere radius + tolerance, contact is broken
        contact_threshold = self._r + 0.003
        is_touching = ball_height_rel <= contact_threshold

        # Estimated load (quasi-static gravity component in simulation unless contact force sensor exists)
        plate_matrix = self._sim.getObjectMatrix(self._h_plate, self._h_base)
        # R_z vector of the plate relative to base: [matrix[2], matrix[6], matrix[10]]
        normal_projection = float(plate_matrix[10])
        normal_force = max(0.0, self._m * self._g * normal_projection) if is_touching else 0.0

        return RawSensorPacket(
            timestamp=t,
            leg_positions=q_meas,
            leg_velocities=dot_q_meas,
            tactile_pos_raw=tactile_pos_raw,
            tactile_force_estimate=normal_force,
            tactile_active=is_touching,
            bus_healthy=True
        )

    def write_actuators(self, command: ActuatorCommandPacket) -> bool:
        """
        Sets joint targets and executes one synchronous simulation step.
        """
        if self._status != PlatformStatus.OPERATIONAL:
            return False

        try:
            # Apply leg targets according to selected mode
            if command.mode == ActuatorMode.CSP:
                for i, motor in enumerate(self._h_motors):
                    self._sim.setJointPosition(motor, float(command.q_send[i]))
            elif command.mode == ActuatorMode.CSV:
                for i, motor in enumerate(self._h_motors):
                    self._sim.setJointTargetVelocity(motor, float(command.dot_q_send[i]))

            # Solve closed-chain kinematic constraints
            self._sim_ik.handleGroup(self._ik_env, self._ik_group, {'syncWorlds': True})

            # Advance simulation clock by exactly dt
            self._client.step()
            return True
        except Exception:
            self._status = PlatformStatus.FAULT
            return False

    def disconnect(self) -> None:
        """Stops simulation and clears remote environment."""
        if self._sim and self._status == PlatformStatus.OPERATIONAL:
            try:
                self._client.setStepping(False)
                self._sim.stopSimulation()
                while self._sim.getSimulationState() != self._sim.simulation_stopped:
                    time.sleep(0.05)
            except Exception:
                pass
        self._status = PlatformStatus.SHUTDOWN