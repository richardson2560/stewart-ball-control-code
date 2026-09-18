"""Isolate scene/IK geometry using the user's original backend, without SBC.
Run with a freshly loaded stopped scene. Outputs raw JSONL, including on failure.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from _baseline_coppelia import CoppeliaBackend


def snapshot(backend, calibration, phase, command=None, ik_result=None):
    sim = backend._sim
    base, plate = backend._h_base, backend._h_plate
    b0, a0, offsets, _ = calibration
    matrix = np.array(sim.getObjectMatrix(plate, base)).reshape(3, 4)
    rotation, position = matrix[:, :3], matrix[:, 3]
    motors = np.array([sim.getObjectPosition(h, base) for h in backend._h_motors])
    tips = np.array([sim.getObjectPosition(h, base) for h in backend._h_tips])
    local_tips = np.array([sim.getObjectPosition(h, plate) for h in backend._h_tips])
    q = np.array([sim.getJointPosition(h) for h in backend._h_motors])
    lengths_fixed = np.linalg.norm(position + a0 @ rotation.T - b0, axis=1)
    lengths_live = np.linalg.norm(tips - motors, axis=1)
    targets = np.array([sim.getObjectPosition(h, base) for h in backend._h_targets])
    data = dict(phase=phase, time=float(sim.getSimulationTime()), q=q.tolist(),
                q_command=None if command is None else command.tolist(),
                position=position.tolist(), rotation=rotation.tolist(),
                motor_origins=motors.tolist(), tips_base=tips.tolist(),
                tips_plate=local_tips.tolist(), lengths_live=lengths_live.tolist(),
                lengths_fixed=lengths_fixed.tolist(),
                closure=(lengths_fixed-offsets-q).tolist(),
                anchor_base_motion=(motors-b0).tolist(),
                anchor_plate_motion=(local_tips-a0).tolist(),
                slave_closure=(tips[:5]-targets).tolist(), ik_result=ik_result)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dt', type=float, default=0.002)
    parser.add_argument('--port', type=int, default=23000)
    parser.add_argument('--pulse-mm', type=float, default=0.0,
                        help='Optional single-leg excursions, capped at 0.2 mm. Default: hold only.')
    parser.add_argument('--output', default='data/logs/scene_contract.jsonl')
    args = parser.parse_args()
    if not (0 < args.dt <= 0.02 and 0 <= args.pulse_mm <= 0.2):
        parser.error('Require 0 < dt <= 0.02 and 0 <= pulse-mm <= 0.2.')
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient(port=args.port)
    sim = client.require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Load the original scene and stop simulation before this audit.')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = CoppeliaBackend(cycle_time=args.dt, port=args.port)
    try:
        backend.connect()
        calibration = backend.extract_kinematic_parameters()
        q0 = np.array([sim.getJointPosition(h) for h in backend._h_motors])
        with path.open('w', encoding='utf-8', buffering=1) as out:
            def record(phase, command=None, result=None):
                row = snapshot(backend, calibration, phase, command, result)
                out.write(json.dumps(row, default=str, allow_nan=False)+'\n')
                out.flush()
                return row
            record('initial')
            commands = [('hold', q0.copy()) for _ in range(50)]
            if args.pulse_mm:
                for leg in range(6):
                    for delta_mm in (0.5, 1.0, 2.0, 5.0, 10.0):
                        q = q0.copy()
                        q[leg] += delta_mm * 1e-3
                        commands.append((f'leg{leg}_change{delta_mm:+g}', q))
            for label, q in commands:
                for h, value in zip(backend._h_motors, q):
                    cyclic, bounds = sim.getJointInterval(h)
                    if cyclic or not bounds[0] <= value <= bounds[0]+bounds[1]:
                        raise RuntimeError('Requested diagnostic pulse outside joint interval.')
                record(label+':before_write', q)
                for h, value in zip(backend._h_motors, q):
                    sim.setJointPosition(h, float(value))
                record(label+':after_write', q)
                result = backend._sim_ik.handleGroup(backend._ik_env, backend._ik_group,
                                                     {'syncWorlds': True})
                record(label+':after_ik', q, result)
                before = float(sim.getSimulationTime())
                backend._client.step()
                after = record(label+':after_step', q)
                if abs(after['time']-before-args.dt) > 1e-7:
                    raise RuntimeError('Simulation step mismatch, see recorded times.')
                # Diagnostic stop, not a safety certificate or an inferred noise bound.
                if max(np.abs(after['closure'])) > 0.002:
                    raise RuntimeError('Closure exceeds 2 mm diagnostic stop; evidence saved.')
        print(f'Acquisition completed: {path}. This is not a control/safety certification.')
    finally:
        # Do not jump joints back to q0 after an unexpected geometry failure.
        backend.disconnect()


if __name__ == '__main__':
    main()
