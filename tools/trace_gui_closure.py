"""Run the installed GUI unchanged and trace geometric closure by execution phase."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import threading
import numpy as np


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


class Trace:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, 'w', encoding='utf-8')
        self.lock = threading.Lock()
        self.cycle = 0

    def emit(self, **record):
        with self.lock:
            self.file.write(json.dumps(record, default=encode) + '\n')
            self.file.flush()

    def snapshot(self, backend, phase, **extra):
        # Deliberately no read_sensors/read_platform_motion: those mutate derivative history.
        s = backend._sim
        matrix = np.asarray(s.getObjectMatrix(backend._h_plate, backend._h_base)).reshape(3, 4)
        native = np.array([s.getJointPosition(h) for h in backend._h_motors])
        signs = np.asarray(getattr(backend, '_motor_length_signs', np.ones(6)))
        b, a, offsets, _ = backend._closure_trace_geometry
        q_ik = np.linalg.norm(matrix[:, 3] + a @ matrix[:, :3].T - b, axis=1) - offsets
        motor_base = np.array([s.getObjectPosition(h, backend._h_base) for h in backend._h_motors])
        tip_plate = np.array([s.getObjectPosition(h, backend._h_plate) for h in backend._h_tips])
        tips = np.array([s.getObjectPosition(h, backend._h_base) for h in backend._h_tips])
        targets = np.array([s.getObjectPosition(h, backend._h_base) for h in backend._h_targets])
        self.emit(phase=phase, cycle=self.cycle, time=s.getSimulationTime(), matrix=matrix,
                  q_native=native, q_model=native*signs, q_ik=q_ik,
                  residual=q_ik-native*signs, motor_base=motor_base, tip_plate=tip_plate,
                  tip_target_residual=tips[:len(targets)]-targets, **extra)


class PhaseProxy:
    def __init__(self, wrapped, name, before, after):
        self.wrapped, self.name, self.before, self.after = wrapped, name, before, after

    def __getattr__(self, name):
        method = getattr(self.wrapped, name)
        if name != self.name:
            return method
        def call(*args, **kwargs):
            self.before()
            result = method(*args, **kwargs)
            self.after(result)
            return result
        return call


def install(cls, trace):
    extract, write = cls.extract_kinematic_parameters, cls.write_actuators
    def traced_extract(self, *args, **kwargs):
        geometry = extract(self, *args, **kwargs)
        self._closure_trace_geometry = tuple(np.array(v, copy=True) for v in geometry)
        trace.emit(phase='calibration', geometry=geometry, motors=self._h_motors, tips=self._h_tips)
        trace.snapshot(self, 'initial')
        return geometry
    def traced_write(self, command):
        trace.cycle += 1
        trace.snapshot(self, 'before_write', command={k:getattr(command, k, None) for k in
                       ('timestamp', 'mode', 'q_send', 'dot_q_send', 'u_q_applied', 'slack_value')})
        ik, client = self._sim_ik, self._client
        self._sim_ik = PhaseProxy(ik, 'handleGroup',
            lambda: trace.snapshot(self, 'after_joint_write_before_ik'),
            lambda result: trace.snapshot(self, 'after_ik', ik_result=result))
        self._client = PhaseProxy(client, 'step',
            lambda: trace.snapshot(self, 'before_step'),
            lambda result: trace.snapshot(self, 'after_step'))
        try:
            result = write(self, command)
            trace.emit(phase='write_return', cycle=trace.cycle, success=result,
                       last_error=getattr(self, 'last_error', ''))
            return result
        finally:
            self._sim_ik, self._client = ik, client
    cls.extract_kinematic_parameters = traced_extract
    cls.write_actuators = traced_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/logs/gui_closure.jsonl')
    args, gui_args = parser.parse_known_args()
    gui = Path(__file__).resolve().with_name('run_gui.py')
    if not gui.exists():
        parser.error('Place this script beside the existing tools/run_gui.py.')
    from sbc.interfaces.backends.coppelia import CoppeliaBackend
    trace = Trace(args.out)
    for path in (gui, Path(inspect.getfile(CoppeliaBackend))):
        trace.emit(phase='source', path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    install(CoppeliaBackend, trace)
    sys.argv = [str(gui)] + gui_args
    print(f'[SBC trace] {Path(args.out).resolve()} (diagnostic RPC overhead; simulation stepping unchanged)')
    try:
        runpy.run_path(str(gui), run_name='__main__')
    finally:
        trace.emit(phase='gui_return')
        # Daemon control thread can still finish its last call during GUI shutdown.
        # Every record is flushed; process teardown closes the file.

if __name__ == '__main__':
    main()
