"""Offline regression: instrumentation must preserve call count, order and return."""
import unittest
from types import SimpleNamespace
from trace_gui_closure import install

class HookTest(unittest.TestCase):
    def test_success_and_failure_restore_objects(self):
        for fail in (False, True):
            events = []
            class Backend:
                def extract_kinematic_parameters(self):
                    return ([1], [2], [3], [4])
                def write_actuators(self, command):
                    events.append('joint_write')
                    self._sim_ik.handleGroup(1, 2, {'syncWorlds': True})
                    if fail:
                        raise ValueError('injected')
                    self._client.step()
                    return True
            class Recorder:
                cycle = 0
                def snapshot(self, backend, phase, **kwargs):
                    events.append(phase)
                def emit(self, **kwargs):
                    pass
            backend = Backend()
            ik = SimpleNamespace(handleGroup=lambda *a: events.append('IK'))
            client = SimpleNamespace(step=lambda: events.append('STEP'))
            backend._sim_ik, backend._client = ik, client
            install(Backend, Recorder())
            if fail:
                with self.assertRaises(ValueError):
                    backend.write_actuators(SimpleNamespace())
            else:
                self.assertTrue(backend.write_actuators(SimpleNamespace()))
                self.assertEqual(events, ['before_write', 'joint_write',
                    'after_joint_write_before_ik', 'IK', 'after_ik', 'before_step', 'STEP', 'after_step'])
            self.assertIs(backend._sim_ik, ik)
            self.assertIs(backend._client, client)
            self.assertEqual(events.count('IK'), 1)
            self.assertEqual(events.count('STEP'), int(not fail))

if __name__ == '__main__':
    unittest.main()
