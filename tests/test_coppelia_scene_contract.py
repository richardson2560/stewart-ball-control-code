import numpy as np

from sbc.interfaces.backends.coppelia import CoppeliaBackend


class _FakeScene:
    stringparam_scene_name = 0

    def __init__(self) -> None:
        names = ["/stewartPlatform", "/stewartPlatform/tip", "/Sphere", "/platformTable"]
        names += [f"/stewartPlatform/motor{i}" for i in range(1, 7)]
        names += [f"/stewartPlatform/downArm{i}Tip" for i in range(1, 6)]
        names += [f"/stewartPlatform/downArm{i}Target" for i in range(1, 6)]
        self.handles = {name: index + 1 for index, name in enumerate(names)}
        self.master_tip = 999
        angles = np.linspace(0.0, 2.0*np.pi, 6, endpoint=False)
        self.motor_positions = {
            self.handles[f"/stewartPlatform/motor{i+1}"]: np.array([np.cos(a), np.sin(a), 0.0])
            for i, a in enumerate(angles)
        }
        tips = [self.handles[f"/stewartPlatform/downArm{i}Tip"] for i in range(1, 6)] + [self.master_tip]
        # Deliberately cross the geometric-nearest association. The joint-axis
        # incidence is the source of truth: motor i owns tip i+1 cyclically.
        motors = [self.handles[f"/stewartPlatform/motor{i}"] for i in range(1, 7)]
        self.parents = {tip: motors[0] for tip in tips}
        self.parents[self.handles["/platformTable"]] = self.master_tip
        self.parents.update({motor: -1 for motor in motors})
        self.tip_for_motor = {
            motor: tips[(index + 1) % 6] for index, motor in enumerate(motors)
        }
        self.tip_positions = {
            handle: 0.9*np.array([np.cos(a), np.sin(a), 0.4])
            for handle, a in zip(tips, angles)
        }

    def getObject(self, path, options):
        return self.handles.get(path, -1)

    def getStringParam(self, parameter):
        return "stewart_platform.ttt"

    def getObjectParent(self, handle):
        return self.parents.get(handle, -1)

    def getObjectPosition(self, handle, relative):
        if handle in self.motor_positions:
            return self.motor_positions[handle].tolist()
        return self.tip_positions[handle].tolist()

    def getObjectMatrix(self, handle, relative):
        # Local z points from the motor toward its topological tip.
        tip = self.tip_for_motor[handle]
        axis = self.tip_positions[tip] - self.motor_positions[handle]
        z = axis / np.linalg.norm(axis)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(helper, z)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        y = np.cross(z, helper)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        p = self.motor_positions[handle]
        return np.column_stack([x, y, z, p]).reshape(-1).tolist()


def test_scene_uses_five_slaves_and_one_master_tip() -> None:
    backend = CoppeliaBackend()
    backend._sim = _FakeScene()
    backend._resolve_scene_handles()
    assert len(backend._h_motors) == 6
    assert len(backend._h_tips) == 6
    assert len(backend._h_ik_tips) == 5
    assert len(backend._h_targets) == 5
    assert backend._sim.master_tip in backend._h_tips
    tips = [backend._sim.handles[f"/stewartPlatform/downArm{i}Tip"] for i in range(1, 6)] + [backend._sim.master_tip]
    assert backend._h_tips == tips[1:] + tips[:1]
    np.testing.assert_allclose(backend._motor_length_signs, np.ones(6))
