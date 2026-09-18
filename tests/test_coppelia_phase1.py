"""Backend contract tests. No running simulator required."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sbc.interfaces.backends.coppelia import CoppeliaBackend
from sbc.interfaces.factory import BackendFactory
from sbc.datatypes import ActuatorMode, PlatformStatus

class FakeSim:
    handle_world=-1
    handleflag_axis=0x100000
    simulation_stopped=0
    object_joint_type=1
    object_dummy_type=4
    object_shape_type=0
    joint_prismatic_subtype=11
    jointmode_dynamic=5
    jointmode_kinematic=0
    jointintparam_dynctrlmode=2001
    jointdynctrl_position=8
    jointdynctrl_velocity=4
    shapeintparam_static=3000
    shapeintparam_respondable=3001
    dummyintparam_link_type=4000
    dummy_linktype_dynamics_loop_closure=6
    def __init__(self):
        self.time=0.; self.targets=[]; self.stopped=False
        self.velocity_handle=None; self.contacts=[]; self.legacy=False
    def getSimulationTime(self): return self.time
    def setJointTargetPosition(self,h,v): self.targets.append((h,v))
    def getJointInterval(self,h): return False,[-.05,.1]
    def getJointPosition(self,h): return 0.
    def getObjectVelocity(self,h):
        self.velocity_handle=h
        return [0.,0.,0.],[.1,.2,.3]
    def getObjectMatrix(self,*args): return [1,0,0,0,0,1,0,0,0,0,1,0]
    def stopSimulation(self): self.stopped=True
    def getSimulationState(self): return 0
    def getObjectPosition(self,*args): return [0,0,0]
    def getObjectType(self,h):
        if 10<=h<16:return self.object_joint_type
        if h>=30:return self.object_shape_type if self.legacy else self.object_dummy_type
        return self.object_shape_type
    def getJointType(self,h): return 11
    def getJointMode(self,h): return 0 if self.legacy else 5
    def getObjectInt32Param(self,h,p):
        if p==self.jointintparam_dynctrlmode:return 0 if self.legacy else 8
        if p==self.shapeintparam_static:return int(h==1 or self.legacy)
        if p==self.shapeintparam_respondable:return 1
        return self.dummy_linktype_dynamics_loop_closure
    def getJointTargetForce(self,h):return 50.
    def getLinkDummy(self,h):
        if self.legacy:raise AssertionError('getLinkDummy called on a shape')
        return h+5 if h<35 else h-5
    def getContactInfo(self,p,h,i):
        assert p==0
        return self.contacts[i] if i<len(self.contacts) else []

class FakeClient:
    def __init__(self,s):self.sim=s;self.steps=0
    def step(self):self.steps+=1;self.sim.time+=.002
    def setStepping(self,e):pass
class ForbiddenIK:
    def handleGroup(self,*args):raise AssertionError('dynamic backend invoked IK')
def backend():
    b=CoppeliaBackend(joint_command_mode='dynamic')
    s=FakeSim();c=FakeClient(s)
    b._sim=s;b._client=c;b._sim_ik=ForbiddenIK()
    b._status=PlatformStatus.OPERATIONAL;b._h_motors=list(range(10,16))
    b._h_base=1;b._h_plate=2;b._h_surface=3;b._h_ball=4
    return b,s,c
def command(values=None,mode=ActuatorMode.CSP):
    return SimpleNamespace(mode=mode,q_send=np.zeros(6) if values is None else values,dot_q_send=np.zeros(6))
class Phase1(unittest.TestCase):
    def test_dynamic_does_not_run_ik(self):
        b,s,c=backend();self.assertTrue(b.write_actuators(command()));self.assertEqual(c.steps,1);self.assertEqual(len(s.targets),6)
    def test_unknown_mode(self):
        b,s,c=backend();self.assertFalse(b.write_actuators(command(mode='bad')));self.assertEqual((s.targets,c.steps),([],0))
    def test_nan_before_first_write(self):
        b,s,c=backend();q=np.zeros(6);q[-1]=np.nan
        self.assertFalse(b.write_actuators(command(q)));self.assertEqual((s.targets,c.steps),([],0))
    def test_wrong_shape_before_first_write(self):
        b,s,c=backend();self.assertFalse(b.write_actuators(command(np.zeros(5))));self.assertEqual(s.targets,[])
    def test_out_of_range_send(self):
        b,s,c=backend();self.assertFalse(b.write_actuators(command(np.ones(6))));self.assertEqual(s.targets,[])
    def test_mode_change_rejected(self):
        b,s,c=backend();self.assertFalse(b.write_actuators(command(mode=ActuatorMode.CSV)));self.assertEqual(c.steps,0)
    def test_axis_flag(self):
        b,s,c=backend();b.read_platform_pose=lambda:(np.zeros(3),np.eye(3))
        _,_,tw=b.read_platform_motion();self.assertEqual(s.velocity_handle,2|s.handleflag_axis);np.testing.assert_allclose(tw[3:],[.1,.2,.3])
    def test_stop_after_fault(self):
        b,s,c=backend();b._status=PlatformStatus.FAULT;b.disconnect();self.assertTrue(s.stopped)
    def test_timestep_mismatch_fault(self):
        b,s,c=backend();c.step=lambda:setattr(s,'time',s.time+.01)
        self.assertFalse(b.write_actuators(command()));self.assertEqual(b.status,PlatformStatus.FAULT)
    def test_legacy_scene_rejected_without_dummy_call(self):
        b,s,c=backend();s.legacy=True;b._h_ik_tips=list(range(30,35));b._h_targets=list(range(35,40))
        with self.assertRaisesRegex(RuntimeError,'requires linked DUMMIES'):b._validate_scene()
    def test_kinematic_scene_accepts_shape_closure_objects(self):
        b,s,c=backend();s.legacy=True;b._joint_command_mode='kinematic'
        b._h_ik_tips=list(range(30,35));b._h_targets=list(range(35,40))
        b._validate_scene()  # Shapes are valid addElementFromScene endpoints.
    def test_valid_configuration_preflight(self):
        b,s,c=backend();b._h_ik_tips=list(range(30,35));b._h_targets=list(range(35,40));b._validate_scene()
    def test_force_uses_only_ball_surface(self):
        b,s,c=backend();s.contacts=[([4,3],[0,0,0],[0,0,1],[0,0,1]),([4,99],[0,0,0],[0,0,20],[0,0,1])]
        self.assertEqual(b._read_contact_load(),(True,1.))
    def test_no_contact_no_fake_gravity_load(self):
        b,s,c=backend();self.assertEqual(b._read_contact_load(),(False,0.))
    def test_side_contact_excluded(self):
        b,s,c=backend();s.contacts=[([4,3],[0,0,-.02],[0,0,3],[0,0,1])];self.assertEqual(b._read_contact_load(),(False,0.))
    def test_factory_defaults(self):
        b=BackendFactory.create('coppelia',{});self.assertFalse(b.ideal_kinematic_csp);self.assertEqual(b.kinematic_closure_tolerance,5e-4)
    def test_geometry_not_silently_inferred(self):
        b,s,c=backend()
        with self.assertRaisesRegex(RuntimeError,'Explicit geometry'):b.extract_kinematic_parameters()
    def test_sign_transforms_command(self):
        b,s,c=backend();b._motor_length_signs=-np.ones(6);self.assertTrue(b.write_actuators(command(np.full(6,.01))));self.assertTrue(all(v==-.01 for h,v in s.targets))
    def test_closure_failure_after_step(self):
        b,s,c=backend();b._h_ik_tips=[30];b._h_targets=[35];s.getObjectPosition=lambda *a:[.02,0,0]
        self.assertFalse(b.write_actuators(command()));self.assertEqual(b.status,PlatformStatus.FAULT)
if __name__=='__main__':unittest.main(verbosity=2)
