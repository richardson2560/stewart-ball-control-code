"""For this partial attachment bundle only: substitute missing packet definitions.
This is NOT an integration test of the real datatypes or runner.
On the complete project run unittest discovery directly instead.
"""
import sys,types,unittest
from pathlib import Path
from enum import Enum
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
class Mode(Enum):CSP=1;CSV=2
class Status(Enum):DISCONNECTED=0;INITIALIZING=1;OPERATIONAL=2;FAULT=3;SHUTDOWN=4
class Packet:
    def __init__(self,**kw):self.__dict__.update(kw)
m=types.ModuleType('sbc.datatypes')
m.ActuatorMode=Mode;m.PlatformStatus=Status
m.RawSensorPacket=Packet;m.ActuatorCommandPacket=Packet
sys.modules['sbc.datatypes']=m
suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_coppelia_phase1.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
