import numpy as np
import pytest

from sbc.datatypes import StateEstimatePacket


def test_state_packet_copies_and_freezes_arrays() -> None:
    source = np.zeros(2)
    packet = StateEstimatePacket(
        timestamp=0.0,
        platform_pos=np.zeros(3),
        platform_rot=np.eye(3),
        platform_twist=np.zeros(6),
        platform_accel_src=np.zeros(3),
        ball_pos=source,
        ball_vel=np.zeros(2),
        yaw_rate=0.0,
        contact_valid=True,
    )
    source[0] = 3.0
    assert packet.ball_pos[0] == 0.0
    with pytest.raises(ValueError):
        packet.ball_pos[0] = 1.0


def test_state_packet_rejects_nonrotation() -> None:
    with pytest.raises(ValueError):
        StateEstimatePacket(0.0, np.zeros(3), np.zeros((3, 3)), np.zeros(6),
                            np.zeros(3), np.zeros(2), np.zeros(2), 0.0, True)

