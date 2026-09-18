import numpy as np

from sbc.datatypes import SupervisorMode
from sbc.safety.fallback import OneCycleFallback
from sbc.core.supervisor import Supervisor, SupervisorInputs


def test_zero_acceleration_is_not_accepted_near_outgoing_stroke() -> None:
    policy = OneCycleFallback(0.002, 8.0, stroke_margin=0.0)
    q = np.zeros(6); q[0] = 0.349
    q_dot = np.zeros(6); q_dot[0] = 1.0
    decision = policy.select(np.zeros(6), q, q_dot, -0.35*np.ones(6), 0.35*np.ones(6))
    assert not decision.one_cycle_admissible
    assert decision.reason == "failsafe_braking_not_certified"


def test_fallback_is_allowed_only_once() -> None:
    supervisor = Supervisor()
    good = SupervisorInputs(True, True, True, True, False, False)
    assert supervisor.update(good) is SupervisorMode.NORMAL_ZERO_SLACK
    fallback = SupervisorInputs(True, True, True, False, True, False)
    assert supervisor.update(fallback) is SupervisorMode.ONE_CYCLE_FALLBACK
    assert supervisor.update(fallback) is SupervisorMode.FAILSAFE_HOLD


def test_model_envelope_reentry_uses_only_strictly_admitted_command() -> None:
    supervisor = Supervisor()
    admitted_recovery = SupervisorInputs(True, True, False, True, False, False)
    assert supervisor.update(admitted_recovery) is SupervisorMode.RELAXED_DEGRADED

    rejected_recovery = SupervisorInputs(True, True, False, False, True, False)
    assert supervisor.update(rejected_recovery) is SupervisorMode.MODEL_EXIT

    nominal = SupervisorInputs(True, True, True, True, False, False)
    assert supervisor.update(nominal) is SupervisorMode.NORMAL_ZERO_SLACK
