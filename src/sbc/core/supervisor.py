# src/sbc/core/supervisor.py

"""Deterministic SBC operating-mode state machine with transient coasting resilience."""

from dataclasses import dataclass
from sbc.datatypes import SupervisorMode


@dataclass(frozen=True)
class SupervisorInputs:
    bus_healthy: bool
    contact_valid: bool
    model_valid: bool
    strict_solution: bool
    fallback_admissible: bool
    yaw_unwind_requested: bool


class Supervisor:
    def __init__(self, max_contact_loss_steps: int = 25) -> None:
        self.mode = SupervisorMode.FAILSAFE_HOLD
        self._fallback_used = False
        self._loss_counter = 0
        self._max_loss = max_contact_loss_steps  # 25 steps @ 2ms = 50 ms coasting window

    def reset(self) -> None:
        self.mode = SupervisorMode.FAILSAFE_HOLD
        self._fallback_used = False
        self._loss_counter = 0

    def update(self, inputs: SupervisorInputs) -> SupervisorMode:
        if not inputs.bus_healthy:
            self.mode = SupervisorMode.FAILSAFE_HOLD
            return self.mode

        # Resilient contact monitoring: Allow transient micro-flickers under coasting
        if not inputs.contact_valid:
            self._loss_counter += 1
            if self._loss_counter > self._max_loss:
                self.mode = SupervisorMode.FAILSAFE_HOLD
                return self.mode
        else:
            self._loss_counter = 0

        if not inputs.model_valid:
            self.mode = (
                SupervisorMode.RELAXED_DEGRADED
                if inputs.strict_solution
                else SupervisorMode.MODEL_EXIT
            )
            return self.mode

        if inputs.strict_solution:
            self._fallback_used = False
            self.mode = (
                SupervisorMode.YAW_UNWIND
                if inputs.yaw_unwind_requested
                else SupervisorMode.NORMAL_ZERO_SLACK
            )
            return self.mode

        if inputs.fallback_admissible and not self._fallback_used:
            self._fallback_used = True
            self.mode = SupervisorMode.ONE_CYCLE_FALLBACK
        else:
            self.mode = SupervisorMode.FAILSAFE_HOLD

        return self.mode