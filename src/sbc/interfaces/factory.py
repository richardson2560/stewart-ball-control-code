# src/sbc/interfaces/factory.py

from typing import Any, Dict
from sbc.interfaces.base_backend import BasePlatformBackend
from sbc.interfaces.backends.coppelia import CoppeliaBackend


class BackendFactory:
    """
    Instantiates platform backends based on declarative configuration.
    Decouples simulation from physical hardware runtime selection.
    """

    @staticmethod
    def create(backend_type: str, config: Dict[str, Any]) -> BasePlatformBackend:
        backend_key = backend_type.lower().strip()

        if backend_key in ("coppelia", "coppeliasim"):
            sim_cfg = config.get("coppelia", {})
            return CoppeliaBackend(
                cycle_time=config.get("cycle_time", 0.002),
                host=sim_cfg.get("host", "localhost"),
                port=sim_cfg.get("port", 23000),
                ball_mass=config.get("ball_mass", 0.065449846949792),
                ball_radius=config.get("ball_radius", 0.025),
                gravity=config.get("gravity", 9.81),
                joint_command_mode=sim_cfg.get("joint_command_mode", "dynamic"),
                kinematic_closure_tolerance=sim_cfg.get(
                    "kinematic_closure_tolerance", 5e-4
                ),
                scene_config=sim_cfg.get("scene", {}),
                actuator_mode=sim_cfg.get("actuator_mode", "CSP"),
                shutdown_timeout=sim_cfg.get("shutdown_timeout", 2.0),
            )

        elif backend_key in ("numerical", "native"):
            from sbc.interfaces.backends.numerical import NumericalBackend
            return NumericalBackend(
                dt=config.get("cycle_time", 0.002),
                ball_mass=config.get("ball_mass", 0.065449846949792),
                ball_radius=config.get("ball_radius", 0.025)
            )

        elif backend_key in ("hardware", "ethercat", "real"):
            raise NotImplementedError(
                "HardwareBackend requires physical EtherCAT master configuration."
            )

        else:
            raise ValueError(f"Unknown backend type: '{backend_type}'.")
