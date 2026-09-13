# src/sbc/controllers/base.py

from abc import ABC, abstractmethod
import numpy as np
from sbc.datatypes import StateEstimatePacket, VirtualControlPacket


class BaseController(ABC):
    """
    Abstract interface for planar decoupled ball tracking controllers.
    Ensures all baseline and robust variants expose an identical signature.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clears internal integrators, filters, or state memory."""
        pass

    @abstractmethod
    def compute_control(
        self,
        state: StateEstimatePacket,
        rho_d: np.ndarray,
        dot_rho_d: np.ndarray,
        ddot_rho_d: np.ndarray
    ) -> VirtualControlPacket:
        """
        Computes virtual planar acceleration u_0 and physical driving target B_des.

        Args:
            state: Reconstructed perception and platform state packet.
            rho_d: Target planar ball position in P [m], shape (2,).
            dot_rho_d: Target planar ball velocity in P [m/s], shape (2,).
            ddot_rho_d: Target planar ball acceleration in P [m/s^2], shape (2,).

        Returns:
            VirtualControlPacket containing u_0 and B_des.
        """
        pass