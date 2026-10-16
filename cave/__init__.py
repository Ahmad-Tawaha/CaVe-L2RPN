"""CaVe (Calibrated Verification): a grid-topology controller for the L2RPN IDF power grid.

A simulator-trained ranker proposes actions; a rule-based controller tests them by simulation (N-1 risk, structure,
look-ahead, planned outages, redispatch); a training stage identifies the line parameters of the real grid; a verifier
checks every critical decision on that identified model before it is executed.
"""
from .agent import CaVeAgent, make_agent
from .config import CaVeConfig, TrainingConfig

__all__ = ["CaVeAgent", "CaVeConfig", "TrainingConfig", "make_agent"]
__version__ = "1.0.0"
