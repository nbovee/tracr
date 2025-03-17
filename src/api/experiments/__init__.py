"""Experiment management module."""

from .base import BaseExperiment, ExperimentPaths, ProcessingTimes
from .local import LocalExperiment
from .networked import NetworkedExperiment
from .manager import ExperimentManager

__all__ = [
    "BaseExperiment",
    "ExperimentPaths",
    "ProcessingTimes",
    "LocalExperiment",
    "NetworkedExperiment",
    "ExperimentManager",
]
