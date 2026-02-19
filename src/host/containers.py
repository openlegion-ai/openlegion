"""Docker container lifecycle management for agent instances.

This module is now a thin facade over ``runtime.DockerBackend``.
Existing code that imports ``ContainerManager`` continues to work
unchanged. New code should use ``RuntimeBackend`` / ``select_backend``
from ``src.host.runtime`` directly.
"""

from __future__ import annotations

from src.host.runtime import DockerBackend

# ContainerManager is now an alias for DockerBackend.
# All methods (start_agent, stop_agent, spawn_agent, health_check,
# get_logs, wait_for_agent, stop_all, etc.) are inherited.
ContainerManager = DockerBackend
