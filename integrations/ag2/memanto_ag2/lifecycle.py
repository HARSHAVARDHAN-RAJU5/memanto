"""Bind a shared SdkClient to one Memanto agent_id (thread-safe)."""

from __future__ import annotations

import logging
import threading

from memanto.cli.client.sdk_client import SdkClient

logger = logging.getLogger(__name__)


class AgentSessionBinder:
    """Create agent once and re-activate when the SDK session is bound elsewhere."""

    def __init__(
        self, client: SdkClient, agent_id: str, *, duration_hours: int = 6
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._duration_hours = duration_hours
        self._lock = threading.Lock()
        self._agent_created = False

    def _bind_unlocked(self) -> None:
        if not self._agent_created:
            try:
                self._client.create_agent(agent_id=self._agent_id, pattern="tool")
            except Exception as exc:
                logger.debug("create_agent ignored for %s: %s", self._agent_id, exc)
            self._agent_created = True

        if self._client.agent_id != self._agent_id:
            try:
                self._client.activate_agent(
                    self._agent_id,
                    duration_hours=self._duration_hours,
                )
            except Exception as exc:
                logger.debug("activate_agent ignored for %s: %s", self._agent_id, exc)

    def bind(self) -> None:
        with self._lock:
            self._bind_unlocked()

    def call(self, operation):
        with self._lock:
            self._bind_unlocked()
            return operation()
