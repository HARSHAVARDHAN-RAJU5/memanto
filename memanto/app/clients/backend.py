"""
Backend selection and protocol for Memanto's Moorcheh client.

Memanto can talk to two backends:
- ``cloud``   - Moorcheh Cloud via the ``moorcheh_sdk`` package (API key).
- ``on-prem`` - A local ``moorcheh`` server (Docker) via the ``moorcheh-client``
  package. No Moorcheh API key needed. Does not support ``answer.generate``.

Both clients are expected to expose the same attribute shape used across
``memanto/app/`` (namespaces/documents/answer with method names matching the
cloud SDK), so service code never branches on backend.
"""

from enum import Enum


class Backend(str, Enum):
    CLOUD = "cloud"
    ON_PREM = "on-prem"


class OnPremFeatureUnavailable(Exception):
    """Raised when on-prem hits a feature that only Moorcheh Cloud supports."""


def parse_backend(value: str | None) -> Backend:
    """Coerce a string (env / yaml) into a Backend, defaulting to cloud."""
    if not value:
        return Backend.CLOUD
    try:
        return Backend(value.strip().lower())
    except ValueError:
        return Backend.CLOUD
