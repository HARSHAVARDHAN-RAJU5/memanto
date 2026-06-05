"""
On-prem Moorcheh client.

Wraps ``moorcheh.MoorchehApiClient`` (from the ``moorcheh-client`` PyPI package)
and exposes the same ``client.namespaces.*`` / ``client.documents.*`` /
``client.similarity_search.*`` / ``client.answer.*`` shape that Memanto's
services use against the cloud SDK, so no service code branches on backend.

Real on-prem method signatures (verified against docs):

    client.create_namespace(payload: dict) -> dict
        payload = {"namespace_name": str, "type": "text"|"vector",
                   "vector_dimension": int (vector only)}

    client.list_namespaces() -> dict
        returns {"namespaces": [{"namespace_name", "type",
                 "vector_dimension", "item_count", "created_at"}, ...]}

    client.delete_namespace(namespace_name: str) -> dict
        returns {"job_id": ...}

    client.upload_namespace_documents(namespace_name: str, payload: dict) -> dict
        payload = {"documents": [{"id": str, "text": str, ...metadata}, ...]}

    client.get_namespace_items(namespace_name: str, payload: dict) -> dict
        payload = {"ids": [str, ...]}  # <= 100 ids

    client.delete_namespace_items(namespace_name: str, payload: dict) -> dict
        payload = {"ids": [str, ...]}

    client.search(payload: dict) -> dict
        payload = {"query": str|list, "namespaces": [str, ...],
                   "top_k": int, "threshold": float, "metadata": dict,
                   "kiosk_mode": bool}

    client.health() -> dict

Cloud-only surface Memanto uses that on-prem does not expose - we surface a
clear ``OnPremFeatureUnavailable`` instead of silently failing:
    - documents.upload_file (no server-side file chunking on on-prem)
    - answer.generate (cloud-only LLM RAG)
"""

from typing import Any

from memanto.app.clients.backend import OnPremFeatureUnavailable

_DEFAULT_URL = "http://localhost:8080"
_ANSWER_DISABLED_MSG = (
    "answer is not available on the on-prem backend. "
    "Switch with: memanto config backend cloud"
)
_FEATURE_DISABLED_MSG = (
    "{feature} is not available on the on-prem backend. "
    "Switch with: memanto config backend cloud"
)


def _import_raw_client() -> Any:
    """Lazy import so the cloud path doesn't require ``moorcheh-client``."""
    try:
        from moorcheh import MoorchehApiClient  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised at runtime only
        raise RuntimeError(
            "moorcheh-client is not installed. Run: pip install moorcheh-client"
        ) from e
    return MoorchehApiClient


class _NamespacesAdapter:
    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def create(self, namespace_name: str, type: str = "text") -> Any:
        return self._raw.create_namespace(
            {"namespace_name": namespace_name, "type": type}
        )

    def list(self) -> Any:
        return self._raw.list_namespaces()

    def delete(self, namespace_name: str) -> Any:
        return self._raw.delete_namespace(namespace_name)


class _DocumentsAdapter:
    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def upload(self, namespace_name: str, documents: list[dict]) -> Any:
        return self._raw.upload_namespace_documents(
            namespace_name, {"documents": documents}
        )

    def get(self, namespace_name: str, ids: list[str]) -> Any:
        return self._raw.get_namespace_items(namespace_name, {"ids": ids})

    def delete(self, namespace_name: str, ids: list[str]) -> Any:
        return self._raw.delete_namespace_items(namespace_name, {"ids": ids})

    def upload_file(self, *_args: Any, **_kwargs: Any) -> Any:
        raise OnPremFeatureUnavailable(
            _FEATURE_DISABLED_MSG.format(feature="upload_file")
        )

    def fetch_text_data(self, namespace_name: str) -> Any:
        """Emulate cloud's ``fetch_text_data`` via on-prem ``search``.

        Cloud returns ``{"items": [...]}`` capped at 100 per namespace; we
        approximate the same shape with a broad search so callers (memory
        export, UI, etc.) keep working without branching.
        """
        last_exc: Exception | None = None
        for probe_query in (" ", "*", "."):
            try:
                resp = self._raw.search(
                    {
                        "query": probe_query,
                        "namespaces": [namespace_name],
                        "top_k": 100,
                    }
                )
                if isinstance(resp, dict):
                    items = resp.get("results", resp.get("items", []))
                    return {"items": items}
                return resp
            except Exception as e:
                last_exc = e
                continue
        raise (
            last_exc
            if last_exc is not None
            else RuntimeError("search failed for fetch_text_data")
        )


class _SimilaritySearchAdapter:
    """Maps cloud's ``client.similarity_search.query(...)`` onto on-prem ``search``."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def query(
        self,
        query: str,
        namespaces: list[str] | None = None,
        top_k: int = 10,
        threshold: float | None = None,
        kiosk_mode: bool = False,
        **kwargs: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "query": query,
            "namespaces": namespaces or [],
            "top_k": top_k,
            "kiosk_mode": kiosk_mode,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        # Pass-through for any future kwargs (e.g. metadata filter).
        payload.update(kwargs)
        return self._raw.search(payload)


class _AnswerAdapter:
    def generate(self, *_args: Any, **_kwargs: Any) -> Any:
        raise OnPremFeatureUnavailable(_ANSWER_DISABLED_MSG)


class OnPremClient:
    """Cloud-shaped facade over ``MoorchehApiClient``."""

    def __init__(self, base_url: str | None = None) -> None:
        client_cls = _import_raw_client()
        self._raw = client_cls(base_url or _DEFAULT_URL)
        self.namespaces = _NamespacesAdapter(self._raw)
        self.documents = _DocumentsAdapter(self._raw)
        self.similarity_search = _SimilaritySearchAdapter(self._raw)
        self.answer = _AnswerAdapter()


class AsyncOnPremClient:
    """Async facade. Memanto only awaits a handful of methods via
    ``asyncio.to_thread`` today, so we expose the same sync ``OnPremClient``
    shape - existing ``await asyncio.to_thread(client.documents.upload, ...)``
    calls keep working unchanged.
    """

    def __init__(self, base_url: str | None = None) -> None:
        client_cls = _import_raw_client()
        self._raw = client_cls(base_url or _DEFAULT_URL)
        self.namespaces = _NamespacesAdapter(self._raw)
        self.documents = _DocumentsAdapter(self._raw)
        self.similarity_search = _SimilaritySearchAdapter(self._raw)
        self.answer = _AnswerAdapter()
