"""
Tests for the backend abstraction (cloud vs on-prem dispatcher).
"""

from unittest.mock import patch

import pytest

from memanto.app.clients.backend import (
    Backend,
    OnPremFeatureUnavailable,
    parse_backend,
)


class TestBackendParse:
    def test_default_is_cloud(self):
        assert parse_backend("") == Backend.CLOUD
        assert parse_backend(None) == Backend.CLOUD

    def test_cloud(self):
        assert parse_backend("cloud") == Backend.CLOUD
        assert parse_backend("Cloud") == Backend.CLOUD

    def test_on_prem(self):
        assert parse_backend("on-prem") == Backend.ON_PREM
        assert parse_backend("ON-PREM") == Backend.ON_PREM

    def test_unknown_falls_back_to_cloud(self):
        assert parse_backend("hybrid") == Backend.CLOUD


class TestOnPremClient:
    def test_answer_generate_always_raises(self):
        """OnPremClient.answer.generate must reject calls with a clear message."""
        # Import is lazy so test runs even without ``moorcheh-client`` installed.
        from memanto.app.clients import onprem

        # Stub _import_raw_client so we don't need the real package.
        class _FakeRaw:
            def __init__(self, base_url):
                self.base_url = base_url

        with patch.object(onprem, "_import_raw_client", return_value=_FakeRaw):
            client = onprem.OnPremClient(base_url="http://localhost:8080")
            with pytest.raises(OnPremFeatureUnavailable) as exc:
                client.answer.generate(namespace="x", query="y")
            assert "memanto config backend cloud" in str(exc.value)


class TestSingletonDispatch:
    def test_cloud_returns_cloud_client(self):
        """On cloud, the dispatcher must not return an OnPremClient."""
        from memanto.app.clients import moorcheh as mclients
        from memanto.app.clients import onprem
        from memanto.app.config import settings

        original = settings.MEMANTO_BACKEND
        settings.MEMANTO_BACKEND = "cloud"
        mclients.moorcheh_client.reset_client()
        try:
            client = mclients.moorcheh_client.get_client()
            assert not isinstance(client, onprem.OnPremClient)
        finally:
            settings.MEMANTO_BACKEND = original
            mclients.moorcheh_client.reset_client()

    def test_on_prem_returns_on_prem_client(self):
        from memanto.app.clients import moorcheh as mclients
        from memanto.app.clients import onprem
        from memanto.app.config import settings

        original = settings.MEMANTO_BACKEND
        settings.MEMANTO_BACKEND = "on-prem"
        mclients.moorcheh_client.reset_client()

        class _FakeRaw:
            def __init__(self, base_url):
                self.base_url = base_url

        try:
            with patch.object(onprem, "_import_raw_client", return_value=_FakeRaw):
                client = mclients.moorcheh_client.get_client()
                assert isinstance(client, onprem.OnPremClient)
        finally:
            settings.MEMANTO_BACKEND = original
            mclients.moorcheh_client.reset_client()


class TestDataDirRouting:
    def test_cloud_uses_default(self, tmp_path, monkeypatch):
        from memanto.app import config as app_config

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(app_config.settings, "MEMANTO_BACKEND", "cloud")
        # Path.home() is cached via os.path.expanduser in some envs; force it
        monkeypatch.setattr(app_config.Path, "home", classmethod(lambda cls: tmp_path))
        assert app_config.get_data_dir() == tmp_path / ".memanto"

    def test_on_prem_uses_subdir(self, tmp_path, monkeypatch):
        from memanto.app import config as app_config

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(app_config.settings, "MEMANTO_BACKEND", "on-prem")
        monkeypatch.setattr(app_config.Path, "home", classmethod(lambda cls: tmp_path))
        result = app_config.get_data_dir()
        assert result == tmp_path / ".memanto" / "on-prem"
        assert result.exists()
