import pytest

from backend.services.ai.config import get_gemini_api_key


def test_gemini_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        get_gemini_api_key()


def test_gemini_api_key_loaded(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    assert get_gemini_api_key() == "test-key"
