from fastapi.testclient import TestClient

from backend.main import app


def test_app_serves_frontend_and_api_from_one_origin():
    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/app.js")
        health = client.get("/health")

    assert page.status_code == 200
    assert "ClauseLens" in page.text
    assert script.status_code == 200
    assert 'window.location.port === "5500" ? "http://127.0.0.1:8000" : ""' in script.text
    assert health.status_code == 200
