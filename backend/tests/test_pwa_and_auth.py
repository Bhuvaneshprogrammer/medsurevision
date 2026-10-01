"""
tests/test_pwa_and_auth.py — Tests for PWA assets, offline shell, and authentication hardening.
"""
import os
from starlette.testclient import TestClient

from app.main import app
from app.auth.security import create_access_token, decode_token, hash_password, verify_password
from app.config import MEDSURE_ADMIN_PASSWORD


def test_pwa_manifest_and_icons():
    client = TestClient(app)
    # Check manifest.json
    r_manifest = client.get("/manifest.json")
    assert r_manifest.status_code == 200
    data = r_manifest.json()
    assert data["name"] == "MedSure Vision"
    assert data["start_url"] == "/"
    assert data["display"] == "standalone"

    # Check icons
    r_192 = client.get("/icon-192.png")
    assert r_192.status_code == 200
    assert r_192.headers.get("content-type") == "image/png"

    r_512 = client.get("/icon-512.png")
    assert r_512.status_code == 200
    assert r_512.headers.get("content-type") == "image/png"


def test_pwa_service_worker_and_root():
    client = TestClient(app)
    # Service worker
    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert "medsure-v2" in r_sw.text
    assert "install" in r_sw.text

    # Root redirect / index.html
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "MedSure Vision" in r_root.text

    r_index = client.get("/index.html")
    assert r_index.status_code == 200
    assert "MedSure Vision" in r_index.text


def test_auth_jwt_and_bcrypt_no_warnings():
    # Verify password hashing with bcrypt
    hashed = hash_password("test-password-123")
    assert verify_password("test-password-123", hashed)
    assert not verify_password("wrong-password", hashed)

    # Verify JWT creation and decoding with python-jose
    payload = {"sub": "1", "role": "admin", "username": "admin"}
    token = create_access_token(payload)
    decoded = decode_token(token)
    assert decoded["sub"] == "1"
    assert decoded["role"] == "admin"
    assert decoded["username"] == "admin"


def test_admin_password_env_var(monkeypatch):
    monkeypatch.setenv("MEDSURE_ADMIN_PASSWORD", "custom-demo-pass-2026")
    import importlib
    import app.config
    importlib.reload(app.config)
    assert app.config.MEDSURE_ADMIN_PASSWORD == "custom-demo-pass-2026"
