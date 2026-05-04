"""Tests for layer0.secrets — SecretsManager."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from layer0.secrets import (
    SecretsManager,
    SecretsVaultCorruptedError,
    SecretNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def vault_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def manager(vault_dir: Path, monkeypatch) -> SecretsManager:
    """Provide a fresh SecretsManager backed by a temp directory."""
    _patch_paths(monkeypatch, vault_dir)
    return SecretsManager("correct-password")


def _patch_paths(monkeypatch, vault_dir: Path) -> None:
    """Redirect vault and salt paths to tmp directory."""
    vault_path = vault_dir / "vault.enc"
    salt_path = vault_dir / "vault.salt"

    # Patch ConfigRegistry so SecretsManager doesn't try to read real config
    import layer0.secrets as sm_module
    orig_init = sm_module.SecretsManager.__init__

    def patched_init(self, master_password: str) -> None:
        from cryptography.fernet import Fernet, InvalidToken
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        import base64, json, tempfile

        self._vault_path = vault_path
        self._salt_path = salt_path
        vault_path.parent.mkdir(parents=True, exist_ok=True)

        if not salt_path.exists():
            salt = os.urandom(16)
            salt_path.write_bytes(salt)

        salt = salt_path.read_bytes()
        self._fernet = self._derive_fernet(master_password, salt)

        if not vault_path.exists():
            self._write_vault({})
        else:
            try:
                self._read_vault()
            except (InvalidToken, json.JSONDecodeError) as exc:
                raise SecretsVaultCorruptedError(str(exc)) from exc

    monkeypatch.setattr(sm_module.SecretsManager, "__init__", patched_init)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSecretsManager:

    def test_new_vault_creation(self, manager: SecretsManager):
        """Fresh vault should be empty and reachable."""
        assert manager.list_keys() == []

    def test_set_and_get(self, manager: SecretsManager):
        manager.set("polygon.api_key", "secret123")
        assert manager.get("polygon.api_key") == "secret123"

    def test_get_missing_key_raises(self, manager: SecretsManager):
        with pytest.raises(SecretNotFoundError):
            manager.get("nonexistent.key")

    def test_delete_key(self, manager: SecretsManager):
        manager.set("foo", "bar")
        manager.delete("foo")
        with pytest.raises(SecretNotFoundError):
            manager.get("foo")

    def test_delete_missing_key_raises(self, manager: SecretsManager):
        with pytest.raises(SecretNotFoundError):
            manager.delete("nonexistent")

    def test_list_keys(self, manager: SecretsManager):
        manager.set("alpha", "1")
        manager.set("beta", "2")
        assert manager.list_keys() == ["alpha", "beta"]

    def test_wrong_password_raises_corrupted(self, vault_dir: Path, monkeypatch):
        """Opening an existing vault with wrong password → SecretsVaultCorruptedError."""
        _patch_paths(monkeypatch, vault_dir)
        m = SecretsManager("correct")
        m.set("x", "y")

        _patch_paths(monkeypatch, vault_dir)
        with pytest.raises(SecretsVaultCorruptedError):
            SecretsManager("wrong-password")

    def test_password_rotation(self, vault_dir: Path, monkeypatch):
        """After rotation the vault is readable with the new password."""
        _patch_paths(monkeypatch, vault_dir)
        m = SecretsManager("old-pass")
        m.set("key", "value")
        m.rotate_master_password("new-pass")

        _patch_paths(monkeypatch, vault_dir)
        m2 = SecretsManager("new-pass")
        assert m2.get("key") == "value"

    def test_recycled_ticker_handling(self, manager: SecretsManager):
        """A key can be overwritten — simulates a recycled ticker scenario."""
        manager.set("ticker.SIVB", "old-nightshade-id")
        manager.set("ticker.SIVB", "new-nightshade-id")
        assert manager.get("ticker.SIVB") == "new-nightshade-id"
