"""Layer 0 — Secrets Manager.

Encrypts all API keys and sensitive credentials using Fernet symmetric
encryption with PBKDF2HMAC key derivation.  The master password is never
stored — it is accepted at runtime only.

Vault location:  ~/.nightshade/vault.enc
Salt location:   ~/.nightshade/vault.salt
Key derivation:  PBKDF2HMAC + SHA256 + 480 000 iterations
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from layer0.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SecretsError(Exception):
    """Base exception for all secrets-manager errors."""


class SecretsVaultCorruptedError(SecretsError):
    """Raised when vault cannot be decrypted (wrong password or corrupted data)."""


class SecretsVaultPermissionError(SecretsError):
    """Raised when vault or salt file has insecure permissions (not 600)."""


class SecretNotFoundError(SecretsError):
    """Raised when a requested key does not exist in the vault."""


# ---------------------------------------------------------------------------
# SecretsManager
# ---------------------------------------------------------------------------

class SecretsManager:
    """Fernet-encrypted secrets vault backed by a local file.

    Args:
        master_password: The password used to derive the encryption key.
                         Never stored anywhere.
    """

    _ITERATIONS = 480_000

    def __init__(self, master_password: str) -> None:
        from layer0.config import ConfigRegistry  # lazy import to avoid circular

        # Resolve paths — try loading from config if available, else use defaults
        try:
            cfg = ConfigRegistry()
            vault_path = Path(cfg.get("secrets.vault_path", "~/.nightshade/vault.enc")).expanduser()
            salt_path = Path(cfg.get("secrets.salt_path", "~/.nightshade/vault.salt")).expanduser()
        except Exception:
            vault_path = Path("~/.nightshade/vault.enc").expanduser()
            salt_path = Path("~/.nightshade/vault.salt").expanduser()

        self._vault_path = vault_path
        self._salt_path = salt_path

        vault_path.parent.mkdir(parents=True, exist_ok=True)

        # Create salt if absent (first-time setup)
        if not salt_path.exists():
            salt = os.urandom(16)
            salt_path.write_bytes(salt)
            _set_permissions_600(salt_path)
            logger.debug("Created new salt file: %s", salt_path)

        # Verify permissions
        _assert_permissions_600(salt_path)
        if vault_path.exists():
            _assert_permissions_600(vault_path)

        salt = salt_path.read_bytes()
        self._fernet = self._derive_fernet(master_password, salt)

        # Create empty vault if absent
        if not vault_path.exists():
            self._write_vault({})
            logger.debug("Created new empty vault: %s", vault_path)
        else:
            # Validate password by attempting a decrypt
            try:
                self._read_vault()
            except (InvalidToken, json.JSONDecodeError, Exception) as exc:
                raise SecretsVaultCorruptedError(
                    f"Cannot decrypt vault at {vault_path}: {exc}"
                ) from exc

        logger.debug("SecretsManager initialised, vault=%s", vault_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> str:
        """Return the value for *key*.  Raises SecretNotFoundError if missing."""
        logger.debug("get secret: key=%s", key)
        vault = self._read_vault()
        if key not in vault:
            raise SecretNotFoundError(f"Secret not found: {key!r}")
        return vault[key]

    def set(self, key: str, value: str) -> None:
        """Store *value* under *key*.  Atomic write."""
        logger.debug("set secret: key=%s", key)
        vault = self._read_vault()
        vault[key] = value
        self._write_vault(vault)

    def delete(self, key: str) -> None:
        """Remove *key* from vault.  Raises SecretNotFoundError if missing."""
        logger.debug("delete secret: key=%s", key)
        vault = self._read_vault()
        if key not in vault:
            raise SecretNotFoundError(f"Secret not found: {key!r}")
        del vault[key]
        self._write_vault(vault)

    def list_keys(self) -> list[str]:
        """Return all key names (values never returned)."""
        vault = self._read_vault()
        return sorted(vault.keys())

    def rotate_master_password(self, new_password: str) -> None:
        """Re-encrypt the entire vault with a new password and new salt.

        Backs up old vault and salt files with a UTC timestamp suffix.
        """
        import time
        vault = self._read_vault()  # decrypt with current key

        ts = str(int(time.time()))
        # Backup old files
        for p in (self._vault_path, self._salt_path):
            if p.exists():
                backup = p.with_suffix(f".{ts}{p.suffix}")
                p.rename(backup)
                logger.debug("Backed up %s → %s", p, backup)

        # Write new salt
        new_salt = os.urandom(16)
        self._salt_path.write_bytes(new_salt)
        _set_permissions_600(self._salt_path)

        self._fernet = self._derive_fernet(new_password, new_salt)
        self._write_vault(vault)
        logger.debug("Master password rotated successfully")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_fernet(self, password: str, salt: bytes) -> Fernet:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
        return Fernet(key)

    def _read_vault(self) -> dict:
        encrypted = self._vault_path.read_bytes()
        raw = self._fernet.decrypt(encrypted)
        return json.loads(raw.decode("utf-8"))

    def _write_vault(self, vault: dict) -> None:
        """Atomic write: serialize → encrypt → temp-file → rename."""
        raw = json.dumps(vault).encode("utf-8")
        encrypted = self._fernet.encrypt(raw)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._vault_path.parent, prefix=".vault_tmp_"
        )
        try:
            os.write(tmp_fd, encrypted)
            os.close(tmp_fd)
            _set_permissions_600(Path(tmp_path))
            os.replace(tmp_path, self._vault_path)
        except Exception:
            os.close(tmp_fd)
            os.unlink(tmp_path)
            raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_permissions_600(path: Path) -> None:
    """Set file permissions to owner read/write only (600)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass  # Windows — no-op


def _assert_permissions_600(path: Path) -> None:
    """Raise SecretsVaultPermissionError if file permissions are not 600.

    On Windows this check is skipped (permissions model differs).
    """
    if os.name == "nt":
        return  # Windows doesn't support Unix permission bits
    mode = oct(stat.S_IMODE(path.stat().st_mode))
    if mode != oct(0o600):
        raise SecretsVaultPermissionError(
            f"Insecure permissions on {path}: {mode} (expected 0o600)"
        )
