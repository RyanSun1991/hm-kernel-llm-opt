"""Operator-owned production settings. Secrets are environment references, never values."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


def endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})
    ):
        raise ValueError("Use HTTPS or loopback HTTP, without URL credentials/query/fragment")
    return value.rstrip("/")


def secret(name: str | None, *, minimum_length: int = 1) -> str | None:
    if name is None:
        return None
    value = os.environ.get(name, "")
    if len(value) < minimum_length or "\n" in value or "\r" in value:
        raise ValueError(f"Configure a valid secret in environment {name}")
    return value


class WorkerConfig(Strict):
    url: str
    directory: str
    provider_id: str = Field(min_length=1, max_length=160)
    model_id: str = Field(min_length=1, max_length=160)
    authorization_env: str | None = None
    concurrency: int = Field(default=2, ge=1, le=16)
    lease_seconds: int = Field(default=60, ge=30, le=600)
    task_timeout_seconds: int = Field(default=900, ge=60, le=7200)
    max_packet_bytes: int = Field(default=256_000, ge=1000, le=1_000_000)
    max_context_rounds: int = Field(default=3, ge=0, le=10)
    max_context_bytes: int = Field(default=128_000, ge=1000, le=1_000_000)
    max_reported_tokens: int = Field(default=100_000, ge=1000, le=1_000_000)

    _url = field_validator("url")(endpoint)

    @field_validator("directory")
    @classmethod
    def absolute_directory(cls, value):
        if not Path(value).is_absolute():
            raise ValueError("Worker directory must be absolute")
        return str(Path(value).resolve())


class Contact(Strict):
    principal: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=320, pattern=r"^[^\r\n]+$")


class ApprovalConfig(Strict):
    contacts: dict[str, Contact] = Field(min_length=1, max_length=2000)
    signing_key_env: str = Field(min_length=1)
    gateway_url: str
    expires_seconds: int = Field(default=259200, ge=60, le=2592000)
    webhook_url: str | None = None
    webhook_authorization_env: str | None = None
    smtp_host: str | None = None
    smtp_port: int = Field(default=465, ge=1, le=65535)
    smtp_security: Literal["ssl", "starttls"] = "ssl"
    smtp_sender: str | None = Field(default=None, pattern=r"^[^\r\n]+$")
    smtp_username: str | None = None
    smtp_password_env: str | None = None

    _gateway = field_validator("gateway_url")(endpoint)

    @field_validator("webhook_url")
    @classmethod
    def webhook(cls, value):
        return endpoint(value) if value else value

    @model_validator(mode="after")
    def transport(self):
        if bool(self.webhook_url) == bool(self.smtp_host):
            raise ValueError("Configure exactly one notification transport: webhook or SMTP TLS")
        if self.smtp_host and not self.smtp_sender:
            raise ValueError("SMTP requires smtp_sender")
        if bool(self.smtp_username) != bool(self.smtp_password_env):
            raise ValueError("SMTP username and password environment reference must be paired")
        if any(not owner or owner != owner.strip() for owner in self.contacts):
            raise ValueError("Owner directory keys must be nonempty canonical owner labels")
        return self


class ValidationRunnerConfig(Strict):
    command: list[str] = Field(min_length=1, max_length=64)
    cwd: str
    resource_id: str = Field(default="business-validation", min_length=1, max_length=160)
    timeout_seconds: int = Field(default=3600, ge=1, le=86400)

    @model_validator(mode="after")
    def paths(self):
        if not Path(self.cwd).is_absolute() or not Path(self.command[0]).is_absolute():
            raise ValueError("Validation adapter cwd and executable must be absolute")
        if any(not p or "\x00" in p for p in self.command):
            raise ValueError("Validation command must be an argv list without NUL")
        return self


class ProductionConfig(Strict):
    worker: WorkerConfig | None = None
    approval: ApprovalConfig | None = None
    validation: ValidationRunnerConfig | None = None


def production_config(value: dict | None) -> ProductionConfig:
    return ProductionConfig.model_validate(value or {})


def check_production(value: dict | None) -> dict:
    """Offline readiness; never contacts a service or echoes a secret."""
    config = production_config(value)
    missing = []
    names = []
    if config.worker:
        names.append(config.worker.authorization_env)
        if not Path(config.worker.directory).is_dir():
            missing.append("worker.directory is missing")
    if config.validation:
        if not Path(config.validation.cwd).is_dir():
            missing.append("validation.cwd is missing")
        if not Path(config.validation.command[0]).is_file():
            missing.append("validation executable is missing")
    if config.approval:
        names += [
            config.approval.signing_key_env,
            config.approval.webhook_authorization_env,
            config.approval.smtp_password_env,
        ]
    for name in names:
        if name:
            try:
                signing = config.approval and name == config.approval.signing_key_env
                secret(name, minimum_length=32 if signing else 1)
            except ValueError:
                missing.append(f"environment:{name}")
    return {
        "worker_configured": config.worker is not None,
        "approval_configured": config.approval is not None,
        "validation_configured": config.validation is not None,
        "missing": missing,
        "ready_for_connection_check": not missing,
        "external_services_checked": False,
    }
