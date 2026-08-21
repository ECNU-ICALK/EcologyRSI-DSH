"""Shared transport policy for model gateway URLs."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class GatewayUrlAssessment:
    """Normalized URL plus its directory and execution policy result."""

    url: str
    reason_code: str | None = None
    reason_message: str | None = None
    gateway_error: str | None = None
    insecure_http_exception: bool = False

    @property
    def directory_reason(self) -> dict[str, str] | None:
        if self.reason_code is None:
            return None
        return {
            "code": self.reason_code,
            "message": self.reason_message or "The model gateway URL is unavailable.",
        }


def assess_gateway_url(
    value: Any,
    *,
    allow_insecure_http: bool = False,
) -> GatewayUrlAssessment:
    """Apply one URL policy for both directory visibility and execution."""

    if not isinstance(value, str) or not value.strip():
        return GatewayUrlAssessment(
            "",
            reason_code="missing_gateway_url",
            reason_message="The DSH provider does not define a gateway URL.",
            gateway_error="gateway_url must be a non-empty string",
        )
    raw_url = value.strip()
    try:
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname
        # Accessing port validates malformed and out-of-range values.
        parsed.port
    except ValueError:
        return _invalid_gateway_url(
            raw_url,
            "The DSH provider gateway URL is invalid.",
            "gateway_url must use HTTPS or loopback HTTP",
        )
    if parsed.username is not None or parsed.password is not None:
        return _invalid_gateway_url(
            raw_url,
            "The DSH provider gateway URL must not contain credentials.",
            "gateway_url must not contain credentials",
        )
    if parsed.query or parsed.fragment:
        return _invalid_gateway_url(
            raw_url,
            "The DSH provider gateway URL must not contain a query or fragment.",
            "gateway_url must not contain a query or fragment",
        )
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not hostname:
        return _invalid_gateway_url(
            raw_url,
            "The DSH provider gateway URL is invalid.",
            "gateway_url must use HTTPS or loopback HTTP",
        )
    normalized_url = raw_url.rstrip("/")
    if scheme == "https":
        return GatewayUrlAssessment(normalized_url)

    normalized_hostname = hostname.rstrip(".").casefold()
    if _is_loopback_hostname(normalized_hostname):
        return GatewayUrlAssessment(normalized_url)
    if allow_insecure_http:
        return GatewayUrlAssessment(
            normalized_url,
            insecure_http_exception=True,
        )
    return GatewayUrlAssessment(
        normalized_url,
        reason_code="insecure_http_blocked",
        reason_message=(
            "Non-loopback HTTP is blocked for this provider; configure HTTPS "
            "or add its exact provider ID to the controlled insecure-HTTP allowlist."
        ),
        gateway_error="plain HTTP is allowed only for loopback addresses",
    )


def _invalid_gateway_url(
    url: str,
    reason_message: str,
    gateway_error: str,
) -> GatewayUrlAssessment:
    return GatewayUrlAssessment(
        url,
        reason_code="invalid_gateway_url",
        reason_message=reason_message,
        gateway_error=gateway_error,
    )


def _is_loopback_hostname(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
