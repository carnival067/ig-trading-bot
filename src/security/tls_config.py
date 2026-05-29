"""TLS 1.2+ configuration for all external connections.

Configures secure TLS settings for:
- IG API connections
- News source API connections
- Notification channel connections (Telegram, Discord, SMTP)

Enforces:
- Minimum TLS 1.2
- Certificate chain validation
- Strong cipher suites only
"""

from __future__ import annotations

import logging
import ssl
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Minimum TLS version
MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_2

# Strong cipher suites (TLS 1.2+)
STRONG_CIPHERS = ":".join(
    [
        "ECDHE+AESGCM",
        "ECDHE+CHACHA20",
        "DHE+AESGCM",
        "DHE+CHACHA20",
        "ECDH+AESGCM",
        "DH+AESGCM",
        "ECDH+AES",
        "DH+AES",
        "!aNULL",
        "!MD5",
        "!DSS",
        "!RC4",
        "!3DES",
    ]
)


def create_ssl_context(
    ca_bundle_path: str | None = None,
    verify_certs: bool = True,
    min_tls_version: ssl.TLSVersion = MIN_TLS_VERSION,
) -> ssl.SSLContext:
    """Create a secure SSL context with TLS 1.2+ enforcement.

    Args:
        ca_bundle_path: Optional path to a custom CA certificate bundle.
            If None, uses the system default CA certificates.
        verify_certs: Whether to verify server certificates (always True in production).
        min_tls_version: Minimum TLS version to accept (default: TLS 1.2).

    Returns:
        Configured ssl.SSLContext with strong security settings.

    Raises:
        FileNotFoundError: If the specified CA bundle path does not exist.
        ssl.SSLError: If the SSL context cannot be created.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    # Enforce minimum TLS version
    context.minimum_version = min_tls_version

    # Disable older protocols explicitly
    context.options |= ssl.OP_NO_SSLv2
    context.options |= ssl.OP_NO_SSLv3
    context.options |= ssl.OP_NO_TLSv1
    context.options |= ssl.OP_NO_TLSv1_1

    # Set strong cipher suites
    context.set_ciphers(STRONG_CIPHERS)

    # Certificate verification
    if verify_certs:
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True

        if ca_bundle_path:
            ca_path = Path(ca_bundle_path)
            if not ca_path.exists():
                raise FileNotFoundError(f"CA bundle not found: {ca_bundle_path}")
            context.load_verify_locations(cafile=str(ca_path))
            logger.info("Loaded custom CA bundle from: %s", ca_bundle_path)
        else:
            context.load_default_certs()
            logger.info("Using system default CA certificates")
    else:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        logger.warning("TLS certificate verification DISABLED - not recommended for production")

    logger.info(
        "SSL context created: min_version=%s, verify=%s",
        min_tls_version.name,
        verify_certs,
    )
    return context


def create_secure_httpx_client(
    ca_bundle_path: str | None = None,
    verify_certs: bool = True,
    timeout: float = 30.0,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Create an httpx AsyncClient with TLS 1.2+ enforcement.

    This is the recommended way to make external HTTP requests in the application.
    All connections to IG API, news sources, and notification channels should use this.

    Args:
        ca_bundle_path: Optional path to a custom CA certificate bundle.
        verify_certs: Whether to verify server certificates.
        timeout: Request timeout in seconds.
        **kwargs: Additional arguments passed to httpx.AsyncClient.

    Returns:
        Configured httpx.AsyncClient with secure TLS settings.
    """
    ssl_context = create_ssl_context(
        ca_bundle_path=ca_bundle_path,
        verify_certs=verify_certs,
    )

    client = httpx.AsyncClient(
        verify=ssl_context,
        timeout=httpx.Timeout(timeout),
        http2=True,
        **kwargs,
    )

    logger.info("Secure HTTPX client created with TLS 1.2+ enforcement")
    return client


def validate_certificate_chain(hostname: str, port: int = 443) -> dict[str, Any]:
    """Validate the certificate chain for a given hostname.

    Connects to the host and verifies:
    - Certificate is valid and not expired
    - Certificate chain is complete
    - Hostname matches the certificate

    Args:
        hostname: The hostname to validate.
        port: The port to connect to (default: 443).

    Returns:
        Dictionary with validation results including:
        - valid: bool indicating if the certificate is valid
        - subject: Certificate subject
        - issuer: Certificate issuer
        - expires: Certificate expiration date
        - error: Error message if validation failed
    """
    result: dict[str, Any] = {
        "hostname": hostname,
        "port": port,
        "valid": False,
        "subject": None,
        "issuer": None,
        "expires": None,
        "error": None,
    }

    try:
        context = create_ssl_context()
        import socket

        with socket.create_connection((hostname, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                if cert:
                    result["valid"] = True
                    result["subject"] = dict(x[0] for x in cert.get("subject", ()))
                    result["issuer"] = dict(x[0] for x in cert.get("issuer", ()))
                    result["expires"] = cert.get("notAfter")
                    logger.info("Certificate valid for %s:%d", hostname, port)
                else:
                    result["error"] = "No certificate returned"

    except ssl.SSLCertVerificationError as exc:
        result["error"] = f"Certificate verification failed: {exc}"
        logger.warning("Certificate validation failed for %s:%d - %s", hostname, port, exc)
    except ssl.SSLError as exc:
        result["error"] = f"SSL error: {exc}"
        logger.warning("SSL error for %s:%d - %s", hostname, port, exc)
    except (OSError, TimeoutError) as exc:
        result["error"] = f"Connection error: {exc}"
        logger.warning("Connection failed for %s:%d - %s", hostname, port, exc)

    return result


def get_tls_config_from_settings() -> dict[str, Any]:
    """Load TLS configuration from application settings.

    Returns:
        Dictionary with TLS configuration parameters.
    """
    import os

    ca_bundle = os.environ.get("TLS_CA_BUNDLE_PATH", "")
    min_version = os.environ.get("TLS_MIN_VERSION", "1.2")
    verify = os.environ.get("TLS_VERIFY_CERTS", "true").lower() == "true"

    tls_version = MIN_TLS_VERSION
    if min_version == "1.3":
        tls_version = ssl.TLSVersion.TLSv1_3

    return {
        "ca_bundle_path": ca_bundle if ca_bundle else None,
        "min_tls_version": tls_version,
        "verify_certs": verify,
    }
