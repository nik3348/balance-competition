"""Thin wrapper around the POST /api/run_game endpoint."""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:3000"

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504}
DEFAULT_HTTP_TIMEOUT = 600.0


class APIError(Exception):
    """Base exception for all API errors."""


class APITimeoutError(APIError):
    """Raised when the server reports a timeout (408) or the HTTP request times out."""


class BadRequestError(APIError):
    """Raised when the server returns a 400 Bad Request."""


class ServerError(APIError):
    """Raised when the server returns a 500 Internal Server Error."""


def run_game(
    game: str,
    params: dict,
    run_type: str,
    timeout_ms: int = 0,
    http_timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> float:
    payload = {
        "game": game,
        "params": params,
        "run_type": run_type,
        "timeout": timeout_ms,
    }

    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{BASE_URL}/api/run_game",
                json=payload,
                timeout=http_timeout,
            )
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                logger.warning("HTTP timeout on attempt %d, retrying in %.1fs", attempt + 1, wait)
                time.sleep(wait)
                continue
            raise APITimeoutError("HTTP request timed out after all retries") from exc
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF ** attempt
                logger.warning("Request failed on attempt %d: %s, retrying in %.1fs", attempt + 1, exc, wait)
                time.sleep(wait)
                continue
            raise APIError(f"Request failed after all retries: {exc}") from exc

        if response.status_code == 200:
            return float(response.json()["score"])

        try:
            error_message = response.json().get("error", response.text)
        except ValueError:
            error_message = response.text

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF ** attempt
            logger.warning(
                "Server returned %d on attempt %d: %s, retrying in %.1fs",
                response.status_code, attempt + 1, error_message, wait,
            )
            time.sleep(wait)
            continue

        if response.status_code == 400:
            raise BadRequestError(error_message)
        if response.status_code == 408:
            raise APITimeoutError(error_message)
        if response.status_code == 500:
            raise ServerError(error_message)

        raise APIError(f"Unexpected status code {response.status_code}: {error_message}")

    raise APIError(f"All {MAX_RETRIES} retries exhausted: {last_exc}")
