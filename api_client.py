"""
api_client.py – thin wrapper around the POST /api/run_game endpoint.
"""

import requests

BASE_URL = "http://localhost:3000"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Base exception for all API errors."""


class TimeoutError(APIError):
    """Raised when the server reports a timeout (408) or the HTTP request times out."""


class BadRequestError(APIError):
    """Raised when the server returns a 400 Bad Request."""


class ServerError(APIError):
    """Raised when the server returns a 500 Internal Server Error."""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run_game(
    game: str,
    params: dict,
    run_type: str,
    timeout_ms: int = 0,
    http_timeout: float = 120.0,
) -> float:
    """Run a game via the API and return the resulting score.

    Args:
        game:         Identifier of the game to run.
        params:       Dictionary of game parameters.
        run_type:     Execution mode (e.g. "fast", "full", …).
        timeout_ms:   Per-game timeout in milliseconds passed to the server.
                      0 means no server-side timeout.
        http_timeout: Seconds to wait for the HTTP response before raising
                      TimeoutError. Defaults to 120 s.

    Returns:
        The ``score`` float returned by the server on a 200 response.

    Raises:
        TimeoutError:    HTTP request timed out, or server returned 408.
        BadRequestError: Server returned 400.
        ServerError:     Server returned 500.
        APIError:        Any other non-200 status code, or a lower-level
                         requests exception.
    """
    payload = {
        "game": game,
        "params": params,
        "run_type": run_type,
        "timeout": timeout_ms,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/run_game",
            json=payload,
            timeout=http_timeout,
        )
    except requests.exceptions.Timeout as exc:
        raise TimeoutError("HTTP request timed out") from exc
    except requests.exceptions.RequestException as exc:
        raise APIError(f"Request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Handle success
    # ------------------------------------------------------------------
    if response.status_code == 200:
        data = response.json()
        return float(data["score"])

    # ------------------------------------------------------------------
    # Handle known error status codes
    # ------------------------------------------------------------------
    try:
        error_message = response.json().get("error", response.text)
    except ValueError:
        error_message = response.text

    if response.status_code == 400:
        raise BadRequestError(error_message)
    if response.status_code == 408:
        raise TimeoutError(error_message)
    if response.status_code == 500:
        raise ServerError(error_message)

    raise APIError(f"Unexpected status code {response.status_code}: {error_message}")
