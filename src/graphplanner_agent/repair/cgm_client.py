from __future__ import annotations

import json
from pathlib import Path
import os
import time
from urllib.error import HTTPError, URLError
import urllib.request

from graphplanner_agent.config import AgentConfig
from graphplanner_agent.infra.http import urlopen_no_proxy_for_localhost

RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


class CgmUnavailableError(RuntimeError):
    """Raised when the CGM service cannot be reached or is temporarily unavailable."""


class CgmClient:
    def generate_patch(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError

    def review_intent(self, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError


class MockCgmClient(CgmClient):
    def __init__(self, response: dict[str, object] | None = None):
        self.response = response

    def generate_patch(self, payload: dict[str, object]) -> dict[str, object]:
        if self.response is not None:
            return self.response
        return {"patch": {"edits": [], "summary": "mock CGM has no configured edit"}}

    def review_intent(self, payload: dict[str, object]) -> dict[str, object]:
        if self.response is not None and isinstance(self.response.get("review"), dict):
            return self.response
        return {
            "summary": "mock CGM review",
            "review": {
                "verdict": "needs_more_evidence",
                "confidence": 0.0,
                "target_assessment": "mock CGM has no configured review",
                "evidence_gaps": [],
                "suggested_next_action": "collect or configure real review evidence",
                "adoption_advice": "revise the evidence package before adopting this mock review",
            },
        }


class StaticCgmClient(CgmClient):
    def __init__(self, response: dict[str, object]):
        self.response = response

    def generate_patch(self, payload: dict[str, object]) -> dict[str, object]:
        return self.response

    def review_intent(self, payload: dict[str, object]) -> dict[str, object]:
        if isinstance(self.response.get("review"), dict):
            return self.response
        return {
            "summary": "static CGM review",
            "review": {
                "verdict": "ready",
                "confidence": 0.5,
                "target_assessment": "static response has no explicit review; using configured patch response as neutral evidence",
                "evidence_gaps": [],
                "suggested_next_action": "repair if planner confidence remains high",
                "adoption_advice": "adopt only if the configured patch response matches the committed evidence",
            },
        }


class HttpCgmClient(CgmClient):
    def __init__(self, endpoint: str, timeout: int = 120, max_attempts: int = 3):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts or 1))

    def generate_patch(self, payload: dict[str, object]) -> dict[str, object]:
        return self._post(payload)

    def review_intent(self, payload: dict[str, object]) -> dict[str, object]:
        review_payload = dict(payload)
        review_payload["mode"] = "intent_review"
        return self._post(review_payload)

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with urlopen_no_proxy_for_localhost(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                if exc.code not in RETRYABLE_HTTP_STATUS:
                    raise RuntimeError(f"CGM HTTP {exc.code}: {body[:1000]}") from exc
                if attempt == self.max_attempts - 1:
                    raise CgmUnavailableError(f"CGM HTTP {exc.code}: {body[:1000]}") from exc
                last_error = RuntimeError(f"CGM HTTP {exc.code}: {body[:1000]}")
            except URLError as exc:
                if attempt == self.max_attempts - 1:
                    raise CgmUnavailableError(f"CGM request failed: {exc}") from exc
                last_error = exc
            except TimeoutError as exc:
                if attempt == self.max_attempts - 1:
                    raise CgmUnavailableError(f"CGM request timed out: {exc}") from exc
                last_error = exc
            time.sleep(0.5 * (2**attempt))
        raise CgmUnavailableError(f"CGM request failed: {last_error}")


def make_cgm_client(config: AgentConfig) -> CgmClient:
    backend = config.cgm_backend.lower()
    if backend == "mock":
        response = None
        if config.cgm_mock_response:
            raw = config.cgm_mock_response
            if raw.startswith("@"):
                raw = Path(raw[1:]).read_text(encoding="utf-8")
            response = json.loads(raw)
        return MockCgmClient(response=response)
    if backend == "http":
        if not config.cgm_endpoint:
            raise ValueError("CGM_ENDPOINT is required when CGM_BACKEND=http")
        timeout = int(
            os.getenv("CGM_HTTP_TIMEOUT")
            or os.getenv("CGM_DASHSCOPE_TIMEOUT")
            or str(config.command_timeout)
        )
        return HttpCgmClient(config.cgm_endpoint, timeout=timeout, max_attempts=config.cgm_http_max_attempts)
    raise ValueError(f"unsupported CGM_BACKEND: {config.cgm_backend}")
