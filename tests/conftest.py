"""Shared test setup.

Every answer the tests serve is built from the examples in the public API
document (tests/fixtures, written by scripts/fixtures_from_openapi.py) or
made up here. Nothing comes from a real account.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from osmund import OsmundClient

FIXTURES = Path(__file__).parent / "fixtures"
CPH = ZoneInfo("Europe/Copenhagen")

KEY = "osm_4f9Kx2Qa_test_only_not_a_real_key"
# The two locations of the documented example: one with access, one without.
OPEN_ID = "0192f3a1-7c2e-7b1a-9d3e-2f6a8c1e4b10"
CLOSED_ID = "0192f3a1-9a41-7d02-8c11-5b7e3d2f6a22"

ME = "/api/v1/me"
LOCATIONS = "/api/v1/locations"
INVOICES = "/api/v1/invoices"


def fixture(name: str) -> Any:
    """A documented example answer, as a fresh copy."""
    return copy.deepcopy(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))


def price_day(day: date, base: float = 1.0, resolution: str = "PT1H") -> dict[str, Any]:
    """A whole Danish day of made-up prices: the hour h costs base + h / 100 kr with VAT."""
    answer = fixture("prices")
    template = answer["intervals"][0]
    start = datetime(day.year, day.month, day.day, tzinfo=CPH)
    step = timedelta(minutes=15) if resolution == "PT15M" else timedelta(hours=1)
    intervals = []
    moment = start
    while moment.date() == day:
        # Step in UTC, so a day with a clock change keeps its 23 or 25 hours.
        end = (moment.astimezone(ZoneInfo("UTC")) + step).astimezone(CPH)
        total = round(base + moment.hour / 100 + moment.minute / 10000, 5)
        interval = copy.deepcopy(template)
        interval.update(
            start=moment.isoformat(),
            end=end.isoformat(),
            totalIncVat=total,
            totalExVat=round(total / 1.25, 5),
            spot=round(total / 2, 5),
        )
        intervals.append(interval)
        moment = end
    answer.update(date=day.isoformat(), resolution=resolution, intervals=intervals, complete=True)
    return answer


@dataclass
class Request:
    """One request the fake API received."""

    path_qs: str
    headers: dict[str, str]


@dataclass
class FakeApi:
    """A local HTTP server that answers each path as it is told, and keeps every request."""

    base_url: str = ""
    answers: dict[str, web.Response] = field(default_factory=dict)
    requests: list[Request] = field(default_factory=list)

    def get(
        self,
        path: str,
        *,
        status: int = 200,
        payload: Any = None,
        text: str | None = None,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Answer GET `path`, whatever its query, with this."""
        body = json.dumps(payload) if text is None else text
        self.answers[path] = web.Response(
            status=status, text=body, content_type=content_type, headers=headers
        )

    async def handle(self, request: web.Request) -> web.Response:
        self.requests.append(Request(request.path_qs, dict(request.headers)))
        if request.path not in self.answers:
            return web.json_response({"code": "not_mocked"}, status=599)
        return self.answers.pop(request.path)


@pytest.fixture
async def api() -> AsyncIterator[FakeApi]:
    """The API's answers; nothing reaches the network."""
    fake = FakeApi()
    app = web.Application()
    app.router.add_get("/{tail:.*}", fake.handle)
    server = TestServer(app)
    await server.start_server()
    fake.base_url = str(server.make_url("")).rstrip("/")
    yield fake
    await server.close()


@pytest.fixture
async def client(api: FakeApi) -> AsyncIterator[OsmundClient]:
    """A client with its own session and the made-up key, talking to the fake API."""
    async with aiohttp.ClientSession() as session:
        yield OsmundClient(session, KEY, base_url=api.base_url)
