"""An async client for the Osmund API: https://osmundenergi.dk/docs/osmund-api/.

It only reads, ignores fields and values it does not know (the API is in beta
and adds them), and branches on the error `code`, never on the message.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from importlib.metadata import version
from typing import Any

import aiohttp

API_BASE = "https://api.osmundenergi.dk"
# When nothing says how long to wait after a 429 or 503.
DEFAULT_RETRY_AFTER = 60

TIMEOUT = aiohttp.ClientTimeout(total=30)
USER_AGENT = f"python-osmund/{version('osmund')}"


class OsmundError(Exception):
    """Anything that went wrong talking to the API."""


class OsmundConnectionError(OsmundError):
    """No usable answer: the network, a timeout, or an answer that is not the API's JSON."""


class OsmundAuthError(OsmundError):
    """401: the key is missing, not valid, expired or revoked."""


class OsmundScopeError(OsmundError):
    """403 scope_required: the key cannot read this route."""


class OsmundAccessError(OsmundError):
    """403 api_access_not_included: this location has no API access."""


class OsmundNotFoundError(OsmundError):
    """404: no such location or invoice among the customer's."""


class OsmundUnavailableError(OsmundError):
    """429 or 503: wait `retry_after` seconds, then try again."""

    def __init__(self, message: str, retry_after: int) -> None:
        """Keep how long to wait."""
        super().__init__(message)
        self.retry_after = retry_after


class OsmundApiError(OsmundError):
    """Any other error the API answered with."""

    def __init__(self, message: str, code: str | None, request_id: str | None) -> None:
        """Keep the code and the request id, which support asks for."""
        super().__init__(message)
        self.code = code
        self.request_id = request_id


@dataclass(frozen=True, slots=True)
class ApiKey:
    """The key itself, from GET /api/v1/me."""

    name: str
    prefix: str
    scopes: frozenset[str]
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class Location:
    """One location, from GET /api/v1/locations."""

    id: str
    label: str | None
    state: str
    api_access: str
    price_area: str | None
    meter_resolution: str | None
    # When the latest meter reading ends; it moves when there is new consumption.
    consumption_through: datetime | None = None
    # The latest day that is rated; it moves when there is a new cost.
    rated_through: date | None = None

    @property
    def has_access(self) -> bool:
        """Whether the API answers for this location."""
        return self.api_access == "open"


@dataclass(frozen=True, slots=True)
class PriceInterval:
    """One hour or quarter hour of a price day, in kr/kWh."""

    start: datetime
    end: datetime
    total_inc_vat: float
    total_ex_vat: float
    spot: float


@dataclass(frozen=True, slots=True)
class PriceDay:
    """The prices of one Danish day at one location."""

    day: date
    resolution: str
    complete: bool
    reason: str | None
    intervals: tuple[PriceInterval, ...]

    def at(self, moment: datetime) -> PriceInterval | None:
        """The interval that holds `moment`, or None."""
        for interval in self.intervals:
            if interval.start <= moment < interval.end:
                return interval
        return None


@dataclass(frozen=True, slots=True)
class Reading:
    """kWh used from `start` to `end`."""

    start: datetime
    end: datetime
    kwh: float


@dataclass(frozen=True, slots=True)
class Amount:
    """An amount in kroner, without and with VAT."""

    ex_vat: float
    inc_vat: float


@dataclass(frozen=True, slots=True)
class RatedDay:
    """What one day cost: the charges per kWh, and with the subscriptions."""

    day: date
    kwh: float
    per_kwh: Amount
    with_subscriptions: Amount


@dataclass(frozen=True, slots=True)
class RatedMonth:
    """The rated cost of one month, from GET .../costs/rated."""

    month: str
    days: tuple[RatedDay, ...]
    total_kwh: float
    total: Amount
    total_with_subscriptions: Amount
    complete: bool
    days_with_data: int
    days_in_month: int


@dataclass(frozen=True, slots=True)
class StandbyMonth:
    """What the location draws when nothing is switched on, in kW."""

    month: str
    kw: float


@dataclass(frozen=True, slots=True)
class InvoiceLine:
    """One line as on the invoice."""

    description: str
    quantity: float
    unit: str
    inc_vat: float
    period_start: date | None
    period_end: date | None


@dataclass(frozen=True, slots=True)
class Invoice:
    """One invoice or credit note."""

    number: str
    type: str
    invoice_date: date | None
    due_date: date | None
    total: Amount
    outstanding: float
    paid: bool
    lines: tuple[InvoiceLine, ...] = ()


def _when(value: Any) -> datetime | None:
    """An ISO 8601 time with its offset, or None."""
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def _day(value: Any) -> date | None:
    """A YYYY-MM-DD date, or None."""
    return date.fromisoformat(value) if isinstance(value, str) else None


def _amount(value: Any) -> Amount:
    """An amount object; a missing one is zero."""
    value = value if isinstance(value, dict) else {}
    return Amount(ex_vat=float(value.get("exVat") or 0), inc_vat=float(value.get("incVat") or 0))


def _invoice(row: dict[str, Any], lines: tuple[InvoiceLine, ...] = ()) -> Invoice:
    return Invoice(
        number=str(row["invoiceNo"]),
        type=str(row.get("type", "")),
        invoice_date=_day(row.get("invoiceDate")),
        due_date=_day(row.get("dueDate")),
        total=_amount(row.get("total")),
        outstanding=float(row.get("outstanding") or 0),
        paid=bool(row.get("paid", False)),
        lines=lines,
    )


class OsmundClient:
    """Reads one customer's data with one personal key."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        api_key: str,
        base_url: str = API_BASE,
        user_agent: str = USER_AGENT,
    ) -> None:
        """Use the caller's session; the key is sent in a header and never logged."""
        self._session = session
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._user_agent = user_agent

    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """GET a route and return its JSON, or raise the error it stands for."""
        try:
            async with self._session.get(
                f"{self._base_url}{path}",
                params=params,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                    "User-Agent": self._user_agent,
                },
                timeout=TIMEOUT,
            ) as response:
                body: Any = None
                if (response.content_type or "").startswith("application/json"):
                    body = await response.json()
                if response.status == 200 and isinstance(body, dict):
                    return body
                raise _error(response.status, body, response.headers.get("Retry-After"))
        except (aiohttp.ClientError, TimeoutError) as err:
            raise OsmundConnectionError(f"Could not reach the Osmund API: {err}") from err

    async def me(self) -> ApiKey:
        """GET /api/v1/me."""
        key = (await self._get("/api/v1/me")).get("key") or {}
        return ApiKey(
            name=str(key.get("name", "")),
            prefix=str(key.get("prefix", "")),
            scopes=frozenset(s for s in key.get("scopes") or [] if isinstance(s, str)),
            expires_at=_when(key.get("expiresAt")),
        )

    async def locations(self) -> list[Location]:
        """GET /api/v1/locations."""
        rows = (await self._get("/api/v1/locations")).get("locations") or []
        return [
            Location(
                id=str(row["id"]),
                label=row.get("label") or None,
                state=str(row.get("state", "")),
                api_access=str(row.get("apiAccess", "")),
                price_area=row.get("priceArea"),
                meter_resolution=row.get("meterResolution"),
                consumption_through=_when(row.get("consumptionThrough")),
                rated_through=_day(row.get("ratedThrough")),
            )
            for row in rows
            if isinstance(row, dict) and row.get("id")
        ]

    async def prices(self, location_id: str, day: date, resolution: str) -> PriceDay:
        """GET /api/v1/locations/{id}/prices for today or tomorrow."""
        body = await self._get(
            f"/api/v1/locations/{location_id}/prices",
            {"date": day.isoformat(), "resolution": resolution},
        )
        intervals = tuple(
            PriceInterval(
                start=datetime.fromisoformat(row["start"]),
                end=datetime.fromisoformat(row["end"]),
                total_inc_vat=float(row["totalIncVat"]),
                total_ex_vat=float(row["totalExVat"]),
                spot=float(row["spot"]),
            )
            for row in body.get("intervals") or []
        )
        return PriceDay(
            day=date.fromisoformat(body.get("date", day.isoformat())),
            resolution=str(body.get("resolution", resolution)),
            complete=bool(body.get("complete", False)),
            reason=body.get("reason"),
            intervals=intervals,
        )

    async def consumption(self, location_id: str, first: date, last: date) -> list[Reading]:
        """GET .../consumption by the hour, `first` to `last` included: at most 31 days."""
        body = await self._get(
            f"/api/v1/locations/{location_id}/consumption",
            {"resolution": "PT1H", "from": first.isoformat(), "to": last.isoformat()},
        )
        return [
            Reading(
                start=datetime.fromisoformat(row["start"]),
                end=datetime.fromisoformat(row["end"]),
                kwh=float(row["kwh"]),
            )
            for row in body.get("values") or []
        ]

    async def standby(self, location_id: str) -> list[StandbyMonth]:
        """GET .../standby: completed months, oldest first."""
        rows = (await self._get(f"/api/v1/locations/{location_id}/standby")).get("months") or []
        months = [StandbyMonth(month=str(row["month"]), kw=float(row["standbyKw"])) for row in rows]
        return sorted(months, key=lambda m: m.month)

    async def rated_cost(self, location_id: str, month: str) -> RatedMonth:
        """GET .../costs/rated for one month, YYYY-MM."""
        body = await self._get(f"/api/v1/locations/{location_id}/costs/rated", {"month": month})
        return RatedMonth(
            month=str(body.get("month", month)),
            days=tuple(
                RatedDay(
                    day=date.fromisoformat(row["date"]),
                    kwh=float(row.get("kwh") or 0),
                    per_kwh=Amount(
                        ex_vat=float(row.get("exVat") or 0), inc_vat=float(row.get("incVat") or 0)
                    ),
                    with_subscriptions=_amount(row.get("totalWithSubscriptions")),
                )
                for row in body.get("days") or []
            ),
            total_kwh=float(body.get("totalKwh") or 0),
            total=_amount(body.get("total")),
            total_with_subscriptions=_amount(body.get("totalWithSubscriptions")),
            complete=bool(body.get("complete", False)),
            days_with_data=int(body.get("daysWithData") or 0),
            days_in_month=int(body.get("daysInMonth") or 0),
        )

    async def invoices(self, per_page: int = 25) -> list[Invoice]:
        """GET /api/v1/invoices: the first page, newest first."""
        rows = (await self._get("/api/v1/invoices", {"page": "1", "perPage": str(per_page)})).get(
            "invoices"
        ) or []
        return [_invoice(row) for row in rows if isinstance(row, dict) and row.get("invoiceNo")]

    async def invoice(self, number: str) -> Invoice:
        """GET /api/v1/invoices/{invoiceNo}, with its lines."""
        body = await self._get(f"/api/v1/invoices/{number}")
        lines = tuple(
            InvoiceLine(
                description=str(row.get("description", "")),
                quantity=float(row.get("quantity") or 0),
                unit=str(row.get("unit", "")),
                inc_vat=float(row.get("incVat") or 0),
                period_start=_day(row.get("periodStart")),
                period_end=_day(row.get("periodEnd")),
            )
            for row in body.get("lines") or []
        )
        return _invoice(body, lines)


def _error(status: int, body: Any, retry_after_header: str | None) -> OsmundError:
    """The error an answer that is not 200 stands for."""
    if not isinstance(body, dict):
        # Not the API's JSON: a proxy, a firewall or an outage in front of it.
        return OsmundConnectionError(f"The Osmund API answered HTTP {status} without its JSON")
    code = body.get("code")
    message = str(body.get("message") or f"HTTP {status}")
    if status == 401:
        return OsmundAuthError(message)
    if status == 403 and code == "scope_required":
        return OsmundScopeError(message)
    if status == 403 and code == "api_access_not_included":
        return OsmundAccessError(message)
    if status == 404:
        return OsmundNotFoundError(message)
    if status in (429, 503):
        return OsmundUnavailableError(message, _retry_after(retry_after_header, body.get("retryAfter")))
    return OsmundApiError(message, code if isinstance(code, str) else None, body.get("requestId"))


def _retry_after(header: str | None, from_body: Any) -> int:
    """Seconds to wait: the header first, then the body, then a default."""
    for value in (header, from_body):
        try:
            seconds = int(value)  # type: ignore[arg-type]
        except TypeError, ValueError:
            continue
        if seconds > 0:
            return seconds
    return DEFAULT_RETRY_AFTER
