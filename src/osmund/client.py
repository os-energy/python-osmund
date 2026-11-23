"""An async client for the Osmund API: https://osmundenergi.dk/docs/osmund-api/.

It only reads, and it represents the API's answers as they are: the fields keep
the API's names (in snake_case), a value the answer leaves out is None, and
fields and values it does not know are ignored, since the API adds them. It
branches on the error `code`, never on the message.
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

# Every personal key starts with this.
KEY_PREFIX = "osm_"
# The scopes a key can have.
SCOPE_PRICES = "prices:read"
SCOPE_CONSUMPTION = "consumption:read"
SCOPE_COSTS = "costs:read"
SCOPE_LABELS = "labels:read"
# Location.api_access when the API answers for the location.
API_ACCESS_OPEN = "open"
# The resolutions of prices and consumption.
RESOLUTION_HOUR = "PT1H"
RESOLUTION_QUARTER_HOUR = "PT15M"

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
    state: str | None
    # "open" when the API answers for this location.
    api_access: str | None
    price_area: str | None
    meter_resolution: str | None
    # When the latest meter reading ends; it moves when there is new consumption.
    consumption_through: datetime | None = None
    # The latest day that is rated; it moves when there is a new cost.
    rated_through: date | None = None


@dataclass(frozen=True, slots=True)
class PriceInterval:
    """One interval of a price day, in DKK/kWh."""

    start: datetime
    end: datetime
    total_inc_vat: float
    total_ex_vat: float
    spot: float


@dataclass(frozen=True, slots=True)
class PriceDay:
    """The prices of one Danish day at one location; `day` is the answer's `date`."""

    day: date
    resolution: str | None
    complete: bool | None
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
    """One value of GET .../consumption: kWh used from `start` to `end`."""

    start: datetime
    end: datetime
    kwh: float | None


@dataclass(frozen=True, slots=True)
class Amount:
    """An amount in DKK, without and with VAT."""

    ex_vat: float | None
    inc_vat: float | None


@dataclass(frozen=True, slots=True)
class RatedDay:
    """One day of GET .../costs/rated; `day` is the answer's `date`."""

    day: date
    kwh: float | None
    ex_vat: float | None
    inc_vat: float | None
    total_with_subscriptions: Amount | None


@dataclass(frozen=True, slots=True)
class RatedMonth:
    """The rated cost of one month, from GET .../costs/rated."""

    month: str
    days: tuple[RatedDay, ...]
    total_kwh: float | None
    total: Amount | None
    total_with_subscriptions: Amount | None
    complete: bool | None
    days_with_data: int | None
    days_in_month: int | None


@dataclass(frozen=True, slots=True)
class StandbyMonth:
    """One month of GET .../standby: what the location draws when nothing is switched on."""

    month: str
    standby_kw: float | None


@dataclass(frozen=True, slots=True)
class InvoiceLine:
    """One line as on the invoice."""

    description: str | None
    quantity: float | None
    unit: str | None
    ex_vat: float | None
    inc_vat: float | None
    period_start: date | None
    period_end: date | None


@dataclass(frozen=True, slots=True)
class Invoice:
    """One invoice or credit note."""

    invoice_no: str
    type: str | None
    invoice_date: date | None
    due_date: date | None
    total: Amount | None
    outstanding: float | None
    paid: bool | None
    lines: tuple[InvoiceLine, ...] = ()


def _when(value: Any) -> datetime | None:
    """An ISO 8601 time with its offset, or None."""
    return datetime.fromisoformat(value) if isinstance(value, str) else None


def _day(value: Any) -> date | None:
    """A YYYY-MM-DD date, or None."""
    return date.fromisoformat(value) if isinstance(value, str) else None


def _float(value: Any) -> float | None:
    """A number, or None when the answer leaves it out."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int(value: Any) -> int | None:
    """A whole number, or None when the answer leaves it out."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bool(value: Any) -> bool | None:
    """A boolean, or None when the answer leaves it out."""
    return value if isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    """A string, or None when the answer leaves it out."""
    return value if isinstance(value, str) else None


def _amount(value: Any) -> Amount | None:
    """An amount object, or None when the answer leaves it out."""
    if not isinstance(value, dict):
        return None
    return Amount(ex_vat=_float(value.get("exVat")), inc_vat=_float(value.get("incVat")))


def _invoice(row: dict[str, Any], lines: tuple[InvoiceLine, ...] = ()) -> Invoice:
    return Invoice(
        invoice_no=str(row["invoiceNo"]),
        type=_str(row.get("type")),
        invoice_date=_day(row.get("invoiceDate")),
        due_date=_day(row.get("dueDate")),
        total=_amount(row.get("total")),
        outstanding=_float(row.get("outstanding")),
        paid=_bool(row.get("paid")),
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
                label=_str(row.get("label")),
                state=_str(row.get("state")),
                api_access=_str(row.get("apiAccess")),
                price_area=_str(row.get("priceArea")),
                meter_resolution=_str(row.get("meterResolution")),
                consumption_through=_when(row.get("consumptionThrough")),
                rated_through=_day(row.get("ratedThrough")),
            )
            for row in rows
            if isinstance(row, dict) and row.get("id")
        ]

    async def prices(self, location_id: str, day: date, resolution: str) -> PriceDay:
        """GET /api/v1/locations/{id}/prices for one day, PT1H or PT15M."""
        body = await self._get(
            f"/api/v1/locations/{location_id}/prices",
            {"date": day.isoformat(), "resolution": resolution},
        )
        return PriceDay(
            day=_day(body.get("date")) or day,
            resolution=_str(body.get("resolution")),
            complete=_bool(body.get("complete")),
            reason=_str(body.get("reason")),
            intervals=tuple(
                PriceInterval(
                    start=datetime.fromisoformat(row["start"]),
                    end=datetime.fromisoformat(row["end"]),
                    total_inc_vat=float(row["totalIncVat"]),
                    total_ex_vat=float(row["totalExVat"]),
                    spot=float(row["spot"]),
                )
                for row in body.get("intervals") or []
            ),
        )

    async def consumption(
        self, location_id: str, first: date, last: date, resolution: str = RESOLUTION_HOUR
    ) -> list[Reading]:
        """GET .../consumption from `first` to `last`, both included."""
        body = await self._get(
            f"/api/v1/locations/{location_id}/consumption",
            {"resolution": resolution, "from": first.isoformat(), "to": last.isoformat()},
        )
        return [
            Reading(
                start=datetime.fromisoformat(row["start"]),
                end=datetime.fromisoformat(row["end"]),
                kwh=_float(row.get("kwh")),
            )
            for row in body.get("values") or []
        ]

    async def standby(self, location_id: str) -> list[StandbyMonth]:
        """GET .../standby: the completed months, in the API's order."""
        rows = (await self._get(f"/api/v1/locations/{location_id}/standby")).get("months") or []
        return [
            StandbyMonth(month=str(row["month"]), standby_kw=_float(row.get("standbyKw"))) for row in rows
        ]

    async def rated_cost(self, location_id: str, month: str) -> RatedMonth:
        """GET .../costs/rated for one month, YYYY-MM."""
        body = await self._get(f"/api/v1/locations/{location_id}/costs/rated", {"month": month})
        return RatedMonth(
            month=str(body.get("month", month)),
            days=tuple(
                RatedDay(
                    day=date.fromisoformat(row["date"]),
                    kwh=_float(row.get("kwh")),
                    ex_vat=_float(row.get("exVat")),
                    inc_vat=_float(row.get("incVat")),
                    total_with_subscriptions=_amount(row.get("totalWithSubscriptions")),
                )
                for row in body.get("days") or []
            ),
            total_kwh=_float(body.get("totalKwh")),
            total=_amount(body.get("total")),
            total_with_subscriptions=_amount(body.get("totalWithSubscriptions")),
            complete=_bool(body.get("complete")),
            days_with_data=_int(body.get("daysWithData")),
            days_in_month=_int(body.get("daysInMonth")),
        )

    async def invoices(self, page: int = 1, per_page: int = 25) -> list[Invoice]:
        """GET /api/v1/invoices: one page, newest first."""
        body = await self._get("/api/v1/invoices", {"page": str(page), "perPage": str(per_page)})
        rows = body.get("invoices") or []
        return [_invoice(row) for row in rows if isinstance(row, dict) and row.get("invoiceNo")]

    async def invoice(self, invoice_no: str) -> Invoice:
        """GET /api/v1/invoices/{invoiceNo}, with its lines."""
        body = await self._get(f"/api/v1/invoices/{invoice_no}")
        lines = tuple(
            InvoiceLine(
                description=_str(row.get("description")),
                quantity=_float(row.get("quantity")),
                unit=_str(row.get("unit")),
                ex_vat=_float(row.get("exVat")),
                inc_vat=_float(row.get("incVat")),
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
