"""The client: what each documented answer becomes."""

from __future__ import annotations

from datetime import date, datetime

import aiohttp
import pytest

from osmund import (
    OsmundAccessError,
    OsmundApiError,
    OsmundAuthError,
    OsmundClient,
    OsmundConnectionError,
    OsmundNotFoundError,
    OsmundScopeError,
    OsmundUnavailableError,
)

from .conftest import (
    CLOSED_ID,
    CPH,
    INVOICES,
    KEY,
    LOCATIONS,
    ME,
    OPEN_ID,
    FakeApi,
    fixture,
    price_day,
)


async def test_the_key_is_sent_as_a_bearer_header(api: FakeApi, client: OsmundClient) -> None:
    api.get(ME, payload=fixture("me"))

    key = await client.me()

    assert key.prefix == "osm_4f9Kx2Qa"
    assert "prices:read" in key.scopes
    assert key.expires_at == datetime(2027, 11, 16, 9, 12, tzinfo=CPH)
    [request] = api.requests
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["User-Agent"].startswith("python-osmund/")


async def test_locations_keep_both_and_say_which_has_access(api: FakeApi, client: OsmundClient) -> None:
    api.get(LOCATIONS, payload=fixture("locations"))

    locations = await client.locations()

    assert [(loc.id, loc.has_access) for loc in locations] == [(OPEN_ID, True), (CLOSED_ID, False)]
    assert locations[0].label == "Hjem"
    assert locations[0].meter_resolution == "PT1H"


async def test_fields_and_values_it_does_not_know_are_ignored(api: FakeApi, client: OsmundClient) -> None:
    """The API is in beta and adds fields and values: none of them may break a caller."""
    answer = fixture("locations")
    answer["locations"][0]["somethingNew"] = {"nested": True}
    answer["locations"][0]["state"] = "a_state_from_the_future"
    answer["locations"][0]["meterResolution"] = "PT5M"
    answer["nextPage"] = None
    api.get(LOCATIONS, payload=answer)

    locations = await client.locations()

    assert locations[0].state == "a_state_from_the_future"
    assert locations[0].meter_resolution == "PT5M"


async def test_prices_are_parsed_with_their_offset(api: FakeApi, client: OsmundClient) -> None:
    day = date(2026, 11, 17)
    api.get(f"{LOCATIONS}/{OPEN_ID}/prices", payload=price_day(day))

    prices = await client.prices(OPEN_ID, day, "PT1H")

    assert prices.day == day
    assert prices.complete
    assert len(prices.intervals) == 24
    ten = prices.at(datetime(2026, 11, 17, 10, 30, tzinfo=CPH))
    assert ten is not None
    assert ten.total_inc_vat == pytest.approx(1.10)
    [request] = api.requests
    assert request.path_qs.endswith("/prices?date=2026-11-17&resolution=PT1H")


async def test_a_day_with_a_clock_change_has_its_25_hours(api: FakeApi, client: OsmundClient) -> None:
    day = date(2026, 10, 25)
    api.get(f"{LOCATIONS}/{OPEN_ID}/prices", payload=price_day(day))

    prices = await client.prices(OPEN_ID, day, "PT1H")

    assert len(prices.intervals) == 25


async def test_consumption_by_the_hour_from_and_to(api: FakeApi, client: OsmundClient) -> None:
    answer = fixture("consumption")
    api.get(f"{LOCATIONS}/{OPEN_ID}/consumption", payload=answer)

    readings = await client.consumption(OPEN_ID, date(2026, 11, 1), date(2026, 11, 3))

    assert [r.kwh for r in readings] == [v["kwh"] for v in answer["values"]]
    assert readings[0].start == datetime.fromisoformat(answer["values"][0]["start"])
    [request] = api.requests
    assert request.path_qs.endswith("/consumption?resolution=PT1H&from=2026-11-01&to=2026-11-03")


async def test_standby_months_oldest_first(api: FakeApi, client: OsmundClient) -> None:
    answer = fixture("standby")
    answer["months"].reverse()
    api.get(f"{LOCATIONS}/{OPEN_ID}/standby", payload=answer)

    months = await client.standby(OPEN_ID)

    assert [m.month for m in months] == sorted(m["month"] for m in answer["months"])


async def test_a_rated_month_with_each_day(api: FakeApi, client: OsmundClient) -> None:
    answer = fixture("costs_rated")
    api.get(f"{LOCATIONS}/{OPEN_ID}/costs/rated", payload=answer)

    rated = await client.rated_cost(OPEN_ID, "2026-11")

    assert rated.month == "2026-11"
    assert not rated.complete
    assert rated.days_with_data == answer["daysWithData"]
    assert rated.total_with_subscriptions.inc_vat == answer["totalWithSubscriptions"]["incVat"]
    assert [d.per_kwh.inc_vat for d in rated.days] == [d["incVat"] for d in answer["days"]]


async def test_invoices_and_one_invoice_with_its_lines(api: FakeApi, client: OsmundClient) -> None:
    api.get(INVOICES, payload=fixture("invoices"))
    detail = fixture("invoice")
    api.get(f"{INVOICES}/{detail['invoiceNo']}", payload=detail)

    invoices = await client.invoices()
    invoice = await client.invoice(invoices[0].number)

    assert invoices[0].number == detail["invoiceNo"]
    assert invoice.total.inc_vat == detail["total"]["incVat"]
    assert [line.description for line in invoice.lines] == [line["description"] for line in detail["lines"]]


@pytest.mark.parametrize(
    ("status", "name", "error"),
    [
        (400, "error_400", OsmundApiError),
        (401, "error_401", OsmundAuthError),
        (403, "error_403", OsmundAccessError),
        (404, "error_404", OsmundNotFoundError),
        (500, "error_500", OsmundApiError),
    ],
)
async def test_each_documented_error_has_its_own_exception(
    api: FakeApi, client: OsmundClient, status: int, name: str, error: type[Exception]
) -> None:
    api.get(ME, status=status, payload=fixture(name))

    with pytest.raises(error):
        await client.me()


async def test_an_error_keeps_its_code_and_request_id(api: FakeApi, client: OsmundClient) -> None:
    answer = fixture("error_500")
    api.get(ME, status=500, payload=answer)

    with pytest.raises(OsmundApiError) as raised:
        await client.me()

    assert raised.value.code == answer["code"]
    assert raised.value.request_id == answer["requestId"]


async def test_a_missing_scope_is_not_a_missing_access(api: FakeApi, client: OsmundClient) -> None:
    api.get(ME, status=403, payload=fixture("error_403") | {"code": "scope_required"})

    with pytest.raises(OsmundScopeError):
        await client.me()


async def test_429_waits_as_long_as_the_header_says(api: FakeApi, client: OsmundClient) -> None:
    api.get(ME, status=429, payload=fixture("error_429"), headers={"Retry-After": "17"})

    with pytest.raises(OsmundUnavailableError) as raised:
        await client.me()

    assert raised.value.retry_after == 17


async def test_without_a_header_the_body_says_how_long(api: FakeApi, client: OsmundClient) -> None:
    api.get(ME, status=429, payload=fixture("error_429"))

    with pytest.raises(OsmundUnavailableError) as raised:
        await client.me()

    assert raised.value.retry_after == 42


async def test_503_is_a_wait_too(api: FakeApi, client: OsmundClient) -> None:
    api.get(ME, status=503, payload=fixture("error_503"))

    with pytest.raises(OsmundUnavailableError):
        await client.me()


async def test_an_answer_that_is_not_the_apis_json_is_a_connection_error(
    api: FakeApi, client: OsmundClient
) -> None:
    """A firewall or proxy in front of the API answers HTML, not the API's error."""
    api.get(ME, status=403, text="<html>Forbidden</html>", content_type="text/html")

    with pytest.raises(OsmundConnectionError):
        await client.me()


async def test_the_network_failing_is_a_connection_error() -> None:
    async with aiohttp.ClientSession() as session:
        # Nothing listens on port 1.
        unreachable = OsmundClient(session, KEY, base_url="http://127.0.0.1:1")

        with pytest.raises(OsmundConnectionError):
            await unreachable.me()
