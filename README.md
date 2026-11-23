# osmund

An async Python client for the [Osmund API](https://osmundenergi.dk/docs/osmund-api/): electricity prices, consumption, cost and invoices for Osmund's customers. It only reads.

```python
import aiohttp
from datetime import date

from osmund import OsmundClient


async def main() -> None:
    async with aiohttp.ClientSession() as session:
        client = OsmundClient(session, "osm_...")
        for location in await client.locations():
            if location.api_access == "open":
                prices = await client.prices(location.id, date.today(), "PT1H")
                print(location.label, [i.total_inc_vat for i in prices.intervals])
```

Create a personal key on [My page](https://osmundenergi.dk/min-side#api). The calls count toward your key's [limits](https://osmundenergi.dk/docs/osmund-api/#tag/limits).

The answers keep the API's field names, in snake_case. A value the API leaves out is `None`.

Each error the API documents has its own exception, all subclasses of `OsmundError`. A 429 or 503 raises `OsmundUnavailableError` with `retry_after` in seconds.
