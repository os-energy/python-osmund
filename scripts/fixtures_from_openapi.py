"""Write the test fixtures from the examples in the public Osmund API document.

Every fixture under tests/fixtures comes from the published OpenAPI document,
never from an account: this repository is public, and so is its history.

    uv run python scripts/fixtures_from_openapi.py
"""

import json
import sys
import urllib.request
from pathlib import Path

OPENAPI_URL = "https://osmundenergi.dk/docs/osmund-api/openapi.json"
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# Route in the document -> fixture name for its 200 example.
ANSWERS = {
    "/api/v1/me": "me",
    "/api/v1/locations": "locations",
    "/api/v1/locations/{id}/prices": "prices",
    "/api/v1/locations/{id}/consumption": "consumption",
    "/api/v1/locations/{id}/standby": "standby",
    "/api/v1/locations/{id}/costs/rated": "costs_rated",
    "/api/v1/invoices": "invoices",
    "/api/v1/invoices/{invoiceNo}": "invoice",
}
# Every error status comes from the one route that documents all of them.
ERRORS_FROM = "/api/v1/locations/{id}/prices"


def example(response: dict) -> object:
    """Return the `default` example of one documented response."""
    return response["content"]["application/json"]["examples"]["default"]["value"]


def main() -> int:
    """Read the document and write one JSON file per example."""
    with urllib.request.urlopen(OPENAPI_URL, timeout=30) as answer:
        document = json.load(answer)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    written = {}
    for route, name in ANSWERS.items():
        written[name] = example(document["paths"][route]["get"]["responses"]["200"])
    for status, response in document["paths"][ERRORS_FROM]["get"]["responses"].items():
        if status != "200":
            written[f"error_{status}"] = example(response)

    for name, value in written.items():
        path = FIXTURES / f"{name}.json"
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(FIXTURES.parent.parent)}")
    print(f"from {OPENAPI_URL}, version {document['info'].get('version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
