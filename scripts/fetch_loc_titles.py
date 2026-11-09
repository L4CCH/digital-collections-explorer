"""
Fetch the title and date of every Library of Congress item that `input/merged_files.csv` lists,
for `create_loc_assets.py` to show on each map's page. Reads the loc.gov JSON API, slowly enough
to stay within its published limits, and writes `input/loc_titles.json`.
"""

import http.client
import json
import os
import time
import urllib.request

import pandas as pd

CSV_PATH = "input/merged_files.csv"
TITLES_PATH = "input/loc_titles.json"
PAGES_DIR = "input/loc_titles_pages"
ITEMS_PATH = "input/loc_titles_items.json"
SEARCH_URL = "https://www.loc.gov/maps/?fo=json&c=1000&at=results,pagination&sp={page}"
ITEM_URL = "{item}?fo=json&at=item"
USER_AGENT = "digital-collections-explorer (https://github.com/L4CCH/digital-collections-explorer)"
# loc.gov allows 80 search requests a minute and 200 item requests a minute.
SEARCH_PAUSE_SECONDS = 3
ITEM_PAUSE_SECONDS = 0.4


def get_json(url: str, attempts: int = 5) -> dict:
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            wait = 60 * (attempt + 1)
            print(f"{url}: {error}; trying again in {wait} seconds")
            time.sleep(wait)
    raise RuntimeError(f"Could not read {url}")


def key_of(url: str) -> str:
    return url.replace("https://", "http://", 1).rstrip("/") + "/"


def described(record: dict) -> dict:
    title = record.get("title")
    date = record.get("date") or next(iter(record.get("dates") or []), None)
    return {
        name: value
        for name, value in (("title", title), ("date", date))
        if isinstance(value, str) and value
    }


def fetch_titles():
    wanted = {
        key_of(url)
        for url in pd.read_csv(CSV_PATH, usecols=["p1_item_id"]).p1_item_id.dropna()
    }
    print(f"{len(wanted)} items to describe")

    os.makedirs(PAGES_DIR, exist_ok=True)
    found = {}
    page, pages = 1, 1
    while page <= pages:
        saved = os.path.join(PAGES_DIR, f"{page}.json")
        if os.path.exists(saved):
            with open(saved) as f:
                kept = json.load(f)
        else:
            data = get_json(SEARCH_URL.format(page=page))
            kept = {
                "pages": data["pagination"]["total"],
                "found": {
                    key_of(record.get("id") or ""): described(record)
                    for record in data["results"]
                },
            }
            with open(saved, "w") as f:
                json.dump(kept, f)
            time.sleep(SEARCH_PAUSE_SECONDS)
        pages = kept["pages"]
        found.update({k: v for k, v in kept["found"].items() if k in wanted})
        print(f"page {page} of {pages}: {len(found)} of {len(wanted)} items found")
        page += 1

    missing = sorted(wanted - found.keys())
    print(
        f"{len(missing)} items are not in the maps search; reading them one at a time"
    )
    items = {}
    if os.path.exists(ITEMS_PATH):
        with open(ITEMS_PATH) as f:
            items = json.load(f)
    for number, key in enumerate(missing, 1):
        if key not in items:
            address = ITEM_URL.format(item=key.replace("http://", "https://", 1))
            items[key] = described(get_json(address).get("item") or {})
            time.sleep(ITEM_PAUSE_SECONDS)
        if number % 100 == 0 or number == len(missing):
            with open(ITEMS_PATH, "w") as f:
                json.dump(items, f)
            print(f"{number} of {len(missing)}")
    found.update(items)

    with open(TITLES_PATH, "w") as f:
        json.dump(found, f, indent=1, ensure_ascii=False)
    titled = sum(1 for value in found.values() if value.get("title"))
    print(f"Saved {TITLES_PATH}: {titled} of {len(wanted)} items have a title")


if __name__ == "__main__":
    fetch_titles()
