"""Look up candidate references in Crossref and print the registry record.

Usage:  python scripts/crossref_lookup.py "free text query" ["another query" ...]
        python scripts/crossref_lookup.py --doi 10.1007/978-3-662-48324-4_26 ...
        python scripts/crossref_lookup.py --from-year 2024 "query"

Nothing is written to the bibliography by this script. It only prints what the
registry holds, so that an entry is never typed from memory.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

UA = {"User-Agent": "puf-sentinel/2.0 (reference verification)"}


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def show(it: dict) -> None:
    authors = ", ".join(f"{a.get('family', '')} {a.get('given', '')[:1]}".strip()
                        for a in it.get("author", [])[:8])
    year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    print(" *", (it.get("title") or ["?"])[0])
    print("   ", authors, "|", year, "|", (it.get("container-title") or [""])[0],
          "| vol", it.get("volume"), "| issue", it.get("issue"), "| pp", it.get("page"),
          "| art", it.get("article-number"), "|", it.get("type"))
    print("    doi:", it.get("DOI"))


def main() -> None:
    args = sys.argv[1:]
    mode, year = "query", None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--doi":
            mode = "doi"
        elif a == "--from-year":
            year = args[i + 1]
            i += 1
        elif mode == "doi":
            try:
                show(get("https://api.crossref.org/works/" + urllib.parse.quote(a))["message"])
            except Exception as e:                      # noqa: BLE001
                print(" ! DOI not resolved:", a, e)
        else:
            url = ("https://api.crossref.org/works?rows=4&query.bibliographic="
                   + urllib.parse.quote(a))
            if year:
                url += f"&filter=from-pub-date:{year}"
            print("QUERY:", a)
            try:
                for it in get(url)["message"]["items"]:
                    show(it)
            except Exception as e:                      # noqa: BLE001
                print(" ! query failed:", e)
        i += 1


if __name__ == "__main__":
    main()
