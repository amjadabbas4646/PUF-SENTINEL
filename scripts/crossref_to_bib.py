"""Build BibTeX entries from Crossref records, so that no field of a new
reference is typed from memory.

Usage:  python scripts/crossref_to_bib.py key1=doi1 key2=doi2 ... > new_entries.bib

The registry record is also stored in literature/crossref_r2.json as evidence.
Container titles are written as registered. Check capitalisation of the title
by eye before use.
"""
from __future__ import annotations

import html
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "puf-sentinel/2.0 (reference verification)"}
LATEX = {"ö": '{\\"o}', "ü": '{\\"u}', "ä": '{\\"a}', "é": "{\\'e}", "è": "{\\`e}", "ć": "{\\'c}",
         "č": "{\\v{c}}", "ğ": "{\\u{g}}", "ı": "{\\i}", "ñ": "{\\~n}", "ó": "{\\'o}", "á": "{\\'a}",
         "í": "{\\'i}", "ú": "{\\'u}", "ø": "{\\o}", "å": "{\\aa}", "ş": "{\\c{s}}", "ç": "{\\c{c}}",
         "Ö": '{\\"O}', "Ü": '{\\"U}', "–": "--", "—": "--", "’": "'", "&": "\\&"}


def tex(s: str) -> str:
    s = html.unescape(s or "")
    return "".join(LATEX.get(ch, ch) for ch in s)


def main() -> None:
    store = ROOT / "literature" / "crossref_r2.json"
    evidence = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
    out = []
    for arg in sys.argv[1:]:
        key, doi = arg.split("=", 1)
        req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi),
                                     headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            it = json.loads(r.read().decode("utf-8"))["message"]
        evidence[key] = {k: it.get(k) for k in ("DOI", "title", "author", "container-title",
                                                "volume", "issue", "page", "article-number",
                                                "issued", "type", "publisher")}
        authors = " and ".join(f"{tex(a.get('family', ''))}, {tex(a.get('given', ''))}"
                               for a in it.get("author", []))
        year = it["issued"]["date-parts"][0][0]
        cont = tex((it.get("container-title") or [""])[0])
        kind = {"journal-article": "article", "proceedings-article": "inproceedings",
                "book-chapter": "incollection"}.get(it.get("type"), "misc")
        fields = [("author", authors), ("title", "{" + tex(it["title"][0]) + "}")]
        if kind == "article":
            fields += [("journal", cont), ("volume", it.get("volume")), ("number", it.get("issue")),
                       ("pages", (it.get("page") or it.get("article-number") or "").replace("-", "--"))]
        elif kind in ("inproceedings", "incollection"):
            fields += [("booktitle", cont), ("pages", (it.get("page") or "").replace("-", "--")),
                       ("publisher", tex(it.get("publisher", "")))]
        fields += [("year", str(year)), ("doi", it["DOI"])]
        body = ",\n".join(f"  {k:<9} = {{{v}}}" for k, v in fields if v)
        out.append(f"@{kind}{{{key},\n{body}\n}}\n")
    store.write_text(json.dumps(evidence, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
