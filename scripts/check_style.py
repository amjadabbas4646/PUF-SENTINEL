"""Residue scan for the house writing rules, applied to the manuscript and to
the Markdown reports.

Rules checked in prose
    no en dash, no em dash, no double hyphen used as a dash
    no slash between words (write "A and B", "A or B", "A per B")
    no comma before "and"
Code spans, code blocks, URLs, file paths, LaTeX commands that carry file
names or keys, math and the bibliography are not prose and are skipped.

Usage:  python scripts/check_style.py [files...]      exit code 1 on findings
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["CEE_SUBMISSION/manuscript.tex", "CEE_SUBMISSION/supplement.tex",
           "CEE_SUBMISSION/highlights.tex", "CEE_SUBMISSION/author_info.tex",
           "CEE_SUBMISSION/RESPONSE_TO_COMMENTS.md", "README.md", "CHANGELOG.md", "REVISION_AUDIT.md",
           "NOVELTY_AND_VENUE_ASSESSMENT.md", "DATA_PROVENANCE.md", "EXPERIMENT_PLAN.md",
           "REFERENCE_VERIFICATION_LOG.md", "FINAL_TECHNICAL_REPORT.md", "data/README.md"]


def strip_md(text: str) -> str:
    text = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "CODE", text)
    text = re.sub(r"https?://\S+", "URL", text)
    text = re.sub(r"\b10\.\d{4,9}/\S+", "DOI", text)
    text = re.sub(r'"[^"\n]*"', "QUOTE", text)       # verbatim quotations
    return text


def strip_tex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\(url|input|includegraphics|label|ref|cite|bibliography|bibliographystyle"
                  r"|usepackage|documentclass|graphicspath|texttt)(\[[^\]]*\])?\{[^}]*\}", "CMD", text)
    text = re.sub(r"\$[^$]*\$", "MATH", text)
    text = re.sub(r"\\begin\{(equation|align|algorithmic|tikzpicture)\*?\}.*?\\end\{\1\*?\}",
                  lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return text


def scan(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    text = strip_tex(raw) if path.suffix == ".tex" else strip_md(raw)
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if "\u2014" in line or "\u2013" in line:
            out.append(f"{path.name}:{i}: en or em dash: {line.strip()[:90]}")
        if re.search(r"(?<![-!<|:])--(?![->|:])", line) and path.suffix == ".tex":
            out.append(f"{path.name}:{i}: double hyphen: {line.strip()[:90]}")
        if re.search(r",\s+and\b", line):
            out.append(f"{path.name}:{i}: comma before and: {line.strip()[:90]}")
        if re.search(r"[A-Za-z]\s?/\s?[A-Za-z]", line):
            out.append(f"{path.name}:{i}: slash in prose: {line.strip()[:90]}")
    return out


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]] or [ROOT / f for f in DEFAULT]
    findings = []
    for f in files:
        if f.exists():
            findings += scan(f)
        else:
            print(f"(missing, skipped) {f}")
    for line in findings:
        print(line)
    print(f"{len(findings)} finding(s) in {len(files)} file(s)")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
