#!/usr/bin/env python
"""
Extract narrow SEC credit-note patterns from document text.

This is a high-precision scaffold for three note families:
  1. Revolver / credit facility availability
  2. Lease cost / liabilities / lease maturity lines
  3. Debt maturity schedules

Inputs:
  - data/inputs_layer/doc_text_map/year=YYYY/part.parquet
  - optional data/inputs_layer/raw_documents/year=YYYY/*.parquet for metadata

Outputs:
  - data/sec/note_extracts/revolver_note_extracts.parquet
  - data/sec/note_extracts/lease_note_extracts.parquet
  - data/sec/note_extracts/debt_maturity_note_extracts.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REVOLVER_OUT_PATH = ROOT / "data" / "sec" / "note_extracts" / "revolver_note_extracts.parquet"
LEASE_OUT_PATH = ROOT / "data" / "sec" / "note_extracts" / "lease_note_extracts.parquet"
MATURITY_OUT_PATH = ROOT / "data" / "sec" / "note_extracts" / "debt_maturity_note_extracts.parquet"

MONEY_RE = re.compile(
    r"(?P<prefix>\$)?\s*(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<unit>billion|million|thousand|bn|mm|mn|m|b|k)?",
    re.IGNORECASE,
)
YEAR_AMOUNT_RE = re.compile(
    r"^\s*(?P<label>(?:20\d{2}|thereafter))\b[^\n$]{0,40}(?P<amount>\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mm|mn|m|b|k))?)",
    re.IGNORECASE,
)

REVOLVER_KEYWORDS = re.compile(
    r"revolving credit|credit facility|line of credit|senior credit facility|abl facility|asset[- ]based lending",
    re.IGNORECASE,
)
LEASE_KEYWORDS = re.compile(
    r"\bleases?\b|lease cost|lease liabilit|future lease payments|maturity analysis of lease liabilities",
    re.IGNORECASE,
)
MATURITY_KEYWORDS = re.compile(
    r"debt maturit|long-term debt maturit|principal maturit|contractual maturit|scheduled maturit|debt due",
    re.IGNORECASE,
)
STRICT_MONEY_CAPTURE = r"(?P<money>(?:\$\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million|thousand|bn|mm|mn|m|b|k))?|\d[\d,]*(?:\.\d+)?\s*(?:billion|million|thousand|bn|mm|mn|m|b|k)))"


def _quoted_paths(paths: Sequence[Path]) -> str:
    return "[" + ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in paths) + "]"


def _parquet_columns(paths: Sequence[Path]) -> set[str]:
    if not paths:
        return set()
    con = duckdb.connect()
    try:
        first_path = paths[0].as_posix().replace("'", "''")
        df = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{first_path}')"
        ).df()
        return set(df["column_name"].astype(str))
    except Exception:
        return set()


def _context_multiplier(text: str) -> float:
    lower = (text or "").lower()
    if "in billions" in lower or "(billions)" in lower:
        return 1_000_000_000.0
    if "in millions" in lower or "(millions)" in lower:
        return 1_000_000.0
    if "in thousands" in lower or "(thousands)" in lower:
        return 1_000.0
    return 1.0


def _money_mentions(text: str) -> List[Dict[str, object]]:
    mentions: List[Dict[str, object]] = []
    if not text:
        return mentions
    default_multiplier = _context_multiplier(text)
    for match in MONEY_RE.finditer(text):
        raw = match.group(0).strip()
        number_raw = (match.group("number") or "").replace(",", "")
        if not number_raw:
            continue
        try:
            number = float(number_raw)
        except ValueError:
            continue
        prefix = match.group("prefix")
        unit = (match.group("unit") or "").lower()
        # Skip plain years and similar false positives unless they look like money.
        if not prefix and not unit and "." not in number_raw and 1900 <= number <= 2100:
            continue
        # Skip tiny bare numbers when there is no currency/unit context. This
        # filters date fragments like "31" in "December 31, 2024" while still
        # allowing plain table values when the block says "(in millions)".
        if not prefix and not unit and "," not in number_raw and default_multiplier == 1.0 and number < 1000:
            continue
        if unit in {"billion", "bn", "b"}:
            multiplier = 1_000_000_000.0
        elif unit in {"million", "mm", "mn", "m"}:
            multiplier = 1_000_000.0
        elif unit in {"thousand", "k"}:
            multiplier = 1_000.0
        else:
            multiplier = default_multiplier
        mentions.append({"raw": raw, "value": number * multiplier, "start": match.start(), "end": match.end()})
    return mentions


def _first_money_value(text: str) -> Optional[float]:
    mentions = _money_mentions(text)
    if not mentions:
        return None
    return float(mentions[0]["value"])


def _match_money_value(match: re.Match[str]) -> Optional[float]:
    money_text = match.groupdict().get("money") or match.group(0)
    return _first_money_value(money_text)


def _candidate_blocks(text: str, keyword_re: re.Pattern, radius: int = 2) -> List[str]:
    lines = [line.strip() for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []
    blocks: List[str] = []
    seen: set[str] = set()
    for idx, line in enumerate(lines):
        if not keyword_re.search(line):
            continue
        lo = max(0, idx - radius)
        hi = min(len(lines), idx + radius + 1)
        block = "\n".join(lines[lo:hi]).strip()
        if block and block not in seen:
            seen.add(block)
            blocks.append(block)
    if blocks:
        return blocks
    # Fallback to a few sentence windows if the document is one long blob.
    sentences = re.split(r"(?<=[.!?])\s+", text or "")
    for idx, sentence in enumerate(sentences):
        if not keyword_re.search(sentence):
            continue
        lo = max(0, idx - 1)
        hi = min(len(sentences), idx + 2)
        block = " ".join(sentences[lo:hi]).strip()
        if block and block not in seen:
            seen.add(block)
            blocks.append(block)
    return blocks


def _is_likely_sec_filing(doc: Dict[str, object]) -> bool:
    hay = " ".join(
        str(doc.get(key) or "")
        for key in ("source_type", "doc_type", "title", "url", "document_id")
    ).lower()
    form_hits = any(token in hay for token in ("10-k", "10q", "10-q", "10k", "annual report", "quarterly report"))
    sec_hits = any(token in hay for token in ("sec", "edgar", "/archives/"))
    return form_hits or sec_hits


def _base_row(doc: Dict[str, object], family: str, metric_key: str, value: float, evidence_text: str, pattern_name: str, confidence: float, bucket_label: Optional[str] = None) -> Dict[str, object]:
    return {
        "document_id": doc.get("document_id"),
        "entity_id": doc.get("entity_id"),
        "source_type": doc.get("source_type"),
        "doc_type": doc.get("doc_type"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "published_at": doc.get("published_at"),
        "effective_at": doc.get("effective_at"),
        "ingested_at": doc.get("ingested_at"),
        "pattern_family": family,
        "metric_key": metric_key,
        "value": float(value),
        "currency": "USD",
        "bucket_label": bucket_label,
        "pattern_name": pattern_name,
        "extraction_confidence": float(confidence),
        "evidence_text": evidence_text[:4000],
        "extraction_method": "regex_note_pattern",
    }


def extract_revolver_note_rows(doc: Dict[str, object]) -> List[Dict[str, object]]:
    text = str(doc.get("raw_text") or "")
    rows: List[Dict[str, object]] = []
    capacity_candidates: List[float] = []
    outstanding_candidates: List[float] = []
    blocks = _candidate_blocks(text, REVOLVER_KEYWORDS, radius=3)
    if not blocks:
        return rows

    patterns = [
        (
            "financial.revolver_undrawn",
            "undrawn_direct",
            [
                re.compile(rf"(?:undrawn|unused commitments?|availability|available borrowings?)[^\n.;$]{{0,80}}{STRICT_MONEY_CAPTURE}", re.IGNORECASE),
                re.compile(rf"{STRICT_MONEY_CAPTURE}[^\n.;]{{0,80}}(?:available under|available on|undrawn under|unused under).{{0,40}}(?:revolving credit|credit facility|line of credit)", re.IGNORECASE),
            ],
            0.92,
        ),
        (
            "financial.revolver_capacity",
            "capacity",
            [
                re.compile(rf"(?:aggregate commitments? of|commitments? of|revolving credit facility (?:of|with|provides)|line of credit (?:of|with)|credit facility (?:of|with))[^\n.;$]{{0,80}}{STRICT_MONEY_CAPTURE}", re.IGNORECASE),
            ],
            0.76,
        ),
        (
            "financial.revolver_outstanding",
            "outstanding",
            [
                re.compile(rf"{STRICT_MONEY_CAPTURE}[^\n.;]{{0,80}}(?:was|were)?[^\n.;]{{0,40}}(?:outstanding|drawn under|borrowings outstanding|amount outstanding)", re.IGNORECASE),
                re.compile(rf"(?:outstanding|drawn under|borrowings outstanding|amount outstanding)[^\n.;$]{{0,80}}{STRICT_MONEY_CAPTURE}", re.IGNORECASE),
            ],
            0.82,
        ),
    ]

    for block in blocks:
        for metric_key, pattern_name, regexes, confidence in patterns:
            for regex in regexes:
                for match in regex.finditer(block):
                    value = _match_money_value(match)
                    if value is None:
                        continue
                    rows.append(_base_row(doc, "revolver", metric_key, value, block, pattern_name, confidence))
                    if metric_key == "financial.revolver_capacity":
                        capacity_candidates.append(value)
                    elif metric_key == "financial.revolver_outstanding":
                        outstanding_candidates.append(value)

    if not any(row["metric_key"] == "financial.revolver_undrawn" for row in rows):
        if capacity_candidates and outstanding_candidates:
            derived = max(capacity_candidates) - min(outstanding_candidates)
            if derived >= 0:
                evidence = f"Derived from capacity={max(capacity_candidates):,.0f} and outstanding={min(outstanding_candidates):,.0f}"
                rows.append(_base_row(doc, "revolver", "financial.revolver_undrawn", derived, evidence, "capacity_minus_outstanding", 0.68))
    return rows


def extract_lease_note_rows(doc: Dict[str, object]) -> List[Dict[str, object]]:
    text = str(doc.get("raw_text") or "")
    rows: List[Dict[str, object]] = []
    blocks = _candidate_blocks(text, LEASE_KEYWORDS, radius=4)
    if not blocks:
        return rows

    patterns = [
        (
            "financial.lease_expense_operating",
            "operating_lease_cost",
            [
                re.compile(rf"operating lease cost[^\n.;$]{{0,60}}{STRICT_MONEY_CAPTURE}", re.IGNORECASE),
                re.compile(rf"{STRICT_MONEY_CAPTURE}[^\n.;]{{0,60}}operating lease cost", re.IGNORECASE),
            ],
            0.90,
        ),
        (
            "financial.lease_expense_finance",
            "finance_lease_cost",
            [
                re.compile(rf"finance lease cost[^\n.;$]{{0,60}}{STRICT_MONEY_CAPTURE}", re.IGNORECASE),
                re.compile(rf"{STRICT_MONEY_CAPTURE}[^\n.;]{{0,60}}finance lease cost", re.IGNORECASE),
            ],
            0.90,
        ),
        (
            "financial.lease_liability_current",
            "lease_liability_current",
            [
                re.compile(rf"(?:current portion of )?lease liabilit(?:y|ies)[^\n.;$]{{0,80}}(?:current|short[- ]term)[^\n.;$]{{0,40}}{STRICT_MONEY_CAPTURE}", re.IGNORECASE),
                re.compile(rf"{STRICT_MONEY_CAPTURE}[^\n.;]{{0,80}}(?:current|short[- ]term)[^\n.;]{{0,40}}lease liabilit(?:y|ies)", re.IGNORECASE),
            ],
            0.86,
        ),
        (
            "financial.lease_liability_noncurrent",
            "lease_liability_noncurrent",
            [
                re.compile(rf"lease liabilit(?:y|ies)[^\n.;$]{{0,80}}(?:noncurrent|long[- ]term)[^\n.;$]{{0,40}}{STRICT_MONEY_CAPTURE}", re.IGNORECASE),
                re.compile(rf"{STRICT_MONEY_CAPTURE}[^\n.;]{{0,80}}(?:noncurrent|long[- ]term)[^\n.;]{{0,40}}lease liabilit(?:y|ies)", re.IGNORECASE),
            ],
            0.86,
        ),
    ]
    schedule_heading = re.compile(r"future lease payments|maturity analysis of lease liabilities", re.IGNORECASE)

    for block in blocks:
        for metric_key, pattern_name, regexes, confidence in patterns:
            for regex in regexes:
                for match in regex.finditer(block):
                    value = _match_money_value(match)
                    if value is None:
                        continue
                    rows.append(_base_row(doc, "lease", metric_key, value, block, pattern_name, confidence))

        if schedule_heading.search(block):
            for line in block.splitlines():
                year_match = YEAR_AMOUNT_RE.search(line)
                if not year_match:
                    continue
                label = year_match.group("label")
                value = _first_money_value(year_match.group("amount"))
                if value is None:
                    continue
                rows.append(
                    _base_row(
                        doc,
                        "lease",
                        "financial.lease_payment_due",
                        value,
                        block,
                        "lease_maturity_schedule",
                        0.84,
                        bucket_label=label,
                    )
                )
    return rows


def extract_debt_maturity_rows(doc: Dict[str, object]) -> List[Dict[str, object]]:
    text = str(doc.get("raw_text") or "")
    rows: List[Dict[str, object]] = []
    blocks = _candidate_blocks(text, MATURITY_KEYWORDS, radius=6)
    if not blocks:
        return rows
    for block in blocks:
        for line in block.splitlines():
            match = YEAR_AMOUNT_RE.search(line)
            if not match:
                continue
            label = match.group("label")
            value = _first_money_value(match.group("amount"))
            if value is None:
                continue
            rows.append(
                _base_row(
                    doc,
                    "debt_maturity",
                    "financial.debt_maturity_bucket",
                    value,
                    block,
                    "debt_maturity_schedule",
                    0.86,
                    bucket_label=label,
                )
            )
    return rows


def extract_note_pattern_rows(doc: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    if not _is_likely_sec_filing(doc):
        return {"revolver": [], "lease": [], "debt_maturity": []}
    return {
        "revolver": extract_revolver_note_rows(doc),
        "lease": extract_lease_note_rows(doc),
        "debt_maturity": extract_debt_maturity_rows(doc),
    }


def load_document_texts(
    *,
    years: Sequence[int],
    doc_text_map_root: Path,
    raw_documents_root: Optional[Path],
    company_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    text_paths = [doc_text_map_root / f"year={year}" / "part.parquet" for year in years if (doc_text_map_root / f"year={year}" / "part.parquet").exists()]
    if not text_paths:
        raise FileNotFoundError(f"No doc_text_map parquet files found under {doc_text_map_root} for years={list(years)}")

    con = duckdb.connect()
    text_expr = _quoted_paths(text_paths)
    text_limit_clause = f"LIMIT {int(limit)}" if limit and int(limit) > 0 and not company_ids else ""

    metadata_join = ""
    metadata_select = ", NULL AS entity_id, NULL AS source_type, NULL AS doc_type, NULL AS title, NULL AS url, NULL AS published_at, NULL AS effective_at, NULL AS ingested_at"
    raw_cols: set[str] = set()
    if raw_documents_root and raw_documents_root.exists():
        raw_paths = sorted((raw_documents_root / f"year={year}").glob("*.parquet") for year in years)
        raw_paths = [path for group in raw_paths for path in group]
        if raw_paths:
            raw_cols = _parquet_columns(raw_paths)
            raw_expr = _quoted_paths(raw_paths)
            entity_expr = "CAST(entity_id AS VARCHAR)" if "entity_id" in raw_cols else "NULL"
            source_type_expr = "CAST(source_type AS VARCHAR)" if "source_type" in raw_cols else "NULL"
            doc_type_expr = "CAST(doc_type AS VARCHAR)" if "doc_type" in raw_cols else "NULL"
            title_expr = "CAST(title AS VARCHAR)" if "title" in raw_cols else "NULL"
            url_expr = "CAST(url AS VARCHAR)" if "url" in raw_cols else "NULL"
            published_expr = "CAST(published_at AS TIMESTAMP)" if "published_at" in raw_cols else "NULL"
            effective_expr = "CAST(effective_at AS TIMESTAMP)" if "effective_at" in raw_cols else "NULL"
            ingested_expr = "CAST(ingested_at AS TIMESTAMP)" if "ingested_at" in raw_cols else "NULL"
            metadata_join = f"""
            LEFT JOIN (
                SELECT
                    document_id,
                    {entity_expr} AS entity_id,
                    {source_type_expr} AS source_type,
                    {doc_type_expr} AS doc_type,
                    {title_expr} AS title,
                    {url_expr} AS url,
                    {published_expr} AS published_at,
                    {effective_expr} AS effective_at,
                    {ingested_expr} AS ingested_at
                FROM read_parquet({raw_expr}, union_by_name=True)
                WHERE document_id IN (SELECT document_id FROM text_docs)
            ) meta USING (document_id)
            """
            metadata_select = ", meta.entity_id, meta.source_type, meta.doc_type, meta.title, meta.url, meta.published_at, meta.effective_at, meta.ingested_at"
            if company_ids and "entity_id" not in raw_cols:
                print(
                    f"[warn] metadata root {raw_documents_root} does not expose entity_id; company filter will be skipped",
                    flush=True,
                )

    filters = ["text_docs.document_id IS NOT NULL", "text_docs.raw_text IS NOT NULL", "length(text_docs.raw_text) > 0"]
    if company_ids and raw_documents_root and raw_documents_root.exists() and "entity_id" in raw_cols:
        quoted = ", ".join("'" + cid.replace("'", "''") + "'" for cid in company_ids)
        filters.append(f"meta.entity_id IN ({quoted})")

    query = f"""
    WITH text_docs AS (
        SELECT document_id, raw_text
        FROM read_parquet({text_expr}, union_by_name=True)
        WHERE document_id IS NOT NULL
          AND raw_text IS NOT NULL
          AND length(raw_text) > 0
        {text_limit_clause}
    )
    SELECT
        text_docs.document_id,
        text_docs.raw_text
        {metadata_select}
    FROM text_docs
    {metadata_join}
    WHERE {' AND '.join(filters)}
    """
    return con.execute(query).fetchdf()


def extract_note_patterns_from_documents(documents: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    revolver_rows: List[Dict[str, object]] = []
    lease_rows: List[Dict[str, object]] = []
    maturity_rows: List[Dict[str, object]] = []

    for record in documents.to_dict(orient="records"):
        extracted = extract_note_pattern_rows(record)
        revolver_rows.extend(extracted["revolver"])
        lease_rows.extend(extracted["lease"])
        maturity_rows.extend(extracted["debt_maturity"])

    return {
        "revolver": pd.DataFrame(revolver_rows),
        "lease": pd.DataFrame(lease_rows),
        "debt_maturity": pd.DataFrame(maturity_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract narrow SEC credit-note patterns from document text.")
    parser.add_argument("--years", required=True, help="Comma-separated years to scan, e.g. 2023,2024")
    parser.add_argument("--doc-text-map-root", default="data/inputs_layer/doc_text_map")
    parser.add_argument("--raw-documents-root", default="data/inputs_layer/raw_documents")
    parser.add_argument("--company-ids", default=None, help="Optional comma-separated entity IDs to retain")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of documents to scan after load")
    parser.add_argument("--out-root", default="data/sec/note_extracts")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    years = [int(token.strip()) for token in args.years.split(",") if token.strip()]
    company_ids = [token.strip() for token in (args.company_ids or "").split(",") if token.strip()] or None

    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    out_paths = {
        "revolver": out_root / "revolver_note_extracts.parquet",
        "lease": out_root / "lease_note_extracts.parquet",
        "debt_maturity": out_root / "debt_maturity_note_extracts.parquet",
    }
    if not args.overwrite:
        existing = [name for name, path in out_paths.items() if path.exists()]
        if existing:
            raise SystemExit(f"Output exists for {existing}; pass --overwrite to replace.")

    documents = load_document_texts(
        years=years,
        doc_text_map_root=ROOT / args.doc_text_map_root,
        raw_documents_root=ROOT / args.raw_documents_root if args.raw_documents_root else None,
        company_ids=company_ids,
        limit=args.limit,
    )
    extracted = extract_note_patterns_from_documents(documents)
    for name, df in extracted.items():
        path = out_paths[name]
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    "document_id",
                    "entity_id",
                    "source_type",
                    "doc_type",
                    "title",
                    "url",
                    "published_at",
                    "effective_at",
                    "ingested_at",
                    "pattern_family",
                    "metric_key",
                    "value",
                    "currency",
                    "bucket_label",
                    "pattern_name",
                    "extraction_confidence",
                    "evidence_text",
                    "extraction_method",
                ]
            )
        df.to_parquet(path, index=False)
        print(f"Wrote {name} note extracts -> {path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
