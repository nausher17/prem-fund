"""OCR extraction of financial line items from Companies House filings.

Why OCR: every filing for the panel's PL clubs is stored by Companies House
as a scanned/image PDF (zero filings offer iXBRL in our window — verified
2026-07-16), so structured extraction is impossible and the text layer is
empty. We use macOS's built-in Vision framework (no external binaries):
pypdfium2 renders pages at 300dpi, VNRecognizeTextRequest (accurate mode)
reads them, and observations are re-joined into table rows by y-position.

Pipeline (subcommands, each cached and idempotent):
  download  fetch every AA filing PDF for CH_REGISTRY clubs with period end
            2015-2024 -> data/raw/companies_house/pdf/
  ocr       OCR each PDF once -> data/raw/companies_house/ocr/{stem}.json
            (list of pages, each a list of row strings)
  extract   locate the group P&L and pull: turnover, staff costs (P&L or
            note), operating result, result for the year; detect units
            (GBP'000 vs GBPm) from statement headers -> ch_financials.csv

Integrity guards:
- the prior-year column of filing N must equal the current-year value of
  filing N-1 (within 1%o) — catches OCR digit errors and wrong-column picks;
- extraction failures are recorded with reasons, never guessed;
- a benchmark set of publicly reported figures is asserted in validation.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from .companies_house import (CH_REGISTRY, accounts_filings, make_session)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw" / "companies_house"
PDF_DIR = RAW / "pdf"
OCR_DIR = RAW / "ocr"
PROCESSED = PROJECT_ROOT / "data" / "processed"


# -- download -----------------------------------------------------------------

def in_window(made_up: str | None) -> bool:
    return bool(made_up) and "2015" <= made_up[:4] <= "2024"


def download_pdfs() -> list[Path]:
    import requests
    session = make_session()
    key = session.session.auth[0]
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for club, number in CH_REGISTRY.items():
        for filing in accounts_filings(session, number):
            made_up = filing.get("description_values", {}).get("made_up_date") \
                or filing.get("action_date")
            if not in_window(made_up):
                continue
            out = PDF_DIR / f"{club.replace(' ', '_')}_{made_up}.pdf"
            paths.append(out)
            if out.exists():
                continue
            meta_link = filing.get("links", {}).get("document_metadata")
            if not meta_link:
                log.error("%s %s: no document metadata", club, made_up)
                continue
            meta = json.loads(session.get(meta_link))
            url = meta["links"]["document"]  # already .../content
            session._throttle(url)
            r = requests.get(url, auth=(key, ""), timeout=180,
                             headers={"Accept": "application/pdf"},
                             allow_redirects=True)
            r.raise_for_status()
            out.write_bytes(r.content)
            log.info("downloaded %s (%.1f MB)", out.name, len(r.content) / 1e6)
    return paths


# -- OCR ------------------------------------------------------------------------

def _ocr_image_rows(pil_img) -> list[str]:
    import Vision
    from Cocoa import NSData
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    handler.performRequests_error_([req], None)
    obs = []
    for o in (req.results() or []):
        c = o.topCandidates_(1)
        if c and len(c):
            bb = o.boundingBox()
            obs.append((1.0 - (bb.origin.y + bb.size.height / 2), bb.origin.x,
                        c[0].string()))
    obs.sort()
    rows, cur, last_y = [], [], None
    for y, x, t in obs:
        if last_y is None or abs(y - last_y) < 0.008:
            cur.append((x, t))
        else:
            rows.append(" | ".join(t for _, t in sorted(cur)))
            cur = [(x, t)]
        last_y = y
    if cur:
        rows.append(" | ".join(t for _, t in sorted(cur)))
    return rows


def ocr_pdf(pdf_path: Path) -> list[list[str]]:
    """OCR all pages (cached as JSON next to the raw layer)."""
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    cache = OCR_DIR / (pdf_path.stem + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf_path)
    pages = []
    for i in range(len(doc)):
        img = doc[i].render(scale=300 / 72, grayscale=True).to_pil()
        pages.append(_ocr_image_rows(img))
    cache.write_text(json.dumps(pages))
    log.info("OCR %s: %d pages", pdf_path.name, len(pages))
    return pages


# -- extraction -------------------------------------------------------------------

NUM = re.compile(r"\(?\d{1,3}(?:,\s?\d{3})+\)?|\(?\d+(?:\.\d+)?\)?")

_PREFIX = r"^(group |total |consolidated )?"
LABELS = {
    "revenue": re.compile(_PREFIX + r"(turnover|revenue)\b", re.I),
    "staff_costs": re.compile(_PREFIX + r"(staff costs|wages and salaries)", re.I),
    "operating_result": re.compile(_PREFIX + r"operating (profit|loss|result)", re.I),
    "result_for_year": re.compile(
        _PREFIX + r"(profit|loss).{0,30}for the (financial )?(year|period)", re.I),
}

PNL_HEADINGS = ("profit and loss account", "income statement",
                "statement of comprehensive income")


def parse_number(tok: str) -> float | None:
    neg = tok.startswith("(") and tok.endswith(")")
    tok = tok.strip("()").replace(",", "")
    try:
        v = float(tok)
    except ValueError:
        return None
    return -v if neg else v


def row_numbers(row: str) -> list[float]:
    """Numeric cells after the label, with the note-reference column dropped.

    Statutory P&L rows read e.g. 'Turnover | 3 | 474,847 | 453,056' where 3 is
    the note number. A leading small bare integer followed by a much larger
    figure is a note ref, not a value. Values are then read right-to-left:
    rightmost = prior year, second-from-right = current year — which is also
    correct for segmented P&Ls (e.g. Chelsea's ops/player-trading/total
    columns, where 'total' sits immediately left of the prior year).
    """
    cells = [c.strip() for c in row.split("|")]
    out = []
    for c in cells[1:] if len(cells) > 1 else []:
        m = NUM.fullmatch(c.replace(" ", "")) or NUM.search(c)
        if m:
            v = parse_number(m.group().replace(" ", ""))
            if v is not None:
                out.append((v, m.group()))
    if (len(out) >= 2 and out[0][0] > 0 and out[0][0] < 60
            and "." not in out[0][1] and "," not in out[0][1]
            and abs(out[1][0]) >= 50 * out[0][0]):
        out = out[1:]  # drop note reference
    return [v for v, _ in out]


def pick_current_prior(nums: list[float]) -> tuple[float | None, float | None]:
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], None
    return nums[-2], nums[-1]


def find_pnl_page(pages: list[list[str]]) -> tuple[int | None, str]:
    """The consolidated P&L page and how it was found.

    Primary: a statement heading (PNL_HEADINGS) + a numeric Turnover row + an
    operating-result row. Fallback: any page with a numeric Turnover row AND a
    result-for-year row (some scans garble the heading); flagged so validation
    scrutinises these harder. Strategic-report KPI tables are avoided by
    requiring the operating/result rows, and the first qualifying page is the
    statements page in practice (group P&L precedes company-only)."""
    fallback = None
    for idx, rows in enumerate(pages):
        text = " \n".join(rows).lower()
        has_turnover = any(LABELS["revenue"].match(r.split("|")[0].strip())
                           and len(row_numbers(r)) >= 1 for r in rows)
        if not has_turnover:
            continue
        has_operating = any(LABELS["operating_result"].match(r.split("|")[0].strip())
                            for r in rows)
        has_result = any(LABELS["result_for_year"].match(r.split("|")[0].strip())
                         for r in rows)
        if any(h in text for h in PNL_HEADINGS) and has_operating:
            return idx, "heading"
        if fallback is None and has_operating and has_result:
            fallback = idx
    return fallback, ("fallback" if fallback is not None else "none")


def detect_units(rows: list[str]) -> float:
    """Multiplier to GBP: 1e3 for £'000 (statutory default), 1e6 for £m.

    OCR regularly reads '£' as '€' or 'E' (Everton's £m columns come out as
    'Em', Brighton's £'000 as €'000), so match on normalised tokens."""
    joined = " ".join(rows).lower().replace("€", "£")
    if re.search(r"\b[£e]\s?m\b", joined):
        return 1e6
    return 1e3


def extract_pdf(pdf_path: Path) -> dict:
    club, made_up = pdf_path.stem.rsplit("_", 1)
    club = club.replace("_", " ")
    pages = ocr_pdf(pdf_path)
    pnl_idx, how = find_pnl_page(pages)
    result = {"club": club, "fy_end": made_up, "source": pdf_path.name,
              "pnl_page": None if pnl_idx is None else pnl_idx + 1,
              "pnl_locator": how}
    if pnl_idx is None:
        result["status"] = "pnl_not_found"
        return result
    rows = pages[pnl_idx]
    units = detect_units(rows)
    result["units"] = units
    for field, pattern in LABELS.items():
        cur, prior = None, None
        for r in rows:
            label = r.split("|")[0].strip()
            if pattern.match(label):
                cur, prior = pick_current_prior(row_numbers(r))
                if cur is not None:
                    cur *= units
                    prior = prior * units if prior is not None else None
                    break
        result[field] = cur
        result[f"{field}_prior"] = prior
    # staff costs often live in the notes, not the P&L face
    if result.get("staff_costs") is None:
        for rows_n in pages[pnl_idx:]:
            for r in rows_n:
                label = r.split("|")[0].strip()
                if LABELS["staff_costs"].match(label):
                    cur, prior = pick_current_prior(row_numbers(r))
                    if cur is not None:
                        result["staff_costs"] = cur * units
                        result["staff_costs_prior"] = (prior * units
                                                       if prior is not None else None)
                        break
            if result.get("staff_costs") is not None:
                break
    result["status"] = "ok" if result.get("revenue") else "revenue_not_found"
    return result


def extract_all() -> pd.DataFrame:
    rows = [extract_pdf(p) for p in sorted(PDF_DIR.glob("*.pdf"))]
    df = pd.DataFrame(rows).sort_values(["club", "fy_end"], ignore_index=True)

    # prior-year cross-check: filing N's prior column vs filing N-1's current
    df["prior_check"] = ""
    for club, grp in df.groupby("club"):
        grp = grp.sort_values("fy_end")
        for prev, cur in zip(grp.index[:-1], grp.index[1:]):
            a, b = df.loc[cur, "revenue_prior"], df.loc[prev, "revenue"]
            if pd.notna(a) and pd.notna(b):
                df.loc[cur, "prior_check"] = (
                    "ok" if abs(a - b) <= 0.001 * max(abs(b), 1) else "MISMATCH")
    out = PROCESSED / "ch_financials.csv"
    df.to_csv(out, index=False)
    ok = (df.status == "ok").sum()
    print(f"Wrote {out}: {len(df)} filings, {ok} with revenue, "
          f"{(df.prior_check == 'MISMATCH').sum()} prior-year mismatches")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["download", "ocr", "extract", "all"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.what in ("download", "all"):
        download_pdfs()
    if args.what in ("ocr", "all"):
        for p in sorted(PDF_DIR.glob("*.pdf")):
            ocr_pdf(p)
    if args.what in ("extract", "all"):
        df = extract_all()
        bad = df[df.status != "ok"]
        if len(bad):
            print("\nFilings needing attention:")
            print(bad[["club", "fy_end", "status"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
