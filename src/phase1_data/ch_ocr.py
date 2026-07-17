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

# comma-separated thousands only: allowing space-separated groups made
# OCR-merged cells ('4 143,137,157') parse as one giant number
NUM = re.compile(r"\(?\d{1,3}(?:,\d{3})+\)?|\(?\d+(?:\.\d+)?\)?")

# Label patterns tolerate common OCR damage seen in real filings: rn->m
# ('tumover'), e->c ('opcrating'), and a lost leading character ('urnover',
# 'perating', 'roup operating').
_PREFIX = r"^(group |roup |total |consolidated )?"
LABELS = {
    "revenue": re.compile(_PREFIX + r"[a-z]?(urnover|umover|evenue)\b", re.I),
    "staff_costs": re.compile(_PREFIX + r"(staff costs|wages and salaries)", re.I),
    "operating_result": re.compile(
        _PREFIX + r"[a-z]?(perating|pcrating) (profit|loss|result)", re.I),
    "result_for_year": re.compile(
        _PREFIX + r"[a-z]?(rofit|oss).{0,30}for the (financial )?(year|period)", re.I),
}
# pre-tax result: used only as a locator signal, not extracted
PRETAX = re.compile(_PREFIX + r"[a-z]?(rofit|oss).{0,20}before tax", re.I)

PNL_HEADINGS = ("profit and loss account", "income statement",
                "statement of comprehensive income", "statement of profit or loss",
                "statement of total income",   # FRS102 1A 'total income and retained earnings'
                "statement of income")         # '... of income and retained earnings'


def norm_label(cell: str) -> str:
    """Normalise a row label for matching: 'Operating (loss)/profit' ->
    'operating loss profit'; leading bullets/marks ('• Turnover') are OCR
    layout noise too."""
    cell = re.sub(r"^[^A-Za-z]+", "", cell)
    return re.sub(r"\s+", " ", re.sub(r"[()\[\]/]", " ", cell)).strip()


def fuzzy_contains(row: str, phrase: str, threshold: float = 0.82) -> bool:
    """True if `phrase` appears in `row` allowing OCR-grade corruption
    ('profii and loss account', 'lncome statement'). Slides a phrase-sized
    window over the row and takes the best SequenceMatcher ratio."""
    from difflib import SequenceMatcher
    row = row.lower()
    if phrase in row:
        return True
    n, m = len(row), len(phrase)
    if n < m * 0.8:
        return False
    best = 0.0
    for start in range(0, max(1, n - m + 1), 3):
        best = max(best, SequenceMatcher(None, row[start:start + m], phrase).ratio())
        if best >= threshold:
            return True
    return False


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
    # OCR renders some thousands separators as semicolons ('107;863,788')
    cells = [c.strip().replace(";", ",") for c in row.split("|")]
    out = []
    for c in cells[1:] if len(cells) > 1 else []:
        # OCR can merge columns into one cell ('4 143,137,157' or
        # '106,752,215 112,450,205') — take every number in the cell
        for m in NUM.finditer(c):
            v = parse_number(m.group())
            if v is not None:
                out.append((v, m.group()))
    # drop leading note references — possibly several ('Notes 1,2' OCRs as
    # two numbers) — recognisable as small bare integers dwarfed by the
    # actual figures in the row
    while (len(out) >= 2 and out[0][0] > 0 and out[0][0] < 60
           and "." not in out[0][1] and "," not in out[0][1]
           and max(abs(v) for v, _ in out[1:]) >= 50 * out[0][0]):
        out = out[1:]
    # OCR sometimes reads a thousands comma as a decimal point ('172.155' for
    # '172,155'); reinterpret exactly-3-decimal values when siblings are >=1000
    vals = [v for v, _ in out]
    big = any(abs(v) >= 1000 for v in vals)
    fixed = []
    for v, tok in out:
        if big and "." in tok and re.search(r"\.\d{3}\)?$", tok) and abs(v) < 1000:
            v *= 1000
        fixed.append(v)
    return fixed


def pick_current_prior(nums: list[float],
                       prev_value: float | None = None) -> tuple[float | None, float | None, str]:
    """Choose (current, prior, how) from a statement row's numbers.

    Statements print current-year column(s) left of prior-year column(s), and
    segmented P&Ls (e.g. Arsenal's football | property | total x two years)
    repeat the block per year. When we know the previous filing's value, we
    find it from the right and take the figure half-a-row to the left — the
    same position in the current-year block:
        [572,599  1,374  573,973 | 428,453  1,457  429,910], prev=429,910
        -> j=5, current = nums[5 - ceil(6/2)] = nums[2] = 573,973.
    Without a prior anchor: two numbers = (current, prior); more = take
    second-from-right and flag as ambiguous for the mismatch audit."""
    if not nums:
        return None, None, "none"
    if len(nums) == 1:
        return nums[0], None, "single"
    if prev_value is not None:
        for j in range(len(nums) - 1, -1, -1):
            if abs(nums[j] - prev_value) <= 0.005 * max(abs(prev_value), 1):
                k = -(-len(nums) // 2)  # ceil
                if j - k >= 0:
                    return nums[j - k], nums[j], "prior_anchor"
                break
    # No usable prior anchor: statements repeat the column block per year
    # (Operations | Player trading | Total, then the same for the comparative
    # year), so with 2k numbers the current-year total is the k-th; a plain
    # two-column row is the k=1 special case. Odd counts arise when OCR merges
    # duplicate cells (e.g. Chelsea) — integer halving still lands on the
    # current block; the prior-year audit polices the residual risk.
    return nums[len(nums) // 2 - 1] if len(nums) > 1 else nums[0], nums[-1], \
        ("two_col" if len(nums) == 2 else "half_split")


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
        # a statement page carries its title near the top; matching the phrase
        # anywhere in the page text lets strategic-report prose hijack the pick
        top_rows = rows[:10]
        has_turnover = any(LABELS["revenue"].match(norm_label(r.split("|")[0]))
                           and len(row_numbers(r)) >= 1 for r in rows)
        if not has_turnover:
            continue
        has_operating = any(LABELS["operating_result"].match(norm_label(r.split("|")[0]))
                            for r in rows)
        has_result = any(LABELS["result_for_year"].match(norm_label(r.split("|")[0]))
                         for r in rows)
        has_pretax = any(PRETAX.match(norm_label(r.split("|")[0])) for r in rows)
        # heading may be split across two OCR rows ('Consolidated Profit' /
        # 'and Loss Account'), so test consecutive pairs too
        candidates = list(top_rows) + [
            f"{a} {b}" for a, b in zip(top_rows, top_rows[1:])]
        has_heading = any(fuzzy_contains(r, h)
                          for r in candidates for h in PNL_HEADINGS)
        # some statements have no clean operating row (straight to 'loss
        # before interest'), so operating OR result-for-year suffices
        if has_heading and (has_operating or has_result):
            return idx, "heading"
        if has_heading:
            # heading + numeric turnover alone (statement rows too garbled to
            # classify); contents/notes pages never combine both
            return idx, "heading_only"
        if fallback is None and has_operating and (has_result or has_pretax):
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


def extract_pdf(pdf_path: Path, prev: dict[str, float] | None = None) -> dict:
    """`prev` = previous filing's extracted values, used as column anchors."""
    prev = prev or {}
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

    # Scale correction: header symbols are unreliable on scans (GBP'000 read
    # as GBPm; Villa files in FULL pounds, which no header hints at). Club
    # turnover must land in [1m, 1.2bn] GBP; the raw turnover figure therefore
    # pins the scale up to the tiny band where two scales both fit, which the
    # header-detected value tie-breaks.
    raw_rev = None
    for r in rows:
        if LABELS["revenue"].match(norm_label(r.split("|")[0])):
            nums = row_numbers(r)
            positives = [n for n in nums if n > 0]
            if positives:
                raw_rev = positives[0]
                break
    # a bare small integer (a stray note ref on a lone row) must never be
    # scale-fitted into a plausible revenue; genuine GBPm figures have
    # decimals and genuine GBP'000/GBP figures are >= thousands
    if raw_rev is not None and (raw_rev >= 60 or raw_rev % 1 != 0):
        fits = [u for u in (1.0, 1e3, 1e6) if 1e6 <= raw_rev * u <= 1.2e9]
        if fits and units not in fits:
            result["units_corrected"] = f"{units}->{fits[0]}"
            units = fits[0]
    result["units"] = units

    def grab(field: str, search_rows) -> tuple[float | None, float | None, str]:
        prev_units = prev.get(field)
        for r in search_rows:
            if LABELS[field].match(norm_label(r.split("|")[0])):
                nums = row_numbers(r)
                if field == "revenue":
                    nums = [n for n in nums if n > 0]  # turnover is never negative
                if nums:
                    cur, prior, how_col = pick_current_prior(
                        nums, None if prev_units is None else prev_units / units)
                    if cur is not None:
                        return (cur * units,
                                prior * units if prior is not None else None,
                                how_col)
        return None, None, "none"

    col_flags = []
    for field in LABELS:
        if field == "staff_costs":
            continue  # handled below against extracted revenue
        cur, prior, how_col = grab(field, rows)
        result[field] = cur
        result[f"{field}_prior"] = prior
        if cur is not None:
            col_flags.append(f"{field}:{how_col}")

    # Staff costs: costs are presented negative on many P&Ls (sign is layout,
    # the quantity is a magnitude), and 'wages and salaries' also labels the
    # DIRECTORS' remuneration note. Collect every candidate across statement
    # and notes, then keep the largest magnitude that is plausible relative
    # to revenue — the total always dominates any subset row.
    rev = result.get("revenue")
    candidates = []
    for rows_n in pages[pnl_idx:]:
        for r in rows_n:
            label = norm_label(r.split("|")[0])
            if LABELS["staff_costs"].match(label):
                cur, prior, how_col = pick_current_prior(row_numbers(r))
                if cur is not None:
                    is_total = "staff costs" in label.lower()
                    candidates.append((is_total, abs(cur) * units,
                                       abs(prior) * units if prior is not None else None))
    staff = staff_prior = None
    if candidates and rev:
        # floor 0.20: directors'-remuneration rows (the classic wrong pick)
        # run ~5-15% of revenue; genuine club wage bills never sit below ~25%
        in_band = [c for c in candidates if 0.20 * rev <= c[1] <= 1.8 * rev]
        # 'staff costs' rows are totals (incl. social security + pensions);
        # 'wages and salaries' is a component — prefer totals for a
        # consistent wage-to-revenue measure across clubs
        totals = [c for c in in_band if c[0]]
        pick_from = totals or in_band
        if pick_from:
            _, staff, staff_prior = max(pick_from, key=lambda c: c[1])
    result["staff_costs"] = staff
    result["staff_costs_prior"] = staff_prior
    if staff is not None:
        col_flags.append("staff_costs:max_in_band")
    result["column_how"] = ";".join(col_flags)
    rev = result.get("revenue")
    if rev is None:
        result["status"] = "revenue_not_found"
    elif not (2e6 <= rev <= 1.2e9):
        # scale-fitting can legitimise junk picks (a stray '4' -> GBP 4m);
        # nothing a UK pro club files sits outside this band
        result["status"] = "revenue_implausible"
    else:
        result["status"] = "ok"
    return result


def extract_all() -> pd.DataFrame:
    # chronological per club so each filing can anchor on its predecessor
    by_club: dict[str, list[Path]] = {}
    for p in sorted(PDF_DIR.glob("*.pdf")):
        by_club.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    rows = []
    for club_paths in by_club.values():
        prev: dict[str, float] = {}
        for p in sorted(club_paths, key=lambda q: q.stem.rsplit("_", 1)[1]):
            rec = extract_pdf(p, prev)
            # continuity gate: consecutive-filing revenue moves stay within
            # ~10x even across promotion windfalls (observed max: Brentford
            # 2021, 9.2x); junk picks are typically 100x+ off. Must not
            # poison the anchor chain either way.
            if (rec["status"] == "ok" and prev.get("revenue")
                    and not 1 / 12 <= rec["revenue"] / prev["revenue"] <= 12):
                rec["status"] = "revenue_discontinuous"
            rows.append(rec)
            if rec["status"] == "ok":
                prev = {f: rec[f] for f in LABELS if rec.get(f) is not None}
    df = pd.DataFrame(rows).sort_values(["club", "fy_end"], ignore_index=True)

    # Reconciliation against the NEXT filing's audited comparative column —
    # stronger evidence than any locator heuristic:
    # 1. a value flagged by the continuity gate but reproduced (±2%) in the
    #    next filing's prior-year column is confirmed OK (the gate fires
    #    spuriously when an earlier junk value poisoned the chain);
    # 2. a missing/failed year whose successor filing shows a plausible
    #    prior-year figure is backfilled from it (status records the source).
    for club, grp in df.groupby("club"):
        idxs = grp.sort_values("fy_end").index
        for cur, nxt in zip(idxs[:-1], idxs[1:]):
            nxt_prior = df.loc[nxt, "revenue_prior"]
            if pd.isna(nxt_prior) or not 2e6 <= nxt_prior <= 1.2e9:
                continue
            cur_rev = df.loc[cur, "revenue"]
            if (df.loc[cur, "status"] in ("revenue_discontinuous", "revenue_implausible")
                    and pd.notna(cur_rev)
                    and abs(cur_rev - nxt_prior) <= 0.02 * nxt_prior):
                df.loc[cur, "status"] = "ok"
                df.loc[cur, "column_how"] = str(df.loc[cur, "column_how"]) + ";confirmed_by_next"
            elif df.loc[cur, "status"] != "ok":
                df.loc[cur, "revenue"] = nxt_prior
                df.loc[cur, "status"] = "ok_from_next_prior"
        # the final filing has no successor: confirm it via its OWN prior
        # column matching its predecessor's (now settled) revenue
        for prv, cur in zip(idxs[:-1], idxs[1:]):
            if (df.loc[cur, "status"] == "revenue_discontinuous"
                    and pd.notna(df.loc[cur, "revenue_prior"])
                    and pd.notna(df.loc[prv, "revenue"])
                    and df.loc[prv, "status"].startswith("ok")
                    and abs(df.loc[cur, "revenue_prior"] - df.loc[prv, "revenue"])
                    <= 0.02 * df.loc[prv, "revenue"]):
                df.loc[cur, "status"] = "ok"
                df.loc[cur, "column_how"] = str(df.loc[cur, "column_how"]) + ";confirmed_by_own_prior"

    # curated per-filing overrides for OCR-mangled statements: values read
    # manually from the cited page of the filing itself (never invented)
    overrides_path = Path(__file__).resolve().parent / "ch_overrides.csv"
    if overrides_path.exists():
        for o in pd.read_csv(overrides_path).itertuples():
            mask = (df.club == o.club) & (df.fy_end == o.fy_end)
            if not mask.any():
                continue
            if o.action == "set_revenue":
                df.loc[mask, "revenue"] = o.revenue
                df.loc[mask, "status"] = "ok_override"
            elif o.action == "suppress":
                df.loc[mask, "revenue"] = float("nan")
                df.loc[mask, "status"] = "suppressed_junk"

    # prior-year cross-check: filing N's prior column vs filing N-1's current
    df["prior_check"] = ""
    for club, grp in df.groupby("club"):
        grp = grp.sort_values("fy_end")
        for prev, cur in zip(grp.index[:-1], grp.index[1:]):
            a, b = df.loc[cur, "revenue_prior"], df.loc[prev, "revenue"]
            if pd.notna(a) and pd.notna(b):
                rel = abs(a - b) / max(abs(b), 1)
                # <=6% divergence is routine (comparative-year restatements —
                # e.g. Brighton FY19 restated 143.1->148.0, Villa FY19
                # 54.3->51.4 — and OCR last-digit noise); beyond = suspect pick
                df.loc[cur, "prior_check"] = (
                    "ok" if rel <= 0.001 else
                    "restated_ok" if rel <= 0.06 else "MISMATCH")
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
