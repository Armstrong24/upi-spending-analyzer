"""Read a UPI statement (CSV or PDF) into one clean table:
    datetime | description | type (DEBIT/CREDIT) | amount
"""
import io
import re

import pandas as pd

# Different apps / banks name their columns differently
COLUMN_ALIASES = {
    "date": ["date", "txn date", "transaction date", "value date", "date & time"],
    "time": ["time", "txn time"],
    "description": ["transaction details", "description", "narration", "details",
                    "remarks", "particulars", "paid to", "name"],
    "type": ["type", "dr/cr", "debit/credit", "txn type", "cr/dr"],
    "amount": ["amount", "amount (inr)", "amount(inr)", "txn amount", "transaction amount"],
    "debit": ["debit", "withdrawal", "withdrawal amt", "debit amount", "dr"],
    "credit": ["credit", "deposit", "deposit amt", "credit amount", "cr"],
}


def _find(cols, key):
    for c in cols:
        if c.strip().lower() in COLUMN_ALIASES[key]:
            return c
    return None


def _to_amount(series):
    s = series.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(raw.columns)
    c_date, c_time = _find(cols, "date"), _find(cols, "time")
    c_desc, c_type, c_amt = _find(cols, "description"), _find(cols, "type"), _find(cols, "amount")
    c_dr, c_cr = _find(cols, "debit"), _find(cols, "credit")
    if c_date is None or c_desc is None:
        raise ValueError(f"Could not find date/description columns in: {cols}")

    dt_text = raw[c_date].astype(str)
    if c_time is not None:
        dt_text = dt_text + " " + raw[c_time].astype(str)
    # ISO dates (2026-10-05) are year-first; Indian bank dates (05/10/2026) are day-first
    iso = raw[c_date].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5
    out = pd.DataFrame({"datetime": pd.to_datetime(dt_text, errors="coerce", dayfirst=not iso, format="mixed"),
                        "description": raw[c_desc].astype(str).str.strip()})

    if c_amt is not None:
        amt = _to_amount(raw[c_amt])
        if c_type is not None:
            t = raw[c_type].astype(str).str.strip().str.upper()
            out["type"] = t.str.startswith("CR").map({True: "CREDIT", False: "DEBIT"})
        else:  # negative = debit
            out["type"] = (amt < 0).map({True: "DEBIT", False: "CREDIT"})
        out["amount"] = amt.abs()
    elif c_dr is not None or c_cr is not None:
        dr = _to_amount(raw[c_dr]).fillna(0) if c_dr else 0
        cr = _to_amount(raw[c_cr]).fillna(0) if c_cr else 0
        out["type"] = (pd.Series(dr) > 0).map({True: "DEBIT", False: "CREDIT"}).values
        out["amount"] = (pd.Series(dr) + pd.Series(cr)).values
    else:
        raise ValueError("No amount / debit / credit column found")

    out = out.dropna(subset=["datetime", "amount"])
    out = out[out["amount"] > 0].sort_values("datetime").reset_index(drop=True)
    return out


def read_csv(file) -> pd.DataFrame:
    return normalize(pd.read_csv(file))


# "Oct 05, 2026  07:45 PM  Paid to Swiggy  DEBIT  INR 250.00"
LINE_RE = re.compile(
    r"^(?P<date>[A-Z][a-z]{2} \d{1,2}, \d{4})\s+(?P<time>\d{1,2}:\d{2}\s?[AP]M)\s+(?P<desc>.+?)\s+"
    r"(?P<type>DEBIT|CREDIT)\s+(?:INR|Rs\.?|₹)\s?(?P<amount>[\d,]+(?:\.\d+)?)$")


def read_pdf(file) -> pd.DataFrame:
    import pdfplumber
    data = file.read() if hasattr(file, "read") else open(file, "rb").read()
    rows = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                m = LINE_RE.match(line.strip())
                if m:
                    rows.append(m.groupdict())
    if not rows:
        raise ValueError("No transactions recognised in the PDF")
    raw = pd.DataFrame(rows).rename(columns={"desc": "description"})
    return normalize(raw)


def load_statement(file, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".pdf"):
        return read_pdf(file)
    return read_csv(file)
