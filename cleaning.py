"""Deterministic cleaning shared by the engine AND every generated proof script.

The proof scripts embed this whole file, so anyone can re-run a number
without needing the rest of the project.
"""
import re

import numpy as np
import pandas as pd

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
NA_TOKENS = {"", "n/a", "na", "nan", "null", "none", "-", "--", "?", "unknown", "missing",
             "not available", "not provided", "n.a."}

_MONEY_RE = re.compile(
    r"^\s*(?P<neg>-)?\s*(?P<sym>[$€£₹])?\s*(?P<num>-?\d[\d,]*\.?\d*)\s*"
    r"(?P<code>USD|EUR|GBP|INR)?\s*$",
    re.I,
)
_SLASH_DATE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$")
_ISO_DATE = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")


def fix_mojibake(v):
    """'â‚¹799' (UTF-8 read as cp1252) -> '₹799'. Leaves normal text alone."""
    if isinstance(v, str) and "â" in v:
        try:
            return v.encode("cp1252").decode("utf-8")
        except Exception:
            return v
    return v


def parse_money(x):
    """Return (number, currency_or_None) or None if x is not numeric-like."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float, np.number)):
        return float(x), None
    if not isinstance(x, str):
        return None
    m = _MONEY_RE.match(x)
    if not m:
        return None
    num = float(m.group("num").replace(",", ""))
    if m.group("neg"):
        num = -num
    cur = CURRENCY_SYMBOLS.get(m.group("sym")) or (
        m.group("code").upper() if m.group("code") else None
    )
    return num, cur


def is_text_col(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def date_format_info(s):
    """Classify a text column of dates written like 03-04-2026.

    Returns None if the column is not date-like, else a dict whose "format" is
    DMY / MDY (clear, minority exceptions allowed), AMBIGUOUS (every value has
    both parts <= 12) or MIXED (real mix of formats).
    """
    vals = s.dropna().astype(str)
    if len(vals) == 0:
        return None
    m = vals.map(_SLASH_DATE.match).dropna()
    if len(m) / len(vals) < 0.5:
        return None
    a = np.array([int(x.group(1)) for x in m])
    b = np.array([int(x.group(2)) for x in m])
    a_big, b_big = int((a > 12).sum()), int((b > 12).sum())
    n_iso = int(vals.map(lambda v: bool(_ISO_DATE.match(v))).sum())
    if n_iso / len(vals) > 0.05:
        fmt = "MIXED"
    elif a_big == 0 and b_big == 0:
        fmt = "AMBIGUOUS"
    elif b_big == 0:
        fmt = "DMY"
    elif a_big == 0:
        fmt = "MDY"
    elif b_big / (a_big + b_big) <= 0.05:
        fmt = "DMY"
    elif a_big / (a_big + b_big) <= 0.05:
        fmt = "MDY"
    else:
        fmt = "MIXED"
    return {"format": fmt, "n_values": len(vals), "n_slash": len(m),
            "n_other": len(vals) - len(m)}


def parse_slash_dates(s, fmt):
    """Parse DD-MM-YYYY (fmt='DMY') or MM-DD-YYYY (fmt='MDY'); bad values become NaT."""
    ex = s.astype("string").str.extract(_SLASH_DATE.pattern)
    first, second = pd.to_numeric(ex[0], errors="coerce"), pd.to_numeric(ex[1], errors="coerce")
    year = pd.to_numeric(ex[2], errors="coerce")
    year = year.where(year >= 100, year + 2000)
    day, month = (first, second) if fmt == "DMY" else (second, first)
    return pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}), errors="coerce")


def clean_tables(tables):
    """Clean every table. Returns (cleaned_tables, log_messages)."""
    cleaned, log = {}, []
    for name, df in tables.items():
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        for col in list(df.columns):
            if not is_text_col(df[col]):
                continue
            df[col] = df[col].map(lambda v: fix_mojibake(v.strip()) if isinstance(v, str) else v)
            df[col] = df[col].map(
                lambda v: np.nan if isinstance(v, str) and v.lower() in NA_TOKENS else v
            )
            if re.fullmatch(r"(?i)currency|curr|ccy", col):
                df[col] = df[col].map(
                    lambda v: CURRENCY_SYMBOLS.get(v, v.upper()) if isinstance(v, str) else v)
            nonnull = df[col].dropna()
            if len(nonnull) == 0:
                continue

            info = date_format_info(df[col])
            if info and info["format"] in ("DMY", "MDY"):
                before = int(df[col].isna().sum())
                df[col] = parse_slash_dates(df[col], info["format"])
                bad = int(df[col].isna().sum()) - before
                log.append(f"{name}.{col}: read as {'DD-MM-YYYY' if info['format'] == 'DMY' else 'MM-DD-YYYY'} dates"
                           + (f"; {bad} value(s) were invalid or in another format and set to missing" if bad else ""))
                continue

            parsed = nonnull.map(parse_money)
            if parsed.notna().mean() >= 0.9:  # numbers stored as text ("$1,200")
                nums = df[col].map(lambda v: (parse_money(v) or (np.nan, None))[0])
                curs = df[col].map(lambda v: (parse_money(v) or (None, None))[1])
                bad_vals = sorted({str(v) for v in nonnull[parsed.isna()]})
                df[col] = nums
                if curs.notna().any():
                    df[col + "_currency"] = curs
                log.append(f"{name}.{col}: converted text to numbers"
                           + (f" (currency kept in {col}_currency)" if curs.notna().any() else "")
                           + (f"; non-numeric entries {bad_vals} set to missing" if bad_vals else ""))

        n_dup = int(df.duplicated().sum())
        if n_dup:
            df = df.drop_duplicates().reset_index(drop=True)
            log.append(f"{name}: removed {n_dup} exact duplicate row(s)")
        cleaned[name] = df
    return cleaned, log


def normalize_result(r):
    """Stable text form of a result, used to compare re-runs."""
    if isinstance(r, (pd.DataFrame, pd.Series)):
        try:
            r = r.round(6)
        except Exception:
            pass
        return r.to_string()
    if isinstance(r, (np.integer, int)) and not isinstance(r, bool):
        return str(int(r))
    if isinstance(r, (np.floating, float)):
        return repr(round(float(r), 6))
    return str(r)
