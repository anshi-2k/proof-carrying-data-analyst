# DataProof AI re-runnable proof
# Question: 'What is the total profit in USD?'
# Run: python proof_1791368371434.py
"""Deterministic cleaning shared by the engine AND every generated proof script.

The proof scripts embed this whole file, so anyone can re-run a number
without needing the rest of the project.
"""
import re

import numpy as np
import pandas as pd

CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
NA_TOKENS = {"", "n/a", "na", "nan", "null", "none", "-", "--", "?", "unknown", "missing"}

_MONEY_RE = re.compile(
    r"^\s*(?P<neg>-)?\s*(?P<sym>[$€£₹])?\s*(?P<num>-?\d[\d,]*\.?\d*)\s*"
    r"(?P<code>USD|EUR|GBP|INR)?\s*$",
    re.I,
)


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


def clean_tables(tables):
    """Clean every table. Returns (cleaned_tables, log_messages)."""
    cleaned, log = {}, []
    for name, df in tables.items():
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        for col in list(df.columns):
            if not is_text_col(df[col]):
                continue
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[col] = df[col].map(
                lambda v: np.nan if isinstance(v, str) and v.lower() in NA_TOKENS else v
            )
            nonnull = df[col].dropna()
            if len(nonnull) == 0:
                continue
            parsed = nonnull.map(parse_money)
            if parsed.notna().mean() >= 0.9:  # numbers stored as text ("$1,200")
                nums = df[col].map(lambda v: (parse_money(v) or (np.nan, None))[0])
                curs = df[col].map(lambda v: (parse_money(v) or (None, None))[1])
                df[col] = nums
                if curs.notna().any():
                    df[col + "_currency"] = curs
                log.append(f"{name}.{col}: converted text to numbers"
                           + (f" (currency kept in {col}_currency)" if curs.notna().any() else ""))

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


import os
HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = ['../data/orders.csv', '../data/products.csv', '../data/region_summary.csv']

def _load(p):
    p = os.path.join(HERE, p)
    if p.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    return pd.read_csv(p, dtype=object, keep_default_na=False)

tables = {re.sub(r"\W+", "_", os.path.splitext(os.path.basename(p))[0]): _load(p) for p in PATHS}
tables, _ = clean_tables(tables)
globals().update(tables)   # each table is a variable named after its file

result = orders[orders['Sales_currency']=='USD']['Profit'].sum()
print("RESULT:", normalize_result(result))
