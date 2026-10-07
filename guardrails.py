"""Trap detectors for DataProof AI.

Every detector returns a list of findings:
    {"type", "severity": "blocking" | "warning" | "info", "message", "table", "column"}
"blocking" = answers touching this issue must be refused or clarified.
"""
import re
import datetime

import pandas as pd

from cleaning import parse_money, is_text_col, NA_TOKENS

MONEY_HINTS = ("sales", "profit", "revenue", "price", "cost", "amount", "money", "income", "spend")
DATE_WORDS = re.compile(
    r"\b(date|dates|day|daily|week|weekly|month|monthly|quarter|q[1-4]|year|yearly|ytd|"
    r"yesterday|today|last|recent|recently|latest|january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\b", re.I)


def F(type_, severity, message, table=None, column=None):
    return {"type": type_, "severity": severity, "message": message,
            "table": table, "column": column}


# ---------------------------------------------------------------- data checks
def check_duplicates(name, df):
    out = []
    exact = int(df.duplicated().sum())
    if exact:
        out.append(F("duplicates", "warning",
                     f"{exact} exact duplicate row(s) in '{name}' (removed before analysis).", name))
    for col in df.columns:
        if re.fullmatch(r"(?i).*(_id|id)", col):
            dedup = df.drop_duplicates()
            conflicts = int(dedup[col].duplicated(keep=False).sum())
            if conflicts:
                out.append(F("duplicate_keys", "warning",
                             f"'{name}.{col}' repeats in {conflicts} row(s) with DIFFERENT values "
                             f"(conflicting records, not simple duplicates).", name, col))
    return out


def check_missing(name, df):
    out = []
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n:
            pct = 100 * n / max(len(df), 1)
            out.append(F("missing", "warning",
                         f"'{name}.{col}' has {n} missing value(s) ({pct:.0f}%).", name, col))
    return out


def check_units(name, raw):
    """Mixed currencies and mixed unit suffixes, on RAW data."""
    out, currencies = [], set()
    for col in raw.columns:
        s = raw[col].dropna()
        if len(s) == 0:
            continue
        if re.search(r"(?i)^(currency|curr|ccy)$", col):
            currencies |= {str(v).strip().upper() for v in s.unique()}
        if is_text_col(raw[col]):
            parsed = s.map(parse_money)
            currencies |= {p[1] for p in parsed if p and p[1]}
            suffix = s.astype(str).str.extract(r"^\s*-?[\d.,]+\s*(kg|g|lb|lbs|oz|km|mi|cm|m|k|%)\s*$",
                                              flags=re.I)[0].dropna().str.lower()
            if suffix.nunique() > 1:
                out.append(F("mixed_units", "blocking",
                             f"'{name}.{col}' mixes units: {sorted(suffix.unique())}.", name, col))
    if len(currencies) > 1:
        out.append(F("mixed_currency", "blocking",
                     f"'{name}' mixes currencies {sorted(currencies)} with no exchange rate table.",
                     name))
    elif not currencies:
        money_cols = [c for c in raw.columns if any(h in c.lower() for h in MONEY_HINTS)]
        if money_cols:
            out.append(F("unspecified_currency", "warning",
                         f"'{name}' money column(s) {money_cols} have no currency stated.", name))
    return out


_SLASH = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$")
_ISO = re.compile(r"^\s*\d{4}-\d{1,2}-\d{1,2}")


def check_dates(name, raw):
    out = []
    for col in raw.columns:
        if not is_text_col(raw[col]):
            continue
        s = raw[col].dropna().astype(str)
        m = s.map(_SLASH.match)
        n_slash, n_iso = int(m.notna().sum()), int(s.map(_ISO.match).notna().sum())
        if n_slash == 0:
            continue
        a = [int(x.group(1)) for x in m.dropna()]
        b = [int(x.group(2)) for x in m.dropna()]
        a_big, b_big = any(v > 12 for v in a), any(v > 12 for v in b)
        if n_iso:
            out.append(F("date_format", "blocking",
                         f"'{name}.{col}' mixes ISO and slash date formats.", name, col))
        elif a_big and b_big:
            out.append(F("date_format", "blocking",
                         f"'{name}.{col}' mixes DD/MM and MM/DD formats.", name, col))
        elif not a_big and not b_big:
            out.append(F("ambiguous_dates", "blocking",
                         f"'{name}.{col}' dates are ambiguous: every value has both parts <= 12, so "
                         f"DD/MM vs MM/DD cannot be determined.", name, col))
        else:
            fmt = "DD/MM/YYYY" if a_big else "MM/DD/YYYY"
            out.append(F("date_format_inferred", "info",
                         f"'{name}.{col}' format inferred as {fmt} (a part > 12 appears).", name, col))
    return out


def check_category_variants(name, df):
    out = []
    for col in df.columns:
        if not is_text_col(df[col]):
            continue
        vals = df[col].dropna().astype(str).unique()
        if len(vals) > 50:
            continue
        groups = {}
        for v in vals:
            groups.setdefault(re.sub(r"\s+", " ", v.strip().lower()), set()).add(v)
        bad = [sorted(g) for g in groups.values() if len(g) > 1]
        if bad:
            out.append(F("category_variants", "warning",
                         f"'{name}.{col}' has spelling/case variants of the same value: {bad}. "
                         f"Group on lower-cased text.", name, col))
    return out


def check_contradictions(tables, tol=0.01):
    """Detail table vs summary table: shared key + numeric column whose totals disagree."""
    out, names = [], list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = tables[a], tables[b]
            (dn, D), (sn, S) = ((a, A), (b, B)) if len(A) >= len(B) else ((b, B), (a, A))
            if len(D) == len(S):
                continue
            for key in [c for c in S.columns if c in D.columns
                        and is_text_col(S[c]) and is_text_col(D[c])]:
                skey = S[key].astype(str).str.strip().str.lower()
                if not skey.is_unique:
                    continue
                for sc in [c for c in S.columns if pd.api.types.is_numeric_dtype(S[c])]:
                    for dc in [c for c in D.columns if pd.api.types.is_numeric_dtype(D[c])
                               and (c.lower() == sc.lower() or c.lower() in sc.lower())]:
                        dtot = D.assign(_k=D[key].astype(str).str.strip().str.lower()) \
                                .groupby("_k")[dc].sum()
                        stot = pd.Series(S[sc].values, index=skey.values)
                        bad = []
                        for k, sv in stot.items():
                            dv = dtot.get(k)
                            if dv is not None and abs(dv - sv) > tol * max(abs(dv), 1):
                                bad.append(f"{k}: {sn}.{sc}={sv:g} vs sum of {dn}.{dc}={dv:g}")
                        if bad:
                            out.append(F("contradiction", "warning",
                                         f"Tables '{dn}' and '{sn}' disagree -> " + "; ".join(bad)
                                         + ". Report both or state which source is used.", dn, dc))
    return out


def analyze_tables(raw_tables, cleaned_tables):
    findings = []
    for name, raw in raw_tables.items():
        findings += check_duplicates(name, raw)
        findings += check_units(name, raw)
        findings += check_dates(name, raw)
    for name, df in cleaned_tables.items():
        findings += check_missing(name, df)
        findings += check_category_variants(name, df)
    findings += check_contradictions(cleaned_tables)
    return findings


# ---------------------------------------------------------------- question checks
TRICK_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) (instructions|rules)",
    r"pretend|roleplay|you are now",
    r"just (guess|make (it|something) up|estimate)",
    r"(don'?t|do not|no need to) (verify|check|run|use code|show (the )?code)",
    r"without (checking|running|looking|code)",
    r"(assume|suppose) (that )?(the )?(answer|total|result)",
    r"(you must|always) (answer|say)",
    r"(round|make it) (up )?to (look|seem)",
    r"confidently",
]
FORECAST = re.compile(r"\b(forecast|predict|projection|projected|will (be|sales|profit|we)|next (year|quarter|month))\b", re.I)
MISSING_CONCEPTS = ["customer", "employee", "salary", "inventory", "supplier", "discount",
                    "churn", "gender", "age", "weather", "stock price"]


def data_years(tables):
    years = set()
    for df in tables.values():
        for col in df.columns:
            if is_text_col(df[col]):
                years |= set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)",
                                        " ".join(df[col].dropna().astype(str).head(2000))))
    return {int(y) for y in years}


def check_question(question, tables):
    out, q = [], question.lower()
    for p in TRICK_PATTERNS:
        if re.search(p, q):
            out.append(F("trick_question", "blocking",
                         "Question tries to bypass verification or pressure a guess; "
                         "every number must come from re-runnable code."))
            break
    if FORECAST.search(q):
        out.append(F("no_valid_answer", "blocking",
                     "Forecast/prediction request: historical data alone cannot give a verifiable value."))
    # column names AND table names count as evidence the concept exists (e.g. employees.csv)
    cols = " ".join([c.lower().replace("_", " ") for df in tables.values() for c in df.columns]
                    + [n.lower().replace("_", " ") for n in tables])
    for concept in MISSING_CONCEPTS:
        if re.search(rf"\b{concept}s?\b", q) and not re.search(rf"\b{concept}", cols):
            out.append(F("no_valid_answer", "blocking",
                         f"Question needs '{concept}' data, but no such column exists in any table."))
    qyears = {int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", q)}
    dyears = data_years(tables)
    if qyears and dyears:
        missing = sorted(y for y in qyears if y not in dyears)
        if missing:
            out.append(F("no_valid_answer", "blocking",
                         f"Question asks about {missing}, but the data only covers years {sorted(dyears)}."))
    if re.search(r"\b(today|yesterday|this (week|month|year)|last (week|month|year|quarter))\b", q):
        out.append(F("relative_date", "blocking",
                     f"Relative date in question; data has no reference 'today' "
                     f"(system date is {datetime.date.today()}). Please give explicit dates."))
    return out


def is_date_question(question):
    return bool(DATE_WORDS.search(question))


def is_money_question(question, tables):
    q = question.lower()
    if any(h in q for h in MONEY_HINTS):
        return True
    return any(c.lower() in q for df in tables.values() for c in df.columns
               if any(h in c.lower() for h in MONEY_HINTS))


def referenced_columns(code, tables):
    found = set()
    for name, df in tables.items():
        for c in df.columns:
            if re.search(rf"""['"]{re.escape(c)}['"]""", code):
                found.add((name, c))
    return found
