"""AI part 2: unusual / suspicious payment detection (fraud awareness).

Four simple checks, each one easy to explain:
 1. Duplicate payment   - same payee + same amount within 10 minutes
 2. Z-score             - amount far above your normal spend (log scale, z > 2.5)
 3. Late-night + high   - Rs 2,000 or more between 12 AM and 5 AM
 4. Scam words          - payee name has words like 'kyc', 'refund', 'lottery'
plus an Isolation Forest (unsupervised ML) that looks at amount + time together.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .merchants import SCAM_WORDS

Z_THRESHOLD = 2.5
NIGHT_AMOUNT = 2000
DUP_MINUTES = 10


def find_duplicates(df: pd.DataFrame) -> pd.Series:
    d = df.sort_values("datetime")
    flag = pd.Series(False, index=df.index)
    for _, g in d[d["type"] == "DEBIT"].groupby(["merchant", "amount"]):
        if len(g) < 2:
            continue
        gaps = g["datetime"].diff().dt.total_seconds().div(60)
        flag.loc[g.index[gaps <= DUP_MINUTES]] = True
    return flag


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Return the debit rows with at least one flag, plus reasons and a risk score."""
    deb = df[df["type"] == "DEBIT"].copy()
    if deb.empty:
        return deb.assign(reasons=[], risk=[])

    log_amt = np.log1p(deb["amount"])
    deb["z_score"] = ((log_amt - log_amt.mean()) / (log_amt.std(ddof=0) or 1)).round(2)
    hour = deb["datetime"].dt.hour + deb["datetime"].dt.minute / 60

    # Isolation Forest on [amount, time of day as a circle]
    feats = np.column_stack([log_amt, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])
    iso = IsolationForest(contamination=contamination, random_state=42).fit(feats)
    deb["iforest_outlier"] = iso.predict(feats) == -1

    dup = find_duplicates(df).loc[deb.index]
    night = (deb["datetime"].dt.hour < 5) & (deb["amount"] >= NIGHT_AMOUNT)
    scam = deb["description"].str.lower().apply(lambda t: any(w in t for w in SCAM_WORDS))

    reasons = []
    for i in deb.index:
        r = []
        if dup[i]:
            r.append("Possible duplicate payment")
        if deb.at[i, "z_score"] > Z_THRESHOLD:
            r.append(f"Unusually high amount (z = {deb.at[i, 'z_score']})")
        if night[i]:
            r.append("High-value payment late at night")
        if scam[i]:
            r.append("Payee name looks like a UPI scam")
        if deb.at[i, "iforest_outlier"]:
            r.append("Isolation Forest outlier")
        reasons.append(r)
    deb["reasons"] = reasons
    # each reason = 1 point; duplicates and scam-like payees are serious, so +1 extra
    deb["risk"] = deb["reasons"].apply(lambda r: len(r) + sum(x.startswith(("Possible duplicate", "Payee name")) for x in r))
    flagged = deb[deb["risk"] > 0].sort_values(["risk", "amount"], ascending=False)
    flagged["reasons"] = flagged["reasons"].apply(", ".join)
    flagged["risk_level"] = flagged["risk"].map(lambda r: "High" if r >= 3 else "Medium" if r == 2 else "Low")
    return flagged
