"""Spending insights: merchant names, monthly summaries, recurring payments,
50/30/20 budget and saving tips."""
import re

import pandas as pd

from .merchants import NEEDS, WANTS, INCOME_CATEGORY


def extract_merchant(description: str) -> str:
    """'UPI/DR/123/SWIGGY/YESB/swiggy@ybl' -> 'Swiggy', 'Paid to Zomato' -> 'Zomato'."""
    d = str(description).strip()
    if d.upper().startswith("UPI/"):
        parts = d.split("/")
        if len(parts) >= 4:
            return parts[3].title()
    d = re.sub(r"^(paid to|received from|payment to|sent to|to)\s+", "", d, flags=re.I)
    d = re.sub(r"\s*(via upi|- upi payment| upi)$", "", d, flags=re.I)
    d = re.sub(r"\s*\(.*\)$", "", d)
    if "@" in d and " " not in d:
        d = d.split("@")[0]
    return d.strip() or "Unknown"


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["merchant"] = out["description"].apply(extract_merchant)
    out["month"] = out["datetime"].dt.to_period("M").astype(str)
    return out


def kpis(df: pd.DataFrame) -> dict:
    deb, cre = df[df["type"] == "DEBIT"], df[df["type"] == "CREDIT"]
    months = max(df["month"].nunique(), 1)
    return {
        "total_spent": deb["amount"].sum(),
        "total_received": cre["amount"].sum(),
        "transactions": len(df),
        "avg_monthly_spend": deb["amount"].sum() / months,
        "biggest_category": deb.groupby("category")["amount"].sum().idxmax() if len(deb) else "-",
        "months": months,
    }


def monthly_by_category(df):
    deb = df[df["type"] == "DEBIT"]
    return deb.pivot_table(index="month", columns="category", values="amount", aggfunc="sum", fill_value=0)


def category_totals(df):
    deb = df[df["type"] == "DEBIT"]
    return deb.groupby("category")["amount"].sum().sort_values(ascending=False)


def top_merchants(df, n=10):
    deb = df[df["type"] == "DEBIT"]
    return (deb.groupby(["merchant", "category"])["amount"].agg(["sum", "count"])
            .sort_values("sum", ascending=False).head(n).reset_index()
            .rename(columns={"sum": "total", "count": "payments"}))


def find_recurring(df: pd.DataFrame, max_cv: float = 0.20) -> pd.DataFrame:
    """A payment is 'recurring' if the same payee is paid 3+ times across 2+ months,
    at most twice a month, roughly every 20+ days on a steady schedule,
    with a nearly fixed amount (coefficient of variation <= 20%)."""
    deb = df[df["type"] == "DEBIT"]
    rows = []
    for merchant, g in deb.groupby("merchant"):
        months = g["month"].nunique()
        if months < 2 or len(g) < 3 or len(g) / months > 2:
            continue
        mean = g["amount"].mean()
        cv = g["amount"].std(ddof=0) / mean if mean else 1
        if cv > max_cv:
            continue
        g = g.sort_values("datetime")
        gaps = g["datetime"].diff().dt.days.dropna()
        gap = gaps.median()
        if gap < 20 or gaps.std(ddof=0) / gap > 0.35:      # must repeat on a steady, monthly-ish schedule
            continue
        rows.append({
            "merchant": merchant,
            "category": g["category"].mode().iat[0],
            "typical_amount": round(mean, 0),
            "every_days": int(gap) if pd.notna(gap) else None,
            "times_paid": len(g),
            "last_paid": g["datetime"].iat[-1].date(),
            "next_expected": (g["datetime"].iat[-1] + pd.Timedelta(days=int(gap))).date() if pd.notna(gap) else None,
            "monthly_cost": round(mean * 30 / gap, 0) if pd.notna(gap) and gap else round(mean, 0),
        })
    return pd.DataFrame(rows).sort_values("monthly_cost", ascending=False) if rows else pd.DataFrame(
        columns=["merchant", "category", "typical_amount", "every_days", "times_paid",
                 "last_paid", "next_expected", "monthly_cost"])


def detect_monthly_income(df) -> float:
    cre = df[df["type"] == "CREDIT"]
    if cre.empty:
        return 0.0
    return float(cre.groupby("month")["amount"].sum().median())


def budget_503020(df: pd.DataFrame, monthly_income: float) -> pd.DataFrame:
    """Compare average monthly Needs / Wants / Savings with the 50/30/20 rule."""
    deb = df[df["type"] == "DEBIT"]
    months = max(df["month"].nunique(), 1)
    needs = deb[deb["category"].isin(NEEDS)]["amount"].sum() / months
    wants = deb[~deb["category"].isin(NEEDS)]["amount"].sum() / months
    savings = monthly_income - needs - wants
    rows = [("Needs (50%)", 0.50, needs), ("Wants (30%)", 0.30, wants), ("Savings (20%)", 0.20, savings)]
    out = pd.DataFrame([{
        "bucket": b, "target": round(monthly_income * p, 0), "actual": round(a, 0),
        "actual_pct": round(100 * a / monthly_income, 1) if monthly_income else 0.0,
    } for b, p, a in rows])
    out["status"] = [
        "OK" if out.at[0, "actual"] <= out.at[0, "target"] else "Over",
        "OK" if out.at[1, "actual"] <= out.at[1, "target"] else "Over",
        "OK" if out.at[2, "actual"] >= out.at[2, "target"] else "Below target",
    ]
    return out


def saving_tips(df, budget: pd.DataFrame, recurring: pd.DataFrame, anomalies: pd.DataFrame) -> list:
    tips = []
    deb = df[df["type"] == "DEBIT"]
    months = max(df["month"].nunique(), 1)
    total = deb["amount"].sum() or 1

    delivery = deb[deb["merchant"].str.lower().isin(["swiggy", "zomato", "dominos pizza"])]
    if len(delivery):
        m = delivery["amount"].sum() / months
        tips.append(f"Food delivery costs you about Rs {m:,.0f} a month ({len(delivery) / months:.0f} orders). "
                    f"Cooking or eating at the canteen twice a week instead could save around Rs {m * 0.3:,.0f}.")
    small = deb[deb["amount"] < 100]
    if len(small) > 20:
        tips.append(f"{len(small)} payments were under Rs 100, adding up to Rs {small['amount'].sum():,.0f}. "
                    "Small UPI payments are easy to forget: this is the 'latte factor'.")
    if len(recurring):
        tips.append(f"You have {len(recurring)} recurring payments costing about Rs {recurring['monthly_cost'].sum():,.0f} "
                    "a month. Cancel any subscription you have not used in the last month.")
    wants = budget.loc[budget["bucket"].str.startswith("Wants")].iloc[0]
    if wants["status"] == "Over":
        tips.append(f"Wants are {wants['actual_pct']}% of income (target 30%). Cutting Rs "
                    f"{wants['actual'] - wants['target']:,.0f} a month brings you back on track.")
    sav = budget.loc[budget["bucket"].str.startswith("Savings")].iloc[0]
    if sav["actual"] > 0:
        tips.append(f"Put your Rs {sav['actual']:,.0f} monthly surplus into a recurring deposit or SIP on "
                    "the day income arrives (pay yourself first).")
    else:
        tips.append("You are spending more than you receive. Set a weekly UPI spending limit in your app.")
    top = deb.groupby("category")["amount"].sum().idxmax()
    tips.append(f"Your biggest category is {top} ({100 * deb.groupby('category')['amount'].sum().max() / total:.0f}% of spending).")
    if len(anomalies) and (anomalies["risk_level"] == "High").any():
        high = anomalies.loc[anomalies["risk_level"] == "High", "amount"].sum()
        tips.append(f"The high-risk payments alone total Rs {high:,.0f}. Without them your average month "
                    f"would leave Rs {sav['actual'] + high / months:,.0f} for savings.")
    if len(anomalies):
        tips.append(f"{len(anomalies)} payments look unusual. Check them, and never approve a UPI 'collect' "
                    "request or share your UPI PIN to receive money.")
    return tips
