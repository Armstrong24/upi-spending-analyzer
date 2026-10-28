"""Synthetic data: (1) a labelled training set for the ML model,
(2) a realistic 3-month UPI statement for the demo (CSV + PDF).

Run directly to regenerate the files in data/:
    python -m analyzer.data_generator
"""
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .merchants import (MERCHANTS, LOCAL_PREFIXES, LOCAL_SUFFIXES, FIRST_NAMES, LAST_NAMES,
                        UPI_HANDLES_MERCHANT, UPI_HANDLES_PERSON)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BANKS = ["YESB", "HDFC", "ICIC", "SBIN", "UTIB", "PYTM", "KKBK"]


def _slug(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _ref(rng):
    return str(rng.randint(10**11, 10**12 - 1))


def _typo(text, rng):
    """Drop or swap one character - real statements are messy."""
    if len(text) < 6:
        return text
    i = rng.randint(1, len(text) - 2)
    return text[:i] + text[i + 1:] if rng.random() < 0.5 else text[:i] + text[i + 1] + text[i] + text[i + 2:]


def describe_merchant(name, rng):
    """One UPI-style description for a merchant, in a random app format."""
    vpa = _slug(name) + rng.choice(UPI_HANDLES_MERCHANT)
    templates = [
        f"Paid to {name}",                                              # PhonePe
        f"UPI/DR/{_ref(rng)}/{name.upper()}/{rng.choice(BANKS)}/{vpa}",  # bank statement
        f"{name} - UPI Payment",
        f"Payment to {name} via UPI",                                   # Paytm
        f"To {name}",                                                   # GPay
        vpa,
        f"{name.upper()} UPI",
    ]
    return rng.choice(templates)


def describe_person(first, last, rng):
    vpa = rng.choice([f"{first.lower()}.{last.lower()}", f"{first.lower()}{rng.randint(1, 999)}",
                      f"{first.lower()}{last.lower()[0]}"]) + rng.choice(UPI_HANDLES_PERSON)
    templates = [
        f"Paid to {first} {last}",
        f"UPI/DR/{_ref(rng)}/{first.upper()} {last.upper()}/{rng.choice(BANKS)}/{vpa}",
        f"To {first} {last}",
        f"Payment to {first} {last} via UPI",
        vpa,
        f"Sent to {first}",
    ]
    return rng.choice(templates)


def build_training_data(n_per_category=300, seed=42):
    """Labelled descriptions -> category. 3,000 rows."""
    rng = random.Random(seed)
    rows = []
    for category, merchants in MERCHANTS.items():
        for _ in range(n_per_category):
            if rng.random() < 0.6:
                m = rng.choice(merchants)                        # big brand
            else:
                m = f"{rng.choice(LOCAL_PREFIXES)} {rng.choice(LOCAL_SUFFIXES[category])}"  # local shop
            d = describe_merchant(m, rng)
            if rng.random() < 0.15:
                d = _typo(d, rng)
            rows.append((d, category, m))
    for _ in range(n_per_category):
        f, l = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        rows.append((describe_person(f, l, rng), "Transfers to People", f"{f} {l}"))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["description", "category", "merchant"])


# ---------------------------------------------------------------- statement
def _t(rng, start_h, end_h):
    h = rng.randint(start_h, end_h)
    return h, rng.choice([0, 5, 12, 18, 23, 31, 37, 44, 50, 56])


def build_sample_statement(seed=7, start="2026-08-01", end="2026-10-22"):
    """A college student's PhonePe-style statement, Aug-Oct 2026, with a few
    planted problems (a duplicate payment, a 2 AM high-value payment, a scam payee)."""
    rng = random.Random(seed)
    start_d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    friends = [("Rohan", "Deshmukh"), ("Sneha", "Kulkarni"), ("Yash", "Pawar"),
               ("Tanvi", "Joshi"), ("Omkar", "Shinde"), ("Isha", "Mehta")]
    tx = []

    def add(day, hm, desc, amount, typ="DEBIT"):
        tx.append({"dt": day.replace(hour=hm[0], minute=hm[1]), "desc": desc,
                   "type": typ, "amount": round(float(amount), 2)})

    day = start_d
    jio_next = start_d + timedelta(days=2)
    while day <= end_d:
        d = day.day
        # --- income
        if d == 1:
            add(day, (9, 15), "Received from Rajesh Patil", 15000, "CREDIT")
        if d == 7:
            add(day, (11, 2), "Received from TechNova Solutions (Internship Stipend)", 8000, "CREDIT")
        # --- fixed monthly payments (subscriptions / bills)
        if d == 1:
            add(day, (18, 30), "Paid to Cult.fit", 999)
        if d == 5:
            add(day, (21, 10), "Paid to Netflix", 199)
        if d == 8:
            add(day, (10, 40), "Paid to ACT Fibernet", 799)
        if d == 12:
            add(day, (8, 5), "Paid to Spotify", 119)
        if d == 15:
            add(day, (18, 20), "Paid to MSEDCL Electricity", rng.randint(610, 940))
        if day.date() == jio_next.date():
            add(day, (13, 45), "Paid to Jio Prepaid Recharge", 299)
            jio_next += timedelta(days=28)
        # --- daily life
        if rng.random() < 0.7:
            add(day, _t(rng, 9, 18), "Paid to " + rng.choice(
                ["College Canteen", "Amruttulya Tea", "Chaayos", "Goodluck Cafe", "Vaishali Restaurant"]),
                rng.randint(30, 160))
        if rng.random() < 0.2:
            add(day, _t(rng, 19, 23), "Paid to " + rng.choice(["Swiggy", "Zomato", "Dominos Pizza"]),
                rng.randint(180, 560))
        if rng.random() < 0.15:
            add(day, _t(rng, 10, 21), "Paid to " + rng.choice(["Blinkit", "Zepto", "DMart", "Chitale Dairy"]),
                rng.randint(90, 850))
        if rng.random() < 0.4:
            add(day, _t(rng, 8, 22), "Paid to " + rng.choice(["Rapido", "Uber", "Ola", "Pune Metro"]),
                rng.randint(30, 280))
        if rng.random() < 0.05:
            add(day, _t(rng, 12, 23), "Paid to " + rng.choice(["Amazon", "Flipkart", "Myntra", "Decathlon", "Meesho"]),
                rng.randint(299, 2400))
        if rng.random() < 0.2:
            f, l = rng.choice(friends)
            add(day, _t(rng, 12, 23), f"Paid to {f} {l}", rng.choice([50, 100, 120, 150, 200, 250, 300, 450, 600]))
        if rng.random() < 0.06:
            add(day, _t(rng, 9, 17), "Paid to " + rng.choice(["Xerox Centre", "Stationery Mart", "Pragati Book Depot", "Udemy"]),
                rng.choice([40, 60, 85, 120, 349, 449]))
        if rng.random() < 0.05:
            add(day, _t(rng, 15, 22), "Paid to " + rng.choice(["BookMyShow", "PVR Cinemas", "Steam Games"]),
                rng.randint(220, 650))
        if rng.random() < 0.03:
            add(day, _t(rng, 10, 20), "Paid to " + rng.choice(["Apollo Pharmacy", "MedPlus", "Tata 1mg"]),
                rng.randint(120, 640))
        day += timedelta(days=1)

    # --- one-off but legitimate big spends
    add(datetime(2026, 8, 20), (16, 25), "Paid to MakeMyTrip", 3450)
    add(datetime(2026, 9, 3), (12, 10), "Paid to Exam Fee Portal", 1500)
    # --- planted problems for the anomaly detector
    add(datetime(2026, 9, 14), (20, 42), "Paid to Zomato", 486)
    add(datetime(2026, 9, 14), (20, 43), "Paid to Zomato", 486)                     # duplicate
    add(datetime(2026, 9, 27), (23, 58), "Paid to kyc-update-help@ybl", 4999)         # scam payee
    add(datetime(2026, 10, 18), (2, 13), "Paid to QuickDeals Online", 18999)          # 2 AM, very high

    tx.sort(key=lambda r: r["dt"])
    df = pd.DataFrame({
        "Date": [r["dt"].strftime("%Y-%m-%d") for r in tx],
        "Time": [r["dt"].strftime("%I:%M %p") for r in tx],
        "Transaction Details": [r["desc"] for r in tx],
        "UPI Transaction ID": [_ref(rng) for _ in tx],
        "Type": [r["type"] for r in tx],
        "Amount": [r["amount"] for r in tx],
    })
    return df


def write_pdf(df, path):
    """PhonePe-like PDF statement (so the PDF upload path can be demoed)."""
    from fpdf import FPDF
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Transaction Statement - Demo UPI App (synthetic data)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Account holder: Demo Student | Period: 01 Aug 2026 - 22 Oct 2026", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    widths = [30, 22, 150, 22, 40]
    headers = ["Date", "Time", "Transaction Details", "Type", "Amount"]
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(widths, headers):
        pdf.cell(w, 7, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, r in df.iterrows():
        date = datetime.strptime(r["Date"], "%Y-%m-%d").strftime("%b %d, %Y")
        vals = [date, r["Time"], str(r["Transaction Details"])[:90], r["Type"], f"INR {r['Amount']:,.2f}"]
        for w, v in zip(widths, vals):
            pdf.cell(w, 6, v, border=1)
        pdf.ln()
    pdf.output(str(path))


def main():
    DATA_DIR.mkdir(exist_ok=True)
    train = build_training_data()
    train.to_csv(DATA_DIR / "training_data.csv", index=False)
    st = build_sample_statement()
    st.to_csv(DATA_DIR / "sample_upi_statement.csv", index=False)
    write_pdf(st, DATA_DIR / "sample_upi_statement.pdf")
    print(f"training rows: {len(train)}, statement rows: {len(st)}")


if __name__ == "__main__":
    main()
