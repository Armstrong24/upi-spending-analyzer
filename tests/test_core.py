"""Run with:  python -m pytest -q"""
import io

import pandas as pd
import pytest

import analyzer as A
from analyzer import parser, categorizer, insights, anomaly


@pytest.fixture(scope="module")
def model():
    return A.load_model()


@pytest.fixture(scope="module")
def analysed(model):
    st = parser.load_statement(str(A.DATA / "sample_upi_statement.csv"), "s.csv")
    return A.analyze(st, model)


# ---------- parser
def test_csv_and_pdf_give_same_transactions():
    c = parser.load_statement(str(A.DATA / "sample_upi_statement.csv"), "s.csv")
    p = parser.load_statement(str(A.DATA / "sample_upi_statement.pdf"), "s.pdf")
    assert len(c) == len(p) > 150
    assert c["amount"].sum() == pytest.approx(p["amount"].sum())


def test_bank_style_debit_credit_columns():
    csv = "Txn Date,Narration,Withdrawal,Deposit\n05/10/2026,UPI/DR/123456789012/SWIGGY/YESB/swiggy@ybl,250,\n" \
          "06/10/2026,Salary,,10000\n"
    df = parser.read_csv(io.StringIO(csv))
    assert list(df["type"]) == ["DEBIT", "CREDIT"]
    assert list(df["amount"]) == [250, 10000]
    assert df["datetime"].iloc[0].month == 10   # 05/10/2026 is 5 October (Indian format)


# ---------- categoriser
def test_model_accuracy_is_high(model):
    assert model.accuracy > 0.9


@pytest.mark.parametrize("text,expected", [
    ("Paid to Swiggy", "Food & Dining"),
    ("UPI/DR/412345678901/IRCTC/SBIN/irctc@sbi", "Travel & Transport"),
    ("Paid to Jio Prepaid Recharge", "Mobile Recharge"),
    ("Paid to Netflix", "Entertainment"),
    ("Paid to Rohan Deshmukh", "Transfers to People"),
    ("Paid to Shree Ganesh Medical", "Health & Fitness"),
])
def test_known_examples(model, text, expected):
    assert categorizer.predict_one(text, model)[0][0] == expected


def test_credits_are_income(analysed):
    df, _, _ = analysed
    assert (df.loc[df["type"] == "CREDIT", "category"] == "Income / Received").all()


def test_rule_fallback():
    assert categorizer.rule_category("Paid to Hotel Sagar") == "Food & Dining"
    assert categorizer.rule_category("Paid to Prathamesh Gawde") == "Transfers to People"


def test_merchant_extraction():
    assert insights.extract_merchant("UPI/DR/123/SWIGGY/YESB/swiggy@ybl") == "Swiggy"
    assert insights.extract_merchant("Paid to Zomato") == "Zomato"
    assert insights.extract_merchant("rahul.sharma@okaxis") == "rahul.sharma"


# ---------- anomalies
def test_planted_anomalies_are_found(analysed):
    _, flagged, _ = analysed
    merchants = set(flagged["merchant"])
    assert {"QuickDeals Online", "kyc-update-help", "Zomato"} <= merchants
    assert (flagged.loc[flagged["merchant"] == "QuickDeals Online", "risk_level"] == "High").all()


def test_duplicate_detection():
    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-09-14 20:42", "2026-09-14 20:43", "2026-09-15 20:43"]),
        "merchant": ["Zomato"] * 3, "amount": [486.0] * 3, "type": ["DEBIT"] * 3,
    })
    assert list(anomaly.find_duplicates(df)) == [False, True, False]


# ---------- recurring + budget
def test_subscriptions_detected(analysed):
    _, _, rec = analysed
    assert {"Netflix", "Spotify", "Jio Prepaid Recharge", "ACT Fibernet"} <= set(rec["merchant"])
    assert "Swiggy" not in set(rec["merchant"])


def test_budget_adds_up(analysed):
    df, _, _ = analysed
    b = insights.budget_503020(df, 20000)
    assert b["target"].sum() == 20000
    assert b["actual"].sum() == pytest.approx(20000, abs=2)
