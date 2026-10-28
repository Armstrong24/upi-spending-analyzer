"""AI-Based UPI Spending Analyzer  -  Streamlit front end.
Run:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import analyzer as A
from analyzer import insights, parser, categorizer, llm

st.set_page_config(page_title="UPI Spending Analyzer", page_icon="💸", layout="wide")


@st.cache_resource(show_spinner="Training the categoriser on 3,000 labelled UPI descriptions...")
def get_model():
    return A.load_model()


@st.cache_data
def load_sample():
    return parser.load_statement(str(A.DATA / "sample_upi_statement.csv"), "sample.csv")


def rs(x):
    return f"₹{x:,.0f}"


model = get_model()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1. Statement")
    up = st.file_uploader("Upload UPI statement (CSV or PDF)", type=["csv", "pdf"])
    st.caption("Exported from PhonePe / Google Pay / Paytm or your bank. "
               "No file? The built-in sample (Aug-Oct 2026) is used.")
    if up is not None:
        try:
            statement = parser.load_statement(up, up.name)
            st.success(f"Read {len(statement)} transactions from {up.name}")
        except Exception as e:  # show a friendly error, fall back to sample
            st.error(f"Could not read that file: {e}")
            statement = load_sample()
    else:
        statement = load_sample()

df, flagged, recurring = A.analyze(statement, model)
auto_income = insights.detect_monthly_income(df)

with st.sidebar:
    st.header("2. Budget")
    income = st.number_input("Monthly income / pocket money (₹)", min_value=0,
                             value=int(auto_income or 20000), step=500,
                             help="Auto-detected from money received; change it if needed.")
    st.header("3. Optional AI summary")
    api_key = st.text_input("OpenAI-compatible API key (optional)", type="password",
                            help="Leave empty: everything else works fully offline.")
    st.divider()
    st.caption(f"Model accuracy: **{model.accuracy:.1%}** on {model.n_test} held-out rows")

budget = insights.budget_503020(df, income)
tips = insights.saving_tips(df, budget, recurring, flagged)
k = insights.kpis(df)

# ------------------------------------------------------------------ header
st.title("💸 AI-Based UPI Spending Analyzer")
st.caption("FLB project · Unit 5: Current Financial Trends & Fintech · upload a UPI statement and let ML "
           "categorise, budget and flag risky payments")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total spent", rs(k["total_spent"]))
c2.metric("Money received", rs(k["total_received"]))
c3.metric("Avg spend / month", rs(k["avg_monthly_spend"]))
c4.metric("Transactions", k["transactions"])
c5.metric("Unusual payments", len(flagged), delta=f"{(flagged['risk_level'] == 'High').sum()} high risk",
          delta_color="inverse")

tabs = st.tabs(["📊 Overview", "🧾 Transactions", "🤖 AI Model", "🚨 Anomalies & Fraud",
                "🔁 Subscriptions", "🎯 Budget 50/30/20 & Tips", "📚 FLB Concepts"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    left, right = st.columns([3, 2])
    m = insights.monthly_by_category(df).reset_index().melt(id_vars="month", var_name="category", value_name="amount")
    m["month"] = pd.to_datetime(m["month"]).dt.strftime("%b %Y")
    fig = px.bar(m, x="month", y="amount", color="category", title="Monthly spending by category",
                 labels={"amount": "₹", "month": ""})
    fig.update_xaxes(type="category")
    left.plotly_chart(fig, use_container_width=True)
    ct = insights.category_totals(df).reset_index()
    right.plotly_chart(px.pie(ct, names="category", values="amount", hole=0.45, title="Where the money went"),
                       use_container_width=True)
    l2, r2 = st.columns([2, 3])
    tm = insights.top_merchants(df)
    l2.subheader("Top merchants")
    l2.dataframe(tm.style.format({"total": "₹{:,.0f}"}), hide_index=True, use_container_width=True)
    daily = df[df["type"] == "DEBIT"].set_index("datetime")["amount"].resample("W").sum().reset_index()
    r2.plotly_chart(px.line(daily, x="datetime", y="amount", markers=True, title="Weekly spending trend",
                            labels={"amount": "₹", "datetime": ""}), use_container_width=True)

# ------------------------------------------------------------------ transactions
with tabs[1]:
    st.subheader("Every transaction, categorised by the model")
    cats = st.multiselect("Filter by category", sorted(df["category"].unique()))
    view = df if not cats else df[df["category"].isin(cats)]
    st.dataframe(view[["datetime", "merchant", "description", "type", "amount", "category", "confidence", "method"]]
                 .sort_values("datetime", ascending=False),
                 hide_index=True, use_container_width=True, height=450,
                 column_config={"amount": st.column_config.NumberColumn("amount (₹)", format="%.0f"),
                                "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1)})
    st.download_button("Download categorised CSV", df.to_csv(index=False), "categorised_upi.csv", "text/csv")

# ------------------------------------------------------------------ model
with tabs[2]:
    st.subheader("How the categoriser works")
    st.markdown(
        "1. **Clean** the description (remove UPI reference numbers and bank codes).\n"
        "2. **TF-IDF on character n-grams** (2-4 letters) turns text into numbers, so `SWIGGY`, `swiggy@ybl` "
        "and `Swigy` look alike.\n"
        "3. **Logistic Regression** gives a probability for each of 10 categories.\n"
        f"4. If the top probability is below **{categorizer.CONFIDENCE_THRESHOLD:.0%}**, keyword rules decide "
        "(the fallback). Money received is always *Income*.")
    a, b, c = st.columns(3)
    a.metric("Test accuracy", f"{model.accuracy:.1%}", help="80/20 split, merchants seen in training")
    b.metric("Accuracy on brand-new merchants", f"{model.unseen_accuracy:.1%}",
             help="Whole merchants hidden from training: a harder, more honest test")
    c.metric("Training / test rows", f"{model.n_train} / {model.n_test}")
    left, right = st.columns(2)
    cm = pd.DataFrame(model.confusion, index=model.labels, columns=model.labels)
    left.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                title="Confusion matrix (test set)", labels={"x": "Predicted", "y": "Actual"}),
                      use_container_width=True)
    rep = pd.DataFrame(model.report).T.drop(["accuracy"], errors="ignore")
    right.markdown("**Per-category precision / recall**")
    right.dataframe(rep[["precision", "recall", "f1-score", "support"]].round(2), use_container_width=True)
    st.subheader("Try it live")
    txt = st.text_input("Type any UPI description", "Paid to Shree Krishna Medical")
    for cat, p in categorizer.predict_one(txt, model):
        st.progress(p, text=f"{cat}: {p:.0%}")

# ------------------------------------------------------------------ anomalies
with tabs[3]:
    st.subheader("Payments worth a second look")
    st.caption("Duplicate check · z-score on amount · late-night high value · scam words in payee · Isolation Forest")
    if flagged.empty:
        st.success("Nothing unusual found.")
    else:
        st.dataframe(flagged[["risk_level", "datetime", "merchant", "amount", "category", "reasons"]],
                     hide_index=True, use_container_width=True,
                     column_config={"amount": st.column_config.NumberColumn("amount (₹)", format="%.0f")})
        deb = df[df["type"] == "DEBIT"].copy()
        deb["flag"] = deb.index.isin(flagged.index)
        deb["hour"] = deb["datetime"].dt.hour + deb["datetime"].dt.minute / 60
        st.plotly_chart(px.scatter(deb, x="hour", y="amount", color="flag", log_y=True, hover_data=["merchant"],
                                   color_discrete_map={True: "crimson", False: "steelblue"},
                                   title="Amount vs time of day (red = flagged)",
                                   labels={"hour": "Hour of day", "amount": "₹ (log scale)"}),
                        use_container_width=True)
    st.info("**UPI safety (RBI / NPCI advice):** you never need your UPI PIN to *receive* money; don't approve "
            "unknown 'collect' requests; no bank asks for KYC updates over UPI; report fraud at 1930 or "
            "cybercrime.gov.in.")

# ------------------------------------------------------------------ recurring
with tabs[4]:
    st.subheader("Recurring payments & subscriptions")
    if recurring.empty:
        st.write("No recurring payments found.")
    else:
        st.metric("Committed every month", rs(recurring["monthly_cost"].sum()))
        st.dataframe(recurring, hide_index=True, use_container_width=True,
                     column_config={"typical_amount": st.column_config.NumberColumn("amount (₹)", format="%.0f"),
                                    "monthly_cost": st.column_config.NumberColumn("per month (₹)", format="%.0f")})
    st.caption("Rule: same payee, 3+ payments across 2+ months, roughly monthly, amount varies by 20% or less.")

# ------------------------------------------------------------------ budget
with tabs[5]:
    st.subheader(f"50/30/20 budget on {rs(income)} a month")
    left, right = st.columns([3, 2])
    bm = budget.melt(id_vars="bucket", value_vars=["target", "actual"], var_name="series", value_name="₹")
    left.plotly_chart(px.bar(bm, x="bucket", y="₹", color="series", barmode="group", labels={"bucket": "", "series": ""},
                             color_discrete_map={"target": "#9ecae1", "actual": "#08519c"},
                             title="Target vs your average month"), use_container_width=True)
    right.dataframe(budget, hide_index=True, use_container_width=True)
    right.caption("Needs: groceries, travel, bills, recharge, education, health. "
                  "Wants: eating out, shopping, entertainment, transfers to friends.")
    st.subheader("Saving tips for you")
    for t in tips:
        st.markdown(f"- {t}")
    if api_key:
        if st.button("✨ Ask the AI coach for a summary"):
            prompt = llm.build_prompt(
                {kk: (round(float(v)) if isinstance(v, (int, float)) else v) for kk, v in k.items()},
                insights.category_totals(df).round().to_dict(), budget.to_dict("records"),
                recurring[["merchant", "monthly_cost"]].to_dict("records"), len(flagged))
            try:
                st.markdown(llm.ask_llm(prompt, api_key))
            except Exception as e:
                st.warning(f"AI summary unavailable ({e}). The offline tips above still apply.")

# ------------------------------------------------------------------ FLB concepts
with tabs[6]:
    st.subheader("How each feature maps to the FLB syllabus")
    st.table(pd.DataFrame([
        ("Reading UPI statements (CSV/PDF)", "Digital payments: UPI, NPCI, payment apps", "Unit 5"),
        ("ML categorisation", "Fintech: AI/ML in personal finance apps", "Unit 5"),
        ("Monthly & category charts", "Personal financial planning, tracking expenses", "Budgeting"),
        ("50/30/20 budget", "Budgeting rules: needs vs wants vs savings", "Budgeting"),
        ("Saving tips, pay-yourself-first", "Savings habits, RD / SIP, compounding", "Savings & investment"),
        ("Recurring payments", "Subscription economy, UPI AutoPay / e-mandates", "Unit 5"),
        ("Anomaly & scam detection", "Banking fraud, cyber safety, RBI customer protection", "Unit 3 / fraud awareness"),
    ], columns=["Feature", "FLB concept", "Syllabus link"]))
