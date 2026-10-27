"""Olist customer experience & retention dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8985"

st.set_page_config(page_title="Olist Customer Dashboard", page_icon="🛒", layout="wide")

ACTIONS = {
    "Repeat – active": "Reward and retain: loyalty perks, early access.",
    "Repeat – lapsed": "Win-back email with a reminder of past purchases.",
    "New – high value": "Second-purchase offer within 30 days.",
    "New": "Welcome flow, delivery follow-up, review request.",
    "Recent one-timers": "Recommendations before they go cold; fix delivery problems.",
    "Lapsed – high value": "Strongest win-back offer: most revenue at stake.",
    "Lapsed one-timers": "Low-cost email reactivation only.",
}

REQUIRED = ["segment_summary.csv", "cluster_profile.csv", "customer_segments.csv",
            "delay_vs_bad_review.csv", "bad_review_lift.csv", "bad_review_importance.csv",
            "churn_lift.csv", "bad_review_metrics.json",
            "bad_review_model_A.joblib", "bad_review_model_B.joblib"]
missing = [f for f in REQUIRED if not (ART / f).exists()]
if missing:
    st.error(f"Missing files in {ART}: {missing}. Run the notebooks that save them first.")
    st.stop()


@st.cache_data
def load_tables():
    t = {
        "segments": pd.read_csv(ART / "segment_summary.csv"),
        "clusters": pd.read_csv(ART / "cluster_profile.csv"),
        "customers": pd.read_csv(ART / "customer_segments.csv"),
        "delay": pd.read_csv(ART / "delay_vs_bad_review.csv"),
        "br_lift": pd.read_csv(ART / "bad_review_lift.csv"),
        "br_imp": pd.read_csv(ART / "bad_review_importance.csv"),
        "churn_lift": pd.read_csv(ART / "churn_lift.csv"),
    }
    t["metrics"] = json.loads((ART / "bad_review_metrics.json").read_text())
    return t


@st.cache_resource
def load_model(name):
    return joblib.load(ART / name)


def style(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), title_x=0,
                      showlegend=False)
    return fig


T = load_tables()
cust, seg = T["customers"], T["segments"]

st.sidebar.title("🛒 Olist dashboard")
page = st.sidebar.radio("Page", ["Overview", "Segments", "Delivery & reviews",
                                 "Bad-review risk calculator", "Repeat-purchase model"])
st.sidebar.caption("Data: Olist Brazilian e-commerce, 2017-01 to 2018-08, loaded from PostgreSQL.")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Customer overview")
    c = st.columns(4)
    c[0].metric("Customers", f"{len(cust):,}")
    c[1].metric("Revenue", f"{cust['monetary'].sum():,.0f}")
    c[2].metric("Repeat customers", f"{(cust['frequency'] > 1).mean():.1%}")
    c[3].metric("Average review", f"{cust['avg_review'].mean():.2f}")

    a, b = st.columns(2)
    s = seg.sort_values("pct_revenue")
    a.plotly_chart(style(px.bar(s, x="pct_revenue", y="segment", orientation="h",
                                title="Share of revenue by segment (%)",
                                color_discrete_sequence=[ORANGE]).update_yaxes(title=None)
                         .update_xaxes(title=None)), width="stretch")
    k = T["clusters"].sort_values("customers")
    b.plotly_chart(style(px.bar(k, x="customers", y="cluster_name", orientation="h",
                                title="Customers by behavioural cluster",
                                color_discrete_sequence=[BLUE]).update_yaxes(title=None)
                         .update_xaxes(title=None)), width="stretch")
    st.info("Only a small share of customers ever order twice, so most growth comes from "
            "first-time buyers. Delivery delays are the clearest driver of unhappy customers.")

# ---------------------------------------------------------------- Segments
elif page == "Segments":
    st.title("Customer segments")
    a, b = st.columns(2)
    s = seg.sort_values("pct_customers")
    a.plotly_chart(style(px.bar(s, x="pct_customers", y="segment", orientation="h",
                                title="% of customers",
                                color_discrete_sequence=[BLUE]).update_yaxes(title=None)
                         .update_xaxes(title=None)), width="stretch")
    s = seg.sort_values("pct_revenue")
    b.plotly_chart(style(px.bar(s, x="pct_revenue", y="segment", orientation="h",
                                title="% of revenue",
                                color_discrete_sequence=[ORANGE]).update_yaxes(title=None)
                         .update_xaxes(title=None)), width="stretch")

    t = seg.copy()
    t["recommended_action"] = t["segment"].map(ACTIONS)
    st.dataframe(t.sort_values("revenue", ascending=False), hide_index=True, width="stretch",
                 column_config={
                     "revenue": st.column_config.NumberColumn(format="%.0f"),
                     "avg_order_value": st.column_config.NumberColumn("avg order value", format="%.0f"),
                     "median_recency": st.column_config.NumberColumn("median recency (days)", format="%.0f"),
                     "avg_review": st.column_config.NumberColumn("avg review", format="%.2f"),
                     "late_share": st.column_config.NumberColumn("late share", format="%.2f"),
                     "pct_customers": st.column_config.NumberColumn("% customers", format="%.1f"),
                     "pct_revenue": st.column_config.NumberColumn("% revenue", format="%.1f"),
                 })

    st.subheader("Behavioural clusters (K-Means)")
    st.dataframe(T["clusters"], hide_index=True, width="stretch")

    st.subheader("Customer explorer")
    f1, f2, f3 = st.columns(3)
    segs = f1.multiselect("Segment", sorted(cust["segment"].unique()),
                          default=["Lapsed – high value"])
    states = f2.multiselect("State", sorted(cust["state"].dropna().unique()))
    n = f3.slider("Rows", 10, 500, 50, 10)
    d = cust[cust["segment"].isin(segs)]
    if states:
        d = d[d["state"].isin(states)]
    d = d.sort_values("monetary", ascending=False).head(n)
    st.dataframe(d, hide_index=True, width="stretch")
    st.download_button("Download this list (CSV)", d.to_csv(index=False).encode(),
                       "customers_selection.csv", "text/csv")

# --------------------------------------------------------- Delivery & reviews
elif page == "Delivery & reviews":
    m = T["metrics"]
    st.title("Delivery and customer satisfaction")
    c = st.columns(4)
    c[0].metric("Bad-review rate (1–2 stars)", f"{m['base_rate']:.1%}")
    c[1].metric("Model B ROC-AUC", f"{m['model_B']['roc_auc']:.3f}")
    c[2].metric("Model B PR-AUC", f"{m['model_B']['pr_auc']:.3f}",
                f"{m['model_B']['pr_auc'] / m['base_rate']:.1f}× base rate", delta_color="off")
    c[3].metric("Model A PR-AUC (order time)", f"{m['model_A']['pr_auc']:.3f}",
                f"{m['model_A']['pr_auc'] / m['base_rate']:.1f}× base rate", delta_color="off")

    a, b = st.columns(2)
    d = T["delay"]
    a.plotly_chart(style(px.bar(d, x="delay_bucket", y="bad_rate",
                                title="Bad-review rate by delivery vs estimate",
                                color_discrete_sequence=[ORANGE],
                                category_orders={"delay_bucket": list(d["delay_bucket"])})
                         .update_yaxes(tickformat=".0%", title=None)
                         .update_xaxes(title=None)), width="stretch")
    lf = T["br_lift"]
    b.plotly_chart(style(px.bar(lf, x="decile", y="rate",
                                title="Actual bad-review rate by predicted-risk decile",
                                color_discrete_sequence=[BLUE])
                         .update_yaxes(tickformat=".0%", title=None)
                         .add_hline(y=m["base_rate"], line_dash="dash", line_color=GRAY)),
                 width="stretch")
    imp = T["br_imp"].sort_values("importance")
    st.plotly_chart(style(px.bar(imp, x="importance", y="feature", orientation="h",
                                 title="What drives bad reviews (permutation importance)",
                                 color_discrete_sequence=[BLUE]).update_yaxes(title=None), 420),
                    width="stretch")
    st.caption("Model B uses the delivery outcome, so it explains dissatisfaction after the fact. "
               "Model A uses only information known at order time and works as an early warning.")

# ------------------------------------------------- Bad-review risk calculator
elif page == "Bad-review risk calculator":
    st.title("Bad-review risk calculator")
    st.write("Estimate the chance an order gets a 1–2 star review. Model A uses order-time "
             "information only; tick the box to add the actual delivery time (model B).")
    use_b = st.checkbox("Include actual delivery outcome (model B)")
    bundle = load_model("bad_review_model_B.joblib" if use_b else "bad_review_model_A.joblib")
    oh = bundle["model"].named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    cats = {k: [x for x in v if x != "missing"]
            for k, v in zip(["payment_type", "region", "category"], oh.categories_)}

    c1, c2, c3 = st.columns(3)
    price = c1.number_input("Items value", 1.0, 5000.0, 100.0)
    freight = c1.number_input("Freight value", 0.0, 500.0, 20.0)
    n_items = c1.slider("Items in order", 1, 8, 1)
    n_sellers = c1.slider("Sellers", 1, 5, 1)
    weight = c2.number_input("Average product weight (g)", 50, 30000, 1500)
    photos = c2.slider("Average product photos", 1, 10, 2)
    desc_len = c2.number_input("Average description length", 0, 4000, 700)
    installments = c2.slider("Installments", 1, 24, 2)
    n_pay = c3.slider("Number of payments", 1, 5, 1)
    voucher = c3.checkbox("Voucher used")
    planned = c3.slider("Estimated delivery time (days)", 3, 60, 24)
    approval = c3.slider("Hours until payment approved", 0, 120, 10)
    month = c3.slider("Purchase month", 1, 12, 6)
    dow = c3.slider("Day of week (0 = Monday)", 0, 6, 2)
    bf = c3.checkbox("Black Friday week")
    d1, d2, d3 = st.columns(3)
    ptype = d1.selectbox("Payment type", cats["payment_type"])
    region = d2.selectbox("Region", cats["region"])
    category = d3.selectbox("Category", cats["category"])

    row = {
        "log_items_value": np.log1p(price), "freight_value": freight,
        "freight_ratio": freight / price, "n_items": n_items, "n_sellers": n_sellers,
        "avg_weight_g": weight, "avg_photos": photos, "avg_desc_len": desc_len,
        "installments": installments, "n_payments": n_pay, "used_voucher": int(voucher),
        "planned_days": planned, "approval_hours": approval, "purchase_month": month,
        "purchase_dow": dow, "black_friday": int(bf),
        "payment_type": ptype, "region": region, "category": category,
    }
    if use_b:
        delivered = st.slider("Actual delivery time (days)", 1, 90, planned)
        row["delivery_days"] = delivered
        row["days_early"] = planned - delivered
        row["is_late"] = int(delivered > planned)

    X = pd.DataFrame([row])[bundle["features"]]
    p = float(bundle["model"].predict_proba(X)[0, 1])
    base = T["metrics"]["base_rate"]
    k1, k2 = st.columns(2)
    k1.metric("Probability of a 1–2 star review", f"{p:.1%}")
    k2.metric("Compared with average order", f"{p / base:.1f}×", f"base rate {base:.1%}",
              delta_color="off")

# ----------------------------------------------------- Repeat-purchase model
else:
    st.title("Repeat-purchase model")
    cl = T["churn_lift"]
    st.write("Predicts whether a customer places a second order 1–180 days after the first, "
             "using first-order information only. Same-day second orders were removed "
             "because they are split shopping sessions, not real returns.")
    c = st.columns(3)
    c[0].metric("Top-decile lift", f"{cl.loc[0, 'lift']:.2f}×")
    c[1].metric("Bottom-decile lift", f"{cl.loc[len(cl) - 1, 'lift']:.2f}×")
    c[2].metric("Repeaters in top 20%", f"{cl.loc[1, 'cum_capture']:.1%}")
    base = cl["repeaters"].sum() / cl["customers"].sum()
    st.plotly_chart(style(px.bar(cl, x="decile", y="rate",
                                 title="Actual repeat rate by predicted decile (cross-validated)",
                                 color_discrete_sequence=[BLUE])
                          .update_yaxes(tickformat=".1%", title=None)
                          .add_hline(y=base, line_dash="dash", line_color=GRAY)),
                    width="stretch")
    st.warning("The signal is weak (cross-validated ROC-AUC about 0.57). The model ranks "
               "customers better than chance but is not precise enough for individual targeting.")