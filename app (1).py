import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pymannkendall as mk
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Demand Forecasting Agent", page_icon="📦", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    div[data-testid="stMetric"] {
        background-color: #1C1F26; border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4); border: 1px solid #2A2E37;
    }
    div[data-testid="stMetricLabel"] { color: #CCCCCC; }
    div[data-testid="stMetricValue"] { color: #FAFAFA; }
    .badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-weight: 600; font-size: 13px;
    }
    .badge-red { background-color: #3A1A1A; color: #FF6B6B; }
    .badge-orange { background-color: #3A2E14; color: #FFC24B; }
    .badge-green { background-color: #14301F; color: #4ADE80; }
    .card {
        background-color: #1C1F26; border-radius: 14px; padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4); border: 1px solid #2A2E37;
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ---------- NAME MAPPINGS (for readability) ----------
PRODUCT_NAMES = [
    "Wireless Mouse", "Bluetooth Speaker", "Cotton T-Shirt", "Running Shoes", "Coffee Mug",
    "Notebook (A5)", "Desk Lamp", "Phone Case", "Water Bottle", "Backpack",
    "Wall Clock", "Sunglasses", "Yoga Mat", "Table Fan", "Ceramic Plate Set",
    "LED Bulb Pack", "Umbrella", "Wristwatch", "Hair Dryer", "Kitchen Knife Set",
    "Throw Pillow", "Laptop Sleeve", "Power Bank", "Earphones", "Board Game",
    "Bath Towel Set", "Cutting Board", "Reading Lamp", "Travel Mug", "Wall Mirror",
    "Storage Box", "Doormat", "Scented Candle", "Picture Frame", "Bookend Set",
    "Desk Organizer", "Plant Pot", "Blanket", "Cushion Cover", "Key Holder",
    "Shower Curtain", "Laundry Basket", "Wall Shelf", "Photo Album", "Alarm Clock",
    "Mixing Bowl Set", "Coasters (Set of 4)", "Spice Rack", "Door Mat", "Pen Stand"
]
STORE_NAMES = [
    "Downtown Branch", "Mall Outlet", "Highway Store", "Suburb Store", "Airport Kiosk",
    "City Center", "Riverside Branch", "Tech Park Store", "University Outlet", "Old Town Shop"
]

def product_label(item_id):
    return f"Item {item_id} — {PRODUCT_NAMES[(item_id-1) % len(PRODUCT_NAMES)]}"

def store_label(store_id):
    return f"Store {store_id} — {STORE_NAMES[(store_id-1) % len(STORE_NAMES)]}"

# ---------- FAST RISK SCAN ----------
@st.cache_data
def risk_scan(df, assumed_stock, lead_time_days):
    recent = (
        df.sort_values("date")
          .groupby(["store", "item"])["sales"]
          .apply(lambda x: x.tail(90).mean())
          .reset_index(name="avg_daily_demand")
    )
    recent["days_of_cover"] = assumed_stock / recent["avg_daily_demand"]
    recent["status"] = np.where(
        recent["days_of_cover"] < lead_time_days, "High Risk",
        np.where(recent["days_of_cover"] < lead_time_days * 1.5, "Medium Risk", "Healthy")
    )
    recent["Store"] = recent["store"].apply(store_label)
    recent["Product"] = recent["item"].apply(product_label)
    return recent.sort_values("days_of_cover")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 📦 DEMAND AI")
    st.caption("Forecasting & Inventory Agent")
    st.divider()
    page = st.radio("View", ["🏠 Risk Dashboard", "🔍 Product Deep-Dive"])
    st.divider()

    if page == "🏠 Risk Dashboard":
        assumed_stock = st.number_input("Assumed stock per product", value=150, min_value=1)
        lead_time_days_dash = st.number_input("Lead time (days)", value=7, min_value=1)
    else:
        store_id = st.selectbox("Store", sorted(df["store"].unique()), format_func=store_label)
        item_id = st.selectbox("Product", sorted(df["item"].unique()), format_func=product_label)
        lead_time_days = st.number_input("Supplier lead time (days)", value=7, min_value=1)
        current_stock = st.number_input("Current stock (units)", value=150, min_value=0)
        demand_adjust = st.slider("What-if: demand change (%)", -50, 50, 0,
                                    help="Simulate a promotion, holiday spike, or slowdown")
        st.divider()
        st.caption("EOQ inputs")
        ordering_cost = st.number_input("Ordering cost per order (₹)", value=500, min_value=1)
        holding_cost = st.number_input("Holding cost per unit/year (₹)", value=20, min_value=1)
        run = st.button("🚀 Run Forecast", type="primary", use_container_width=True)

# =========================================================
# PAGE 1: RISK DASHBOARD
# =========================================================
if page == "🏠 Risk Dashboard":
    st.markdown("## Inventory Risk Dashboard")
    st.caption("Scanning all stores and products for stockout risk, based on recent demand trends.")

    scan = risk_scan(df, assumed_stock, lead_time_days_dash)

    c1, c2, c3 = st.columns(3)
    c1.metric("Products Scanned", len(scan))
    c2.metric("High Risk", (scan["status"]=="High Risk").sum())
    c3.metric("Healthy", (scan["status"]=="Healthy").sum())

    st.write("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Top 15 Highest-Risk Products**")

    top15 = scan.head(15).copy()
    def badge(s):
        cls = {"High Risk":"badge-red","Medium Risk":"badge-orange","Healthy":"badge-green"}[s]
        return f'<span class="badge {cls}">{s}</span>'
    top15["Risk"] = top15["status"].apply(badge)
    top15_display = top15[["Store","Product","avg_daily_demand","days_of_cover","Risk"]].copy()
    top15_display.columns = ["Store","Product","Avg Daily Demand","Days of Cover","Risk"]
    top15_display["Avg Daily Demand"] = top15_display["Avg Daily Demand"].round(1)
    top15_display["Days of Cover"] = top15_display["Days of Cover"].round(1)
    st.write(top15_display.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("👉 Switch to **Product Deep-Dive** in the sidebar to forecast any of these in detail.")

# =========================================================
# PAGE 2: PRODUCT DEEP-DIVE
# =========================================================
if page == "🔍 Product Deep-Dive":
    st.markdown("## Product Deep-Dive")
    st.caption("Full forecast with uncertainty, statistical trend testing, and EOQ-based inventory recommendation.")

    if not run:
        st.info("👈 Choose a store/product in the sidebar, then click **Run Forecast**.")

    if run:
        data = df[(df["store"]==store_id) & (df["item"]==item_id)].copy()
        data = data.sort_values("date")[["date","sales"]].reset_index(drop=True)

        data["rolling_mean"] = data["sales"].rolling(7, center=True).mean()
        data["rolling_std"] = data["sales"].rolling(7, center=True).std()
        data["is_anomaly"] = abs(data["sales"]-data["rolling_mean"]) > 2*data["rolling_std"]
        data["clean_sales"] = data["sales"].where(~data["is_anomaly"], data["rolling_mean"])
        ts = data.set_index("date")["clean_sales"]

        with st.spinner("Training model, testing trend significance, and computing uncertainty..."):
            train, test = ts[:-30], ts[-30:]

            hw_fit = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=365).fit()
            hw_pred = hw_fit.forecast(30)
            hw_error = (abs(test.values - hw_pred.values)/test.values).mean()*100

            naive_pred = train[-395:-365].values if len(train) > 395 else train[-30:].values
            naive_pred = naive_pred[:len(test)]
            naive_error = (abs(test.values - naive_pred)/test.values).mean()*100 if len(naive_pred)==len(test) else 999

            if hw_error <= naive_error:
                chosen_model, chosen_error = "Holt-Winters (Trend + Seasonality)", hw_error
            else:
                chosen_model, chosen_error = "Seasonal Naive Baseline", naive_error

            final_fit = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=365).fit()
            fc_point = final_fit.forecast(30) * (1 + demand_adjust/100)

            residuals = (ts - final_fit.fittedvalues).dropna().values
            n_sims = 300
            sims = np.array([
                fc_point.values + np.random.choice(residuals, size=30, replace=True)
                for _ in range(n_sims)
            ])
            p10 = np.percentile(sims, 10, axis=0)
            p50 = np.percentile(sims, 50, axis=0)
            p90 = np.percentile(sims, 90, axis=0)

            mk_result = mk.original_test(ts.resample("M").mean())

        avg_daily = fc_point.mean()
        std_daily = fc_point.std()
        z = 1.65
        safety_stock = z*std_daily*np.sqrt(lead_time_days)
        reorder_point = avg_daily*lead_time_days + safety_stock
        days_of_cover = current_stock/avg_daily
        recommended_order_simple = max(0, (avg_daily*30+safety_stock)-current_stock)
        n_anomalies = int(data["is_anomaly"].sum())

        annual_demand = avg_daily * 365
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

        st.markdown(f"### {store_label(store_id)} · {product_label(item_id)}")

        st.markdown(
            f'<span class="badge badge-green">Model: {chosen_model} · Validated Error: {chosen_error:.1f}%</span>',
            unsafe_allow_html=True
        )
        trend_badge = "badge-green" if mk_result.h else "badge-orange"
        trend_text = f"Trend: {mk_result.trend.title()} (statistically significant, p={mk_result.p:.3f})" if mk_result.h else f"Trend: {mk_result.trend.title()} (not statistically significant, p={mk_result.p:.3f})"
        st.markdown(f'<span class="badge {trend_badge}">{trend_text}</span>', unsafe_allow_html=True)
        if demand_adjust != 0:
            st.markdown(f'<span class="badge badge-orange">What-if applied: {demand_adjust:+d}% demand</span>', unsafe_allow_html=True)
        st.write("")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📊 Avg Daily Demand", f"{avg_daily:.0f} units")
        c2.metric("📅 Days of Stock Left", f"{days_of_cover:.1f} days")
        c3.metric("🚨 Anomalies Cleaned", f"{n_anomalies}")
        c4.metric("📦 EOQ (Optimal Order)", f"{eoq:.0f} units")

        st.write("")
        badge_class = "badge-red" if days_of_cover < lead_time_days else "badge-green"
        badge_text = "⚠️ Reorder Now — Stockout Risk" if days_of_cover < lead_time_days else "✅ Healthy Stock Level"
        st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)
        st.write(""); st.write("")

        col_left, col_right = st.columns([2,1])
        with col_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**30-Day Demand Forecast — with 80% Confidence Band**")
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(ts[-180:], label="Historical", color="#4C6FFF")
            ax2.plot(fc_point.index, p50, color="#FF4B4B", label="Forecast (median)")
            ax2.fill_between(fc_point.index, p10, p90, color="#FF4B4B", alpha=0.15, label="80% interval (P10–P90)")
            ax2.legend(); ax2.spines[['top','right']].set_visible(False)
            st.pyplot(fig2)
            st.caption("Shaded band = range of likely outcomes, from bootstrapped forecast residuals — not just a single guess.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Stockout Risk Gauge**")
            risk_pct = min(100, max(0, (1 - days_of_cover/(lead_time_days*2))*100))
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_pct, number={'suffix':"%"},
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color': "#FF4B4B" if risk_pct>50 else "#1B8A45"},
                       'steps':[{'range':[0,40],'color':"#E6F6EC"},
                                {'range':[40,70],'color':"#FFF3CD"},
                                {'range':[70,100],'color':"#FDE8E8"}]}))
            gauge.update_layout(height=280, margin=dict(l=20,r=20,t=10,b=10))
            st.plotly_chart(gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Detected Anomalies**")
        at = data[data["is_anomaly"]][["date","sales","rolling_mean"]].copy()
        at.columns = ["Date","Actual Sales","Expected (avg)"]
        st.dataframe(at, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Inventory Calculation Breakdown**")
        d1,d2 = st.columns(2)
        with d1:
            st.write(f"Average daily demand: **{avg_daily:.1f} units**")
            st.write(f"Demand variability (σ): **{std_daily:.1f} units**")
            st.write(f"Supplier lead time: **{lead_time_days} days**")
            st.write(f"Safety stock (95% service level): **{safety_stock:.1f} units**")
        with d2:
            st.write(f"Reorder point: **{reorder_point:.1f} units**")
            st.write(f"Current stock: **{current_stock} units** → covers **{days_of_cover:.1f} days**")
            st.write(f"Simple 30-day order qty: **{recommended_order_simple:.1f} units**")
            st.write(f"**EOQ — economically optimal order size: {eoq:.1f} units**")
        st.caption("EOQ balances ordering cost vs. holding cost to minimize total inventory cost.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        report = pd.DataFrame([{
            "Store": store_label(store_id), "Product": product_label(item_id), "Model": chosen_model,
            "Validated Error %": round(chosen_error,1),
            "Trend": mk_result.trend, "Trend p-value": round(mk_result.p,4),
            "Avg Daily Demand": round(avg_daily,1),
            "Safety Stock": round(safety_stock,1),
            "Reorder Point": round(reorder_point,1),
            "Current Stock": current_stock,
            "Days of Cover": round(days_of_cover,1),
            "EOQ Recommended Order": round(eoq,1),
        }])
        st.download_button("⬇️ Download Recommendation (CSV)",
                            report.to_csv(index=False), file_name=f"recommendation_{store_id}_{item_id}.csv")
