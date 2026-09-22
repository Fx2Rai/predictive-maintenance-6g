"""
Predictive Maintenance & Anomaly Detection in 6G Smart Manufacturing
Thales Group - Streamlit Dashboard
(Fully Dynamic Risk Threshold Integration)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thales | Predictive Maintenance Analytics Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0a0e1a; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1525 50%, #0a1628 100%); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #0a1525 100%);
    border-right: 1px solid rgba(59,130,246,0.2);
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(15,25,50,0.9), rgba(20,35,70,0.8));
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 16px;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(59,130,246,0.2); }
.kpi-value { font-size: 2rem; font-weight: 700; margin: 8px 0 4px 0; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 500; }
.kpi-delta { font-size: 0.85rem; margin-top: 4px; }

/* Risk badges */
.risk-high  { color: #ff4757; font-weight: 700; }
.risk-med   { color: #ffa502; font-weight: 700; }
.risk-low   { color: #2ed573; font-weight: 700; }

/* Alert cards */
.alert-card {
    background: rgba(255,71,87,0.08);
    border: 1px solid rgba(255,71,87,0.4);
    border-left: 4px solid #ff4757;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 24px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: "";
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(59,130,246,0.4), transparent);
    margin-left: 12px;
}

/* Module tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,25,50,0.6);
    border-radius: 12px;
    gap: 4px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    color: #94a3b8;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
    color: white !important;
}
h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "anomaly_results.csv"), parse_dates=["DateTime"])
    mrs = pd.read_csv(os.path.join(BASE_DIR, "machine_risk_summary.csv"))
    hrt = pd.read_csv(os.path.join(BASE_DIR, "hourly_risk_trends.csv"), parse_dates=["Hour"])
    post_maint_ep = pd.read_csv(os.path.join(BASE_DIR, "post_maintenance_analysis.csv"), parse_dates=["Episode_Start", "Episode_End"])
    post_maint_sum = pd.read_csv(os.path.join(BASE_DIR, "post_maintenance_summary.csv"))
    fi_path = os.path.join(BASE_DIR, "feature_importance.csv")
    fi_df = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    with open(os.path.join(BASE_DIR, "kpi_report.json")) as f:
        kpis = json.load(f)
    return df, mrs, hrt, post_maint_ep, post_maint_sum, kpis, fi_df

# Check if outputs exist
results_path = os.path.join(BASE_DIR, "anomaly_results.csv")
if not os.path.exists(results_path):
    st.error("⚠️ **Analysis not yet run!** Please run `python analysis.py` first to generate the anomaly results.")
    st.code("python analysis.py", language="bash")
    st.stop()

df, machine_risk_summary_static, hourly_risk, post_maint_ep, post_maint_sum, kpis_static, fi_df = load_data()

# ─── SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Thales Group")
    st.markdown("**Maintenance Analytics**")
    st.markdown("*Interactive Risk Explorer*")
    st.divider()

    st.markdown("### ⚙️ Interactive Controls")

    all_machines = sorted(df["Machine_ID"].unique().tolist())
    selected_machines = st.multiselect(
        "🔧 Machine Selector",
        options=all_machines,
        default=all_machines[:10],
        help="Select machines to analyze"
    )
    if not selected_machines:
        selected_machines = all_machines

    risk_threshold = st.slider(
        "⚠️ High-Risk Threshold",
        min_value=0.10, max_value=0.95, value=0.65, step=0.01,
        help="Dynamic anomaly score threshold to classify as High Risk"
    )
    
    # Proportional medium threshold
    med_threshold = round(risk_threshold * 0.55, 3)
    st.caption(f"⚡ **Thresholds:** High ≥ `{risk_threshold:.2f}` | Medium ≥ `{med_threshold:.2f}` | Low < `{med_threshold:.2f}`")

    date_min = df["DateTime"].min().date()
    date_max = df["DateTime"].max().date()
    date_range = st.date_input(
        "📅 Time Window",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )

    operation_modes = st.multiselect(
        "🔄 Operation Mode",
        options=["Active","Idle","Maintenance"],
        default=["Active","Idle","Maintenance"]
    )
    if not operation_modes:
        operation_modes = ["Active","Idle","Maintenance"]

    st.divider()
    st.markdown(f"""
    <div style="font-size:0.75rem; color:#64748b; line-height:1.8">
    📊 Dataset Total: <b>{len(df):,}</b><br>
    🤖 Model: <b>Isolation Forest</b><br>
    📡 Connectivity: <b>6G Real-time</b><br>
    🏭 Monitored Machines: <b>{df['Machine_ID'].nunique()}</b>
    </div>
    """, unsafe_allow_html=True)

# ─── DYNAMIC RISK COMPUTATION ─────────────────────────────────────────────────
# Apply filter criteria
if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    start_date, end_date = df["DateTime"].min(), df["DateTime"].max()

mask = (
    df["Machine_ID"].isin(selected_machines) &
    (df["DateTime"] >= start_date) & (df["DateTime"] <= end_date) &
    df["Operation_Mode"].isin(operation_modes)
)
df_filt = df[mask].copy()

# Consistently calculate dynamic risk across df_filt
def evaluate_dynamic_risk(score):
    if score >= risk_threshold:
        return "High"
    elif score >= med_threshold:
        return "Medium"
    else:
        return "Low"

# Update BOTH column names so all components dynamically update
df_filt["Maintenance_Risk"] = df_filt["Anomaly_Score"].apply(evaluate_dynamic_risk)
df_filt["Maintenance_Risk_Dynamic"] = df_filt["Maintenance_Risk"]

# Compute dynamic machine risk summary from df_filt
machine_risk_summary = df_filt.groupby("Machine_ID").agg(
    Avg_Anomaly_Score=("Anomaly_Score", "mean"),
    Max_Anomaly_Score=("Anomaly_Score", "max"),
    High_Risk_Count=("Maintenance_Risk", lambda x: (x == "High").sum()),
    Medium_Risk_Count=("Maintenance_Risk", lambda x: (x == "Medium").sum()),
    Low_Risk_Count=("Maintenance_Risk", lambda x: (x == "Low").sum()),
    Total_Records=("Anomaly_Score", "count"),
).reset_index()

machine_risk_summary["High_Risk_Rate_%"] = (
    machine_risk_summary["High_Risk_Count"] / machine_risk_summary["Total_Records"].replace(0, 1) * 100
)

# Merge with risk trend slope from precomputed trends if available
if "Risk_Trend_Slope" in machine_risk_summary_static.columns:
    machine_risk_summary = machine_risk_summary.merge(
        machine_risk_summary_static[["Machine_ID", "Risk_Trend_Slope"]],
        on="Machine_ID",
        how="left"
    )
else:
    machine_risk_summary["Risk_Trend_Slope"] = 0.0

# Sort machines by High Risk count descending, then Avg Score
machine_risk_summary = machine_risk_summary.sort_values(
    ["High_Risk_Count", "Avg_Anomaly_Score"],
    ascending=[False, False]
)

# ─── DYNAMIC KPIS ─────────────────────────────────────────────────────────────
dyn_mean_anomaly_score = df_filt["Anomaly_Score"].mean() if len(df_filt) else 0.0
dyn_high_risk_records = (df_filt["Maintenance_Risk"] == "High").sum()
dyn_high_risk_machines = machine_risk_summary[machine_risk_summary["High_Risk_Count"] > 0]["Machine_ID"].nunique()

# Medium-to-High Risk Escalation Lead Duration:
# Measures how long a machine operates in early-warning 'Medium' state immediately before escalating to 'High'.
# (Methodological Note: Represents early-warning runway before critical anomaly, NOT physical failure countdown).
df_sorted = df_filt.sort_values(["Machine_ID", "DateTime"]).reset_index(drop=True)
lead_cycle_counts = []
lead_elapsed_minutes = []

for mid, grp in df_sorted.groupby("Machine_ID"):
    grp = grp.reset_index(drop=True)
    risk_seq = grp["Maintenance_Risk"].values
    times = grp["DateTime"].values
    
    for i in range(1, len(risk_seq)):
        if risk_seq[i] == "High":
            count = 0
            j = i - 1
            while j >= 0 and risk_seq[j] == "Medium":
                count += 1
                j -= 1
            if count > 0:
                lead_cycle_counts.append(count)
                dt_first_med = pd.to_datetime(times[j + 1])
                dt_high = pd.to_datetime(times[i])
                lead_elapsed_minutes.append((dt_high - dt_first_med).total_seconds() / 60.0)

dyn_lead_cycles_avg = float(np.mean(lead_cycle_counts)) if len(lead_cycle_counts) > 0 else 0.0
dyn_lead_elapsed_avg = float(np.mean(lead_elapsed_minutes)) if len(lead_elapsed_minutes) > 0 else 0.0
dyn_dpi_proxy = (len(lead_cycle_counts) / max(dyn_high_risk_records, 1)) * 100.0

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(30,64,175,0.4), rgba(59,130,246,0.2));
     border: 1px solid rgba(59,130,246,0.3); border-radius: 20px; padding: 28px 36px; margin-bottom: 24px;">
<h1 style="margin:0; font-size:2rem; background: linear-gradient(90deg,#60a5fa,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
🏭 Predictive Maintenance Analytics Dashboard</h1>
<p style="margin:8px 0 0 0; color:#94a3b8; font-size:0.95rem;">
Interactive Predictive-Maintenance Analytics | Thales Group 6G Manufacturing Telemetry | Dynamic Anomaly & Risk Exploration
</p>
<div style="margin-top:10px; display:inline-block; background:rgba(30,58,138,0.4); border:1px solid rgba(59,130,246,0.3);
            border-radius:8px; padding:4px 12px; color:#cbd5e1; font-size:0.78rem;">
ℹ️ <b>Demonstration Platform:</b> Interactive analytics dashboard evaluating precomputed multi-sensor telemetry, out-of-sample Isolation Forest models, dynamic threshold tuning, and post-service lifecycle drill-downs.
</div>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS (FULLY DYNAMIC) ────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, "🎯", f"{dyn_mean_anomaly_score:.4f}", "Mean Anomaly Score", "Fleet-wide average health index", "#60a5fa"),
    (k2, "🚨", f"{dyn_high_risk_records:,}", f"High-Risk Events (≥{risk_threshold:.2f})", "Active alerts requiring priority inspection", "#ff4757"),
    (k3, "🏭", f"{dyn_high_risk_machines}", "High-Risk Machines", "Assets recording at least one critical alert", "#ffa502"),
    (k4, "⏱️", f"{dyn_lead_cycles_avg:.1f} cycles", "Med → High Escalation Runway", f"Observed elapsed time between risk-state transitions (~{dyn_lead_elapsed_avg:.0f} min wall-clock)", "#2ed573"),
    (k5, "🛡️", f"{dyn_dpi_proxy:.1f}%", "Pre-Warned Ratio (DPI Proxy)", "% of High-risk events preceded by a Medium alert — pre-warning coverage proxy only, not measured downtime prevented", "#a78bfa"),
]
for col, icon, value, label, subtext, color in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
        <div style="font-size:1.8rem">{icon}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-label">{label}</div>
        <div style="color:#64748b; font-size:0.75rem; margin-top:4px;">{subtext}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Maintenance Overview",
    "🔬 Machine Anomaly Dashboard",
    "🚨 Maintenance Alert Panel",
    "📈 Historical Risk Analysis"
])

color_map = {"High": "#ff4757", "Medium": "#ffa502", "Low": "#2ed573"}

# ════════════════════════════════════════════════════════
# TAB 1: PREDICTIVE MAINTENANCE OVERVIEW
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">📊 Dynamic Risk Distribution Across Machines</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        risk_counts = df_filt["Maintenance_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig_pie = px.pie(
            risk_counts, values="Count", names="Risk",
            color="Risk", color_discrete_map=color_map,
            title=f"Maintenance Risk Distribution (Threshold ≥ {risk_threshold:.2f})",
            hole=0.5
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_size=15,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label+value",
                               textfont_color="#e2e8f0")
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        top_bar_machines = machine_risk_summary.head(20)
        fig_bar = px.bar(
            top_bar_machines, x="Machine_ID", y="High_Risk_Count",
            color="High_Risk_Count", color_continuous_scale="Reds",
            title=f"High-Risk Event Count per Machine (Threshold ≥ {risk_threshold:.2f})",
            labels={"High_Risk_Count": "High-Risk Events", "Machine_ID": "Machine ID"}
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_size=15, showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", type='category'),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">📡 6G Network vs Anomaly Correlation</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        sample = df_filt.sample(min(3000, len(df_filt)), random_state=42) if len(df_filt) > 0 else df_filt
        fig_scatter = px.scatter(
            sample, x="Network_Latency_ms", y="Anomaly_Score",
            color="Maintenance_Risk", color_discrete_map=color_map,
            opacity=0.6, title="Network Latency vs Anomaly Score (Dynamic Risk)",
            size_max=6
        )
        fig_scatter.add_hline(y=risk_threshold, line_dash="dash", line_color="#ff4757",
                              annotation_text=f"High Risk ({risk_threshold:.2f})")
        fig_scatter.add_hline(y=med_threshold, line_dash="dash", line_color="#ffa502",
                              annotation_text=f"Medium ({med_threshold:.2f})")
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with c4:
        mode_risk = df_filt.groupby(["Operation_Mode", "Maintenance_Risk"]).size().reset_index(name="Count")
        fig_mode = px.bar(
            mode_risk, x="Operation_Mode", y="Count",
            color="Maintenance_Risk", color_discrete_map=color_map,
            barmode="group", title="Dynamic Risk Breakdown by Operation Mode"
        )
        fig_mode.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_mode, use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 2: MACHINE ANOMALY DASHBOARD
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔬 Anomaly Score Timeline per Machine</div>', unsafe_allow_html=True)

    machine_sel = st.selectbox(
        "Select Machine", options=selected_machines, key="anomaly_machine_sel",
        help="Choose a specific machine to inspect"
    )
    df_machine = df_filt[df_filt["Machine_ID"] == machine_sel].copy()

    c5, c6, c7 = st.columns(3)
    with c5:
        avg_sc = df_machine["Anomaly_Score"].mean() if len(df_machine) else 0.0
        badge_color = "#ff4757" if avg_sc >= risk_threshold else ("#ffa502" if avg_sc >= med_threshold else "#2ed573")
        st.markdown(f"""<div class="kpi-card"><div style="font-size:1.5rem">🎯</div>
        <div class="kpi-value" style="color:{badge_color}">{avg_sc:.4f}</div>
        <div class="kpi-label">Avg Anomaly Score</div></div>""", unsafe_allow_html=True)
    with c6:
        hr_cnt = (df_machine["Maintenance_Risk"] == "High").sum()
        st.markdown(f"""<div class="kpi-card"><div style="font-size:1.5rem">🚨</div>
        <div class="kpi-value" style="color:#ff4757">{hr_cnt:,}</div>
        <div class="kpi-label">High-Risk Events (≥ {risk_threshold:.2f})</div></div>""", unsafe_allow_html=True)
    with c7:
        pct_rate = (df_machine["Maintenance_Risk"] == "High").mean() * 100 if len(df_machine) else 0.0
        st.markdown(f"""<div class="kpi-card"><div style="font-size:1.5rem">📊</div>
        <div class="kpi-value" style="color:#ffa502">{pct_rate:.1f}%</div>
        <div class="kpi-label">High-Risk Rate</div></div>""", unsafe_allow_html=True)

    # Anomaly score timeline
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=df_machine["DateTime"], y=df_machine["Anomaly_Score"],
        mode="lines", name="Anomaly Score",
        line=dict(color="#3b82f6", width=1.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"
    ))
    # Dynamic high risk points
    hr_pts = df_machine[df_machine["Maintenance_Risk"] == "High"]
    if len(hr_pts):
        fig_timeline.add_trace(go.Scatter(
            x=hr_pts["DateTime"], y=hr_pts["Anomaly_Score"],
            mode="markers", name=f"High Risk (≥ {risk_threshold:.2f})",
            marker=dict(color="#ff4757", size=6, symbol="circle")
        ))
    # Dynamic medium risk points
    med_pts = df_machine[df_machine["Maintenance_Risk"] == "Medium"]
    if len(med_pts):
        fig_timeline.add_trace(go.Scatter(
            x=med_pts["DateTime"], y=med_pts["Anomaly_Score"],
            mode="markers", name=f"Medium Warning (≥ {med_threshold:.2f})",
            marker=dict(color="#ffa502", size=4, symbol="circle-open"),
            opacity=0.5
        ))
    fig_timeline.add_hline(y=risk_threshold, line_dash="dash", line_color="#ff4757",
                            annotation_text=f"High Risk Threshold ({risk_threshold:.2f})")
    fig_timeline.add_hline(y=med_threshold, line_dash="dash", line_color="#ffa502",
                            annotation_text=f"Medium Threshold ({med_threshold:.2f})")
    fig_timeline.update_layout(
        title=f"Machine {machine_sel} — Dynamic Anomaly Score Over Time",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=380,
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Time"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Anomaly Score", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Sensor deviation radar
    st.markdown('<div class="section-header">📡 Sensor Deviation Profile</div>', unsafe_allow_html=True)
    c8, c9 = st.columns(2)

    zscore_cols = [c + "_zscore" for c in ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Error_Rate_%", "Predictive_Maintenance_Score"]]
    labels = ["Temperature", "Vibration", "Power", "Error Rate", "Maint Score"]

    with c8:
        if len(df_machine) and all(c in df_machine.columns for c in zscore_cols):
            avg_dev = df_machine[zscore_cols].mean().values
            fig_radar = go.Figure(go.Scatterpolar(
                r=np.abs(avg_dev).tolist() + [np.abs(avg_dev[0])],
                theta=labels + [labels[0]],
                fill="toself", name="Avg |Z-Score|",
                line_color="#3b82f6", fillcolor="rgba(59,130,246,0.2)"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.1)"),
                           bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                title=f"Machine {machine_sel} — Sensor Deviation (|Z-Score|)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with c9:
        sensor_box_cols = ["Temperature_C", "Vibration_Hz", "Power_Consumption_kW", "Error_Rate_%", "Network_Latency_ms"]
        fig_box = go.Figure()
        sample_pool = df_filt.sample(min(5000, len(df_filt)), random_state=42) if len(df_filt) > 0 else df_filt
        for col in sensor_box_cols:
            if col in sample_pool.columns:
                fig_box.add_trace(go.Box(
                    y=sample_pool[col], name=col.replace("_", " ").replace(" C", "(°C)"),
                    marker_color="#3b82f6", opacity=0.7
                ))
        fig_box.update_layout(
            title="Sensor Distribution Across Selected Machines",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", showlegend=False,
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ─── FEATURE IMPORTANCE SECTION ─────────────────────────────
    if fi_df is not None and len(fi_df) > 0:
        st.markdown("---")
        st.markdown('<div class="section-header">🧠 Isolation Forest Feature Importance (Tree-Depth & Split Analysis)</div>', unsafe_allow_html=True)
        c_fi1, c_fi2 = st.columns([3, 2])
        with c_fi1:
            fi_sorted = fi_df.sort_values("Depth_Weighted_Importance_%", ascending=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_sorted["Depth_Weighted_Importance_%"],
                y=fi_sorted["Feature"],
                orientation="h",
                marker=dict(
                    color=fi_sorted["Depth_Weighted_Importance_%"],
                    colorscale="Blues",
                    line=dict(color="rgba(255,255,255,0.2)", width=1)
                ),
                text=[f"{v:.2f}%" for v in fi_sorted["Depth_Weighted_Importance_%"]],
                textposition="auto"
            ))
            fig_fi.update_layout(
                title="Empirical Depth-Weighted Importance across 200 Trees",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(title="Depth-Weighted Importance (%)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        with c_fi2:
            st.markdown("#### 📊 Empirical Tree Depth Rankings")
            st.dataframe(
                fi_df.rename(columns={
                    "Depth_Weighted_Importance_%": "Importance (%)",
                    "Mean_Split_Depth": "Mean Depth",
                    "Split_Count": "Splits"
                }),
                use_container_width=True, hide_index=True, height=380
            )
            st.caption("Weighting formula: splits are weighted inversely by tree depth ($2^{-\\text{depth}}$) across all 200 estimators. Features partitioning anomalies near root nodes achieve higher discriminative power.")

# ════════════════════════════════════════════════════════
# TAB 3: MAINTENANCE ALERT PANEL
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🚨 Dynamic High-Risk Maintenance Alerts</div>', unsafe_allow_html=True)

    hr_machines = machine_risk_summary[
        machine_risk_summary["Machine_ID"].isin(selected_machines)
    ].copy()

    # Alert cards for top 5 machines with highest High_Risk_Count
    st.markdown(f"#### 🔴 Priority Maintenance Targets (Threshold ≥ {risk_threshold:.2f})")
    top5 = hr_machines.head(5)
    alert_cols = st.columns(min(len(top5), 5) if len(top5) > 0 else 1)
    
    if len(top5) > 0:
        for i, (_, row) in enumerate(top5.iterrows()):
            has_high = row["High_Risk_Count"] > 0
            risk_badge = "HIGH" if has_high else ("MEDIUM" if row["Medium_Risk_Count"] > 0 else "LOW")
            color = "#ff4757" if risk_badge == "HIGH" else ("#ffa502" if risk_badge == "MEDIUM" else "#2ed573")
            
            with alert_cols[i % len(alert_cols)]:
                st.markdown(f"""
                <div style="background:rgba({('255,71,87' if risk_badge=='HIGH' else ('255,165,2' if risk_badge=='MEDIUM' else '46,213,115'))},0.1);
                     border:1px solid {color}; border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:1.8rem">⚠️</div>
                <div style="font-size:1.3rem; font-weight:700; color:{color}">Machine {int(row['Machine_ID'])}</div>
                <div style="color:#94a3b8; font-size:0.78rem; margin:4px 0">Avg Score</div>
                <div style="font-size:1.1rem; font-weight:600; color:#e2e8f0">{row['Avg_Anomaly_Score']:.4f}</div>
                <div style="color:#94a3b8; font-size:0.78rem; margin-top:6px">High Events: <b>{int(row['High_Risk_Count']):,}</b></div>
                <div style="background:{color}; border-radius:6px; padding:4px 8px; margin-top:8px;
                            font-size:0.75rem; font-weight:600; color:white">{risk_badge} RISK</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No machines match the selected filter criteria.")

    st.markdown("---")
    st.markdown(f"#### 📋 Dynamic Inspection Priority Ranking (Threshold: {risk_threshold:.2f})")

    display_cols = ["Machine_ID", "Avg_Anomaly_Score", "Max_Anomaly_Score",
                    "High_Risk_Count", "Medium_Risk_Count", "High_Risk_Rate_%", "Risk_Trend_Slope"]
    display_df = hr_machines[display_cols].copy()
    display_df["Avg_Anomaly_Score"] = display_df["Avg_Anomaly_Score"].round(4)
    display_df["Max_Anomaly_Score"] = display_df["Max_Anomaly_Score"].round(4)
    display_df["High_Risk_Rate_%"]  = display_df["High_Risk_Rate_%"].round(2)
    display_df["Risk_Trend_Slope"]  = display_df["Risk_Trend_Slope"].round(6)
    display_df["Priority"] = range(1, len(display_df) + 1)
    display_df = display_df[["Priority"] + display_cols]

    st.dataframe(
        display_df.rename(columns={
            "Machine_ID": "Machine", "Avg_Anomaly_Score": "Avg Score",
            "Max_Anomaly_Score": "Max Score", "High_Risk_Count": "High Risk Events",
            "Medium_Risk_Count": "Medium Events",
            "High_Risk_Rate_%": "High Risk %", "Risk_Trend_Slope": "Trend Slope"
        }),
        use_container_width=True, height=420, hide_index=True
    )

    st.markdown("---")
    st.markdown(f"#### 📊 Recent High-Risk Events (Anomaly Score ≥ {risk_threshold:.2f})")
    recent_hr = df_filt[df_filt["Maintenance_Risk"] == "High"].sort_values("DateTime", ascending=False).head(50)
    if len(recent_hr):
        show_cols = ["DateTime", "Machine_ID", "Operation_Mode", "Temperature_C",
                     "Vibration_Hz", "Error_Rate_%", "Anomaly_Score", "Maintenance_Risk"]
        st.dataframe(recent_hr[show_cols].round(3), use_container_width=True, height=320, hide_index=True)
    else:
        st.info(f"No high-risk events detected under threshold {risk_threshold:.2f}. Try lowering the risk threshold slider in the sidebar.")

# ════════════════════════════════════════════════════════
# TAB 4: HISTORICAL RISK ANALYSIS
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">📈 Risk Escalation Timelines</div>', unsafe_allow_html=True)

    c10, c11 = st.columns(2)

    with c10:
        # Daily average anomaly score and high risk event counts
        df_filt["Date"] = df_filt["DateTime"].dt.date
        daily_avg = df_filt.groupby("Date").agg(
            mean_score=("Anomaly_Score", "mean"),
            max_score=("Anomaly_Score", "max"),
            high_count=("Maintenance_Risk", lambda x: (x == "High").sum())
        ).reset_index()

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_avg["Date"], y=daily_avg["mean_score"],
            mode="lines+markers", name="Daily Avg Score",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=6)
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily_avg["Date"], y=daily_avg["max_score"],
            mode="lines", name="Daily Max Score",
            line=dict(color="#ff4757", width=1.5, dash="dot")
        ))
        fig_daily.add_hline(y=risk_threshold, line_dash="dash", line_color="#ff4757",
                            annotation_text=f"Threshold ({risk_threshold:.2f})")
        fig_daily.update_layout(
            title=f"Daily Anomaly Score Trend (Active High Threshold: {risk_threshold:.2f})",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Anomaly Score")
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with c11:
        # Hourly risk heatmap (machines x hour-of-day)
        df_filt["HourOfDay"] = df_filt["DateTime"].dt.hour
        heatmap_data = df_filt.groupby(["Machine_ID", "HourOfDay"])["Anomaly_Score"].mean().reset_index()
        pivot = heatmap_data.pivot(index="Machine_ID", columns="HourOfDay", values="Anomaly_Score")
        pivot = pivot.reindex(sorted(selected_machines)[:20])
        fig_heat = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            title="Anomaly Score Heatmap (Machine × Hour of Day)",
            labels=dict(x="Hour of Day", y="Machine ID", color="Avg Score"),
            aspect="auto"
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-header">📊 Multi-Machine Risk Timeline Comparison</div>', unsafe_allow_html=True)

    compare_machines = st.multiselect(
        "Select machines to compare",
        options=selected_machines,
        default=selected_machines[:5],
        key="compare_sel"
    )

    if compare_machines:
        hr_filt = hourly_risk[hourly_risk["Machine_ID"].isin(compare_machines)].copy()
        hr_filt = hr_filt[(hr_filt["Hour"] >= start_date) & (hr_filt["Hour"] <= end_date)]
        fig_multi = px.line(
            hr_filt, x="Hour", y="Avg_Anomaly_Score",
            color="Machine_ID", title="Hourly Anomaly Score — Multi-Machine Comparison",
            labels={"Avg_Anomaly_Score": "Avg Anomaly Score", "Hour": "Time", "Machine_ID": "Machine"}
        )
        fig_multi.add_hline(y=risk_threshold, line_dash="dash", line_color="#ff4757",
                            annotation_text=f"High Risk ({risk_threshold:.2f})")
        fig_multi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=400,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)")
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # ACTUAL POST-MAINTENANCE BEHAVIOR ANALYSIS (LIFECYCLE)
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🔄 Post-Maintenance Behavior Comparison: Before → During → After Maintenance</div>', unsafe_allow_html=True)
    st.markdown("""
    This analysis evaluates equipment performance across three operational phases:
    **Before Maintenance** (pre-service operating baseline) $\\rightarrow$ **Maintenance Event** (servicing downtime) $\\rightarrow$ **After Maintenance** (post-service recovery window).
    """)

    # Filter post-maintenance dataset by machine selection
    pm_scope = post_maint_ep[post_maint_ep["Machine_ID"].isin(selected_machines)].copy()

    maint_scope_choice = st.radio(
        "Maintenance Lifecycle Scope:",
        options=["Fleet Summary (Selected Machines)", "Single Machine Drill-down"],
        horizontal=True
    )

    if maint_scope_choice == "Single Machine Drill-down":
        chosen_maint_machine = st.selectbox("Select Machine for Post-Maintenance Drill-down:", options=selected_machines, key="pm_mach_sel")
        active_pm_episodes = pm_scope[pm_scope["Machine_ID"] == chosen_maint_machine]
    else:
        chosen_maint_machine = None
        active_pm_episodes = pm_scope

    lifecycle_metric_keys = [
        ("Anomaly_Score", "Anomaly Score", ""),
        ("Temperature_C", "Temperature", "°C"),
        ("Vibration_Hz", "Vibration", "Hz"),
        ("Error_Rate_%", "Error Rate", "%"),
        ("Predictive_Maintenance_Score", "Maintenance Readiness Score", ""),
        ("Network_Stress", "Network Stress Index", "")
    ]

    if len(active_pm_episodes) > 0:
        b_means = [active_pm_episodes[f"{m}_Before"].mean() for m, _, _ in lifecycle_metric_keys]
        d_means = [active_pm_episodes[f"{m}_During"].mean() for m, _, _ in lifecycle_metric_keys]
        a_means = [active_pm_episodes[f"{m}_After"].mean() for m, _, _ in lifecycle_metric_keys]

        # 3 Lifecycle Funnel Phase Cards for Anomaly Score
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
                <div style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase;">1. Before Maintenance</div>
                <div class="kpi-value" style="color:#60a5fa;">{b_means[0]:.4f}</div>
                <div class="kpi-label">Pre-Service Baseline Score</div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid #ff4757;">
                <div style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase;">2. Maintenance Event</div>
                <div class="kpi-value" style="color:#ff4757;">{d_means[0]:.4f}</div>
                <div class="kpi-label">During Intervention Score</div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            delta_val = a_means[0] - b_means[0]
            delta_color = "#2ed573" if delta_val <= 0 else "#ffa502"
            st.markdown(f"""
            <div class="kpi-card" style="border-left: 4px solid #2ed573;">
                <div style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase;">3. After Maintenance</div>
                <div class="kpi-value" style="color:{delta_color};">{a_means[0]:.4f}</div>
                <div class="kpi-label">Net Delta: {delta_val:+.4f} (Post Recovery)</div>
            </div>
            """, unsafe_allow_html=True)

        # Multi-Metric Comparison Grouped Bar Chart
        c12, c13 = st.columns([1.2, 0.8])

        with c12:
            chart_data = []
            for i, (m, name, unit) in enumerate(lifecycle_metric_keys):
                chart_data.append({"Metric": f"{name} {unit}".strip(), "Phase": "Before Maintenance", "Value": b_means[i]})
                chart_data.append({"Metric": f"{name} {unit}".strip(), "Phase": "Maintenance Event", "Value": d_means[i]})
                chart_data.append({"Metric": f"{name} {unit}".strip(), "Phase": "After Maintenance", "Value": a_means[i]})

            df_chart = pd.DataFrame(chart_data)

            fig_pm_bars = px.bar(
                df_chart, x="Metric", y="Value", color="Phase",
                barmode="group",
                title=f"Multi-Metric Lifecycle Comparison ({'Machine ' + str(chosen_maint_machine) if chosen_maint_machine else 'Selected Fleet'})",
                color_discrete_map={
                    "Before Maintenance": "#3b82f6",
                    "Maintenance Event": "#ff4757",
                    "After Maintenance": "#2ed573"
                }
            )
            fig_pm_bars.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickangle=-20),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_pm_bars, use_container_width=True)

        with c13:
            pm_table_rows = []
            for i, (m, name, unit) in enumerate(lifecycle_metric_keys):
                b = b_means[i]
                d = d_means[i]
                a = a_means[i]
                delta = a - b
                pct_chg = (delta / (abs(b) + 1e-9)) * 100
                pm_table_rows.append({
                    "Metric": f"{name} ({unit})" if unit else name,
                    "Before": round(b, 3),
                    "During": round(d, 3),
                    "After": round(a, 3),
                    "Delta": round(delta, 4),
                    "% Change": f"{pct_chg:+.2f}%"
                })

            st.markdown("##### 📋 Metric Shift Summary (After vs Before)")
            st.dataframe(pd.DataFrame(pm_table_rows), use_container_width=True, hide_index=True)

        # Individual Maintenance Episodes Explorer
        st.markdown(f"##### 🔍 Individual Maintenance Episodes Explorer ({len(active_pm_episodes):,} total episodes)")

        ep_display = active_pm_episodes[[
            "Machine_ID", "Episode_Start", "Episode_End", "Duration_Records",
            "Anomaly_Score_Before", "Anomaly_Score_During", "Anomaly_Score_After", "Anomaly_Score_Delta",
            "Temperature_C_Before", "Temperature_C_After",
            "Vibration_Hz_Before", "Vibration_Hz_After",
            "Error_Rate_%_Before", "Error_Rate_%_After"
        ]].sort_values("Episode_Start", ascending=False).head(50)

        st.dataframe(
            ep_display.rename(columns={
                "Machine_ID": "Machine",
                "Duration_Records": "Duration (min)",
                "Anomaly_Score_Before": "Score (Before)",
                "Anomaly_Score_During": "Score (During)",
                "Anomaly_Score_After": "Score (After)",
                "Anomaly_Score_Delta": "Score Delta",
                "Temperature_C_Before": "Temp Before (°C)",
                "Temperature_C_After": "Temp After (°C)",
                "Vibration_Hz_Before": "Vib Before (Hz)",
                "Vibration_Hz_After": "Vib After (Hz)",
                "Error_Rate_%_Before": "Error Before (%)",
                "Error_Rate_%_After": "Error After (%)"
            }).round(3),
            use_container_width=True,
            height=300,
            hide_index=True
        )
    else:
        st.info("No maintenance episodes found in the selected machines and time window.")

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.8rem; padding:16px 0">
    🏭 Thales Group | Predictive Maintenance & Anomaly Detection | 6G Smart Manufacturing |
    Interactive Dynamic Risk Engine
</div>
""", unsafe_allow_html=True)
