"""
Predictive Maintenance and Anomaly Detection in 6G-Integrated Smart Manufacturing
Thales Group - Core Analysis Pipeline
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ─── CONFIGURATION ─────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "Thales_Group_Manufacturing.csv")
OUTPUT_DIR   = BASE_DIR
CONTAMINATION   = 0.05
ROLLING_WINDOW  = 30
RISK_LOW_THRESH  = 0.35
RISK_HIGH_THRESH = 0.65
RANDOM_STATE    = 42

SENSOR_COLS = [
    "Temperature_C","Vibration_Hz","Power_Consumption_kW",
    "Network_Latency_ms","Packet_Loss_%","Quality_Control_Defect_Rate_%",
    "Production_Speed_units_per_hr","Predictive_Maintenance_Score","Error_Rate_%"
]

BASELINE_COLS = [
    "Temperature_C","Vibration_Hz","Power_Consumption_kW",
    "Error_Rate_%","Predictive_Maintenance_Score"
]

print("="*70)
print("  Predictive Maintenance & Anomaly Detection - Thales Group")
print("="*70)

# ─── STEP 1: LOADING ────────────────────────────────────────────────────────
print("\n[1/10] Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df):,} rows x {df.shape[1]} columns")

df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Timestamp"], format="%d-%m-%Y %H:%M:%S")
df = df.sort_values(["Machine_ID","DateTime"]).reset_index(drop=True)
df["Operation_Mode_Enc"] = df["Operation_Mode"].map({"Active":0,"Idle":1,"Maintenance":2})
df["Efficiency_Label"] = df["Efficiency_Status"].map({"Low":0,"Medium":1,"High":2})
print(f"  Date range : {df['DateTime'].min().date()} -> {df['DateTime'].max().date()}")
print(f"  Machines   : {df['Machine_ID'].nunique()} unique")

# ─── STEP 2: EDA ────────────────────────────────────────────────────────────
print("\n[2/10] Exploratory Data Analysis...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("EDA - Sensor Feature Distributions", fontsize=16, fontweight="bold")
colors = plt.cm.tab10.colors
for ax, col, color in zip(axes.flat, SENSOR_COLS, colors):
    ax.hist(df[col], bins=60, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.set_xlabel("Value"); ax.set_ylabel("Frequency"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(12, 9))
corr = df[SENSOR_COLS + ["Efficiency_Label"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, annot_kws={"size":8})
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  EDA plots saved.")

# ─── STEP 3: BASELINE MODELING (STRICT HISTORICAL WINDOW) ─────────────────────
print("\n[3/10] Computing per-machine rolling baselines (strictly preceding 30 observations)...")
# Using x.shift(1) ensures the current observation at time t is compared against
# the 30 preceding observations [t-30, t-1] for that machine, avoiding self-contamination.
# NOTE: Because this dataset has ~1,900–2,100 observations per machine over ~69 days,
# the median inter-observation gap is ~34 minutes, so the 30-observation window spans
# approximately 16–17 hours of machine history, not 30 minutes.
for col in BASELINE_COLS:
    roll_mean = df.groupby("Machine_ID")[col].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    # For the initial observation where no history exists, fallback to current reading
    df[f"{col}_roll_mean"] = roll_mean.fillna(df[col])

    roll_std = df.groupby("Machine_ID")[col].transform(
        lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).std()
    )
    # For early samples with insufficient history (< 2 samples), fallback to machine-level std
    machine_std = df.groupby("Machine_ID")[col].transform("std")
    df[f"{col}_roll_std"] = roll_std.fillna(machine_std)

print(f"  Rolling baselines computed with x.shift(1) (window={ROLLING_WINDOW}).")

# ─── STEP 4: FEATURE ENGINEERING ────────────────────────────────────────────
print("\n[4/10] Engineering anomaly features...")
for col in BASELINE_COLS:
    safe_std = df[f"{col}_roll_std"].replace(0, 1e-6)
    df[f"{col}_zscore"] = (df[col] - df[f"{col}_roll_mean"]) / safe_std

df["VibPow_Instability"] = df["Vibration_Hz"] / (df["Power_Consumption_kW"] + 1e-6)

df["Error_Escalation"] = df.groupby("Machine_ID")["Error_Rate_%"].transform(
    lambda x: x.rolling(10, min_periods=1).apply(
        lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=False))

df["Maint_Score_Decay"] = df.groupby("Machine_ID")["Predictive_Maintenance_Score"].transform(
    lambda x: x.rolling(10, min_periods=1).apply(
        lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0, raw=False))

df["Network_Stress"] = (
    (df["Network_Latency_ms"] / df["Network_Latency_ms"].max()) +
    (df["Packet_Loss_%"] / df["Packet_Loss_%"].max())) / 2

df["Quality_Pressure"] = df["Quality_Control_Defect_Rate_%"] * df["Error_Rate_%"] / 100

FEATURE_COLS = (
    [f"{c}_zscore" for c in BASELINE_COLS] +
    ["VibPow_Instability","Error_Escalation","Maint_Score_Decay",
     "Network_Stress","Quality_Pressure","Operation_Mode_Enc"]
)
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
print(f"  Engineered {len(FEATURE_COLS)} features.")

# ─── STEP 5: ISOLATION FOREST ───────────────────────────────────────────────
print("\n[5/10] Training Isolation Forest...")
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURE_COLS])
iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                      max_samples="auto", random_state=RANDOM_STATE, n_jobs=-1)
iso.fit(X)
df["IF_Prediction"] = iso.predict(X)
df["IF_Anomaly"] = (df["IF_Prediction"] == -1).astype(int)
df["IF_Score_Raw"] = iso.decision_function(X)
n_anom = df["IF_Anomaly"].sum()
print(f"  Anomalies detected: {n_anom:,} ({n_anom/len(df)*100:.2f}%)")
joblib.dump(iso, os.path.join(OUTPUT_DIR, "isolation_forest_model.pkl"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

# ─── STEP 5B: FEATURE IMPORTANCE (Tree Depth & Split Frequency Analysis) ──────
print("\n[5b] Computing Isolation Forest Feature Importance (Tree Depth Analysis)...")
n_features = len(FEATURE_COLS)
split_counts = np.zeros(n_features)
depth_sums = np.zeros(n_features)
depth_counts = np.zeros(n_features)
weighted_importance = np.zeros(n_features)

for estimator in iso.estimators_:
    tree = estimator.tree_
    feature = tree.feature
    children_left = tree.children_left
    children_right = tree.children_right
    
    depths = np.zeros(tree.node_count)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        if children_left[node_id] != -1:
            stack.append((children_left[node_id], depth + 1))
        if children_right[node_id] != -1:
            stack.append((children_right[node_id], depth + 1))
            
    for node_id in range(tree.node_count):
        feat = feature[node_id]
        if feat >= 0:
            d = depths[node_id]
            split_counts[feat] += 1
            depth_sums[feat] += d
            depth_counts[feat] += 1
            weighted_importance[feat] += 1.0 / (2.0 ** d)

mean_depths = np.where(depth_counts > 0, depth_sums / depth_counts, 0)
normalized_importance = (weighted_importance / weighted_importance.sum()) * 100

fi_df = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Split_Count": split_counts.astype(int),
    "Mean_Split_Depth": np.round(mean_depths, 3),
    "Depth_Weighted_Importance_%": np.round(normalized_importance, 2)
}).sort_values("Depth_Weighted_Importance_%", ascending=False).reset_index(drop=True)
fi_df["Rank"] = range(1, len(fi_df) + 1)
fi_df = fi_df[["Rank", "Feature", "Depth_Weighted_Importance_%", "Mean_Split_Depth", "Split_Count"]]
fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(fi_df["Feature"], fi_df["Depth_Weighted_Importance_%"], color="#3b82f6", edgecolor="white")
ax.set_xlabel("Depth-Weighted Importance (%)")
ax.set_title("Isolation Forest Feature Importance (Tree Depth Analysis)", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for b, v in zip(bars, fi_df["Depth_Weighted_Importance_%"]):
    ax.text(b.get_width() + 0.1, b.get_y() + b.get_height()/2, f"{v:.2f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print(fi_df.to_string(index=False))

# ─── STEP 6: NORMALIZE SCORES ───────────────────────────────────────────────
print("\n[6/10] Normalizing anomaly scores...")
raw = df["IF_Score_Raw"].values
df["Anomaly_Score"] = (raw.max() - raw) / (raw.max() - raw.min() + 1e-9)
print(f"  Score range: {df['Anomaly_Score'].min():.4f} - {df['Anomaly_Score'].max():.4f}")

# ─── STEP 7: RISK CLASSIFICATION ────────────────────────────────────────────
print("\n[7/10] Classifying maintenance risk...")
def classify_risk(s):
    if s >= RISK_HIGH_THRESH: return "High"
    elif s >= RISK_LOW_THRESH: return "Medium"
    else: return "Low"

df["Maintenance_Risk"] = df["Anomaly_Score"].apply(classify_risk)
print(df["Maintenance_Risk"].value_counts().to_string())

risk_colors = {"High":"#e74c3c","Medium":"#f39c12","Low":"#2ecc71"}
rc = df["Maintenance_Risk"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(rc.index, rc.values, color=[risk_colors[r] for r in rc.index],
                   edgecolor="white", linewidth=1.2)
axes[0].set_title("Maintenance Risk Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Record Count")
for b, v in zip(bars, rc.values):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+100, f"{v:,}", ha="center", va="bottom")
axes[1].hist(df["Anomaly_Score"], bins=80, color="#3498db", edgecolor="white", linewidth=0.3, alpha=0.85)
axes[1].axvline(RISK_LOW_THRESH, color="#f39c12", linestyle="--", linewidth=2, label=f"Medium threshold")
axes[1].axvline(RISK_HIGH_THRESH, color="#e74c3c", linestyle="--", linewidth=2, label=f"High threshold")
axes[1].set_title("Anomaly Score Distribution", fontsize=13, fontweight="bold")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "risk_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─── STEP 8: TEMPORAL ANALYSIS ──────────────────────────────────────────────
print("\n[8/10] Temporal risk escalation analysis...")
df["Hour"] = df["DateTime"].dt.floor("h")
hourly_risk = df.groupby(["Machine_ID","Hour"])["Anomaly_Score"].mean().reset_index()
hourly_risk.rename(columns={"Anomaly_Score":"Avg_Anomaly_Score"}, inplace=True)

def trend_slope(s):
    return np.polyfit(range(len(s)), s, 1)[0] if len(s) > 1 else 0.0

machine_trend = hourly_risk.groupby("Machine_ID").apply(
    lambda g: trend_slope(g["Avg_Anomaly_Score"].values)).reset_index()
machine_trend.columns = ["Machine_ID","Risk_Trend_Slope"]

machine_risk_summary = df.groupby("Machine_ID").agg(
    Avg_Anomaly_Score=("Anomaly_Score","mean"),
    Max_Anomaly_Score=("Anomaly_Score","max"),
    High_Risk_Count=("Maintenance_Risk", lambda x: (x=="High").sum()),
    Medium_Risk_Count=("Maintenance_Risk", lambda x: (x=="Medium").sum()),
    Total_Records=("Anomaly_Score","count"),
).reset_index()
machine_risk_summary["High_Risk_Rate_%"] = (
    machine_risk_summary["High_Risk_Count"] / machine_risk_summary["Total_Records"] * 100)
machine_risk_summary = machine_risk_summary.merge(machine_trend, on="Machine_ID")
machine_risk_summary = machine_risk_summary.sort_values("Avg_Anomaly_Score", ascending=False)

top10 = machine_risk_summary.head(10)
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(top10["Machine_ID"].astype(str), top10["Avg_Anomaly_Score"],
               color="#e74c3c", edgecolor="white", linewidth=0.8)
ax.set_xlabel("Average Anomaly Score")
ax.set_title("Top 10 Highest-Risk Machines", fontsize=13, fontweight="bold")
ax.invert_yaxis()
for b, v in zip(bars, top10["Avg_Anomaly_Score"]):
    ax.text(b.get_width()+0.001, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top10_risk_machines.png"), dpi=150, bbox_inches="tight")
plt.close()

# ─── STEP 9: KPIS ────────────────────────────────────────────────────────────
print("\n[9/10] Computing KPIs...")
kpi_mean_score = df["Anomaly_Score"].mean()
kpi_high_count = (df["Maintenance_Risk"]=="High").sum()
kpi_high_machines = machine_risk_summary[machine_risk_summary["High_Risk_Count"]>0]["Machine_ID"].nunique()

# Medium-to-High Risk Escalation Lead Duration:
# Measures how long a machine remains in the early-warning 'Medium' state immediately before escalating to 'High'.
# (Methodological note: This represents risk escalation runway, NOT physical failure time, as no ground-truth breakdown labels exist).
df_s = df.sort_values(["Machine_ID","DateTime"]).reset_index(drop=True)
lead_cycle_counts = []
lead_elapsed_minutes = []

for mid, grp in df_s.groupby("Machine_ID"):
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

kpi_lead_cycles = float(np.mean(lead_cycle_counts)) if lead_cycle_counts else 0.0
kpi_lead_elapsed_min = float(np.mean(lead_elapsed_minutes)) if lead_elapsed_minutes else 0.0

# Downtime Prevention Potential Proxy (Pre-Warned Anomaly Coverage Ratio):
# Evaluates the percentage of High-Risk events that exhibited an identifiable pre-escalation warning.
# (Methodological note: Represents theoretical pre-warning coverage, NOT measured downtime prevention,
# as the dataset lacks physical downtime, breakdown event, and intervention outcome logs).
kpi_dpi_proxy = (len(lead_cycle_counts) / max(kpi_high_count, 1)) * 100.0

print(f"  Mean Anomaly Score                        : {kpi_mean_score:.4f}")
print(f"  High-Risk Records                         : {kpi_high_count:,}")
print(f"  Machines with High Risk Events            : {kpi_high_machines}")
print(f"  Avg Med-to-High Escalation Cycles         : {kpi_lead_cycles:.2f} observation cycles")
print(f"  Avg Med-to-High Escalation Elapsed Time   : {kpi_lead_elapsed_min:.1f} minutes")
print(f"  Downtime Prevention Potential Proxy (DPI) : {kpi_dpi_proxy:.1f}% (Pre-warned anomaly coverage)")

# ─── STEP 10: POST-MAINTENANCE BEHAVIOR ANALYSIS (Before -> During -> After) ─
print("\n[10/10] Post-Maintenance Behavior Analysis (Before -> During -> After)...")
lifecycle_metrics = [
    "Anomaly_Score", "Temperature_C", "Vibration_Hz",
    "Error_Rate_%", "Predictive_Maintenance_Score", "Network_Stress"
]

mat = df[lifecycle_metrics].values
m_id = df["Machine_ID"].values
is_maint = (df["Operation_Mode"] == "Maintenance").values
dt_series = df["DateTime"].values

same_prev = np.r_[False, m_id[1:] == m_id[:-1]]
same_next = np.r_[m_id[:-1] == m_id[1:], False]
prev_maint = np.r_[False, is_maint[:-1]]
next_maint = np.r_[is_maint[1:], False]

starts = np.where(is_maint & (~prev_maint | ~same_prev))[0]
ends = np.where(is_maint & (~next_maint | ~same_next))[0]

W = 5  # window of records immediately before and after
episode_records = []
b_list, d_list, a_list = [], [], []

for s, e in zip(starts, ends):
    mid = int(m_id[s])
    b_start = max(0, s - W)
    a_end = min(len(df), e + 1 + W)
    if b_start < s and m_id[b_start] == mid and a_end > e + 1 and m_id[a_end - 1] == mid:
        b_mean = mat[b_start:s].mean(axis=0)
        d_mean = mat[s:e+1].mean(axis=0)
        a_mean = mat[e+1:a_end].mean(axis=0)
        
        b_list.append(b_mean)
        d_list.append(d_mean)
        a_list.append(a_mean)
        
        ep_entry = {
            "Machine_ID": mid,
            "Episode_Start": dt_series[s],
            "Episode_End": dt_series[e],
            "Duration_Records": int(e - s + 1)
        }
        for i, m in enumerate(lifecycle_metrics):
            ep_entry[f"{m}_Before"] = float(b_mean[i])
            ep_entry[f"{m}_During"] = float(d_mean[i])
            ep_entry[f"{m}_After"] = float(a_mean[i])
            ep_entry[f"{m}_Delta"] = float(a_mean[i] - b_mean[i])
        episode_records.append(ep_entry)

episodes_df = pd.DataFrame(episode_records)
episodes_df.to_csv(os.path.join(OUTPUT_DIR, "post_maintenance_analysis.csv"), index=False)

b_arr = np.array(b_list)
d_arr = np.array(d_list)
a_arr = np.array(a_list)

post_maint_summary = pd.DataFrame({
    "Metric": lifecycle_metrics,
    "Before_Maintenance": b_arr.mean(axis=0),
    "During_Maintenance": d_arr.mean(axis=0),
    "After_Maintenance": a_arr.mean(axis=0),
})
post_maint_summary["Delta_After_Before"] = post_maint_summary["After_Maintenance"] - post_maint_summary["Before_Maintenance"]
post_maint_summary["Relative_Change_%"] = (
    post_maint_summary["Delta_After_Before"] / post_maint_summary["Before_Maintenance"].abs() * 100
)
post_maint_summary.to_csv(os.path.join(OUTPUT_DIR, "post_maintenance_summary.csv"), index=False)

# ── Recovery Success Rate ──────────────────────────────────────────────────
# Definition: An episode is "successful" if the post-intervention Anomaly_Score
# (mean of the W records immediately after maintenance) is strictly lower than
# the pre-intervention Anomaly_Score (mean of the W records immediately before).
# This is a conservative, reproducible criterion derived entirely from the data.
if len(episodes_df) > 0:
    recovered = (episodes_df["Anomaly_Score_After"] < episodes_df["Anomaly_Score_Before"]).sum()
    total_ep  = len(episodes_df)
    kpi_recovery_rate = (recovered / total_ep) * 100.0
else:
    recovered, total_ep, kpi_recovery_rate = 0, 0, 0.0

print(f"  Analyzed {len(episodes_df):,} full maintenance episodes.")
print(f"  Recovery Success Rate (Anomaly_Score After < Before): {kpi_recovery_rate:.1f}%  ({recovered:,} / {total_ep:,} episodes)")
print(post_maint_summary.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Post-Maintenance Behavior: Before -> During -> After Maintenance", fontsize=15, fontweight="bold")
phase_colors = ["#3498db", "#e74c3c", "#2ecc71"]

for ax, metric in zip(axes.flat, lifecycle_metrics):
    row = post_maint_summary[post_maint_summary["Metric"] == metric].iloc[0]
    vals = [row["Before_Maintenance"], row["During_Maintenance"], row["After_Maintenance"]]
    phases = ["Before", "During", "After"]
    bars = ax.bar(phases, vals, color=phase_colors, edgecolor="white", linewidth=1)
    ax.set_title(metric.replace("_", " "), fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "post_maintenance_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved: post_maintenance_comparison.png")

# ─── SAVE OUTPUTS ─────────────────────────────────────────────────────────────
print("\n[OK] Saving core tables...")
out_cols = [
    "Date","Timestamp","DateTime","Machine_ID","Operation_Mode",
    "Temperature_C","Vibration_Hz","Power_Consumption_kW","Network_Latency_ms","Packet_Loss_%",
    "Quality_Control_Defect_Rate_%","Production_Speed_units_per_hr","Predictive_Maintenance_Score",
    "Error_Rate_%","Efficiency_Status","Anomaly_Score","IF_Anomaly","Maintenance_Risk",
    "VibPow_Instability","Error_Escalation","Maint_Score_Decay","Network_Stress","Quality_Pressure",
] + [f"{c}_zscore" for c in BASELINE_COLS]

df[out_cols].to_csv(os.path.join(OUTPUT_DIR, "anomaly_results.csv"), index=False)
machine_risk_summary.to_csv(os.path.join(OUTPUT_DIR, "machine_risk_summary.csv"), index=False)
hourly_risk.to_csv(os.path.join(OUTPUT_DIR, "hourly_risk_trends.csv"), index=False)

import json
kpis = {
    "Mean_Anomaly_Score": round(kpi_mean_score, 4),
    "High_Risk_Records": int(kpi_high_count),
    "Machines_with_High_Risk": int(kpi_high_machines),
    "Avg_Medium_to_High_Lead_Duration_cycles": round(kpi_lead_cycles, 2),
    "Avg_Medium_to_High_Lead_Elapsed_min": round(kpi_lead_elapsed_min, 1),
    "Downtime_Prevention_Potential_Proxy_pct": round(kpi_dpi_proxy, 1),
    "Total_Maintenance_Episodes": int(len(episodes_df)),
    "Recovery_Success_Rate_pct": round(kpi_recovery_rate, 1),
    "Recovery_Success_Episodes": int(recovered),
    "Methodological_Notes": {
        "Lead_Duration": "Measures early warning runway (consecutive Medium-risk states before High-risk escalation); does not represent physical breakdown time as dataset contains no failure ground-truth labels.",
        "Downtime_Prevention_Index_Proxy": f"Calculated as ratio of High-risk anomalies preceded by an early Medium warning ({kpi_dpi_proxy:.1f}%). Represents theoretical pre-warning coverage, NOT measured downtime prevention, as no breakdown or downtime logs exist in the data.",
        "Recovery_Success_Rate": f"Fraction of maintenance episodes ({recovered}/{total_ep}) where mean Anomaly_Score in the {W} records immediately after maintenance is strictly less than mean Anomaly_Score in the {W} records immediately before maintenance. Criterion is purely data-derived; no failure ground-truth labels are used."
    }
}
with open(os.path.join(OUTPUT_DIR, "kpi_report.json"), "w") as f:
    json.dump(kpis, f, indent=2)

print("="*70)
print("  Analysis COMPLETE! All outputs and post-maintenance datasets saved.")
print("="*70)
