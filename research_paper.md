# Predictive Maintenance Risk and Anomaly Detection in 6G-Integrated Smart Manufacturing Systems

*An Unsupervised Data-Driven Framework for Smart Manufacturing Maintenance Risk Assessment*

**Author:** Fabian Biju  
**Project Framework reference:** Unified Mentor Analytics Series  
**Industry reference:** Thales Group

---

## Abstract

This paper presents an unsupervised machine learning architecture for predictive maintenance risk assessment and anomaly detection in 6G-enabled smart manufacturing environments. Operating on Thales Group's industrial sensor dataset (100,000 records across 50 machines), the system addresses the operational reality that industrial telemetry streams rarely contain confirmed breakdown labels. Rather than attempting supervised prognostic time-to-failure prediction on unlabelled breakdown events, our framework implements an end-to-end unsupervised pipeline: $\text{Sensor Telemetry} \to \text{Per-Machine Rolling Baselines} \to \text{Domain Feature Engineering} \to \text{Isolation Forest} \to \text{Continuous Anomaly Scoring} \to \text{Dynamic Maintenance Risk Classification}$. Utilizing strictly preceding historical baselines ($\text{shift}(1)$ rolling windows over 30 preceding observations to prevent self-contamination), the system isolates 1,130 high-risk maintenance records across the fleet, establishes an average pre-escalation warning runway of 1.48 observation cycles (~73.1 minutes measured wall-clock elapsed time) of Medium-risk operation prior to High-risk escalation, and demonstrates a 33.4% Downtime Prevention Potential Proxy (pre-warned anomaly coverage ratio). We provide an empirical tree-depth feature importance ranking across all 200 estimators, evaluate 8,895 post-maintenance recovery episodes, and articulate the rigorous methodological boundary between operational anomaly detection and supervised failure prognostics.

---

## 1. Introduction

Modern smart manufacturing systems generate large volumes of machine and operational data — including temperature, vibration, power consumption, network latency, packet loss, quality-control defect rates, production speed, maintenance readiness score, and operational error rate. However, identifying abnormal machine behavior early from these multiple, correlated data streams is a non-trivial challenge.

### 1.1 Problem Context

The following inter-related problems motivate this work:

1. **Unexpected machine breakdowns:** Machine failures cause unplanned interruptions to manufacturing operations, with cascading effects on throughput, delivery schedules, and safety.

2. **High maintenance and downtime costs:** When abnormal behavior is not detected early, maintenance becomes reactive rather than preventive, significantly increasing corrective intervention costs.

3. **Difficulty identifying early warning signals:** Small deviations in sensor and operational measurements may precede serious failures, but are difficult to recognize within high-dimensional, noisy telemetry streams.

4. **Non-uniform machine behavior:** The dataset spans 50 distinct machines across three operating modes — Active, Idle, and Maintenance. A sensor reading that is normal for one machine or operating mode may represent abnormal behavior for another, making global fixed-threshold rules unreliable.

5. **Limitations of fixed threshold rules:** Simple univariate rules (e.g., "temperature above $X$ is abnormal") fail to capture anomalies that manifest only as co-deviations across multiple variables — a single sensor in isolation may remain within acceptable bounds while the machine's overall multivariate operating state has shifted significantly.

6. **Need to detect multidimensional anomalies:** Abnormal behavior is frequently encoded in the joint distribution of temperature, vibration, power, network, quality, production, maintenance, and error characteristics — not in any single variable alone.

7. **Need for maintenance-risk monitoring:** Beyond detection, the system must translate continuous anomaly scores into actionable Low / Medium / High risk classifications, track how risk evolves over time per machine, and identify assets requiring immediate attention.

### 1.2 Precise Problem Statement

This project addresses the challenge of **detecting and monitoring abnormal behavior in 6G-enabled smart manufacturing machines using multidimensional sensor and operational data**. Because machine behavior varies across machines and operating modes, conventional fixed-threshold monitoring does not reliably identify subtle anomalies. The proposed system therefore establishes machine-specific behavioral baselines from preceding historical observations, engineers sensor-deviation and trend features that capture multivariate shifts, and applies unsupervised anomaly detection to translate abnormal patterns into maintenance-risk levels for early intervention.

### 1.3 Methodological Distinction: Anomaly Detection & Risk Scoring vs. Failure Prognostics

A fundamental methodological distinction must be established between **unsupervised anomaly detection / risk scoring** and **supervised failure prediction**:

- **Supervised Failure Prediction (Prognostics):**
  $$\text{Current Sensor Data} \longrightarrow \text{Future Failure Prediction} \longrightarrow \text{Failure Probability} \longrightarrow \text{Time-to-Failure (RUL)}$$
  This approach requires confirmed historical failure labels (ground-truth records of catastrophic seizures, component fractures, or forced shutdowns). When such positive breakdown events are absent from the dataset, supervised diagnostic and prognostic metrics—such as **Precision, Recall, F1-Score, ROC-AUC, true failure prediction accuracy, and empirical time-to-failure lead time**—cannot be scientifically calculated or claimed.

- **Implemented Architecture (Predictive Maintenance Risk & Anomaly Detection):**
  $$\text{Sensor Data} \longrightarrow \text{Machine-Specific Baseline} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Isolation Forest} \longrightarrow \text{Anomaly Score} \longrightarrow \text{Maintenance Risk}$$
  Because the Thales manufacturing dataset contains continuous sensor readings, efficiency status grades, and operational modes without labeled machine breakdown events, this project is purposefully and rigorously designed as an **unsupervised anomaly detection and maintenance risk escalation system**. It quantifies how severely a machine departs from its nominal operating profile, alerting maintenance crews to high-risk anomalous behavior well before physical degradation results in operational disruption.

---

## 2. Dataset Description

### 2.1 Data Overview

| Property | Value |
|---|---|
| Total Records | 100,000 |
| Unique Machines | 50 (Machine IDs 1–50) |
| Date Range | January 1 – March 10, 2025 |
| Sampling Interval | Variable (~34 min median per machine; 100,000 records across 50 machines over 69 days) |
| Missing Values | None |
| Features | 14 raw + 11 engineered |

### 2.2 Feature Descriptions

| Column | Description | Type |
|---|---|---|
| Date / Timestamp | Recording datetime | Temporal |
| Machine_ID | Unique machine identifier (1–50) | Categorical |
| Operation_Mode | Active / Idle / Maintenance | Categorical |
| Temperature_C | Machine temperature (°C), range: 30–90 | Continuous |
| Vibration_Hz | Vibration frequency (Hz), range: 0.1–5.0 | Continuous |
| Power_Consumption_kW | Electrical power (kW), range: 1.5–10.0 | Continuous |
| Network_Latency_ms | 6G network latency (ms), range: 1–50 | Continuous |
| Packet_Loss_% | 6G packet loss (%), range: 0–5 | Continuous |
| Quality_Control_Defect_Rate_% | Defect rate (%), range: 0–10 | Continuous |
| Production_Speed_units_per_hr | Throughput (units/hr), range: 50–500 | Continuous |
| Predictive_Maintenance_Score | AI maintenance readiness (0–1) | Continuous |
| Error_Rate_% | Operational error rate (%), range: 0–15 | Continuous |
| Efficiency_Status | Target label: Low / Medium / High | Categorical |

### 2.3 Class Distribution

- **Low Efficiency:** 77,825 (77.8%)
- **Medium Efficiency:** 19,189 (19.2%)
- **High Efficiency:** 2,986 (3.0%)

The strong class imbalance toward "Low" efficiency underscores that most records represent suboptimal but non-critical conditions — making unsupervised anomaly detection more appropriate than classification.

### 2.4 Operation Mode Distribution

- **Active:** 70,054 (70.1%)
- **Idle:** 20,057 (20.1%)
- **Maintenance:** 9,889 (9.9%)

---

## 3. Exploratory Data Analysis

### 3.1 Univariate Analysis

All nine sensor features follow approximately uniform distributions, indicating synthetic generation with controlled ranges. Key observations:

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| Temperature_C | 60.04 | 17.32 | 30.0 | 90.0 |
| Vibration_Hz | 2.55 | 1.41 | 0.1 | 5.0 |
| Power_Consumption_kW | 5.75 | 2.45 | 1.5 | 10.0 |
| Network_Latency_ms | 25.56 | 14.12 | 1.0 | 50.0 |
| Packet_Loss_% | 2.49 | 1.44 | 0.0 | 5.0 |
| Defect_Rate_% | 5.01 | 2.88 | 0.0 | 10.0 |
| Production_Speed | 275.92 | 130.10 | 50.0 | 500.0 |
| Maint_Score | 0.499 | 0.289 | 0.0 | 1.0 |
| Error_Rate_% | 7.50 | 4.34 | 0.0 | 15.0 |

### 3.2 Correlation Analysis

Empirical correlation analysis reveals two crucial structural properties of the manufacturing dataset:

1. **Near-Zero Linear Dependency Among Sensor Variables:**
   Pairwise correlations among physical and network telemetry streams (`Temperature_C`, `Vibration_Hz`, `Power_Consumption_kW`, `Network_Latency_ms`, `Packet_Loss_%`) are virtually zero ($|r| < 0.008$ across all pairs). This absence of simple collinearity confirms that sensor channels capture orthogonal phenomena, rendering simple linear regression or thresholding ineffective and motivating multidimensional tree-based anomaly detection (Isolation Forest).

2. **Divergence Between Operational Efficiency and Physical Machine Health:**
   Evaluating linear correlations against the target label `Efficiency_Label` (encoded as Low: 0, Medium: 1, High: 2) demonstrates a sharp divergence:

| Feature | Pearson Correlation ($r$) with `Efficiency_Label` | Nature of Relationship |
|---|---|---|
| **`Error_Rate_%`** | **$-0.6037$** | Strong inverse linear driver of efficiency |
| **`Production_Speed_units_per_hr`** | **$+0.3336$** | Moderate positive linear driver of throughput |
| `Quality_Control_Defect_Rate_%` | $-0.0059$ | Near zero |
| `Predictive_Maintenance_Score` | $-0.0034$ | Near zero |
| `Network_Latency_ms` | $-0.0016$ | Near zero |
| `Vibration_Hz` | $+0.0003$ | Near zero |
| `Packet_Loss_%` | $+0.0010$ | Near zero |
| `Power_Consumption_kW` | $+0.0012$ | Near zero |
| `Temperature_C` | $+0.0038$ | Near zero |

**Strategic Analytical Insight:**
While current efficiency is heavily dictated by operational throughput and error rates ($r = -0.604$ and $+0.334$), physical hardware telemetry (temperature, vibration, power, and network degradation) exhibits negligible correlation with instantaneous efficiency status ($|r| < 0.006$). This empirically validates the core premise of this architecture: **monitoring operational efficiency alone cannot detect developing mechanical risk**. Subtle mechanical deterioration and network stress develop independently of current production throughput, requiring an unsupervised, sensor-driven predictive maintenance anomaly detection approach rather than retrospective efficiency classification.

### 3.3 Machine-Level Variability

Grouping by `Machine_ID` reveals per-machine behavioral differences in baseline temperature (±5°C), vibration amplitude, and error escalation rates — validating the need for per-machine baseline modeling rather than a global threshold approach.

---

## 4. Methodology

### 4.1 Data Preprocessing

1. **Datetime Parsing:** Combined Date + Timestamp into a single DateTime column
2. **Sorting:** Sorted by Machine_ID and DateTime to preserve temporal structure
3. **Encoding:** Operation_Mode → {Active:0, Idle:1, Maintenance:2}

### 4.2 Baseline Behavior Modeling (Strictly Preceding Historical Window)

For each of 5 key telemetry streams (`Temperature_C`, `Vibration_Hz`, `Power_Consumption_kW`, `Error_Rate_%`, `Predictive_Maintenance_Score`), we compute dynamic, machine-specific rolling baselines using strictly preceding historical observations via `x.shift(1).rolling(30)`:
- **Rolling Historical Mean ($\mu_{m,t}$)**: Evaluated over the preceding $W = 30$ observations $[t-30, t-1]$ per machine.
- **Rolling Historical Standard Deviation ($\sigma_{m,t}$)**: Evaluated over the preceding $W = 30$ observations $[t-30, t-1]$ per machine.

> **Note on temporal span:** The dataset contains approximately 1,900–2,100 observations per machine across 69 days (~34-minute median inter-observation gap). Therefore the 30-observation baseline window spans approximately **16–17 hours of machine history** in the median case, not 30 minutes. The window is thus better characterized as a *preceding-observation count* window rather than a fixed-duration time window.

**Methodological Advantage Over Concurrent Baselines:**
Standard rolling window implementations that include the current observation $x_t$ suffer from *self-contamination*: an acute sensor surge at time $t$ inflates both the rolling mean and variance of its own reference baseline, suppressing the resulting $z$-score and diminishing detection sensitivity. By enforcing an out-of-sample temporal shift ($\text{shift}(1)$):
1. The baseline represents strictly prior expected machine behavior: $\mathbb{E}[x_{m,t} \mid x_{m, t-1}, \dots, x_{m, t-W}]$.
2. The current sensor reading $x_{m,t}$ is evaluated purely out-of-sample, matching true streaming edge telemetry where concurrent spikes cannot be permitted to redefine normal behavior.
3. Early observations with insufficient historical depth ($< 2$ samples) fall back gracefully to the asset's overall sensor standard deviation prior.

### 4.3 Feature Engineering

We engineered 11 features:

| Feature | Formula | Rationale |
|---|---|---|
| `*_zscore` (×5) | (x − roll_mean) / roll_std | Deviation from machine's own baseline |
| `VibPow_Instability` | Vibration_Hz / Power_kW | Mechanical instability indicator |
| `Error_Escalation` | Rolling slope of Error_Rate (10-observation window) | Rising error trend signal |
| `Maint_Score_Decay` | Rolling slope of Maint_Score (10-observation window) | Deteriorating maintenance readiness |
| `Network_Stress` | (Latency/max + PktLoss/max) / 2 | Composite 6G network health |
| `Quality_Pressure` | Defect_Rate × Error_Rate / 100 | Combined quality degradation |
| `Operation_Mode_Enc` | Encoded mode | Context for normal behavior range |

### 4.4 Anomaly Detection: Isolation Forest

**Algorithm:** Isolation Forest (Liu et al., 2008)

**Why Isolation Forest?**
- Works without labeled anomaly data (fully unsupervised)
- Effective on high-dimensional, non-linear data
- Scales efficiently to 100,000 records
- Naturally handles multivariate anomalies (combinations of slightly elevated features)

**Hyperparameters:**
- `n_estimators = 200` (more trees → more stable scores)
- `contamination = 0.05` (5% expected anomaly rate)
- `max_samples = auto` (sub-sampling for efficiency)
- `random_state = 42` (reproducibility)

**How it works:** Trees are built by randomly partitioning features. Points that require fewer splits to isolate are anomalies — they occupy sparse regions in feature space.

### 4.5 Anomaly Scoring

Raw Isolation Forest decision function scores are negated and min-max normalized to [0, 1]:

```
Anomaly_Score = (max_raw - score_raw) / (max_raw - min_raw)
```

- Score = 0.0 → Completely normal behavior
- Score = 1.0 → Maximum anomaly severity

### 4.6 Risk Classification

Scores are thresholded into three risk categories:

| Risk Level | Threshold | Action |
|---|---|---|
| **Low** | Score < 0.35 | Normal operation, routine monitoring |
| **Medium** | 0.35 ≤ Score < 0.65 | Early warning, schedule inspection |
| **High** | Score ≥ 0.65 | Urgent maintenance required |

---

## 5. Results

### 5.1 Anomaly Detection Results

| Metric | Value |
|---|---|
| Total records analyzed | 100,000 |
| Anomalies detected | 5,000 (5.0%) |
| High-Risk records | 1,130 (1.13%) |
| Medium-Risk records | 28,836 (28.84%) |
| Low-Risk records | 70,034 (70.03%) |

### 5.2 Key Performance Indicators

| KPI | Value | Operational Interpretation |
|---|---|---|
| **Mean Anomaly Score** (all records) | 0.2905 | Overall fleet operates in nominal health range |
| **Total High-Risk Records** | 1,130 (1.13%) | Acute anomalous operating conditions |
| **Machines with High-Risk Events** | 50 / 50 | All assets experience periodic risk excursions |
| **Avg Med-to-High Risk Escalation Runway** | **1.48 obs. cycles; ~73.1 min elapsed** | Observed wall-clock time between first Medium-risk observation and subsequent High-risk breach (datetime difference between risk-state transitions, not time-to-failure) |
| **DPI Proxy (Pre-Warned Ratio)** | **33.4%** | Critical anomalies preceded by early Medium alert |

> [!IMPORTANT]
> **Methodological Clarifications on Escalation Runway and Downtime Prevention Index:**
> 1. **No Failure Ground-Truth:** The dataset does not record physical machine breakdown or catastrophic destruction timestamps. Hence, the "escalation runway" represents the **Medium-to-High Risk Escalation Runway** (the operational period during which early-warning Medium Risk telemetry precedes a critical High Risk breach), rather than true prognostic time-to-failure.
> 2. **DPI as a Prevention Potential Proxy:** The Downtime Prevention Index ($33.4\%$) measures the proportion of High-Risk anomalies that exhibited prior Medium warning. It does not measure empirically prevented downtime hours or confirmed avoided failures, but rather the theoretical coverage ceiling of anomalies that are actionable prior to critical threshold crossing.

### 5.3 Top 10 Highest-Risk Machines

| Rank | Machine ID | Avg Anomaly Score | High-Risk Events | Trend Slope |
|---|---|---|---|---|
| 1 | 9 | 0.2971 | 28 | +4.09e-07 (escalating) |
| 2 | 37 | 0.2960 | 24 | -4.13e-06 (recovering) |
| 3 | 32 | 0.2953 | 34 | +1.09e-06 (escalating) |
| 4 | 7 | 0.2944 | 21 | -1.87e-06 (recovering) |
| 5 | 47 | 0.2942 | 26 | -5.24e-06 (recovering) |
| 6 | 33 | 0.2940 | 30 | +1.34e-06 (escalating) |
| 7 | 39 | 0.2935 | 19 | -3.74e-07 (recovering) |
| 8 | 24 | 0.2931 | 23 | -6.16e-06 (recovering) |
| 9 | 8 | 0.2928 | 25 | -1.78e-05 (recovering) |
| 10 | 27 | 0.2927 | 23 | +5.71e-06 (escalating) |

**Priority alert:** Machines 9, 32, 33, and 27 show positive (escalating) risk trend slopes — they require immediate inspection.

### 5.4 Temporal Analysis

- Risk events are distributed across all hours of the day with no strong time-of-day bias.
- Machine risk escalation follows gradual multi-hour drifts punctuated by acute mode transitions.

### 5.5 Post-Maintenance Behavior Lifecycle Analysis

To evaluate the operational effectiveness of maintenance interventions, we tracked machine performance across 8,895 distinct maintenance episodes, comparing the pre-service operating window (**Before Maintenance**), the servicing downtime window (**During Maintenance**), and the recovery window (**After Maintenance**):

| Metric | Before Maintenance | During Maintenance | After Maintenance | Net Delta (After − Before) | Relative Change (%) |
|---|---|---|---|---|---|
| **Anomaly Score** | 0.2873 | 0.4560 | 0.2870 | −0.0003 | −0.12% |
| **Temperature (°C)** | 60.0412 | 59.9983 | 59.9935 | −0.0476 | −0.08% |
| **Vibration (Hz)** | 2.5436 | 2.5572 | 2.5464 | +0.0028 | +0.11% |
| **Error Rate (%)** | 7.5000 | 7.5406 | 7.5075 | +0.0075 | +0.10% |
| **Maint Readiness Score** | 0.4985 | 0.4976 | 0.4990 | +0.0006 | +0.12% |
| **Network Stress Index** | 0.5063 | 0.5031 | 0.5052 | −0.0011 | −0.22% |

**Key Findings:**
1. **Intervention Spike:** During maintenance servicing, the anomaly score spikes from $0.2873$ to $0.4560$ ($+58.8\%$), reflecting diagnostic stress testing, off-nominal motor cycling, and sensor re-calibration routines — an expected artifact of active servicing recorded in the telemetry.
2. **Post-Service Return to Baseline:** Across all six metrics, After-Maintenance values are within $\pm 0.22\%$ of Before-Maintenance values. This demonstrates that maintenance interventions do not introduce lasting degradation and that the fleet returns to its pre-service operating envelope; it does not, however, constitute measurable improvement above the pre-service baseline given the small magnitude of change.
3. **Recovery Success:** A per-episode recovery criterion is applied: an episode is classified as *successful* if the mean Anomaly Score across the $W = 5$ records immediately following maintenance is strictly lower than the mean across the $W = 5$ records immediately preceding it. The computed recovery success rate across all analyzed episodes is reported in the supplementary `kpi_report.json` output (field: `Recovery_Success_Rate_pct`). This criterion is purely data-derived; no failure ground-truth labels are required.

---

## 6. Discussion

### 6.1 6G Network Telemetry Contribution

Empirical evaluation reveals that raw network metrics show essentially zero linear correlation with overall anomaly scores:
- `Network_Latency_ms` $\leftrightarrow$ `Anomaly_Score`: $r = +0.0043$
- `Packet_Loss_%` $\leftrightarrow$ `Anomaly_Score`: $r = +0.0022$
- `Network_Stress` $\leftrightarrow$ `Anomaly_Score`: $r = +0.0046$

**Analytical Clarification:**
These near-zero linear coefficients demonstrate that 6G network degradation does not directly induce or linearly track mechanical machine anomalies. Instead, within the Isolation Forest model:
1. **Multivariate Subspace Partitioning:** Network telemetry operates as an orthogonal contextual feature. While network stress alone does not trigger high-risk alerts, rare co-occurrences of elevated latency/packet loss with mechanical vibrations or error spikes are isolated earlier by random tree partitions.
2. **Operational Transport Enabler:** The true industrial contribution of 6G in this architecture is not causal anomaly prediction, but rather its role as a **designed high-throughput, low-latency transport layer**. The ITU-R IMT-2030 framework specifies sub-millisecond latency targets for 6G industrial use cases — a design envelope that would support streaming high-frequency multi-sensor telemetry to edge compute nodes and enabling real-time anomaly inference without edge-to-cloud bottlenecks. This project's architecture is explicitly designed for that 6G-enabled deployment context; the network latency values in the dataset (`Network_Latency_ms`) are synthetic sensor readings and do not represent a measured 6G network deployment.

### 6.2 Anomaly Feature Importance (Tree Depth & Split Analysis)

To empirically quantify feature importance within the unsupervised ensemble, we traversed all 200 decision trees in the fitted Isolation Forest ($T = 200$, total internal split nodes $> 15,000$). Each feature's importance is computed by weighting splits by their depth $d$:
$$\text{Importance}(f) = \frac{\sum_{t \in \text{Trees}} \sum_{n \in \text{Nodes}(t), \text{feat}(n)=f} 2^{-\text{depth}(n)}}{\sum_{\text{all splits}} 2^{-\text{depth}}}$$

Features that isolate anomalous records near the root of the tree (shallower split depths $d$) contribute disproportionately to the anomaly score:

| Rank | Feature Name | Depth-Weighted Importance (%) | Mean Split Depth | Total Split Count |
|---|---|---|---|---|
| 1 | **`Network_Stress`** | **10.47%** | 5.216 | 1,389 |
| 2 | **`Power_Consumption_kW_zscore`** | **10.15%** | 5.204 | 1,440 |
| 3 | **`Predictive_Maintenance_Score_zscore`** | **9.38%** | 5.264 | 1,366 |
| 4 | **`VibPow_Instability`** | **9.35%** | 5.243 | 1,388 |
| 5 | **`Error_Escalation`** | **9.28%** | 5.288 | 1,396 |
| 6 | **`Quality_Pressure`** | **9.26%** | 5.254 | 1,416 |
| 7 | **`Vibration_Hz_zscore`** | **9.12%** | 5.243 | 1,378 |
| 8 | **`Maint_Score_Decay`** | **9.01%** | 5.297 | 1,413 |
| 9 | **`Error_Rate_%_zscore`** | **8.87%** | 5.290 | 1,396 |
| 10 | **`Temperature_C_zscore`** | **8.40%** | 5.314 | 1,386 |
| 11 | **`Operation_Mode_Enc`** | **6.72%** | 5.100 | 884 |

**Key Observations:**
- The top 5 discriminative features (`Network_Stress`, `Power_Consumption_kW_zscore`, `Predictive_Maintenance_Score_zscore`, `VibPow_Instability`, and `Error_Escalation`) account for nearly **$48.6\%$** of all anomaly isolation power.
- Shifting the rolling baseline by 1 (`shift(1)`) significantly enhances the discriminative power of baseline deviations (e.g., `Power_Consumption_kW_zscore` increased to $10.15\%$ importance) by preventing acute anomalies from dampening their own $z$-scores.
- The relatively balanced distribution across the top 10 features (ranging between $8.4\%$ and $10.5\%$) confirms that industrial anomalies are genuinely **multidimensional** — isolated not by any single rogue sensor, but by compound deviations across power, vibration, error rates, and communication telemetry.

### 6.3 Methodological Boundaries & Limitations

1. **Unsupervised Anomaly Detection vs. Supervised Failure Prediction:**
   A paramount scientific boundary of this study is that it implements **unsupervised anomaly detection and operational risk scoring**, not supervised failure forecasting. The architecture executes:
   $$\text{Sensor Telemetry} \longrightarrow \text{Machine-Specific Baseline} \longrightarrow \text{Feature Engineering} \longrightarrow \text{Isolation Forest} \longrightarrow \text{Anomaly Score} \longrightarrow \text{Maintenance Risk}$$
   It explicitly does not and cannot execute:
   $$\text{Current Sensor Data} \longrightarrow \text{Future Failure Prediction} \longrightarrow \text{Failure Probability} \longrightarrow \text{Time-to-Failure (RUL)}$$
   Because the dataset lacks ground-truth failure timestamps or component seizure logs, standard supervised classification metrics—including **Precision, Recall, F1-Score, ROC-AUC, true failure lead time, and empirical failure prediction accuracy**—are mathematically undefined and cannot be reported.
2. **Escalation Runway Interpretation:**
   The reported average Med-to-High Risk Escalation Runway of $1.48$ observation cycles ($\approx 73.1$ minutes observed wall-clock elapsed time) represents the observed duration of consecutive Medium-risk states preceding an acute High-risk breach. It is measured as the real datetime difference between the onset of the first Medium-risk observation and the subsequent High-risk transition — not a countdown to mechanical failure, and not derived from the 30-observation window size. Its operational value is the actionable triage window it provides maintenance crews.
3. **Synthetic Sensor Distributions:**
   The uniform marginal distributions of raw sensor features reflect synthetic data generation. In industrial deployments, true mechanical fatigue presents Weibull or log-normal degradation dynamics.
4. **Contamination Parameter Sensitivity:**
   The $5\%$ contamination rate reflects an operational prior for outlier filtering; in production plants, this should be tuned against empirical work-order records.
5. **Operation-Mode Confounding in Anomaly Scores:**
   The rolling baseline is conditioned on `Machine_ID` only, not on `Machine_ID × Operation_Mode`. Empirically, Maintenance-mode records exhibit a mean Anomaly Score of $0.4565$ and a High-Risk rate of $6.03\%$, compared with $0.2547$ / $0.41\%$ during Active operation. Since `Operation_Mode_Enc` is included as a model feature and Isolation Forest partitions the full feature space, some proportion of High-Risk detections during Maintenance mode may reflect the mode itself rather than abnormal mechanical behavior within that mode. A stronger implementation would either (a) maintain separate per-machine, per-mode rolling baselines, or (b) score anomalies only within Active and Idle modes and treat Maintenance records as a separate stratum. This limitation does not invalidate results for Active and Idle operating records, which constitute $90.1\%$ of the dataset.

---

## 7. Recommendations

1. **Immediate action:** Schedule physical inspection for Machines 9, 32, 33, and 27 (escalating risk trend).
2. **Operational triage:** Prioritize assets whose rolling anomaly score exceeds the $0.65$ dynamic risk threshold.
3. **Ground-Truth Calibration:** Pair this anomaly detection engine with CMMS (Computerized Maintenance Management System) breakdown logs to evaluate empirical failure correlation.
4. **Extended Baseline Modeling:** Evaluate 60-observation and multi-day rolling baselines to capture seasonal thermal cycles.
5. **Deep Learning Sequence Modeling:** Benchmark unsupervised Temporal Convolutional Networks (TCN) or LSTM autoencoders against the Isolation Forest baseline.
6. **6G Edge Deployment (Future Work):** Deploy the lightweight Isolation Forest inference engine on 6G industrial edge compute nodes — a target deployment context enabled by the sub-millisecond latency budget of the ITU-R IMT-2030 framework — to enable near-real-time anomaly alerting directly at the production cell.

---

## 8. Conclusion

This research designs and validates an unsupervised **Predictive Maintenance Risk and Anomaly Detection System** tailored for 6G-integrated smart manufacturing. Operating without the requirement for scarce or unlabelled catastrophic failure records, the architecture combines a 30-observation machine-specific rolling baseline (approximately 16–17 hours of history at the dataset's median ~34-minute inter-observation interval) with an Isolation Forest ensemble trained on 11 domain-engineered physical and network features. The system successfully isolates 1,130 high-risk operational anomalies across 50 industrial machines, provides an actionable early-warning runway of ~73 minutes wall-clock elapsed time before high-risk breaches, and quantifies post-service stabilization across 8,895 maintenance episodes. An interactive predictive-maintenance analytics dashboard delivers these capabilities in an accessible format for operations teams and maintenance engineers, enabling dynamic threshold exploration and asset triage. By maintaining strict methodological fidelity—distinguishing unsupervised anomaly risk from supervised failure prediction—this framework provides an operationally robust, mathematically defensible foundation for intelligent shop-floor maintenance.

---

## References

1. Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). *Isolation Forest*. Proceedings of the 2008 IEEE International Conference on Data Mining (ICDM), 413–422.

2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

3. ITU-R. (2023). *Framework and Overall Objectives of the Future Development of IMT for 2030 and Beyond*. Recommendation ITU-R M.2160-0.

4. Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys, 41(3), Article 15.

5. Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying Density-Based Local Outliers*. Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data, 93–104.

6. Jardine, A. K. S., Lin, D., & Banjevic, D. (2006). *A Review on Machinery Diagnostics and Prognostics Implementing Condition-Based Maintenance*. Mechanical Systems and Signal Processing, 20(7), 1483–1510.

7. Lei, Y., Li, N., Guo, L., Li, N., Yan, T., & Wang, J. (2018). *Machinery Health Prognostics: A Systematic Review from Data Acquisition to RUL Prediction*. Mechanical Systems and Signal Processing, 104, 799–834.

8. Tao, F., Qi, Q., Liu, A., & Kusiak, A. (2018). *Data-Driven Smart Manufacturing*. Journal of Manufacturing Systems, 48, 157–169.

9. Tao, F., Zhang, M., Liu, Y., & Nee, A. Y. C. (2019). *Digital Twin Driven Smart Manufacturing*. Academic Press.

10. Letaief, K. B., Chen, W., Shi, Y., Zhang, J., & Zhang, Y.-J. A. (2019). *The Roadmap to 6G: AI Empowered Wireless Networks*. IEEE Communications Magazine, 57(8), 84–90.

11. Saad, W., Bennis, M., & Chen, M. (2019). *A Vision of 6G Wireless Systems: Applications, Trends, Technologies, and Open Research Problems*. IEEE Network, 34(3), 134–142.

12. ISO. (2014). *Condition Monitoring and Diagnostics of Machines — Data Interpretation and Diagnostics Techniques — Part 1: General Guidelines*. ISO 13379-1:2014.
