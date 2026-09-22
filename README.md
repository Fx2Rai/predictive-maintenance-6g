# Predictive Maintenance Risk & Anomaly Detection in 6G Smart Manufacturing

> **An unsupervised data-driven framework for smart manufacturing maintenance risk assessment using machine-specific sensor baselines, engineered telemetry features, Isolation Forest anomaly detection, temporal risk analysis, and an interactive Streamlit dashboard.**

**Author:** Fabian Biju
**Project Framework Reference:** Unified Mentor Analytics Series
**Industry Reference:** Thales Group

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [System Architecture](#4-system-architecture)
5. [Dataset](#5-dataset)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Methodology](#7-methodology)
8. [Feature Engineering](#8-feature-engineering)
9. [Anomaly Detection Model](#9-anomaly-detection-model)
10. [Risk Classification](#10-risk-classification)
11. [Temporal Risk Analysis](#11-temporal-risk-analysis)
12. [Post-Maintenance Analysis](#12-post-maintenance-analysis)
13. [Key Results](#13-key-results)
14. [Feature Importance](#14-feature-importance)
15. [Dashboard](#15-dashboard)
16. [Repository Structure](#16-repository-structure)
17. [Installation](#17-installation)
18. [Running the Project](#18-running-the-project)
19. [Output Files](#19-output-files)
20. [Research Paper](#20-research-paper)
21. [Methodological Notes and Limitations](#21-methodological-notes-and-limitations)
22. [Future Improvements](#22-future-improvements)
23. [References](#23-references)
24. [Project Information](#24-project-information)

---

## 1. Project Overview

Modern manufacturing environments generate large volumes of operational telemetry from industrial machines, sensors, production systems, and connected infrastructure.

Traditional maintenance strategies often rely on:

- Fixed maintenance schedules
- Manual inspection
- Static alarm thresholds
- Reactive maintenance after abnormal events occur

These approaches can become difficult to scale when machines operate under different conditions and generate high-dimensional telemetry.

This project develops an **unsupervised predictive-maintenance risk and anomaly-detection framework** for smart manufacturing.

The system analyzes machine telemetry and identifies unusual operating behavior without requiring confirmed machine-failure labels.

The framework combines:

```
Industrial Telemetry
        |
        v
Machine-Specific Historical Baselines
        |
        v
Feature Engineering
        |
        v
Isolation Forest
        |
        v
Normalized Anomaly Score
        |
        v
Maintenance Risk Classification
        |
        v
Temporal & Maintenance Analysis
        |
        v
Interactive Streamlit Dashboard
```

The project contains a complete analytical pipeline from raw data processing through anomaly detection, risk analysis, visualization, and reporting.

---

## 2. Problem Statement

Smart manufacturing systems continuously generate telemetry such as:

- Machine temperature
- Vibration
- Power consumption
- Network latency
- Packet loss
- Production speed
- Quality-control defect rate
- Predictive-maintenance score
- Operational error rate

A major analytical challenge is determining whether a machine is behaving abnormally relative to its own historical operating behavior.

A simple global threshold can be problematic because different machines may naturally operate at different levels.

For example:

```
Machine A  —  Normal temperature ~ 65 C
Machine B  —  Normal temperature ~ 80 C
```

A fixed temperature threshold could therefore incorrectly classify normal behavior for one machine while failing to detect abnormal behavior in another.

This project addresses the problem by creating **machine-specific rolling historical baselines** and using multivariate anomaly detection.

The system is designed to answer questions such as:

- Which machines are showing abnormal behavior?
- How severe are the detected anomalies?
- Which machines experience repeated high-risk events?
- How frequently does risk escalate?
- Which telemetry features contribute most strongly to anomaly detection?
- What happens to machine behavior after maintenance events?
- Can Medium-risk states provide an operational early-warning signal?

---

## 3. Project Objectives

### Objective 1 — Develop machine-specific behavioral baselines

Instead of comparing all machines against one global baseline, historical telemetry is calculated separately for each machine.

### Objective 2 — Engineer meaningful anomaly features

Sensor deviations, trends, interactions, and network/quality stress indicators are transformed into model-ready features.

### Objective 3 — Detect multivariate anomalies

Isolation Forest is used to identify observations that differ from normal operating patterns.

### Objective 4 — Generate continuous anomaly scores

The model output is converted into a normalized score between 0 and 1.

### Objective 5 — Translate anomaly scores into maintenance risk

Records are classified into Low, Medium, and High risk categories.

### Objective 6 — Analyze temporal risk escalation

The project examines how machines transition from Medium to High risk.

### Objective 7 — Analyze post-maintenance behavior

Machine behavior before and after maintenance events is compared.

### Objective 8 — Provide an interactive analytical interface

A Streamlit dashboard allows users to explore machine risk, anomaly patterns, temporal trends, and maintenance results.

---

## 4. System Architecture

```
                    +---------------------------+
                    |  Manufacturing Telemetry  |
                    |      100,000 records      |
                    +------------+--------------+
                                 |
                                 v
                    +---------------------------+
                    |  Data Preprocessing       |
                    |  Cleaning & Validation    |
                    +------------+--------------+
                                 |
                                 v
                    +---------------------------+
                    |  Machine-Specific Rolling |
                    |  Historical Baselines     |
                    |  30 observations          |
                    +------------+--------------+
                                 |
                                 v
                    +---------------------------+
                    |  Feature Engineering      |
                    |  Z-scores / Trends /      |
                    |  Composite Indicators     |
                    +------------+--------------+
                                 |
                                 v
                    +---------------------------+
                    |  Isolation Forest         |
                    |  200 estimators           |
                    |  contamination = 0.05     |
                    +------------+--------------+
                                 |
                                 v
                    +---------------------------+
                    |  Normalized Anomaly Score |
                    |        0.0 to 1.0         |
                    +------------+--------------+
                                 |
                                 v
                    +---------------------------+
                    |  Maintenance Risk         |
                    |  Classification           |
                    +-----+----------+----------+
                          |          |          |
                          v          v          v
                       Low       Medium       High
                          |          |          |
                          +-----+----+----------+
                                |
                                v
                    +---------------------------+
                    |  Temporal Risk Analysis   |
                    |  Maintenance Analysis     |
                    |  Feature Importance       |
                    +------------+--------------+
                                 |
                                 v
                    +---------------------------+
                    |  Streamlit Dashboard      |
                    +---------------------------+
```

---

## 5. Dataset

The project uses the supplied **Thales Group manufacturing dataset**.

### Dataset Characteristics

| Property          | Value                     |
|---|---|
| Records           | 100,000                   |
| Machines          | 50                        |
| Time period       | January–March 2025        |
| Operation modes   | Active, Idle, Maintenance |
| Modeling approach | Unsupervised              |

### Main Telemetry Variables

| Variable                        | Description                  |
|---|---|
| `Machine_ID`                    | Machine identifier           |
| `Timestamp`                     | Observation timestamp        |
| `Operation_Mode`                | Machine operating state      |
| `Temperature_C`                 | Machine temperature          |
| `Vibration_Hz`                  | Machine vibration            |
| `Power_Consumption_kW`          | Electrical power consumption |
| `Network_Latency_ms`            | Network latency              |
| `Packet_Loss_%`                 | Network packet loss          |
| `Quality_Control_Defect_Rate_%` | Quality defect rate          |
| `Production_Speed_units_per_hr` | Production throughput        |
| `Predictive_Maintenance_Score`  | Maintenance-related score    |
| `Error_Rate_%`                  | Operational error rate       |

### Operation Modes

**Active** — The machine is actively producing.

**Idle** — The machine is not actively producing.

**Maintenance** — The machine is undergoing maintenance-related activity.

Operation mode is important because machine telemetry can naturally differ between these operating states.

### Dataset and 6G Context

The project incorporates network telemetry such as network latency and packet loss. These variables demonstrate how manufacturing analytics can incorporate communication-system conditions within a connected smart-manufacturing architecture.

> **Important:** The network telemetry in the supplied dataset is synthetic. The project therefore represents a 6G-oriented analytical architecture rather than a live measured 6G network deployment.

---

## 6. Exploratory Data Analysis

EDA is performed before modeling to understand the structure and behavior of the manufacturing telemetry. The analysis includes:

- Distribution analysis
- Correlation analysis
- Risk distribution
- Machine-level risk comparison
- Feature importance visualization
- Post-maintenance comparison

Generated EDA files:

```
eda_distributions.png
eda_correlation.png
```

---

## 7. Methodology

### 7.1 Data Preprocessing

The data is organized chronologically by machine so that historical observations are available when calculating rolling baselines.

### 7.2 Machine-Specific Rolling Baseline

A key component of the methodology is the use of historical machine-specific baselines using:

```python
x.shift(1).rolling(30)
```

The `shift(1)` operation ensures the current observation is excluded from the historical baseline, preventing the current observation from influencing the baseline against which it is evaluated.

The window covers **30 preceding observations**, which spans approximately **16–17 hours** of wall-clock history at the dataset's median inter-observation gap of ~34 minutes per machine.

### 7.3 Baseline Statistics

Historical rolling statistics are used to calculate deviations:

```
Current Temperature
        -
Historical Temperature Mean
        /
Historical Temperature Standard Deviation
```

This creates a machine-relative deviation measure.

---

## 8. Feature Engineering

The pipeline creates **11 engineered features** for anomaly detection:

| # | Feature | Description |
|---|---|---|
| 1 | `Temperature_C_zscore` | Temperature deviation from machine baseline |
| 2 | `Vibration_Hz_zscore` | Vibration deviation from machine baseline |
| 3 | `Power_Consumption_kW_zscore` | Power consumption deviation |
| 4 | `Error_Rate_%_zscore` | Error rate deviation |
| 5 | `Predictive_Maintenance_Score_zscore` | Maintenance score deviation |
| 6 | `VibPow_Instability` | Combined vibration-power instability index |
| 7 | `Error_Escalation` | Abnormal error escalation indicator |
| 8 | `Maint_Score_Decay` | Maintenance score deterioration indicator |
| 9 | `Network_Stress` | Combined network latency and packet-loss stress |
| 10 | `Quality_Pressure` | Quality-control degradation indicator |
| 11 | `Operation_Mode_Enc` | Encoded operational mode |

---

## 9. Anomaly Detection Model

The project uses **Isolation Forest** for unsupervised anomaly detection. Isolation Forest is suitable because the dataset does not contain confirmed machine-failure labels, and it identifies observations that are unusual relative to the broader feature space.

### Model Configuration

| Parameter       | Value            |
|---|---|
| Algorithm       | Isolation Forest |
| `n_estimators`  | 200              |
| `contamination` | 0.05             |
| `random_state`  | 42               |

The 5% contamination rate produces approximately 5,000 anomaly records from the 100,000-record dataset.

### Why Isolation Forest?

- Does not require failure labels
- Supports multivariate anomaly detection
- Captures unusual combinations of variables
- Computationally practical for large datasets
- Works well with engineered telemetry features

---

## 10. Risk Classification

The normalized anomaly score (0.0 → 1.0) is converted into three operational risk categories:

| Risk   | Score Range     | Operational Interpretation        |
|---|---|---|
| Low    | `< 0.35`        | Normal monitoring                 |
| Medium | `0.35 – < 0.65` | Early warning / inspection        |
| High   | `>= 0.65`       | High-priority maintenance review  |

> These thresholds are project-defined operational categories, not universally applicable industrial standards.

---

## 11. Temporal Risk Analysis

### Medium → High Escalation Runway

| Metric | Value |
|---|---|
| Average transition | 1.48 observation cycles |
| Approximate elapsed time | ~73.1 minutes |

> **Important:** The 73.1-minute figure is the observed elapsed wall-clock time between consecutive Medium-risk and High-risk state transitions (computed as a datetime difference). It is **not** time-to-failure, Remaining Useful Life, or a guaranteed warning time. No confirmed machine-failure timestamps are available in the dataset.

---

## 12. Post-Maintenance Analysis

A total of **8,895 maintenance episodes** were identified and analyzed.

The objective is to examine whether machine telemetry changes following maintenance by comparing before and after operating behavior.

**Recovery Success Rate: 50.5%** — defined as episodes where the mean Anomaly Score in the 5 records after maintenance is strictly lower than the mean in the 5 records before maintenance. This is a data-derived criterion; no failure ground-truth labels are used.

---

## 13. Key Results

| KPI | Result |
|---|---|
| Records analyzed | **100,000** |
| Machines | **50** |
| Anomaly records (5% contamination) | **5,000 (5.0%)** |
| Low-risk records | **70,034 (70.03%)** |
| Medium-risk records | **28,836 (28.84%)** |
| High-risk records | **1,130 (1.13%)** |
| Mean anomaly score | **0.2905** |
| Machines with high-risk events | **50 / 50** |
| Med → High escalation runway | **1.48 obs. cycles (~73.1 min elapsed)** |
| Pre-warning coverage proxy (DPI) | **33.4%** |
| Maintenance episodes analyzed | **8,895** |
| Recovery success rate | **50.5% (4,490 / 8,895)** |

---

## 14. Feature Importance

Feature importance is calculated via tree-depth analysis across all 200 Isolation Forest estimators. Features that appear at shallower splits contribute disproportionately to anomaly isolation.

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Network_Stress` | 10.47% |
| 2 | `Power_Consumption_kW_zscore` | 10.15% |
| 3 | `Predictive_Maintenance_Score_zscore` | 9.38% |
| 4 | `VibPow_Instability` | 9.35% |
| 5 | `Error_Escalation` | 9.28% |
| 6 | `Quality_Pressure` | 9.26% |
| 7 | `Vibration_Hz_zscore` | 9.12% |
| 8 | `Maint_Score_Decay` | 9.01% |
| 9 | `Error_Rate_%_zscore` | 8.87% |
| 10 | `Temperature_C_zscore` | 8.40% |
| 11 | `Operation_Mode_Enc` | 6.72% |

> Feature importance indicates model contribution, not causal influence on machine degradation.

---

## 15. Dashboard

The interactive Streamlit dashboard (`app.py`) loads all outputs generated by `analysis.py` and provides:

- **Overall KPIs** — anomaly counts, high-risk records, mean score, maintenance statistics
- **Machine Risk** — per-machine risk levels, high-risk events, fleet comparisons
- **Temporal Analysis** — hourly risk trends, escalation patterns
- **Feature Analysis** — interactive feature importance chart
- **Maintenance Analysis** — post-maintenance behavior and recovery statistics

---

## 16. Repository Structure

```
Project/
├── analysis.py                         # Main analysis pipeline (run first)
├── app.py                              # Streamlit interactive dashboard
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Excludes __pycache__, *.pyc, generated files
│
├── Thales_Group_Manufacturing.csv      # Raw dataset (100,000 records)
│
├── anomaly_results.csv                 # Record-level anomaly scores and risk labels
├── machine_risk_summary.csv            # Per-machine risk statistics and trend slopes
├── hourly_risk_trends.csv              # Hourly aggregated risk for time-series charts
├── feature_importance.csv              # Tree-depth feature importance results
├── post_maintenance_analysis.csv       # Per-episode before/during/after metrics
├── post_maintenance_summary.csv        # Aggregate post-maintenance comparison
├── kpi_report.json                     # Key performance indicators
│
├── isolation_forest_model.pkl          # Serialised trained Isolation Forest
├── scaler.pkl                          # Serialised MinMaxScaler
│
├── eda_distributions.png
├── eda_correlation.png
├── risk_distribution.png
├── top10_risk_machines.png
├── feature_importance.png
├── post_maintenance_comparison.png
│
├── research_paper.md                   # Full technical research paper
├── executive_summary.md                # Non-technical executive summary
└── README.md                           # This file
```

---

## 17. Installation

### Prerequisites

- Python **3.9 or later**
- pip

### Clone the Repository

```bash
git clone https://github.com/Fx2Rai/predictive-maintenance-6g.git
cd predictive-maintenance-6g
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Min Version | Purpose |
|---|---|---|
| pandas | 2.0 | Data manipulation |
| numpy | 1.24 | Numerical operations |
| scikit-learn | 1.3 | Isolation Forest, scaling |
| joblib | 1.3 | Model serialisation |
| matplotlib | 3.7 | Static plots |
| seaborn | 0.12 | EDA heatmaps |
| streamlit | 1.35 | Interactive dashboard |
| plotly | 5.18 | Interactive charts |

---

## 18. Running the Project

### Step 1 — Run the Analysis Pipeline

```bash
python analysis.py
```

The script performs: data loading, preprocessing, machine-specific baseline creation, feature engineering, scaling, Isolation Forest training, anomaly scoring, risk classification, temporal analysis, maintenance analysis, feature importance, visualization, KPI generation, and model serialization.

Expected runtime: **~2–4 minutes** on a standard laptop.

### Step 2 — Launch the Dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

> **Important:** `analysis.py` must be run before `app.py`. The dashboard reads the CSV/JSON/PKL outputs produced by the analysis pipeline.

---

## 19. Output Files

| File | Contents |
|---|---|
| `anomaly_results.csv` | Full record-level results with scores, risk labels, features |
| `machine_risk_summary.csv` | Per-machine aggregated risk statistics and trend slopes |
| `hourly_risk_trends.csv` | Temporal hourly risk aggregation |
| `feature_importance.csv` | Model feature importance (depth-weighted) |
| `post_maintenance_analysis.csv` | Episode-level before/during/after comparison |
| `post_maintenance_summary.csv` | Aggregate maintenance behavior summary |
| `kpi_report.json` | KPIs consumed by the dashboard |
| `isolation_forest_model.pkl` | Trained Isolation Forest (serialised) |
| `scaler.pkl` | Fitted MinMaxScaler (serialised) |

---

## 20. Research Paper

The repository includes full technical documentation:

- `research_paper.md` — Editable Markdown source covering abstract, introduction, problem context, methodology, EDA, feature engineering, model, results, 6G context, limitations, recommendations, and references
- `executive_summary.md` — Non-technical summary for stakeholders

---

## 21. Methodological Notes and Limitations

### 21.1 Anomaly Detection, Not Supervised Failure Prediction

The dataset contains no confirmed machine-failure labels. The system is **unsupervised anomaly detection and maintenance-risk scoring** — not supervised failure prediction or RUL estimation. Standard supervised metrics (Precision, Recall, F1, ROC-AUC) are mathematically undefined and are not reported.

### 21.2 Rolling Baseline Window

The 30-observation rolling baseline (`x.shift(1).rolling(30)`) spans approximately **16–17 hours** of real elapsed time at the dataset's ~34-minute median inter-observation interval — not 30 minutes.

### 21.3 DPI Proxy — Not Actual Downtime Prevention

The 33.4% pre-warning coverage proxy is the proportion of High-risk events preceded by a Medium-risk observation. It does **not** mean 33.4% of failures were prevented. No downtime logs or work-order outcomes exist in the dataset.

### 21.4 Escalation Runway — Not Time-to-Failure

The ~73.1-minute figure is the observed elapsed wall-clock time between Medium-risk and High-risk state transitions. It is **not** time-to-failure, RUL, or a guaranteed intervention window.

### 21.5 Operation-Mode Confounding

The rolling baseline is conditioned on `Machine_ID` only, not `Machine_ID × Operation_Mode`. Maintenance-mode records have a High-Risk rate of 6.03% vs. 0.41% for Active mode. Some High-Risk detections during Maintenance mode may reflect the operating mode itself rather than abnormal mechanical degradation. Active + Idle records constitute 90.1% of the dataset and are not affected by this limitation.

### 21.6 Synthetic Network Telemetry

`Network_Latency_ms` and `Packet_Loss_%` are synthetic dataset columns, not measurements from a live 6G deployment. The architecture is designed for future 6G-enabled deployment (ITU-R IMT-2030 target context).

### 21.7 Feature Importance Is Not Causality

High feature importance indicates model contribution to anomaly isolation — not a causal relationship with machine degradation.

---

## 22. Future Improvements

1. **Mode-specific baselines** — Condition rolling windows on `Machine_ID × Operation_Mode`
2. **Confirmed failure labels** — Extend to supervised prognostics if CMMS records become available
3. **RUL estimation** — Introduce survival analysis or deep-learning sequence models with degradation labels
4. **Model comparison** — Benchmark against LOF, One-Class SVM, and autoencoder approaches
5. **Streaming detection** — Implement real-time feature engineering and online inference
6. **Real 6G integration** — Replace synthetic latency with live 6G network telemetry
7. **Alerting system** — Add email/webhook notifications and automated maintenance-ticket generation

---

## 23. References

1. Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. IEEE ICDM.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12.
3. ITU-R. (2023). *IMT-2030 (6G) Framework and Overall Objectives*. Rec. ITU-R M.2160-0.
4. Chandola, V., Banerjee, A., & Kumar, V. (2009). *Anomaly Detection: A Survey*. ACM CSUR, 41(3).
5. Breunig, M. M., et al. (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD.
6. Jardine, A. K. S., Lin, D., & Banjevic, D. (2006). *Machinery Diagnostics and Prognostics for CBM*. Mechanical Systems and Signal Processing, 20(7).
7. Lei, Y., et al. (2018). *Machinery Health Prognostics: A Systematic Review*. Mechanical Systems and Signal Processing, 104.
8. Tao, F., et al. (2018). *Data-Driven Smart Manufacturing*. Journal of Manufacturing Systems, 48.
9. Letaief, K. B., et al. (2019). *The Roadmap to 6G: AI Empowered Wireless Networks*. IEEE Communications Magazine, 57(8).
10. Saad, W., Bennis, M., & Chen, M. (2019). *A Vision of 6G Wireless Systems*. IEEE Network, 34(3).
11. ISO 13379-1:2014. *Condition Monitoring and Diagnostics of Machines*.

---

## 24. Project Information

| Item | Detail |
|---|---|
| **Project Title** | Predictive Maintenance Risk & Anomaly Detection in 6G Smart Manufacturing |
| **Author** | Fabian Biju |
| **Project Framework** | Unified Mentor Analytics Series |
| **Industry Reference** | Thales Group |
| **Primary Language** | Python |
| **ML Algorithm** | Isolation Forest (unsupervised) |
| **Dashboard** | Streamlit |
| **Analysis Type** | Unsupervised Anomaly Detection |
| **Dataset Size** | 100,000 records, 50 machines |

---

*Predictive Maintenance Risk & Anomaly Detection System — Thales Group Industrial AI*
