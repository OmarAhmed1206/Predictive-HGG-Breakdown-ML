# HGG Predictive Maintenance System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Reliability Engineering](https://img.shields.io/badge/methodology-Weibull%20Survival%20Analysis-success.svg)](https://en.wikipedia.org/wiki/Weibull_distribution)
[![Status](https://img.shields.io/badge/status-Prototype%20v1.0-orange.svg)]()
[![License](https://img.shields.io/badge/license-Proprietary%20%2F%20AIC%20Steel-lightgrey.svg)]()

> **Predictive maintenance prototype for the HGG CNC Profile Coping Machine (RPC-1200B) at AIC Steel.**

---

## 1. Overview

The **HGG CNC Profile Coping Machine (Model RPC-1200B)** is a mission-critical asset in the fabrication line at AIC Steel, performing automated 3D plasma cutting, beveling, and coping on structural steel profiles (beams, tubes, and channels). Unscheduled machine downtime introduces severe operational bottlenecks across downstream assembly, fitting, and welding workflows.

This predictive maintenance system analyzes **~89 historical work order and downtime records spanning January 2020 through August 2026** (~6.6 years). By utilizing **parametric survival analysis (Weibull distribution)**, the system models the stochastic nature of machine failures to estimate remaining useful life (RUL), calculate Mean Time Between Failures (MTBF), and forecast failure probabilities over upcoming operational windows (7, 14, and 30 days).

The objective is to empower plant maintenance engineers to shift from **reactive firefighting** to **proactive condition-based scheduling**, executing preventive interventions during planned downtime rather than suffering catastrophic unplanned line stoppages.

---

## 2. Project Structure

```
predictive_maintenance/
├── data/
│   └── HGG_downtime.xlsx          # Historical maintenance logs and work orders
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_preprocessing.py      # Ingestion, timestamp parsing, downtime calculation, keyword categorization
│   ├── feature_engineering.py     # Inter-failure times (TBF), rolling frequencies, temporal features
│   ├── modeling.py                # Weibull, Exponential, Log-Normal MLE survival distribution fitting
│   ├── evaluation.py              # Chronological 80/20 train/test split, rolling-window validation, AIC/BIC
│   └── predictions.py             # Conditional failure probabilities, hazard rates, time-to-next-failure
├── dashboard/
│   └── app.py                     # Streamlit interactive monitoring and analytical dashboard
├── models/                        # Serialized model parameters and distribution artifacts
├── requirements.txt               # Production Python dependencies
└── README.md                      # Comprehensive project documentation
```

### Module Descriptions

| Module / Directory | Responsibility |
| :--- | :--- |
| `data/` | Stores raw and cleaned maintenance spreadsheets containing work order dates, downtime hours, and directive narratives. |
| `src/data_preprocessing.py` | Validates timestamps, computes outage durations in hours/days, handles missing fields, and applies domain-specific keyword taxonomy to classify failures (Mechanical, Electrical, Hydraulic, Optical/Torch, Software/Control). |
| `src/feature_engineering.py` | Transforms event sequences into Time-Between-Failures (TBF), cumulative operating spans, rolling 30/90-day event counts, and seasonality metrics. |
| `src/modeling.py` | Fits survival distributions (Weibull 2-parameter, Exponential, Log-Normal) via Maximum Likelihood Estimation (MLE) and computes cumulative hazard functions. |
| `src/evaluation.py` | Enforces strict chronological train/test partitioning (no future-data leakage) and expanding-window backtesting; benchmarks models via Log-Likelihood, AIC, BIC, and interval MAE/RMSE. |
| `src/predictions.py` | Computes conditional survival probabilities $S(t + \Delta t \mid t)$ given elapsed time since last failure, deriving upcoming risk scores and expected time-to-next-failure. |
| `dashboard/app.py` | Multi-view Streamlit dashboard featuring risk gauges, interactive timelines, Pareto downtime analysis, survival curves, and data inspection tools. |

---

## 3. Installation

### Prerequisites
- **Python**: Version 3.10 or higher
- **Virtual Environment**: Recommended (`venv` or `conda`)

### Step-by-Step Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd C:\Users\omara\Downloads\predictive_maintenance
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD):**
     ```cmd
     python -m venv venv
     .\venv\Scripts\activate.bat
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Usage

### A. Launching the Interactive Dashboard Locally

Launch the Streamlit web application from the project root:

```bash
cd predictive_maintenance
streamlit run dashboard/app.py
```

Once started, access the dashboard at `http://localhost:8501` in your browser.

### B. Sharing the Dashboard with Colleagues (No Local Server)

If you want your colleagues to view the dashboard without installing Python or running a local server, you have three primary options:

#### Option 1: Streamlit Community Cloud (Recommended & Free)
This is the easiest way to share a Streamlit app publicly or privately.
1. Create a free account at [share.streamlit.io](https://share.streamlit.io/).
2. Push this entire `predictive_maintenance` folder to a GitHub repository.
3. In Streamlit Community Cloud, click "New App", select your GitHub repository, and set the Main file path to `dashboard/app.py`.
4. Click "Deploy". Streamlit will host the app for free and give you a URL (e.g., `https://hgg-predictive.streamlit.app`) that you can send to any colleague.

#### Option 2: Internal Company Server (Docker)
If your company data is strictly confidential and cannot be hosted externally, your IT department can host it internally.
1. Create a `Dockerfile` in this directory.
2. The IT team can deploy the Docker container to an internal company server (e.g., using AWS, Azure, or an on-premise server).
3. Colleagues can access it via an internal IP address (e.g., `http://10.0.0.55:8501`).

#### Option 3: Static HTML Export
If you only need to share a "snapshot" of the current dashboard without interactivity:
1. Open the dashboard locally.
2. Most browsers allow you to "Print to PDF" or save the webpage as a complete HTML file to send via email.

### C. Programmatic API Usage

The core analytical pipeline can be integrated into custom scripts, automated jobs, or external enterprise systems:

```python
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import compute_inter_failure_times
from src.modeling import fit_weibull, fit_competing_models
from src.predictions import predict_next_breakdown

# 1. Load and preprocess work order data
df = load_and_clean_data("data/HGG_downtime.xlsx")

# 2. Extract inter-failure times (TBF) and temporal features
features_df = compute_inter_failure_times(df)

# 3. Fit 2-parameter Weibull model (MLE)
weibull_model = fit_weibull(features_df["tbf_days"])
print(f"Shape (beta): {weibull_model.shape:.3f}")
print(f"Scale (eta):  {weibull_model.scale:.3f} days")

# 4. Predict breakdown risk based on days elapsed since last failure
days_since_last_failure = 18.5
prediction = predict_next_breakdown(
    model=weibull_model, 
    elapsed_days=days_since_last_failure,
    forecast_horizons=[7, 14, 30]
)

print(f"Expected Time to Next Failure: {prediction['expected_ttnf_days']:.1f} days")
print(f"Conditional Failure Probability in next 7 days:  {prediction['risk_7d'] * 100:.1f}%")
print(f"Conditional Failure Probability in next 14 days: {prediction['risk_14d'] * 100:.1f}%")
print(f"Conditional Failure Probability in next 30 days: {prediction['risk_30d'] * 100:.1f}%")
```

---

## 5. Methodology

The end-to-end analytical framework follows rigorous reliability engineering principles tailored to low-frequency, high-impact industrial maintenance events:

```
[Raw Excel Logs] 
       │
       ▼
[Data Preprocessing] ──► Parse dates, compute downtime duration, clean nulls, keyword categorization
       │
       ▼
[Feature Engineering] ─► Calculate Time-Between-Failures (TBF), cumulative run spans, rolling counts
       │
       ▼
[Statistical Modeling] ─► Maximum Likelihood Estimation (MLE) for Weibull, Exponential, Log-Normal
       │
       ▼
[Validation & Split] ──► Chronological 80/20 train-test split & expanding-window backtesting
       │
       ▼
[Inference Engine] ────► Conditional probability S(t + Δt | t), remaining life, risk gauges
       │
       ▼
[Streamlit Dashboard] ─► Executive KPIs, survival curves, breakdown timelines, Pareto charts
```

1. **Data Preprocessing**:
   - Ingests tabular maintenance records from `HGG_downtime.xlsx`.
   - Parses heterogeneous date-time stamps, resolving format irregularities.
   - Derives total downtime duration per event in hours and calendar days.
   - Cleans duplicate records, zero-duration logging artifacts, and invalid entries.
   - Applies regex keyword matching against the free-text `Directive` field to categorize issues into functional subsystems: *Mechanical, Electrical, Hydraulic, Plasma/Torch/Optical, and Software/CNC*.

2. **Feature Engineering**:
   - Calculates **Time Between Failures (TBF)**: $TBF_i = t_i - t_{i-1}$ where $t_i$ represents failure event timestamps.
   - Computes rolling breakdown frequencies across 30-day and 90-day lookback windows.
   - Generates temporal features (day-of-week, month, quarter, elapsed machine age).

3. **Parametric Modeling**:
   - Fits parametric survival distributions (Weibull, Exponential, Log-Normal) via Maximum Likelihood Estimation (MLE).
   - Generates the survival function $S(t) = P(T > t)$, cumulative failure distribution $F(t) = 1 - S(t)$, and hazard rate function $h(t) = \frac{f(t)}{S(t)}$.

4. **Evaluation Strategy**:
   - **Chronological 80/20 Partitioning**: Standard randomized cross-validation causes temporal leakage in time-series reliability data. Instead, the first 80% chronological events form the training corpus, and the subsequent 20% serve as the out-of-sample holdout.
   - **Expanding-Window Cross-Validation**: Incrementally steps through time, fitting on historical windows and evaluating predictions on upcoming events.
   - **Metrics**: Goodness-of-fit evaluated via Log-Likelihood (LL), Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and Mean Absolute Error (MAE) on predicted vs. actual failure intervals.

5. **Prediction Engine**:
   - Evaluates conditional failure probability given that the machine has operated without incident for $t$ days:
     $$P(t < T \le t + \Delta t \mid T > t) = \frac{S(t) - S(t + \Delta t)}{S(t)} = 1 - \exp\left[ \left(\frac{t}{\eta}\right)^\beta - \left(\frac{t + \Delta t}{\eta}\right)^\beta \right]$$
   - Computes conditional probabilities for operational planning horizons ($\Delta t \in \{7, 14, 30\}$ days).

---

## 6. Model Details

### Why the Weibull Distribution?

The 2-parameter Weibull distribution is the cornerstone of reliability engineering and life data analysis (MIL-HDBK-338B / ISO 14224 standard) for modeling time-to-failure phenomena:

$$f(t) = \frac{\beta}{\eta} \left( \frac{t}{\eta} \right)^{\beta - 1} e^{-\left(\frac{t}{\eta}\right)^\beta}, \quad t \ge 0$$

$$S(t) = e^{-\left(\frac{t}{\eta}\right)^\beta}$$

The Weibull formulation was selected for several fundamental reasons:

| Property | Advantage for HGG Machine Analysis |
| :--- | :--- |
| **Small-Sample Robustness** | Unlike deep learning or complex gradient boosting models that require tens of thousands of samples, Weibull MLE parameter estimation provides stable, unbiased estimates on small datasets ($n < 100$). |
| **Physical Interpretability ($\beta$)** | The shape parameter $\beta$ directly reflects the underlying physical aging mechanism of the machine. |
| **Scale Parameter ($\eta$)** | Represents the characteristic life (the duration by which 63.2% of failures occur), serving as a benchmark for component longevity. |
| **Full Continuous Distribution** | Generates exact failure probabilities across any arbitrary planning window $\Delta t$, rather than single binary or point estimates. |

### Diagnostic Meaning of the Shape Parameter ($\beta$)

The shape parameter ($\beta$) reveals the failure dynamics of the HGG machine:

```
Hazard Rate h(t)
     ▲
     │  \             β < 1 (Infant Mortality / Burn-in)
     │   \
     │    ──────────  β = 1 (Random Failures / Exponential)
     │           /
     │          /     β > 1 (Wear-out / Aging / Fatigue)
     └────────────────────────► Time (t)
```

- **$\beta < 1$ (Infant Mortality / Burn-in)**: Failure rate decreases over time. Indicates defective replacement parts, improper installation, or break-in stress.
- **$\beta = 1$ (Random Failures / Memoryless)**: Constant hazard rate. Failures are caused by external random shocks (power surges, operator error, material jams). Equivalent to the Exponential distribution.
- **$\beta > 1$ (Wear-out / Component Aging)**: Failure rate increases with operating time. Governed by mechanical wear, thermal fatigue, bearing degradation, or torch consumable erosion. *Maintenance action: Schedule preventive overhaul before the wear-out knee.*

---

## 7. Dashboard Features

The Streamlit dashboard (`dashboard/app.py`) provides an executive and operational monitoring console:

- **KPI Cards**: Real-time high-level indicators:
  - Total historical breakdown events ($N \approx 89$)
  - Empirical Mean Time Between Failures (MTBF)
  - Mean Time to Repair (MTTR)
  - Cumulative Downtime (hours/days)
  - Current Risk Status (Low / Moderate / High)
- **Interactive Breakdown Timeline**: Chronological Plotly visualization mapping historical incidents with downtime duration, severity, and category tags.
- **Frequency Analysis**: Monthly breakdown distribution, 90-day rolling failure rate, and historical MTBF evolution trend over the 6.6-year span.
- **Downtime Analysis**: Distribution of repair durations (box plots / histograms), downtime breakdown by failure category (Mechanical, Electrical, Hydraulic, etc.), and Pareto analysis (80/20 rule of downtime contributors).
- **Survival Curves**: Empirical Kaplan-Meier non-parametric survival curve alongside fitted parametric Weibull survival function $S(t)$ with confidence bands.
- **Current Risk Gauge**: Real-time gauge chart indicating the probability of breakdown within the next 7, 14, and 30 days based on elapsed days since the last recorded event.
- **Model Performance Comparison**: Side-by-side comparison table of Weibull vs Exponential vs Log-Normal (AIC, BIC, Log-Likelihood).
- **Data Quality Report & Interactive Data Table**: Audit of missing timestamps, anomalies, filtered records, and searchable/sortable work order ledger.

---

## 8. Limitations

While this prototype provides actionable reliability intelligence, users must understand the following technical limitations:

1. **Sample Size ($n \approx 89$)**:
   Spanning 6.6 years, ~89 recorded events represent a small statistical sample. Sub-segmenting failures into individual failure modes (e.g., plasma torch vs. hydraulic clamp) reduces sample sizes further, yielding wider confidence bounds.
2. **Absence of Sensor / IoT Telemetry**:
   The model does not have real-time sensor streams (vibration spectrum, motor current draw, bearing temperatures, hydraulic pressures). All predictions are based purely on event timing and survival distributions.
3. **Calendar Time vs. Operational Spindle Hours**:
   Failure intervals are computed in calendar days. Variations in production scheduling, single vs. double shift operations, holidays, and weekend shutdowns are not differentiated from active cutting hours.
4. **Independent & Identically Distributed (i.i.d.) Assumption**:
   Renewal survival models assume that after repair, the machine returns to an "as good as new" state. In industrial practice, repairs may be minimal ("as bad as old") or major overhauls ("better than old"), causing non-stationary hazard rates.
5. **Heuristic Keyword Categorization**:
   Failure categorization depends on text matching from technician entries in the `Directive` field. Incomplete descriptions, colloquial language, or missing notes may lead to misclassifications.

---

## 9. Recommended Next Steps

To transition this prototype into a full-scale industrial condition-based maintenance platform:

1. **PLC / SCADA Integration**:
   - Extract high-frequency telemetry from the machine's CNC controller (Beckhoff / Siemens PLC) via OPC-UA or MQTT protocols.
   - Stream critical parameters: axis servo torque, motor current, plasma arc voltage, chiller coolant temperature, and hydraulic pump pressure.
2. **Operating Hours Tracking**:
   - Integrate automated spindle-on and cutting-time hour meters into the data pipeline to calculate true operational MTBF.
3. **Fleet-Level Expansion**:
   - Replicate the predictive modeling architecture across other fabrication machinery at AIC Steel, including the Zeman robotic beam assembler and automated plate processing lines.
4. **Condition-Based Monitoring (CBM)**:
   - Combine survival probability models with real-time anomaly detection algorithms (Isolation Forests, Autoencoders, Mahalanobis distance) to detect mechanical anomalies prior to functional failure.
5. **Spare Parts & ERP Integration**:
   - Connect risk thresholds directly with AIC Steel's ERP / CMMS (SAP or custom systems) to automatically verify spare parts inventory (plasma consumables, servo drives, hydraulic valves) and trigger purchase orders.
6. **Automated Alerting System**:
   - Configure automatic notification channels (Email, Microsoft Teams, SMS) notifying maintenance managers whenever machine risk crosses 70% for the upcoming 14-day window.

---

## 10. Industrial Context

### Prototype Definition
This application is a specialized engineering tool designed to augment the expertise of AIC Steel's maintenance and plant reliability teams. 

### What is "Breakdown Risk"?
In this system, **Breakdown Risk** represents the mathematically derived conditional probability $P(T \le t + \Delta t \mid T > t)$ that the HGG machine will experience an unscheduled stoppage within window $\Delta t$, given that it has operated without breakdown for $t$ calendar days since the previous incident.

### Maintenance Workflow Integration
- **Risk < 30% (Normal Operations)**: Maintain standard autonomous maintenance routines, daily lubrication checks, and optical alignment inspections.
- **Risk 30% - 60% (Elevated Watch)**: Review spare parts availability for high-wear subassemblies; prepare maintenance kits.
- **Risk > 60% (Action Required)**: Coordinate with production planning to schedule targeted inspection and consumable replacement during the nearest planned shift changeover, preventing high-cost in-cut plasma stoppages.

### Retraining Cycle
Because reliability characteristics evolve as machines age and maintenance policies mature, the statistical models should be retrained on a recurring basis (e.g., quarterly or whenever 5 new downtime incidents are logged).

---

*Developed for AIC Steel Reliability & Maintenance Engineering.*
