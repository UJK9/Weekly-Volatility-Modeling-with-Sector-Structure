# Weekly Volatility Modeling with Sector Structure  
**Bayesian Student-t Regression vs Classical Time-Series Baselines**

This capstone models **weekly equity volatility** using sector structure and compares **Bayesian Student-t regression** (with and without an autoregressive lag) against classical forecasting baselines (**AR(1), ARMA(1,1), HAR**).  
Focus is not only point accuracy (RMSE) but also **uncertainty quality** via **predictive interval calibration** (coverage + PIT).

---

## 1. Project Summary

### Goal
Forecast weekly log-volatility proxies across four sectors:
- Automotive  
- Finance  
- Healthcare  
- Industrial  

### Why it matters
Most “good” volatility forecasts look good on RMSE but fail when used for **risk** because their predictive intervals are miscalibrated. This project explicitly evaluates both:
- **Point forecast accuracy** (RMSE)
- **Probabilistic calibration** (95% interval coverage + PIT histograms)

### Models
**Bayesian (primary):**
- Student-t regression (base): market predictors → weekly log volatility
- Student-t regression (lagged): adds AR term \(y_{t-1}\) (robust ARX)

**Classical baselines:**
- AR(1)
- ARMA(1,1)
- HAR (heterogeneous autoregressive)

---

## 2. Data + Target Construction

### Input
Daily equity prices grouped by sector.

### Weekly volatility proxy (WRV proxy)
Within each week:
\[
WRV = \sqrt{\sum_{d \in week} r_d^2}
\quad,\quad
y_t = \log(WRV_t)
\]
where \(r_d\) is the daily log return.

(An alternate proxy may also be evaluated if present in the pipeline.)

---

## 3. Evaluation Design

### Time-series cross-validation
Chronological folds to avoid leakage.
- Train on earlier weeks
- Test on later weeks

### Final holdout evaluation
A final holdout window tests generalization to the most recent regime.

### Metrics
- **RMSE** on log scale
- **95% interval coverage**
- **PIT histograms** (calibration diagnostics)

---

## 4. Key Results (What to Look For)

- Student-t likelihood generally improves realism of tails and uncertainty statements.
- Lagged Student-t (robust ARX) can improve stability when persistence exists.
- HAR may remain competitive on RMSE but can fail interval calibration on holdout (important: this is not a “mistake”, it’s a finding).

---

## 5. Repository Structure

# Weekly Volatility Modeling with Sector Structure  
**Bayesian Student-t Regression vs Classical Time-Series Baselines**

This capstone models **weekly equity volatility** using sector structure and compares **Bayesian Student-t regression** (with and without an autoregressive lag) against classical forecasting baselines (**AR(1), ARMA(1,1), HAR**).  
Focus is not only point accuracy (RMSE) but also **uncertainty quality** via **predictive interval calibration** (coverage + PIT).

---

## 1. Project Summary

### Goal
Forecast weekly log-volatility proxies across four sectors:
- Automotive  
- Finance  
- Healthcare  
- Industrial  

### Why it matters
Most “good” volatility forecasts look good on RMSE but fail when used for **risk** because their predictive intervals are miscalibrated. This project explicitly evaluates both:
- **Point forecast accuracy** (RMSE)
- **Probabilistic calibration** (95% interval coverage + PIT histograms)

### Models
**Bayesian (primary):**
- Student-t regression (base): market predictors → weekly log volatility
- Student-t regression (lagged): adds AR term \(y_{t-1}\) (robust ARX)

**Classical baselines:**
- AR(1)
- ARMA(1,1)
- HAR (heterogeneous autoregressive)

---

## 2. Data + Target Construction

### Input
Daily equity prices grouped by sector.

### Weekly volatility proxy (WRV proxy)
Within each week:
\[
WRV = \sqrt{\sum_{d \in week} r_d^2}
\quad,\quad
y_t = \log(WRV_t)
\]
where \(r_d\) is the daily log return.

(An alternate proxy may also be evaluated if present in the pipeline.)

---

## 3. Evaluation Design

### Time-series cross-validation
Chronological folds to avoid leakage.
- Train on earlier weeks
- Test on later weeks

### Final holdout evaluation
A final holdout window tests generalization to the most recent regime.

### Metrics
- **RMSE** on log scale
- **95% interval coverage**
- **PIT histograms** (calibration diagnostics)

---

## 4. Key Results (What to Look For)

- Student-t likelihood generally improves realism of tails and uncertainty statements.
- Lagged Student-t (robust ARX) can improve stability when persistence exists.
- HAR may remain competitive on RMSE but can fail interval calibration on holdout (important: this is not a “mistake”, it’s a finding).

---

## 5. Repository Structure

CAPSTONE4/
code/ # R scripts and helpers (modeling + evaluation)
data/
raw/ # raw daily returns / prices used by pipeline
processed/ # intermediate weekly datasets (if saved)
outputs/
figures/ # exported figures used in report
tables/ # exported CSVs + generated LaTeX tables
report/
figures/ # Overleaf/LaTeX figures folder (if mirrored)
tables/ # Overleaf/LaTeX tables folder (if mirrored)
FREPORT.tex # main report source
main.tex # (if used as wrapper)
models/ # fitted objects / brms models / ARIMA objects (if saved)
logs/ # run logs, diagnostics
cmdstan/ # CmdStan installation artifacts (if applicable)
00_run_all.R # end-to-end reproducible pipeline
