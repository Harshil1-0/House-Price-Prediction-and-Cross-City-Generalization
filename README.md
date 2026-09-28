# House Price Regression Pipeline
### MA2221 – Mathematics for Machine Learning

A three-phase regression project implementing OLS, Ridge, and Lasso from scratch, applied to Indian and US (Ames) housing datasets, culminating in cross-market transfer learning with Bayesian uncertainty quantification.

---

## Project Structure

```
.
├── part1_regression_pipeline.py   # Phase 1 – OLS, Ridge, Lasso from scratch (Ames Housing)
├── Phase_02_code.py               # Phase 2 – Full pipeline on India Housing dataset
└── Phase_03_code.py               # Phase 3 – Cross-market transfer + Bayesian regression
```

---

## Phase 1 — Regression Foundations (`part1_regression_pipeline.py`)

**Dataset:** Ames Housing (Kaggle) — 1,460 rows, 79 features  
**Target:** `SalePrice` (log-transformed via `log1p`)

### What it does
Derives and implements three regression estimators purely in NumPy, then verifies each against scikit-learn:

- **OLS** — Normal equation: `β̂ = (XᵀX)⁻¹ Xᵀy`
- **Ridge** — Closed form with L2 penalty: `β̂ = (XᵀX + λI)⁻¹ Xᵀy`
- **Lasso** — Coordinate descent with soft-thresholding (no closed form due to non-differentiable L1 penalty)

### Key functions

| Function | Description |
|---|---|
| `load_ames(path)` | Load CSV, impute medians, standardise, return `(X, y)` |
| `ols(X, y)` | OLS via `lstsq` (numerically stable normal equation) |
| `ridge(X, y, lam)` | Ridge closed form; intercept not penalised |
| `lasso_cd(X, y, lam)` | Lasso coordinate descent with soft-threshold operator |
| `kfold_cv(...)` | 5-fold CV for any of the three estimators |
| `verify_against_sklearn(...)` | MAE comparison vs scikit-learn |
| `ridge_reg_path(X, y)` | Ridge coefficients over a log-spaced λ grid |

### Running

```bash
# Place Kaggle train.csv in the same directory, then:
python part1_regression_pipeline.py
```

If `train.csv` is not found, synthetic data (1,460 × 36) is generated automatically for a demo run.

### Output
Console table of 5-fold CV RMSE for OLS / Ridge / Lasso, sklearn verification results, and regularisation path data ready for plotting.

---

## Phase 2 — India Housing Pipeline (`Phase_02_code.py`)

**Dataset:** `india_house_price.csv` (ankushpanday1 / Kaggle)  
**Target:** `Price` (log-transformed via `log1p`)

### What it does
End-to-end pipeline on the India dataset including EDA, feature engineering, model training, hyperparameter tuning, and visualisation.

### Pipeline steps

1. **EDA** — Missing value audit, target distribution, correlation heatmap, outlier boxplots, price-by-city bar chart
2. **Feature engineering** — Median imputation for numeric columns (`Bathroom`, `Parking`, `Age`), mode imputation for `Furnishing`, one-hot encoding for `City`, `Location`, `Furnishing`, `Status`, StandardScaler
3. **Models** — OLS (scratch), Ridge (scratch), Lasso (scikit-learn), all with 5-fold CV over a log-spaced λ grid
4. **Hyperparameter selection** — Best λ chosen by minimum mean CV RMSE
5. **Plots** — 8 plots saved to disk (see below)

### Saved plots

| File | Description |
|---|---|
| `plot1_price_distribution.png` | Raw vs log-transformed price histograms |
| `plot2_correlation_heatmap.png` | Correlation matrix of numeric features |
| `plot3_outliers.png` | Boxplots for Price, Area, Age |
| `plot4_price_by_city.png` | Average price by city |
| `plot5_rmse_vs_lambda.png` | CV RMSE vs λ for Ridge and Lasso |
| `plot6_regularisation_path.png` | Ridge coefficient paths (top 8 features) |
| `plot7_predicted_vs_actual.png` | Scatter plots for all three models |
| `plot8_lasso_feature_importance.png` | Top 15 Lasso non-zero coefficients |

### Running

```bash
# Place india_house_price.csv in the same directory, then:
python Phase_02_code.py
```

---

## Phase 3 — Transfer Learning & Bayesian Regression (`Phase_03_code.py`)

**Datasets:** Synthetic India housing (n=500) + Ames Housing proxy  
**Research question:** Which pricing features are *universal* (transfer across markets) vs *market-specific* (collapse on out-of-distribution data)?

### Three contributions

1. **Cross-market transfer experiment** — Train on India, test on Ames (and vice versa); measure RMSE degradation
2. **Bayesian linear regression** — Conjugate Normal-Inverse-Gamma prior; posterior predictive mean and uncertainty intervals
3. **Feature universality analysis** — Cosine similarity of weight vectors across markets; per-feature sign stability

### Key components

| Component | Description |
|---|---|
| `generate_india_dataset(n)` | Synthetic India dataset (8 cities, realistic price distributions) |
| `preprocess_india(df)` | Imputation, one-hot encoding, StandardScaler |
| `ridge_scratch(X, y, lam)` | Ridge closed form (reused from Phase 1) |
| `BayesianLinearRegression` | Conjugate NIG model; `.fit()`, `.predict(return_std=True)` |
| `universal_features` | 4 cross-market features: Area, BHK, Bathroom, Age |
| PCA alignment | Aligns India feature space onto Ames via SVD before transfer |

### Bugs fixed in this phase

- **Bug 1** — `furnish` was a NumPy fixed-width string array; `None` was silently stored as the string `'None'`. Fixed by casting to `object` dtype before assignment.
- **Bug 2** — City sample counts summed to 450, not 500. Fixed by adjusting city counts.
- **Bug 3** — `bayes_r[3]` (Ames→India Bayesian transfer) incorrectly reused the in-market India score. Fixed by training a separate `blr_ames` model.
- **Bug 4** — Plot title claimed "transfer produces wider uncertainty." Factually incorrect; σ stays flat while RMSE explodes. Title corrected to reflect the actual result.
- **Bug 5** — Spurious percentage change in credible-interval width. Replaced with a direct comparison note.

### Saved plots

| File | Description |
|---|---|
| `p3_plot1_transfer_rmse.png` | In-market vs transfer RMSE bar chart |
| `p3_plot2_weight_comparison.png` | Universal feature weights: India vs Ames |
| `p3_plot3_bayesian_intervals.png` | Posterior predictive intervals (in-market vs transfer) |
| `p3_plot4_pred_actual_grid.png` | 2×2 predicted vs actual grid |
| `p3_plot5_universality_heatmap.png` | Feature universality heatmap |
| `p3_plot6_reg_paths.png` | Ridge regularisation paths: India vs Ames |
| `p3_plot7_uncertainty_vs_error.png` | Posterior σ vs absolute prediction error |
| `p3_plot8_summary.png` | Full model comparison (OLS / Ridge / Bayesian, all scenarios) |

### Running

```bash
python Phase_03_code.py
```

No external CSV is required — India data is generated synthetically; Ames data is proxied internally.

---

## Theory Connections

All three phases are grounded in Deisenroth, Faisal & Ong — *Mathematics for Machine Learning* (MML, 2020):

| Concept | MML Chapter | Implementation |
|---|---|---|
| OLS / MLE | Ch. 9 | Normal equation, `lstsq` |
| Ridge / MAP (Gaussian prior) | Ch. 9 | Closed form with λI |
| Lasso / MAP (Laplace prior) | Ch. 9 | Coordinate descent, soft-threshold |
| Bayesian linear regression | Ch. 9.3–9.4 | Conjugate NIG posterior, predictive distribution |
| Regularisation & optimisation | Ch. 7 | Convex loss, gradient, shrinkage paths |
| PCA / SVD alignment | Ch. 10 | Feature space alignment for transfer |

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## Data Sources

- **Ames Housing** — [Kaggle: House Prices Advanced Regression](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (`train.csv`)
- **India Housing** — [Kaggle: ankushpanday1 India House Price](https://www.kaggle.com/datasets/ankushpanday1/india-house-price) (`india_house_price.csv`)
- **Phase 3 India data** — Generated synthetically via `generate_india_dataset()` (no download needed)

---

## Quick Results Reference

| Phase | Model | Dataset | Notes |
|---|---|---|---|
| 1 | OLS / Ridge / Lasso | Ames (numeric only) | Sklearn-verified, 5-fold CV |
| 2 | OLS / Ridge / Lasso | India (full features) | Best λ by CV, 8 diagnostic plots |
| 3 | Ridge / Bayesian | India ↔ Ames transfer | Universality analysis, uncertainty calibration |
