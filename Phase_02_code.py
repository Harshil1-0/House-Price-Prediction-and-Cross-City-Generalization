"""
============================================================
House Price Prediction with Regularised Regression
MA2221 – Mathematics for Machine Learning
Dataset  : ankushpanday1 / India House Price (Kaggle)
Author   : [Your Name]
============================================================
Theory Connection (Deisenroth, Faisal & Ong – Ch. 7, 9):
  OLS   : minimise ||y - Xw||^2  →  MLE of w
  Ridge : minimise ||y - Xw||^2 + λ||w||^2  →  MAP with Gaussian prior
  Lasso : minimise ||y - Xw||^2 + λ||w||_1  →  MAP with Laplace prior
============================================================
"""
# ──────────────────────────────────────────────────────────
# 0.  IMPORTS
# ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
# Set consistent plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
# ──────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────────────────
df_raw = pd.read_csv("india_house_price.csv")
print("=" * 60)
print("SECTION 1 — RAW DATA OVERVIEW")
print("=" * 60)
print(f"Shape          : {df_raw.shape}  ({df_raw.shape[0]} rows, {df_raw.shape[1]} columns)")
print("\nColumn names   :", df_raw.columns.tolist())
print("\nData types:\n", df_raw.dtypes)
print("\nFirst 5 rows:\n", df_raw.head())
# ──────────────────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2 — EDA")
print("=" * 60)
# ── 2a. Missing values ──────────────────────────────────
print("\n--- Missing Values ---")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "% Missing": missing_pct})
print(missing_df[missing_df["Missing Count"] > 0])
# ── 2b. Target distribution ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df_raw["Price"] / 1e6, bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("Raw Price Distribution (₹ millions)", fontsize=13)
axes[0].set_xlabel("Price (₹ millions)")
axes[0].set_ylabel("Frequency")
axes[1].hist(np.log1p(df_raw["Price"]), bins=40, color="#55A868", edgecolor="white")
axes[1].set_title("Log-Transformed Price Distribution", fontsize=13)
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot1_price_distribution.png", bbox_inches="tight")
plt.close()
print("\n[Plot saved] plot1_price_distribution.png")
# ── 2c. Correlation heatmap ─────────────────────────────
num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
corr = df_raw[num_cols].corr()
plt.figure(figsize=(10, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5,
    annot_kws={"size": 9}
)
plt.title("Correlation Heatmap — Numeric Features", fontsize=14)
plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("[Plot saved] plot2_correlation_heatmap.png")
# ── 2d. Outlier detection (boxplots) ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["Price", "Area", "Age"]):
    data = df_raw[col].dropna()
    if col == "Price":
        data = data / 1e6
        label = "Price (₹M)"
    else:
        label = col
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax.set_title(f"Outliers – {label}", fontsize=12)
    ax.set_ylabel(label)
plt.tight_layout()
plt.savefig("plot3_outliers.png", bbox_inches="tight")
plt.close()
print("[Plot saved] plot3_outliers.png")
# ── 2e. Price by City (bar chart) ───────────────────────
city_avg = (df_raw.groupby("City")["Price"]
              .mean()
              .sort_values(ascending=False) / 1e6)
plt.figure(figsize=(10, 5))
city_avg.plot(kind="bar", color="#4C72B0", edgecolor="white")
plt.title("Average Price by City (₹ millions)", fontsize=13)
plt.xlabel("City")
plt.ylabel("Avg Price (₹M)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("plot4_price_by_city.png", bbox_inches="tight")
plt.close()
print("[Plot saved] plot4_price_by_city.png")
# ──────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3 — FEATURE ENGINEERING")
print("=" * 60)
df = df_raw.copy()
# Step 1: Fill numeric missing values with median
num_feats = ["Bathroom", "Parking", "Age"]
for col in num_feats:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  Filled NaN in '{col}' with median = {median_val:.1f}")
# Step 2: Fill categorical missing values with mode
cat_feats = ["Furnishing"]
for col in cat_feats:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"  Filled NaN in '{col}' with mode  = {mode_val}")
# Step 3: Log-transform target (Price is right-skewed)
y_raw = df["Price"].values
y = np.log1p(y_raw)          # log(1 + Price) so predictions stay positive
print(f"\n  Target skewness (raw)  : {pd.Series(y_raw).skew():.2f}")
print(f"  Target skewness (log)  : {pd.Series(y).skew():.2f}")
print("  → Log transform applied to reduce right-skew.")
# Step 4: One-hot encode categorical columns
cat_cols = ["City", "Location", "Furnishing", "Status"]
df_encoded = pd.get_dummies(df.drop("Price", axis=1), columns=cat_cols, drop_first=True)
print(f"\n  Shape after one-hot encoding: {df_encoded.shape}")
# Step 5: Scale numeric features
X_raw = df_encoded.values.astype(float)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)
print(f"  Features standardised  : mean≈0, std≈1")
print(f"  Final feature matrix X : {X.shape}")
feature_names = df_encoded.columns.tolist()
# ──────────────────────────────────────────────────────────
# 4.  MODEL IMPLEMENTATIONS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4 — MODEL IMPLEMENTATIONS")
print("=" * 60)
def ols_from_scratch(X, y):
    """
    Ordinary Least Squares (OLS) using the Normal Equation.
    Solves: w* = (X^T X)^{-1} X^T y
    This gives the MLE estimate of the weights — it minimises
    the sum of squared residuals with NO regularisation.
    Parameters
    ---------
    X : np.ndarray, shape (n_samples, n_features)
        Design matrix (already scaled).
    y : np.ndarray, shape (n_samples,)
        Target vector.
    Returns
    ------
    w : np.ndarray, shape (n_features,)
        Learned weight vector.
    """
    # Add intercept column of ones
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    # Normal equation
    w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
    return w
def predict_ols(X, w):
    """
    Predict targets using OLS weights.
    Parameters
    ---------
    X : np.ndarray  – design matrix (NOT augmented yet)
    w : np.ndarray  – weight vector (includes bias at index 0)
    Returns
    ------
    y_pred : np.ndarray
    """
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_b @ w
def ridge_from_scratch(X, y, lam):
    """
    Ridge Regression using the closed-form solution.
    Solves: w* = (X^T X + λI)^{-1} X^T y
    MML Theory: Ridge is equivalent to MAP estimation with a
    Gaussian prior N(0, σ²/λ · I) on the weights (Ch. 9).
    Larger λ → stronger regularisation → weights shrink toward 0.
    Parameters
    ---------
    X   : np.ndarray, shape (n_samples, n_features)
    y   : np.ndarray, shape (n_samples,)
    lam : float  – regularisation strength (λ ≥ 0)
    Returns
    ------
    w : np.ndarray, shape (n_features + 1,)  (includes bias)
    """
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    n_cols = X_b.shape[1]
    # Build λI but do NOT regularise the bias term (index 0)
    I = np.eye(n_cols)
    I[0, 0] = 0
    w = np.linalg.solve(X_b.T @ X_b + lam * I, X_b.T @ y)
    return w
def predict_ridge(X, w):
    """Predict with Ridge weights (same structure as OLS)."""
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    return X_b @ w
# ──────────────────────────────────────────────────────────
# 5.  5-FOLD CROSS-VALIDATION
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5 — 5-FOLD CROSS-VALIDATION (RMSE on log-price)")
print("=" * 60)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lambdas = np.logspace(-3, 3, 60)   # 60 values from 0.001 to 1000
def cross_validate_model(X, y, kf, model_fn, lam=None):
    """
    Run K-fold cross-validation and return mean RMSE.
    Parameters
    ---------
    X        : feature matrix
    y        : target vector
    kf       : KFold splitter
    model_fn : function(X_train, y_train, lam) → weights
               or function(X_train, y_train) for OLS
    lam      : lambda value (None for OLS)
    Returns
    ------
    mean_rmse : float
    """
    fold_rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        if lam is not None:
            w = model_fn(X_tr, y_tr, lam)
            y_pred = predict_ridge(X_val, w)   # ridge & OLS share same predict
        else:
            w = model_fn(X_tr, y_tr)
            y_pred = predict_ols(X_val, w)
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        fold_rmses.append(rmse)
    return float(np.mean(fold_rmses))
# ── OLS (no λ) ──────────────────────────────────────────
ols_rmse = cross_validate_model(X, y, kf, ols_from_scratch, lam=None)
print(f"\n  OLS  RMSE (5-fold)  : {ols_rmse:.4f}")
# ── Ridge: sweep λ ──────────────────────────────────────
ridge_rmses = []
for lam in lambdas:
    rmse = cross_validate_model(X, y, kf, ridge_from_scratch, lam=lam)
    ridge_rmses.append(rmse)
best_ridge_idx = np.argmin(ridge_rmses)
best_ridge_lam = lambdas[best_ridge_idx]
best_ridge_rmse = ridge_rmses[best_ridge_idx]
print(f"  Ridge RMSE (best)   : {best_ridge_rmse:.4f}  (λ = {best_ridge_lam:.4f})")
# ── Lasso: sweep α (sklearn) ────────────────────────────
lasso_rmses = []
for lam in lambdas:
    fold_rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = Lasso(alpha=lam, max_iter=10000)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        fold_rmses.append(np.sqrt(np.mean((y_val - y_pred) ** 2)))
    lasso_rmses.append(float(np.mean(fold_rmses)))
best_lasso_idx = np.argmin(lasso_rmses)
best_lasso_lam = lambdas[best_lasso_idx]
best_lasso_rmse = lasso_rmses[best_lasso_idx]
print(f"  Lasso RMSE (best)   : {best_lasso_rmse:.4f}  (α = {best_lasso_lam:.4f})")
# ── Summary table ────────────────────────────────────────
print("\n  ┌────────────────────┬──────────────┬────────────────┐")
print("  │ Model              │ Best λ / α   │  5-fold RMSE   │")
print("  ├────────────────────┼──────────────┼────────────────┤")
print(f"  │ OLS                │      —       │  {ols_rmse:.4f}       │")
print(f"  │ Ridge              │  {best_ridge_lam:9.4f}   │  {best_ridge_rmse:.4f}       │")
print(f"  │ Lasso              │  {best_lasso_lam:9.4f}   │  {best_lasso_rmse:.4f}       │")
print("  └────────────────────┴──────────────┴────────────────┘")
# ──────────────────────────────────────────────────────────
# 6.  REQUIRED PLOTS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 6 — GENERATING REQUIRED PLOTS")
print("=" * 60)
# ── Plot 5: RMSE vs λ (Ridge and Lasso) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].semilogx(lambdas, ridge_rmses, color="#4C72B0", linewidth=2)
axes[0].axvline(best_ridge_lam, color="red", linestyle="--", label=f"Best λ={best_ridge_lam:.3f}")
axes[0].set_title("Ridge: RMSE vs λ (5-fold CV)", fontsize=13)
axes[0].set_xlabel("λ (log scale)")
axes[0].set_ylabel("Mean RMSE (log-price)")
axes[0].legend()
axes[1].semilogx(lambdas, lasso_rmses, color="#55A868", linewidth=2)
axes[1].axvline(best_lasso_lam, color="red", linestyle="--", label=f"Best α={best_lasso_lam:.3f}")
axes[1].set_title("Lasso: RMSE vs α (5-fold CV)", fontsize=13)
axes[1].set_xlabel("α (log scale)")
axes[1].set_ylabel("Mean RMSE (log-price)")
axes[1].legend()
plt.suptitle("Cross-Validation: RMSE vs Regularisation Strength", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plot5_rmse_vs_lambda.png", bbox_inches="tight")
plt.close()
print("[Plot saved] plot5_rmse_vs_lambda.png")
# ── Plot 6: Regularisation Path (Ridge) ─────────────────
w_paths = []
for lam in lambdas:
    w = ridge_from_scratch(X, y, lam)
    w_paths.append(w[1:])            # drop bias
w_paths = np.array(w_paths)
# Pick top 8 features by absolute weight at best λ
best_w = ridge_from_scratch(X, y, best_ridge_lam)
top_idx = np.argsort(np.abs(best_w[1:]))[-8:]
plt.figure(figsize=(11, 6))
for i in top_idx:
    plt.semilogx(lambdas, w_paths[:, i], label=feature_names[i])
plt.axvline(best_ridge_lam, color="black", linestyle="--", alpha=0.5, label=f"Best λ={best_ridge_lam:.3f}")
plt.title("Ridge Regularisation Path — Top 8 Features", fontsize=13)
plt.xlabel("λ (log scale)")
plt.ylabel("Coefficient Weight")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("plot6_regularisation_path.png", bbox_inches="tight")
plt.close()
print("[Plot saved] plot6_regularisation_path.png")
# ── Plot 7: Predicted vs Actual (all three models on full data) ──
w_ols   = ols_from_scratch(X, y)
w_ridge = ridge_from_scratch(X, y, best_ridge_lam)
lasso_model = Lasso(alpha=best_lasso_lam, max_iter=10000).fit(X, y)
y_pred_ols   = predict_ols(X, w_ols)
y_pred_ridge = predict_ridge(X, w_ridge)
y_pred_lasso = lasso_model.predict(X)
# Back-transform from log-space for interpretability
y_actual_orig  = np.expm1(y)
y_ols_orig     = np.expm1(y_pred_ols)
y_ridge_orig   = np.expm1(y_pred_ridge)
y_lasso_orig   = np.expm1(y_pred_lasso)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, y_pred, name, color in zip(
    axes,
    [y_ols_orig, y_ridge_orig, y_lasso_orig],
    ["OLS", "Ridge", "Lasso"],
    ["#4C72B0", "#DD8452", "#55A868"]
):
    lim_min = min(y_actual_orig.min(), y_pred.min()) / 1e6
    lim_max = max(y_actual_orig.max(), y_pred.max()) / 1e6
    ax.scatter(y_actual_orig / 1e6, y_pred / 1e6, alpha=0.4, color=color, s=20)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5)
    ax.set_title(f"{name}: Predicted vs Actual", fontsize=12)
    ax.set_xlabel("Actual Price (₹M)")
    ax.set_ylabel("Predicted Price (₹M)")
plt.suptitle("Predicted vs Actual Prices — OLS, Ridge, Lasso", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plot7_predicted_vs_actual.png", bbox_inches="tight")
plt.close()
print("[Plot saved] plot7_predicted_vs_actual.png")
# ── Plot 8: Feature Importance (Lasso non-zero coefficients) ──
lasso_coef = pd.Series(lasso_model.coef_, index=feature_names)
nonzero = lasso_coef[lasso_coef != 0].sort_values(key=abs, ascending=False).head(15)
plt.figure(figsize=(10, 6))
colors = ["#4C72B0" if v > 0 else "#C44E52" for v in nonzero.values]
nonzero.plot(kind="barh", color=colors, edgecolor="white")
plt.title("Lasso: Top 15 Non-Zero Feature Coefficients", fontsize=13)
plt.xlabel("Coefficient Value")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("plot8_lasso_feature_importance.png", bbox_inches="tight")
plt.close()
print("[Plot saved] plot8_lasso_feature_importance.png")
# ──────────────────────────────────────────────────────────
# 7.  INTERPRETATION SUMMARY
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 7 — INTERPRETATION SUMMARY")
print("=" * 60)
models = {"OLS": ols_rmse, "Ridge": best_ridge_rmse, "Lasso": best_lasso_rmse}
best_model = min(models, key=models.get)
print(f"""
  Best performing model   : {best_model}  (RMSE = {models[best_model]:.4f})
  OLS  RMSE  : {ols_rmse:.4f}
  Ridge RMSE : {best_ridge_rmse:.4f}  (λ = {best_ridge_lam:.4f})
  Lasso RMSE : {best_lasso_rmse:.4f}  (α = {best_lasso_lam:.4f})
  MML Theory connection:
  • Ridge = MAP estimation with Gaussian prior on weights (Ch. 9)
    Adding λ||w||² penalises large weights, equivalent to assuming
    weights follow N(0, σ²/λ). Shrinks all weights toward zero.
  • Lasso = MAP estimation with Laplace prior on weights (Ch. 9)
    Adding λ||w||_1 produces sparse solutions: many weights become
    exactly zero, giving automatic feature selection.
  • OLS = Maximum Likelihood Estimation (MLE) with no prior.
    Finds the unique minimum of ||y - Xw||², but can overfit
    on high-dimensional data.
  Top features by Lasso (non-zero coefficients):
""")
for feat, coef in nonzero.head(10).items():
    direction = "↑ price" if coef > 0 else "↓ price"
    print(f"    {feat:<35}  coef = {coef:+.4f}  ({direction})")
print("\n  Conclusion:")
print("  Regularised models (Ridge, Lasso) outperform OLS when the")
print("  number of features is large relative to samples. Lasso also")
print("  provides built-in feature selection by zeroing irrelevant weights.")
print("\n[All done!] 8 plots saved. See plot1_*.png through plot8_*.png")