"""
=============================================================================
PART 1 — Regression Pipeline: Foundations
=============================================================================

OBJECTIVE
---------
Build a complete regression pipeline from scratch implementing:
  - Ordinary Least Squares (OLS)
  - Ridge Regression (L2 regularisation)
  - Lasso Regression (L1 regularisation, coordinate descent)

Dataset: Ames Housing — Kaggle (1,460 rows, 79 features)
Target  : SalePrice (log-transformed via log1p)
Goal    : Derive each estimator mathematically, implement it in NumPy,
          verify against scikit-learn, and perform 5-fold cross-validation.

=============================================================================
SEC 3 — KEY MATHEMATICS (derivations, referenced by Sections 3 & 5)
=============================================================================

1. OLS — Normal Equation
------------------------
Minimise the residual sum of squares:

    L(β) = ||y - Xβ||²

Taking the gradient and setting to zero:

    ∂L/∂β = -2 Xᵀ(y - Xβ) = 0
    Xᵀ y = Xᵀ X β
    β̂_OLS = (XᵀX)⁻¹ Xᵀy

2. Ridge — Closed Form (MAP with Gaussian prior, Ch. 9)
--------------------------------------------------------
Add an L2 penalty λ||β||² to the OLS loss:

    L_ridge(β) = ||y - Xβ||² + λ||β||²

Gradient and closed form:

    ∂L/∂β = -2Xᵀ(y - Xβ) + 2λβ = 0
    β̂_ridge = (XᵀX + λI)⁻¹ Xᵀy

Bayesian interpretation: this is the MAP estimate when the prior on β is
N(0, σ²/λ · I) — a Gaussian prior that shrinks all coefficients toward zero.

3. Lasso — Coordinate Descent (MAP with Laplace prior, Ch. 6)
--------------------------------------------------------------
Add an L1 penalty λ||β||₁:

    L_lasso(β) = ||y - Xβ||² + λ||β||₁

The L1 penalty is non-differentiable at 0, so there is no closed form.
We use coordinate descent: cycle over each βⱼ, minimising L_lasso with
all other coefficients fixed. The update is the soft-thresholding operator:

    r_j  = y - Xβ + βⱼ · xⱼ          (partial residual)
    ρ_j  = xⱼᵀ r_j / (xⱼᵀ xⱼ)       (unconstrained OLS update for j)
    z_j  = xⱼᵀ xⱼ                     (column norm squared)

    βⱼ  ← S(ρ_j, λ / z_j)
    where S(a, δ) = sign(a) · max(|a| − δ, 0)   (soft-threshold)

Bayesian interpretation: this is the MAP estimate when the prior on β is
Laplace(0, 1/λ) — a Laplace prior that induces sparsity (exact zeros).

=============================================================================
SEC 4 — DATA (Ames Housing)
=============================================================================
Source  : https://www.kaggle.com/c/house-prices-advanced-regression-techniques
Rows    : 1,460
Features: 79 (mix of numeric and categorical; we use only numeric here)
Target  : SalePrice — right-skewed; apply log1p for normality

=============================================================================
"""

# ---------------------------------------------------------------------------
# 0. IMPORTS
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as SkRidge, Lasso as SkLasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def load_ames(path: str = "train.csv") -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Ames Housing dataset, keep numeric features, impute medians,
    apply log1p to the target, and return (X, y) as NumPy arrays.

    Parameters
    ----------
    path : str
        Path to the Kaggle train.csv file.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)  — standardised features
    y : ndarray of shape (n_samples,)              — log1p(SalePrice)
    """
    df = pd.read_csv(path)

    # Target: log-transform to reduce right skew
    y = np.log1p(df["SalePrice"].values).astype(np.float64)

    # Features: numeric columns only, drop Id
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("Id", "SalePrice")]
    X_raw = df[num_cols].fillna(df[num_cols].median())

    # Standardise (zero mean, unit variance) before regularised regression
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float64)

    print(f"Dataset loaded: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y


# ---------------------------------------------------------------------------
# 2. SCRATCH IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    OLS via the normal equation: β̂ = (XᵀX)⁻¹ Xᵀy

    An intercept column of ones is prepended internally.
    """
    n = X.shape[0]
    Xb = np.c_[np.ones(n), X]          # prepend intercept
    beta = np.linalg.lstsq(Xb, y, rcond=None)[0]   # numerically stable
    return beta


def ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """
    Ridge regression (closed form): β̂ = (XᵀX + λI)⁻¹ Xᵀy

    The identity matrix is NOT applied to the intercept term (index 0).

    Parameters
    ----------
    X   : feature matrix (n × p), already standardised
    y   : target vector (n,)
    lam : regularisation strength λ ≥ 0

    Returns
    -------
    beta : coefficient vector of length p+1 (intercept at index 0)
    """
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]           # (n × p+1)

    # Regularisation matrix — do NOT penalise the intercept
    I = np.eye(p + 1)
    I[0, 0] = 0.0

    A = Xb.T @ Xb + lam * I
    b = Xb.T @ y
    beta = np.linalg.solve(A, b)
    return beta


def lasso_cd(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Lasso via coordinate descent with soft-thresholding.

    Minimises: ||y - Xβ||² + λ||β||₁
    Intercept fitted separately (unpenalised) as the mean of the residuals.

    Parameters
    ----------
    X        : feature matrix (n × p), already standardised
    y        : target vector (n,)
    lam      : regularisation strength λ ≥ 0
    max_iter : maximum coordinate descent iterations
    tol      : convergence tolerance on max |Δβ|

    Returns
    -------
    beta : coefficient vector of length p+1 (intercept at index 0)
    """
    n, p = X.shape
    beta = np.zeros(p)
    intercept = np.mean(y)

    # Column norms squared (constant across iterations)
    col_norms_sq = np.sum(X ** 2, axis=0)   # shape (p,)

    for _ in range(max_iter):
        beta_old = beta.copy()

        # Cycle over each coordinate
        for j in range(p):
            # Partial residual: y minus contribution of all other features
            r_j = y - intercept - X @ beta + beta[j] * X[:, j]
            rho_j = X[:, j] @ r_j                     # numerator
            z_j = col_norms_sq[j]                      # denominator

            if z_j == 0:
                beta[j] = 0.0
            else:
                # Soft-thresholding: S(ρ/z, λ/z)
                threshold = lam / z_j
                beta[j] = np.sign(rho_j / z_j) * max(abs(rho_j / z_j) - threshold, 0.0)

        # Unpenalised intercept update
        intercept = np.mean(y - X @ beta)

        # Convergence check
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return np.r_[intercept, beta]   # prepend intercept


# ---------------------------------------------------------------------------
# 3. PREDICTION HELPER
# ---------------------------------------------------------------------------

def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Predict y given X and beta (intercept at index 0).
    """
    n = X.shape[0]
    Xb = np.c_[np.ones(n), X]
    return Xb @ beta


# ---------------------------------------------------------------------------
# 4. 5-FOLD CROSS-VALIDATION LOOP
# ---------------------------------------------------------------------------

def kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    estimator: str = "ridge",
    lam: float = 1.0,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Perform K-fold cross-validation for OLS, Ridge, or Lasso (scratch).

    Parameters
    ----------
    X           : standardised feature matrix
    y           : log1p-transformed target
    estimator   : one of 'ols', 'ridge', 'lasso'
    lam         : regularisation parameter (ignored for OLS)
    n_splits    : number of folds (default 5)
    random_state: random seed for reproducibility

    Returns
    -------
    dict with keys 'rmse_per_fold' and 'mean_rmse'
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmse_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Fit on training fold
        if estimator == "ols":
            beta = ols(X_tr, y_tr)
        elif estimator == "ridge":
            beta = ridge(X_tr, y_tr, lam=lam)
        elif estimator == "lasso":
            beta = lasso_cd(X_tr, y_tr, lam=lam)
        else:
            raise ValueError(f"Unknown estimator: {estimator!r}")

        # Validate
        y_pred = predict(X_val, beta)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_folds.append(rmse)
        print(f"  Fold {fold}: RMSE = {rmse:.5f}")

    mean_rmse = np.mean(rmse_folds)
    print(f"  → Mean RMSE: {mean_rmse:.5f}\n")
    return {"rmse_per_fold": rmse_folds, "mean_rmse": mean_rmse}


# ---------------------------------------------------------------------------
# 5. SKLEARN VERIFICATION
# ---------------------------------------------------------------------------

def verify_against_sklearn(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    """
    Train scratch Ridge and Lasso on the full dataset and compare
    predictions against scikit-learn equivalents.

    A mean absolute error < 1e-4 on predictions confirms correctness.
    """
    n = X.shape[0]
    print("=" * 60)
    print("VERIFICATION: Scratch vs sklearn")
    print("=" * 60)

    # ---- OLS ----
    beta_ols = ols(X, y)
    sk_ols = LinearRegression(fit_intercept=True).fit(X, y)
    pred_scratch = predict(X, beta_ols)
    pred_sk = sk_ols.predict(X)
    mae_ols = np.mean(np.abs(pred_scratch - pred_sk))
    print(f"OLS   — MAE between scratch and sklearn predictions: {mae_ols:.2e}",
          "✓" if mae_ols < 1e-4 else "✗ CHECK")

    # ---- Ridge ----
    # sklearn's alpha = λ/n when fit_intercept=True; we work in the raw-λ space
    # so we match by using alpha = lam/(2) [sklearn uses 0.5 * ||·||² convention]
    beta_r = ridge(X, y, lam=lam)
    # sklearn Ridge: alpha matches our λ when using sum-of-squares (not MSE)
    sk_r = SkRidge(alpha=lam, fit_intercept=True).fit(X, y)
    pred_r_scratch = predict(X, beta_r)
    pred_r_sk = sk_r.predict(X)
    mae_ridge = np.mean(np.abs(pred_r_scratch - pred_r_sk))
    print(f"Ridge — MAE between scratch and sklearn predictions: {mae_ridge:.2e}",
          "✓" if mae_ridge < 1e-3 else "✗ CHECK")

    # ---- Lasso ----
    # sklearn Lasso uses 1/(2n) * ||·||² + alpha*||β||₁; ours uses ||·||² + λ||β||₁
    # so set sklearn alpha = lam * n / 2 … or compare at matched λ
    sk_l = SkLasso(alpha=lam / (2 * n), fit_intercept=True, max_iter=10000).fit(X, y)
    beta_l = lasso_cd(X, y, lam=lam)
    pred_l_scratch = predict(X, beta_l)
    pred_l_sk = sk_l.predict(X)
    mae_lasso = np.mean(np.abs(pred_l_scratch - pred_l_sk))
    print(f"Lasso — MAE between scratch and sklearn predictions: {mae_lasso:.2e}",
          "✓" if mae_lasso < 1e-2 else "✗ CHECK (coordinate descent may need more iters)")
    print()


# ---------------------------------------------------------------------------
# 6. REGULARISATION PATH (Ridge): weights vs log(λ)
# ---------------------------------------------------------------------------

def ridge_reg_path(X: np.ndarray, y: np.ndarray):
    """
    Compute Ridge coefficients for a log-spaced grid of λ values.

    Returns
    -------
    lambdas : array of λ values
    coefs   : (n_lambdas × p) array of coefficient vectors (no intercept)
    """
    lambdas = np.logspace(-3, 5, 80)
    coefs = []
    for lam in lambdas:
        beta = ridge(X, y, lam=lam)
        coefs.append(beta[1:])           # drop intercept
    return lambdas, np.array(coefs)


# ---------------------------------------------------------------------------
# 7. MAIN — tie everything together
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Load data (update path if needed) ----
    # Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
    DATA_PATH = "train.csv"

    try:
        X, y = load_ames(DATA_PATH)
    except FileNotFoundError:
        print(f"[INFO] '{DATA_PATH}' not found — generating synthetic data for demo.")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1460, 36))   # 36 numeric-ish features
        y = X @ rng.standard_normal(36) + 0.1 * rng.standard_normal(1460)
        print(f"Synthetic data: {X.shape[0]} rows × {X.shape[1]} features\n")

    LAM = 10.0   # default λ; tuned in Sec 5/6

    # ---- Cross-validation ----
    print("=" * 60)
    print("5-FOLD CV — OLS (scratch)")
    print("=" * 60)
    cv_ols = kfold_cv(X, y, estimator="ols", n_splits=5)

    print("=" * 60)
    print(f"5-FOLD CV — Ridge (λ={LAM}, scratch)")
    print("=" * 60)
    cv_ridge = kfold_cv(X, y, estimator="ridge", lam=LAM, n_splits=5)

    print("=" * 60)
    print(f"5-FOLD CV — Lasso (λ={LAM}, scratch)")
    print("=" * 60)
    cv_lasso = kfold_cv(X, y, estimator="lasso", lam=LAM, n_splits=5)

    # ---- Sklearn verification ----
    verify_against_sklearn(X, y, lam=LAM)

    # ---- Ridge regularisation path ----
    lambdas, coef_path = ridge_reg_path(X, y)
    print("Regularisation path computed.")
    print(f"  λ range : [{lambdas[0]:.1e}, {lambdas[-1]:.1e}]")
    print(f"  coef_path shape: {coef_path.shape}")
    print("  → Pass (lambdas, coef_path) to Sec 5 for the reg. path plot.\n")

    # ---- Summary table ----
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Estimator':<10} {'Mean CV RMSE':>14}")
    print("-" * 26)
    for name, cv in [("OLS", cv_ols), ("Ridge", cv_ridge), ("Lasso", cv_lasso)]:
        print(f"{name:<10} {cv['mean_rmse']:>14.5f}")
