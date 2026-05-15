import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, io
from datetime import datetime

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

PAGE = landscape(A4)
W, H = PAGE

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
PASS_COLOR  = colors.HexColor('#1a7a4a')
FAIL_COLOR  = colors.HexColor('#c0392b')
LIGHT_GRAY  = colors.HexColor('#f4f6f7')
MID_GRAY    = colors.HexColor('#bdc3c7')
DARK        = colors.HexColor('#2c3e50')
NAVY        = colors.HexColor('#0d2b55')


# ─────────────────────────────────────────────
# 1. CSV IMPORT
# ─────────────────────────────────────────────

def load_data(filepath):
    """
    Load a CSV file and return a DataFrame.
    Edit TARGET_COL and PREDICTOR_COLS in __main__ to match your data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[DATA] Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"[DATA] Columns: {list(df.columns)}\n")
    return df


# ─────────────────────────────────────────────
# 2. MODEL SELECTION — polynomials deg 1-4, log, exp, log-log,
#    reciprocal, spline, lag-1; ranked by orig-scale BIC
# ─────────────────────────────────────────────

def _model_type_label(degree, model_variant='poly'):
    """Return a human-readable model type label."""
    if model_variant == 'log':
        return 'Logarithmic'
    if model_variant == 'exp':
        return 'Exponential'
    if degree == 1:
        return 'Linear'
    return f'Polynomial (deg {degree})'


def select_top_models(df, target_col, predictor_cols, max_degree=4):
    """
    Fit a comprehensive set of OLS model types and rank by orig-scale BIC.

    Single-predictor models tried:
      - Polynomial deg 1-4
      - Logarithmic  : y  ~ ln(x)
      - Exponential  : ln(y) ~ x        (needs y > 0)
      - Log-log      : ln(y) ~ ln(x)    (needs x,y > 0; power law)
      - Reciprocal   : y  ~ 1/x         (needs x != 0)
      - Spline (natural cubic, 4 knots at quartiles)
      - Lag-1        : y_t ~ x_t + x_{t-1}

    Multi-predictor models tried:
      - Linear (degree 1)
      - Interaction  : all pairwise x_i * x_j terms added
    """
    from patsy import dmatrix

    results = []
    n_obs   = len(df)
    single  = (len(predictor_cols) == 1)
    col     = predictor_cols[0] if single else None

    # ── helper: append one result dict ──────────────────────────────────────
    def _add(model_type, mdl, Xc, orig_fitted=None):
        """orig_fitted: predicted y on original scale (pass for transformed-response models)."""
        res  = mdl.resid
        sw_p = shapiro(res)[1]
        bp_p = het_breuschpagan(res, mdl.model.exog)[1]
        results.append({
            'Degree':        1,
            'ModelType':     model_type,
            'Equation':      _equation_str(mdl, target_col,
                                           list(Xc.columns[1:]), 1),
            'R2':            round(mdl.rsquared, 4),
            'Adj_R2':        round(mdl.rsquared_adj, 4),
            'AIC':           round(mdl.aic, 2),
            'BIC':           round(mdl.bic, 2),
            'SW_p':          round(sw_p, 4),
            'BP_p':          round(bp_p, 4),
            '_model':        mdl,
            '_X':            Xc,
            '_orig_fitted':  orig_fitted,   # None means fittedvalues IS original scale
        })

    # ── 1. Polynomials (deg 1-4) ─────────────────────────────────────────────
    for deg in range(1, max_degree + 1):
        X = df[predictor_cols].copy()
        if deg > 1 and single:
            for d in range(2, deg + 1):
                X[f'{col}^{d}'] = df[col] ** d
        Xc  = sm.add_constant(X)
        mdl = sm.OLS(df[target_col], Xc).fit()
        res = mdl.resid
        sw_p = shapiro(res)[1]
        bp_p = het_breuschpagan(res, mdl.model.exog)[1]
        results.append({
            'Degree':       deg,
            'ModelType':    _model_type_label(deg, 'poly'),
            'Equation':     _equation_str(mdl, target_col, predictor_cols, deg),
            'R2':           round(mdl.rsquared, 4),
            'Adj_R2':       round(mdl.rsquared_adj, 4),
            'AIC':          round(mdl.aic, 2),
            'BIC':          round(mdl.bic, 2),
            'SW_p':         round(sw_p, 4),
            'BP_p':         round(bp_p, 4),
            '_model':       mdl,
            '_X':           Xc,
            '_orig_fitted': None,
        })

    if single:
        x_vals  = df[col].values
        y_vals  = df[target_col].values
        x_pos   = (x_vals > 0).all()
        y_pos   = (y_vals > 0).all()
        x_nz    = (x_vals != 0).all()

        # ── 2. Logarithmic  y ~ ln(x) ────────────────────────────────────────
        if x_pos:
            try:
                Xc = sm.add_constant(pd.DataFrame({f'ln({col})': np.log(x_vals)}))
                mdl = sm.OLS(y_vals, Xc).fit()
                _add('Logarithmic', mdl, Xc)
            except Exception:
                pass

        # ── 3. Exponential  ln(y) ~ x ────────────────────────────────────────
        if y_pos:
            try:
                Xc  = sm.add_constant(df[[col]])
                mdl = sm.OLS(np.log(y_vals), Xc).fit()
                orig_fitted = np.exp(mdl.fittedvalues.values)
                # Build equation with correct ln(y) label on LHS
                eq_str = _equation_str(mdl, f'ln({target_col})', [col], 1)
                res  = mdl.resid
                sw_p = shapiro(res)[1]
                bp_p = het_breuschpagan(res, mdl.model.exog)[1]
                results.append({
                    'Degree':        1,
                    'ModelType':     'Exponential',
                    'Equation':      eq_str,
                    'R2':            round(mdl.rsquared, 4),
                    'Adj_R2':        round(mdl.rsquared_adj, 4),
                    'AIC':           round(mdl.aic, 2),
                    'BIC':           round(mdl.bic, 2),
                    'SW_p':          round(sw_p, 4),
                    'BP_p':          round(bp_p, 4),
                    '_model':        mdl,
                    '_X':            Xc,
                    '_orig_fitted':  orig_fitted,
                })
            except Exception:
                pass

        # ── 4. Log-log  ln(y) ~ ln(x)  (power law) ──────────────────────────
        if x_pos and y_pos:
            try:
                Xc  = sm.add_constant(pd.DataFrame({f'ln({col})': np.log(x_vals)}))
                mdl = sm.OLS(np.log(y_vals), Xc).fit()
                orig_fitted = np.exp(mdl.fittedvalues.values)
                eq_str = _equation_str(mdl, f'ln({target_col})', [f'ln({col})'], 1)
                res  = mdl.resid
                sw_p = shapiro(res)[1]
                bp_p = het_breuschpagan(res, mdl.model.exog)[1]
                results.append({
                    'Degree':        1,
                    'ModelType':     'Log-Log (Power Law)',
                    'Equation':      eq_str,
                    'R2':            round(mdl.rsquared, 4),
                    'Adj_R2':        round(mdl.rsquared_adj, 4),
                    'AIC':           round(mdl.aic, 2),
                    'BIC':           round(mdl.bic, 2),
                    'SW_p':          round(sw_p, 4),
                    'BP_p':          round(bp_p, 4),
                    '_model':        mdl,
                    '_X':            Xc,
                    '_orig_fitted':  orig_fitted,
                })
            except Exception:
                pass

        # ── 5. Reciprocal  y ~ 1/x ───────────────────────────────────────────
        if x_nz:
            try:
                Xc  = sm.add_constant(pd.DataFrame({f'1/{col}': 1.0 / x_vals}))
                mdl = sm.OLS(y_vals, Xc).fit()
                _add('Reciprocal', mdl, Xc)
            except Exception:
                pass

        # ── 6. Natural cubic spline (4 knots at quartiles) ───────────────────
        try:
            knots = np.quantile(x_vals, [0.25, 0.50, 0.75])
            spline_df = dmatrix(
                f'cr(x, knots={list(knots)})',
                {'x': x_vals}, return_type='dataframe')
            # patsy includes an intercept column; drop it and add our own
            spline_df = spline_df.iloc[:, 1:]
            Xc  = sm.add_constant(spline_df)
            mdl = sm.OLS(y_vals, Xc).fit()
            _add('Spline (Natural Cubic)', mdl, Xc)
        except Exception:
            pass

        # ── 7. Lag-1  y ~ x_t + x_{t-1} ─────────────────────────────────────
        try:
            lag_col = f'{col}_lag1'
            X_lag   = pd.DataFrame({
                col:    x_vals[1:],
                lag_col: x_vals[:-1],
            })
            Xc  = sm.add_constant(X_lag)
            mdl = sm.OLS(y_vals[1:], Xc).fit()
            _add('Lag-1', mdl, Xc)
        except Exception:
            pass

    else:
        # ── Multi-predictor: interaction terms ───────────────────────────────
        try:
            X_int = df[predictor_cols].copy()
            from itertools import combinations
            for c1, c2 in combinations(predictor_cols, 2):
                X_int[f'{c1}*{c2}'] = df[c1] * df[c2]
            Xc  = sm.add_constant(X_int)
            mdl = sm.OLS(df[target_col], Xc).fit()
            res = mdl.resid
            sw_p = shapiro(res)[1]
            bp_p = het_breuschpagan(res, mdl.model.exog)[1]
            results.append({
                'Degree':       1,
                'ModelType':    'Interaction',
                'Equation':     _equation_str(mdl, target_col,
                                              list(X_int.columns), 1),
                'R2':           round(mdl.rsquared, 4),
                'Adj_R2':       round(mdl.rsquared_adj, 4),
                'AIC':          round(mdl.aic, 2),
                'BIC':          round(mdl.bic, 2),
                'SW_p':         round(sw_p, 4),
                'BP_p':         round(bp_p, 4),
                '_model':       mdl,
                '_X':           Xc,
                '_orig_fitted': None,
            })
        except Exception:
            pass

    # ── Unified orig-scale BIC ───────────────────────────────────────────────
    # All models scored with BIC = n*ln(RSS/n) + k*ln(n) on the ORIGINAL y
    # scale, making them directly comparable regardless of response transform.
    res_df = pd.DataFrame(results)

    actual = df[target_col].values

    def _orig_scale_bic(row):
        mt  = row['ModelType']
        of  = row['_orig_fitted']
        mdl = row['_model']

        if of is not None:
            # Transformed-response model (Exponential, Log-Log) —
            # _orig_fitted already holds back-transformed predictions on y scale
            fitted       = np.asarray(of)
            fitted_actual = actual
        else:
            fitted = mdl.fittedvalues.values
            if mt == 'Lag-1':
                # Lag model drops first observation — align to actual[1:]
                fitted_actual = actual[1:]
            else:
                fitted_actual = actual

        rss = np.sum((fitted_actual - fitted) ** 2)
        k   = int(mdl.df_model) + 1
        n   = len(fitted)
        return n * np.log(rss / n) + k * np.log(n) if rss > 0 else -np.inf

    res_df['_bic_orig'] = res_df.apply(_orig_scale_bic, axis=1)
    combined = res_df.sort_values('_bic_orig').reset_index(drop=True)

    top10     = combined.head(10).copy()
    best_row = combined.iloc[0]

    return top10, best_row['_model'], best_row['_X'], int(best_row['Degree']), str(best_row['ModelType'])



def _equation_str(model, target_col, predictor_cols, degree):
    """Build a human-readable equation string from fitted coefficients."""
    params = model.params
    parts  = []
    intercept = params.get('const', None)
    if intercept is not None:
        parts.append(f"{intercept:.3f}")

    for name, coef in params.items():
        if name == 'const':
            continue
        sign = '+' if coef >= 0 else '-'
        parts.append(f"{sign} {abs(coef):.4f}*{name}")

    return f"{target_col} = " + " ".join(parts)


# ─────────────────────────────────────────────
# 3. DIAGNOSTICS
# ─────────────────────────────────────────────

def run_diagnostics(model, X):
    residuals = model.resid
    n         = len(residuals)

    f_stat, f_pval = model.fvalue, model.f_pvalue
    df_model       = int(model.df_model)
    df_resid       = int(model.df_resid)

    # VIF — PASS/FAIL only (threshold 5)
    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == 'const':
            continue
        v    = variance_inflation_factor(X.values, i)
        flag = "PASS" if v < 5 else "FAIL"
        vif_rows.append((col, round(v, 2), flag))

    sw_stat, sw_p                    = shapiro(residuals)
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(residuals)
    bp_stat, bp_p, _, _              = het_breuschpagan(residuals, model.model.exog)
    dw                               = durbin_watson(residuals)

    influence     = model.get_influence()
    cooks_d       = influence.cooks_distance[0]
    threshold     = 4 / n
    n_influential = int(np.sum(cooks_d > threshold))

    return dict(
        n=n,
        r2=model.rsquared, adj_r2=model.rsquared_adj,
        f_stat=f_stat, f_pval=f_pval, df_model=df_model, df_resid=df_resid,
        f_pass=(f_pval < 0.05),
        vif_rows=vif_rows,
        vif_pass=all(r[2] == "PASS" for r in vif_rows),
        sw_stat=sw_stat, sw_p=sw_p, sw_pass=(sw_p > 0.05),
        jb_stat=jb_stat, jb_p=jb_p, jb_skew=jb_skew, jb_kurt=jb_kurt,
        jb_pass=(jb_p > 0.05),
        bp_stat=bp_stat, bp_p=bp_p, bp_pass=(bp_p > 0.05),
        dw=dw, dw_pass=(1.5 < dw < 2.5),
        cooks_d=cooks_d, cooks_threshold=threshold,
        n_influential=n_influential,
        residuals=residuals,
        fitted=model.fittedvalues,
    )


# ─────────────────────────────────────────────
# 5. MATPLOTLIB FIGURES
# ─────────────────────────────────────────────

def _fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def make_ols_plot(df, model, target_col, predictor_cols, degree, diag,
                  model_type='Linear'):
    """
    Draw actual data vs the fitted curve for all supported model types.
    Transformed-response models (Exponential, Log-Log) are back-transformed
    so the curve is always on the original y scale.
    """
    from patsy import dmatrix

    fig, ax = plt.subplots(figsize=(10, 4.5))
    actual  = df[target_col].values

    if len(predictor_cols) == 1:
        x_col  = predictor_cols[0]
        x_vals = df[x_col].values
        x_min, x_max = x_vals.min(), x_vals.max()
        x_smooth = np.linspace(x_min, x_max, 400)
        params   = model.params

        ax.scatter(x_vals, actual, color='#4a90d9', alpha=0.5,
                   s=28, label='Actual', zorder=3)

        try:
            if model_type == 'Exponential':
                a_val = params.get('const', 0)
                b_key = [k for k in params.index if k != 'const'][0]
                y_smooth = np.exp(a_val + params[b_key] * x_smooth)

            elif model_type == 'Logarithmic':
                a_val = params.get('const', 0)
                b_key = [k for k in params.index if k != 'const'][0]
                y_smooth = a_val + params[b_key] * np.log(np.maximum(x_smooth, 1e-9))

            elif model_type == 'Log-Log (Power Law)':
                a_val = params.get('const', 0)
                b_key = [k for k in params.index if k != 'const'][0]
                y_smooth = np.exp(a_val + params[b_key] * np.log(np.maximum(x_smooth, 1e-9)))

            elif model_type == 'Reciprocal':
                a_val = params.get('const', 0)
                b_key = [k for k in params.index if k != 'const'][0]
                safe  = np.where(np.abs(x_smooth) < 1e-9, np.nan, x_smooth)
                y_smooth = a_val + params[b_key] * (1.0 / safe)

            elif model_type == 'Spline (Natural Cubic)':
                knots = np.quantile(x_vals, [0.25, 0.50, 0.75])
                spline_smooth = dmatrix(
                    f'cr(x, knots={list(knots)})',
                    {'x': x_smooth}, return_type='dataframe')
                spline_smooth = spline_smooth.iloc[:, 1:]
                Xs = sm.add_constant(spline_smooth)
                # align columns
                Xs = Xs.reindex(columns=model.model.exog_names, fill_value=0)
                y_smooth = model.predict(Xs)

            elif model_type == 'Lag-1':
                # Lag model: y_t ~ x_t + x_{t-1}; plot as fitted vs x (not smooth curve)
                lag_col = f'{x_col}_lag1'
                X_lag = pd.DataFrame({
                    x_col:   x_vals[1:],
                    lag_col: x_vals[:-1],
                })
                Xc_lag = sm.add_constant(X_lag)
                sort_idx = np.argsort(x_vals[1:])
                y_smooth = model.fittedvalues.values
                ax.plot(x_vals[1:][sort_idx], y_smooth[sort_idx],
                        color='#e74c3c', linewidth=2,
                        label=f'OLS Fit ({model_type})', zorder=4)
                ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
                ax.set_ylabel(target_col, fontsize=11, fontweight='bold')
                ax.set_title(f'OLS Regression: {x_col} -> {target_col}  [{model_type}]',
                             fontsize=12, fontweight='bold')
                ax.legend(fontsize=9, prop={'weight': 'bold'})
                ax.text(0.02, 0.97,
                        f'R²={diag["r2"]:.4f}   Adj R²={diag["adj_r2"]:.4f}',
                        transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
                fig.tight_layout()
                return _fig_to_bytes(fig)

            else:
                # Linear / Polynomial
                y_smooth = np.full_like(x_smooth, params.get('const', 0))
                for d in range(1, degree + 1):
                    term = f'{x_col}^{d}' if d > 1 else x_col
                    if term in params:
                        y_smooth = y_smooth + params[term] * (x_smooth ** d)

            ax.plot(x_smooth, y_smooth, color='#e74c3c', linewidth=2,
                    label=f'OLS Fit ({model_type})', zorder=4)

        except Exception as e:
            # Fallback: plot sorted fitted values
            sort_idx = np.argsort(x_vals)
            ax.plot(x_vals[sort_idx], diag['fitted'].values[sort_idx],
                    color='#e74c3c', linewidth=2,
                    label=f'OLS Fit ({model_type}) [fallback]', zorder=4)

        ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
        ax.set_ylabel(target_col, fontsize=11, fontweight='bold')
        ax.set_title(f'OLS Regression: {x_col} -> {target_col}  [{model_type}]',
                     fontsize=12, fontweight='bold')
    else:
        # Multi-predictor: Actual vs Fitted scatter
        fitted_y = diag['fitted'].values
        ax.scatter(fitted_y, actual, color='#4a90d9', alpha=0.5, s=28, zorder=3)
        mn = min(fitted_y.min(), actual.min())
        mx = max(fitted_y.max(), actual.max())
        ax.plot([mn, mx], [mn, mx], color='#e74c3c', linewidth=2,
                linestyle='--', label='Perfect Fit')
        ax.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual Values', fontsize=11, fontweight='bold')
        ax.set_title('Actual vs Fitted', fontsize=12, fontweight='bold')

    ax.legend(fontsize=9, prop={'weight': 'bold'})
    ax.text(0.02, 0.97,
            f'R²={diag["r2"]:.4f}   Adj R²={diag["adj_r2"]:.4f}',
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    fig.tight_layout()
    return _fig_to_bytes(fig)


def make_diagnostic_plots(diag):
    residuals = diag['residuals']
    fitted    = diag['fitted']
    cooks_d   = diag['cooks_d']
    threshold = diag['cooks_threshold']
    n         = diag['n']

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # Residuals vs Fitted
    ax1  = fig.add_subplot(gs[0, 0])
    infl = cooks_d > threshold
    # All points blue — influential ones get a red circle overlay, not recolored
    ax1.scatter(fitted, residuals,
                s=np.where(infl, 50, 25),
                c='#4a90d9',
                alpha=0.65, zorder=3)
    for idx in np.where(infl)[0]:
        ax1.scatter(fitted.iloc[idx], residuals.iloc[idx],
                    s=260, facecolors='none', edgecolors='#e74c3c',
                    linewidths=1.8, zorder=4)
    ax1.axhline(0, color='black', lw=1, ls='--')
    ax1.set_xlabel('Fitted Values', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Residuals', fontsize=9, fontweight='bold')
    ax1.set_title("Residuals vs Fitted\n(red circles = influential, Cook's D > 4/n)",
                  fontsize=9, fontweight='bold')

    # Q-Q
    ax2 = fig.add_subplot(gs[0, 1])
    sm.qqplot(residuals, line='s', ax=ax2, alpha=0.55,
              marker='o', markerfacecolor='#4a90d9', markeredgewidth=0)
    ax2.set_title("Q-Q Plot (Normality of Residuals)", fontsize=9, fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Sample Quantiles', fontsize=9, fontweight='bold')

    # Cook's Distance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(n), cooks_d,
            color=['#e74c3c' if c > threshold else '#4a90d9' for c in cooks_d],
            alpha=0.75, width=0.8)
    ax3.axhline(threshold, color='#e74c3c', ls='--', lw=1.2,
                label=f'Threshold 4/n={threshold:.4f}')
    ax3.set_xlabel('Observation Index', fontsize=9, fontweight='bold')
    ax3.set_ylabel("Cook's Distance", fontsize=9, fontweight='bold')
    ax3.set_title("Cook's Distance (Influential Observations)", fontsize=9, fontweight='bold')
    ax3.legend(fontsize=8, prop={'weight': 'bold'})

    # Residual histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(residuals, bins=20, color='#4a90d9', edgecolor='white', alpha=0.8)
    ax4.set_xlabel('Residual Value', fontsize=9, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=9, fontweight='bold')
    ax4.set_title('Residual Distribution', fontsize=9, fontweight='bold')

    fig.suptitle('OLS Diagnostic Plots', fontsize=13, fontweight='bold')
    return _fig_to_bytes(fig)


# ─────────────────────────────────────────────
# 6. PDF STYLES
# ─────────────────────────────────────────────

def _styles():
    return {
        'title': ParagraphStyle('title', fontSize=17, leading=21,
                                textColor=NAVY, alignment=TA_CENTER,
                                fontName='Helvetica-Bold', spaceAfter=4),
        'sub':   ParagraphStyle('sub', fontSize=9, leading=13,
                                textColor=colors.HexColor('#7f8c8d'),
                                alignment=TA_CENTER, spaceAfter=10),
        'h2':    ParagraphStyle('h2', fontSize=11, leading=15,
                                textColor=NAVY, fontName='Helvetica-Bold',
                                spaceBefore=8, spaceAfter=4),
        'h3':    ParagraphStyle('h3', fontSize=9.5, leading=13,
                                textColor=NAVY, fontName='Helvetica-Bold',
                                spaceBefore=6, spaceAfter=3),
        'body':  ParagraphStyle('body', fontSize=8.5, leading=12, textColor=DARK),
        'note':  ParagraphStyle('note', fontSize=7.5, leading=11,
                                textColor=colors.HexColor('#555555'), leftIndent=6),
        'small': ParagraphStyle('small', fontSize=7, leading=10,
                                textColor=colors.HexColor('#7f8c8d')),
        'medal': ParagraphStyle('medal', fontSize=9, leading=12,
                                fontName='Helvetica-Bold', alignment=TA_CENTER),
        'interp_head': ParagraphStyle('interp_head', fontSize=10, leading=14,
                                      textColor=NAVY, fontName='Helvetica-Bold',
                                      spaceBefore=5, spaceAfter=2),
        'interp_body': ParagraphStyle('interp_body', fontSize=8.5, leading=13,
                                      textColor=DARK, leftIndent=8, spaceAfter=3),
        'hdr_cell': ParagraphStyle('hdr_cell', fontSize=8, leading=11,
                                   textColor=colors.white, fontName='Helvetica-Bold'),
    }


def _status(passed):
    """PASS/FAIL only - no WARN."""
    if passed:
        txt, c = 'PASS', PASS_COLOR
    else:
        txt, c = 'FAIL', FAIL_COLOR
    hex_c = c.hexval()[2:]
    return Paragraph(f'<font color="#{hex_c}"><b>{txt}</b></font>',
                     ParagraphStyle('s', fontSize=8, leading=11, alignment=TA_CENTER))


def _tbl_style(header_color=None):
    hc = header_color or NAVY
    return TableStyle([
        ('FONTNAME',       (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',       (0,0), (-1,-1), 8),
        ('BACKGROUND',     (0,0), (-1,0),  hc),
        ('TEXTCOLOR',      (0,0), (-1,0),  colors.white),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GRAY]),
        ('GRID',           (0,0), (-1,-1), 0.4, MID_GRAY),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING',    (0,0), (-1,-1), 5),
        ('RIGHTPADDING',   (0,0), (-1,-1), 5),
        ('TOPPADDING',     (0,0), (-1,-1), 3),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 3),
    ])


# ─────────────────────────────────────────────
# 7. INTERPRETATION HELPERS
# ─────────────────────────────────────────────

def _interpretation_bullets(diag, best_degree, target_col, predictor_cols, best_model_type=''):
    """Return a list of (heading, text) tuples for the interpretation section."""
    bullets = []

    # Overall fit — use model type name for non-polynomial models
    poly_types = {'Linear', 'Polynomial'}
    is_poly = any(best_model_type.startswith(p) for p in poly_types)
    model_label = (f'degree {best_degree}' if is_poly
                   else best_model_type if best_model_type else f'degree {best_degree}')

    r2  = diag['r2']
    ar2 = diag['adj_r2']
    fit_quality = 'strong' if ar2 >= 0.80 else ('moderate' if ar2 >= 0.50 else 'weak')
    bullets.append(("Overall Fit",
        f"The best model ({model_label}) explains {r2*100:.1f}% of variance in "
        f"{target_col} (R\u00b2 = {r2:.4f}, Adj R\u00b2 = {ar2:.4f}), indicating a "
        f"{fit_quality} fit."))

    # F-test
    fp = diag['f_pval']
    f_interp = "statistically significant (p < 0.05)" if diag['f_pass'] else "not significant (p \u2265 0.05)"
    bullets.append(("Model Significance (F-Test)",
        f"F({diag['df_model']}, {diag['df_resid']}) = {diag['f_stat']:.3f}, p = {fp:.4f}. "
        f"The overall regression is {f_interp}."))

    # Normality
    sw_txt = ("Residuals appear normally distributed - the normality assumption is met."
              if diag['sw_pass'] else
              "Residuals deviate from normality (Shapiro-Wilk p \u2264 0.05). "
              "Consider robust SE or transformation.")
    bullets.append(("Normality of Residuals (Shapiro-Wilk)",
        f"W = {diag['sw_stat']:.4f}, p = {diag['sw_p']:.4f}. {sw_txt}"))

    # Homoscedasticity
    bp_txt = ("Residual variance appears constant (homoscedastic)."
              if diag['bp_pass'] else
              "Evidence of heteroscedasticity detected. Consider WLS or robust standard errors.")
    bullets.append(("Homoscedasticity (Breusch-Pagan)",
        f"BP stat = {diag['bp_stat']:.4f}, p = {diag['bp_p']:.4f}. {bp_txt}"))

    # Autocorrelation
    dw_val = diag['dw']
    dw_txt = ("No significant autocorrelation detected."
              if diag['dw_pass'] else
              f"Durbin-Watson = {dw_val:.3f} suggests possible autocorrelation "
              f"({'positive' if dw_val < 1.5 else 'negative'}). Consider time-series methods.")
    bullets.append(("Autocorrelation (Durbin-Watson)",
        f"DW = {dw_val:.4f}. {dw_txt}"))

    # Multicollinearity
    if len(diag['vif_rows']) > 1:
        vif_max = max(r[1] for r in diag['vif_rows'])
        vif_txt = ("All VIF < 5; no multicollinearity concern."
                   if diag['vif_pass'] else
                   f"Max VIF = {vif_max:.2f} \u2265 5. Multicollinearity may inflate standard errors.")
        bullets.append(("Multicollinearity (VIF)", vif_txt))

    # Influential observations
    ni = diag['n_influential']
    infl_txt = (f"No influential observations detected (Cook's D > 4/n = {diag['cooks_threshold']:.4f})."
                if ni == 0 else
                f"{ni} influential observation(s) detected (Cook's D > {diag['cooks_threshold']:.4f}). "
                "Review flagged points for data entry errors or genuine outliers.")
    bullets.append(("Influential Observations (Cook's Distance)", infl_txt))

    return bullets


# ─────────────────────────────────────────────
# 8. BUILD PDF
# ─────────────────────────────────────────────

def build_pdf(diag, top10, ols_png, diag_png, out_path,
              target_col, predictor_cols, best_degree, best_model_type='Linear'):

    doc   = SimpleDocTemplate(out_path, pagesize=PAGE,
                              leftMargin=1.4*cm, rightMargin=1.4*cm,
                              topMargin=1.1*cm,  bottomMargin=1.1*cm)
    S     = _styles()
    story = []
    uw    = W - 2.8*cm

    # ═══════════════════════════════════════════
    # PAGE 1  — OLS Plot + Diagnostics
    # ═══════════════════════════════════════════

    # ── Page 1 Header ────────────────────────────────────────────
    story.append(Paragraph("OLS Regression - Diagnostic Audit Report", S['title']))
    story.append(Paragraph(
        f"Target: <b>{target_col}</b>  |  "
        f"Predictors: <b>{', '.join(predictor_cols)}</b>  |  "
        f"Best Model: <b>{best_model_type}</b>  |  n = <b>{diag['n']}</b>",
        S['sub']))
    story.append(HRFlowable(width=uw, thickness=1.5, color=NAVY, spaceAfter=8))

    # ── Full-width OLS Plot ───────────────────────────────────────
    story.append(Paragraph(f"OLS Regression Plot - Best Model ({best_model_type})", S['h2']))
    ols_img = RLImage(ols_png, width=uw, height=uw * 0.34)
    story.append(ols_img)
    story.append(Spacer(1, 6))

    # ── Diagnostic Scorecard ─────────────────────────────────────
    story.append(Paragraph(
        f"Diagnostic Scorecard - Best Model ({best_model_type})", S['h2']))

    vif_max  = max((r[1] for r in diag['vif_rows']), default=0)
    vif_pass = diag['vif_pass']

    score_header = [
        Paragraph('<b>Test</b>', S['hdr_cell']),
        Paragraph('<b>Statistic</b>', S['hdr_cell']),
        Paragraph('<b>p-value</b>', S['hdr_cell']),
        Paragraph('<b>Threshold</b>', S['hdr_cell']),
        Paragraph('<b>Result</b>', S['hdr_cell']),
        Paragraph('<b>Notes</b>', S['hdr_cell']),
    ]
    score_rows = [score_header]

    # F-test row
    score_rows.append([
        Paragraph("F-Test (Overall)", S['note']),
        Paragraph(f"{diag['f_stat']:.4f}", S['small']),
        Paragraph(f"{diag['f_pval']:.4f}", S['small']),
        Paragraph("p < 0.05", S['small']),
        _status(diag['f_pass']),
        Paragraph(f"df model={diag['df_model']}, resid={diag['df_resid']}", S['small']),
    ])
    # Shapiro-Wilk
    score_rows.append([
        Paragraph("Normality - Shapiro-Wilk", S['note']),
        Paragraph(f"{diag['sw_stat']:.4f}", S['small']),
        Paragraph(f"{diag['sw_p']:.4f}",   S['small']),
        Paragraph("p > 0.05", S['small']),
        _status(diag['sw_pass']),
        Paragraph("Tests residual normality", S['small']),
    ])
    # Jarque-Bera
    score_rows.append([
        Paragraph("Normality - Jarque-Bera", S['note']),
        Paragraph(f"{diag['jb_stat']:.4f}", S['small']),
        Paragraph(f"{diag['jb_p']:.4f}",   S['small']),
        Paragraph("p > 0.05", S['small']),
        _status(diag['jb_pass']),
        Paragraph(f"Skew={diag['jb_skew']:.3f}, Kurt={diag['jb_kurt']:.3f}", S['small']),
    ])
    # Breusch-Pagan
    score_rows.append([
        Paragraph("Homoscedasticity - Breusch-Pagan", S['note']),
        Paragraph(f"{diag['bp_stat']:.4f}", S['small']),
        Paragraph(f"{diag['bp_p']:.4f}",   S['small']),
        Paragraph("p > 0.05", S['small']),
        _status(diag['bp_pass']),
        Paragraph("Tests constant variance", S['small']),
    ])
    # Durbin-Watson
    score_rows.append([
        Paragraph("Autocorrelation - Durbin-Watson", S['note']),
        Paragraph(f"{diag['dw']:.4f}", S['small']),
        Paragraph("-", S['small']),
        Paragraph("1.5 – 2.5", S['small']),
        _status(diag['dw_pass']),
        Paragraph("Tests residual independence", S['small']),
    ])
    # VIF
    if diag['vif_rows']:
        score_rows.append([
            Paragraph("Multicollinearity - Max VIF", S['note']),
            Paragraph(f"{vif_max:.2f}", S['small']),
            Paragraph("-", S['small']),
            Paragraph("< 5", S['small']),
            _status(vif_pass),
            Paragraph(f"Predictors: {', '.join(r[0] for r in diag['vif_rows'])}", S['small']),
        ])
    # Cook's Distance
    score_rows.append([
        Paragraph("Influential Obs - Cook's D", S['note']),
        Paragraph(f"{diag['n_influential']} flagged", S['small']),
        Paragraph("-", S['small']),
        Paragraph(f"4/n = {diag['cooks_threshold']:.4f}", S['small']),
        _status(diag['n_influential'] == 0),
        Paragraph("Red circles on residual plot", S['small']),
    ])

    sc_col_w = [uw*p for p in [0.25, 0.10, 0.09, 0.10, 0.08, 0.38]]
    sc_tbl   = Table(score_rows, colWidths=sc_col_w, repeatRows=1)
    sc_tbl.setStyle(_tbl_style(NAVY))
    story.append(sc_tbl)
    story.append(Spacer(1, 6))

    # ─────────────────── PAGE BREAK ───────────────────────────────
    story.append(PageBreak())

    # ═══════════════════════════════════════════
    # PAGE 2  — Interpretation + Top-3 Equations
    # ═══════════════════════════════════════════

    # ── Page 2 Header ────────────────────────────────────────────
    story.append(Paragraph("OLS Regression - Interpretation & Model Comparison", S['title']))
    story.append(Paragraph(
        f"Target: <b>{target_col}</b>  |  "
        f"Predictors: <b>{', '.join(predictor_cols)}</b>  |  "
        f"Best Model: <b>{best_model_type}</b>  |  n = <b>{diag['n']}</b>",
        S['sub']))
    story.append(HRFlowable(width=uw, thickness=1.5, color=NAVY, spaceAfter=8))

    # ── Interpretation ────────────────────────────────────────────
    story.append(Paragraph("Interpretation of Results", S['h2']))

    interp_items = _interpretation_bullets(diag, best_degree, target_col, predictor_cols, best_model_type)
    interp_rows  = []
    for heading, text in interp_items:
        interp_rows.append([
            Paragraph(f"<b>{heading}</b>", S['interp_head']),
            Paragraph(text, S['interp_body']),
        ])

    interp_col_w = [uw * 0.25, uw * 0.75]
    interp_tbl   = Table(interp_rows, colWidths=interp_col_w)
    interp_tbl.setStyle(TableStyle([
        ('FONTNAME',       (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE',       (0,0), (-1,-1), 8.5),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, LIGHT_GRAY]),
        ('GRID',           (0,0), (-1,-1), 0.3, MID_GRAY),
        ('VALIGN',         (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',    (0,0), (-1,-1), 6),
        ('RIGHTPADDING',   (0,0), (-1,-1), 6),
        ('TOPPADDING',     (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',  (0,0), (-1,-1), 4),
        ('BACKGROUND',     (0,0), (0,-1),  colors.HexColor('#eaf0fb')),
    ]))
    story.append(interp_tbl)
    story.append(Spacer(1, 10))

    # ── Top 10 Models Ranked by BIC ─────────────────────────────
    story.append(Paragraph("Top 10 Models Ranked by BIC", S['h2']))
    story.append(Paragraph(
        "Models ranked by BIC within each family (polynomial vs log/exponential) - "
        "lower BIC = better fit penalised for complexity.  Cross-family comparison "
        "uses original-scale R\u00b2 since BIC is not comparable across different response "
        "variable transformations.  "
        "SW p = Shapiro-Wilk p-value (normality); "
        "BP p = Breusch-Pagan p-value (homoscedasticity).  "
        "Green = PASS (p > 0.05), Red = FAIL (p \u2264 0.05).",
        S['note']))
    story.append(Spacer(1, 4))

    top3_header = [
        Paragraph('<b>Rank</b>', S['hdr_cell']),
        Paragraph('<b>Model Type</b>', S['hdr_cell']),
        Paragraph('<b>Fitted Equation</b>', S['hdr_cell']),
        Paragraph('<b>R²</b>', S['hdr_cell']),
        Paragraph('<b>Adj R²</b>', S['hdr_cell']),
        Paragraph('<b>AIC</b>', S['hdr_cell']),
        Paragraph('<b>BIC</b>', S['hdr_cell']),
        Paragraph('<b>SW p</b>', S['hdr_cell']),
        Paragraph('<b>BP p</b>', S['hdr_cell']),
    ]
    top3_rows = [top3_header]

    for i, (_, row) in enumerate(top10.iterrows()):
        rank_txt = f'<b>{i+1}</b>' if i == 0 else str(i + 1)
        rank_p = Paragraph(
            rank_txt,
            ParagraphStyle('rp', fontSize=8, leading=10, alignment=TA_CENTER,
                           textColor=DARK))

        # Model type label — use ModelType column if present
        model_type_str = str(row.get('ModelType', _model_type_label(int(row['Degree']))))
        type_p = Paragraph(
            f'<font size="7.5"><b>{model_type_str}</b></font>',
            ParagraphStyle('tp', fontSize=7.5, leading=10, alignment=TA_CENTER,
                           textColor=NAVY))

        eq_font = 3.25 if 'Spline' in model_type_str else 6.5
        eq_p = Paragraph(
            f'<font size="{eq_font}">{row["Equation"]}</font>',
            ParagraphStyle('eq', fontSize=eq_font, leading=max(5, eq_font * 1.3)))

        sw_pass = row['SW_p'] > 0.05
        bp_pass = row['BP_p'] > 0.05
        sw_hex  = PASS_COLOR.hexval()[2:] if sw_pass else FAIL_COLOR.hexval()[2:]
        bp_hex  = PASS_COLOR.hexval()[2:] if bp_pass else FAIL_COLOR.hexval()[2:]

        top3_rows.append([
            rank_p, type_p, eq_p,
            Paragraph(str(row['R2']),     S['small']),
            Paragraph(str(row['Adj_R2']), S['small']),
            Paragraph(str(row['AIC']),    S['small']),
            Paragraph(str(row['BIC']),    S['small']),
            Paragraph(f'<font color="#{sw_hex}"><b>{row["SW_p"]}</b></font>', S['small']),
            Paragraph(f'<font color="#{bp_hex}"><b>{row["BP_p"]}</b></font>', S['small']),
        ])

    top3_col_w = [uw*p for p in [0.08, 0.11, 0.37, 0.07, 0.08, 0.08, 0.08, 0.065, 0.065]]
    t3_tbl     = Table(top3_rows, colWidths=top3_col_w, repeatRows=1)
    t3_style   = _tbl_style(NAVY)
    t3_style.add('BACKGROUND', (0,1), (0,1), colors.HexColor('#fef9e7'))  # gold highlight best row
    t3_tbl.setStyle(t3_style)
    story.append(t3_tbl)
    story.append(Spacer(1, 10))

    # ── Diagnostic Plots (page 3, centered) ───────────────────────
    story.append(PageBreak())

    img_w = uw
    img_h = img_w * 0.52
    # vertical center: total usable height minus heading + spacer + image
    page_usable = H - 2.2 * cm
    heading_h   = 20
    v_pad       = max(0, (page_usable - heading_h - 8 - img_h) / 2)

    story.append(Spacer(1, v_pad))

    heading_tbl = Table([[Paragraph("Diagnostic Plots - Best Model", S['h2'])]],
                        colWidths=[uw])
    heading_tbl.setStyle(TableStyle([('ALIGN', (0,0), (0,0), 'CENTER')]))
    story.append(heading_tbl)
    story.append(Spacer(1, 8))

    diag_img = RLImage(diag_png, width=img_w, height=img_h)
    img_tbl  = Table([[diag_img]], colWidths=[uw])
    img_tbl.setStyle(TableStyle([
        ('ALIGN',   (0,0), (0,0), 'CENTER'),
        ('VALIGN',  (0,0), (0,0), 'MIDDLE'),
        ('LEFTPADDING',  (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING',   (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0), (-1,-1), 0),
    ]))
    story.append(img_tbl)

    doc.build(story)
    print(f"[PDF] Saved → {out_path}")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # ── Configure these for your dataset ──────────────────────────
    CSV_PATH       = 'data.csv'       # path to your CSV file
    TARGET_COL     = 'y'              # name of the dependent variable column
    PREDICTOR_COLS = ['x']            # list of predictor column names
    datestamp      = datetime.now().strftime('%d-%m-%Y')
    OUT_PDF        = f'OLS Regression Analysis {datestamp}.pdf'
    # ──────────────────────────────────────────────────────────────

    df = load_data(CSV_PATH)

    top10, best_model, best_X, best_degree, best_model_type = select_top_models(
        df, TARGET_COL, PREDICTOR_COLS)

    diag = run_diagnostics(best_model, best_X)

    ols_png  = make_ols_plot(df, best_model, TARGET_COL, PREDICTOR_COLS,
                              best_degree, diag, model_type=best_model_type)
    diag_png = make_diagnostic_plots(diag)

    build_pdf(diag, top10, ols_png, diag_png, OUT_PDF,
              TARGET_COL, PREDICTOR_COLS, best_degree, best_model_type)
