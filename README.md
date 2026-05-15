# OLS Regressions

## How It Works

The tool fits multiple regression model families to your data (polynomial degrees 1 through 4, natural cubic spline, log, reciprocal, power law, and exponential), then ranks them by BIC (Bayesian Info. Criterion) to identify the best-fitting OLS while punishing overfitting. Overfitting is a problem I kept falling into and I got sick of doing this over & over again. So, this tool exists!

Just hook up your .csv, and you're ready to cook.

## Model Selection

BIC penalizes model complexity, so a degree-3 polynomial won't beat a degree-2 just because it fits the training data marginally better. The top-ranked model is selected as the "best model" for all downstream diagnostics. BIC comparisons are only made within model families that share the same response variable scale; log-transformed models (log-log, exponential) are evaluated separately on original-scale R² since their BIC values aren't directly comparable.

## Diagnostics

Once the best model is selected, the code runs 6 diagnostic tests:

- **F-Test**
  Checks whether the model as a whole explains statistically significant variance in y. A pass (p < 0.05) checks for noise.

- **Shapiro-Wilk & Jarque-Bera**
  Test whether residuals are normally distributed, which you need for inference on coefficients and p-values.

- **Breusch-Pagan**
  Tests for heteroscedasticity (non-constant residual variance - in plain English, this is the spread between the variances). Failing this means standard errors are unreliable and confidence intervals are suspect.

- **Durbin-Watson**
  Detects autocorrelation in residuals, primarily relevant for time-ordered data. Values near 2.0 indicate no autocorrelation.

- **Variance Inflation Factor (VIF)**
  Flags multicollinearity among predictors. In polynomial models, x and x² are inherently correlated, so VIF will routinely exceed 5 or even 20. This is a known limitation of polynomial regression and doesn't invalidate the model, but it does mean individual coefficient standard errors are inflated.

- **Cook's Distance**
  Identifies observations that disproportionately influence the fitted coefficients. Points exceeding the 4/n threshold are circled for review as outliers.

## Output

The pipeline produces a 3-page PDF report - I've attached one with mock data to GitHub. 

It has a regression plot with the fitted curve overlaid on the data, a diagnostic scorecard with pass/fail results for each test, a ranked model comparison table, and 4 diagnostic plots (residuals vs. fitted, Q-Q plot, Cook's distance, residual distribution).
