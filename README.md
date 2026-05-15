# Regression-Template
How It Works
The tool fits multiple regression model families to your data (polynomial degrees 1 through 4, natural cubic spline, logarithmic, reciprocal, power law, and exponential), then ranks them by BIC (Bayesian Information Criterion) to identify the best-fitting specification that isn't just overfitting.
Model Selection
BIC penalizes model complexity, so a degree-3 polynomial won't beat a degree-2 just because it fits the training data marginally better. The top-ranked model is selected as the "best model" for all downstream diagnostics. BIC comparisons are only made within model families that share the same response variable scale; log-transformed models (log-log, exponential) are evaluated separately on original-scale R² since their BIC values aren't directly comparable.
Diagnostics
Once the best model is selected, six diagnostic tests are run:

**F-Test
**
Checks whether the model as a whole explains statistically significant variance in y. A pass (p < 0.05) confirms the predictors aren't just noise.


**Shapiro-Wilk & Jarque-Bera
**
Test whether residuals are normally distributed, an assumption required for valid inference on coefficients and p-values.


**Breusch-Pagan
**
Tests for heteroscedasticity (non-constant residual variance). Failing this means standard errors are unreliable and confidence intervals are suspect.


**Durbin-Watson
**
Detects autocorrelation in residuals, primarily relevant for time-ordered data. Values near 2.0 indicate no autocorrelation.


**Variance Inflation Factor (VIF)
**
Flags multicollinearity among predictors. In polynomial models, x and x² are inherently correlated, so VIF will routinely exceed 5 or even 20. This is a known limitation of polynomial regression and doesn't invalidate the model, but it does mean individual coefficient standard errors are inflated.


**Cook's Distance
**
Identifies observations that disproportionately influence the fitted coefficients. Points exceeding the 4/n threshold are flagged for review; they may represent genuine outliers or data entry errors.



Output
The pipeline produces a three-page PDF audit report: a regression plot with the fitted curve overlaid on the data, a diagnostic scorecard with pass/fail results for each test, a ranked model comparison table, and four diagnostic plots (residuals vs. fitted, Q-Q plot, Cook's distance, residual distribution).
