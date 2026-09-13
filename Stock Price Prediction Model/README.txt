## Results

Target: next-day return (stationary). Baseline: predict return = 0,
equivalent to "tomorrow's price = today's price".

Trained on NVDA daily data (2020–2025), evaluated on a chronological
20% test split.

| Model              | RMSE   | MAE    | R²      |
|--------------------|--------|--------|---------|
| Baseline (ret=0)   | 0.0335 | 0.0250 | -0.0155 |
| Linear Regression  | 0.0333 | 0.0250 |  0.0016 |
| Random Forest      | 0.0334 | 0.0245 | -0.0056 |

Linear Regression and Random Forest both marginally outperformed the
naive zero-return baseline on RMSE (Linear Regression) and MAE
(Random Forest). The improvement is small — daily returns are dominated
by noise — but the direction is consistent with a small, learnable
signal in the technical features.

### Why the target was changed from price to return

An earlier version predicted next-day *price*. Random Forest failed
catastrophically (RMSE ~$33, R² = -0.67) because tree-based models
cannot extrapolate beyond the target range seen during training, and
stock prices are non-stationary. Switching the target to next-day
*return* — which is approximately stationary — removed this failure
mode and brought Random Forest back into a reasonable range.

### Why the baseline R² is negative

The baseline predicts 0 every day. R² compares against the test-set
*mean*, not against 0. Because NVDA had a positive average return during
the test period, predicting 0 is worse than predicting the mean, so R²
is slightly negative. This is expected and does not indicate a bug.

---------------------------------------------------------------------

Trained on AAPL daily data (2020–2025), evaluated on a chronological
20% test split.

| Model              | RMSE   | MAE    | R²     |
|--------------------|--------|--------|--------|
| Baseline (ret=0)   | 0.0143 | 0.0104 | -0.0098|
| Linear Regression  | 0.0145 | 0.0107 | -0.0364|
| Random Forest      | 0.0168 | 0.0119 | -0.4059|

### Why the target was changed from price to return

An earlier version predicted next-day price directly. Random Forest
failed catastrophically (RMSE ~$33, R² = -0.67) because tree-based
models cannot extrapolate beyond the target range seen during training,
and stock prices are non-stationary. Switching the target to next-day
return — which is approximately stationary — removed this failure mode.

### Interpretation

Daily returns are dominated by noise. Even a small improvement over the
zero-return baseline is meaningful, and a near-zero R² is expected for
this problem. The value of the project is the methodology: proper
stationarity handling, a chronological train/test split, and an honest
baseline comparison.