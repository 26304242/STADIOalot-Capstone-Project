# Model 1 – Logistic Regression

## Objective

The first model is Logistic Regression.

Logistic Regression is suitable for this project because the target variable is binary:

- `returned = 0` = not returned
- `returned = 1` = returned

The implementation is in:

`models/model1_logistic_regression.py`

## Technique

Logistic Regression is used as a relatively simple and interpretable binary-classification model.

Because numerical variables can have different scales, the preprocessing pipeline for this model standardises numerical variables using `StandardScaler`.

## Configuration

```python
LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
```

## Hyperparameters

| Hyperparameter | Value | Purpose |
|---|---:|---|
| `C` | `1.0` | Controls regularisation strength |
| `penalty` | `l2` | Applies L2 regularisation |
| `solver` | `liblinear` | Suitable for binary classification |
| `max_iter` | `1000` | Allows sufficient iterations for convergence |
| `random_state` | `42` | Reproducibility |

## Code and usage

Run from the repository root:

```bash
python models/model1_logistic_regression.py
```

The script loads the Logistic Regression preprocessing outputs, trains the model and evaluates it on the held-out test set.

Outputs are saved to:

```text
data/results/logistic_regression_metrics.txt
data/results/logistic_regression_model.joblib
```
