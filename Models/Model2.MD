# Model 2 – Random Forest Classification

## Objective

The second model is Random Forest Classification.

Random Forest is valuable because it can model nonlinear relationships and interactions between variables.

For example, return probability might depend on a combination of:

- product category
- discount
- order value
- delivery time
- customer behaviour

rather than one variable independently.

The implementation is in:

`models/model2_random_forest.py`

## Technique

Random Forest combines multiple decision trees to create an ensemble classification model.

Unlike Logistic Regression, numerical variables do not need to be standardised for Random Forest because tree-based splits are based on thresholds rather than numeric distance measurements.

## Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

## Hyperparameters

| Hyperparameter | Value | Purpose |
|---|---:|---|
| `n_estimators` | `300` | Number of trees |
| `max_depth` | `15` | Limits tree complexity |
| `min_samples_split` | `5` | Minimum observations needed to split a node |
| `min_samples_leaf` | `2` | Minimum observations per leaf |
| `max_features` | `sqrt` | Number of features considered at each split |
| `class_weight` | `balanced` | Helps account for class imbalance |
| `random_state` | `42` | Reproducibility |
| `n_jobs` | `-1` | Uses available CPU cores |

## Code and usage

Run from the repository root:

```bash
python models/model2_random_forest.py
```

The script loads the Random Forest preprocessing outputs, trains the model and evaluates it on the held-out test set.

Outputs are saved to:

```text
data/results/random_forest_metrics.txt
data/results/random_forest_model.joblib
```
