# Preprocessing

## Objective

The preprocessing stage prepares the public dataset for the two classification models.

The preprocessing techniques are:

- Missing-value handling
- Duplicate-record checking
- Categorical-variable encoding
- Numerical-variable processing
- Train/test splitting
- Separate preprocessing pipelines for Logistic Regression and Random Forest

The implementation is in:

`preprocessing/preprocessing.py`

## 1. Missing values

Missing values are checked using:

```python
df.isnull().sum()
```

Numerical variables are imputed using the median:

```python
SimpleImputer(strategy='median')
```

Categorical variables are imputed using the most frequent category:

```python
SimpleImputer(strategy='most_frequent')
```

## 2. Duplicate records

Duplicate records are checked using:

```python
df.duplicated().sum()
```

Duplicates are removed using:

```python
df = df.drop_duplicates()
```

This is applied before the train/test split.

## 3. Categorical variables

Machine-learning models require categorical variables to be represented numerically.

One-hot encoding is therefore applied using:

```python
OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)
```

`handle_unknown='ignore'` is useful because the test set may contain a category that was not present in the training set.

## 4. Numerical variables

For Logistic Regression, numerical variables are standardised using:

```python
StandardScaler()
```

Random Forest does not require feature scaling because its tree splits are based on thresholds rather than numeric distance measurements.

Therefore, separate preprocessing pipelines are created for the two models.

## 5. Train/test split

The dataset is split before fitting preprocessing transformations in order to avoid data leakage.

The configuration is:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

`stratify=y` helps maintain the class distribution between the training and testing datasets, which is particularly important if returned orders are less common than non-returned orders.

## Code and usage

The relevant code is:

`preprocessing/preprocessing.py`

Run from the repository root:

```bash
python preprocessing/preprocessing.py
```

Expected input:

`data/engineered_data.csv`

Expected output:

```text
data/processed/
├── X_train_lr.csv
├── X_test_lr.csv
├── X_train_rf.csv
├── X_test_rf.csv
├── y_train.csv
├── y_test.csv
├── lr_preprocessor.joblib
├── rf_preprocessor.joblib
└── feature_names.txt
```
