# Feature Engineering

## Objective

Feature engineering involves transforming existing variables into more useful and explainable predictor variables, allowing a more thorough understanding of trends and relationships within the data.

The implementation is in:
`feature_engineering/feature_engineering.py`

The features below are created when the required source columns are available in the public dataset.

## 1. Order Value

Order value provides the price of the entire order rather than only using individual product prices.
```python
df['order_value'] = df['quantity'] * df['unit_price']
```

## 2. Discount Percentage

If the dataset contains both the original and discounted selling price, discount percentage can be calculated as:
```python
df['discount_percent'] = (
    (df['original_price'] - df['selling_price'])
    / df['original_price']) * 100
```

The implementation avoids division by zero.

## 3. Customer Purchase Frequency

If customer ID and transaction dates are available, the number of previous transactions associated with a customer can be calculated.
The implementation calculates the number of transactions that occurred before the current transaction.

## 4. Previous Return Rate

If customer ID, transaction date and the return target are available, the percentage of the customer's historical transactions that resulted in a return can be calculated.
Only previous transactions are used. The current transaction's return outcome is not included, which helps prevent target leakage.

## 5. Delivery Time

If order date and delivery date are available, delivery time can be calculated as:
```python
df['delivery_time'] = (
    df['delivery_date'] - df['order_date']).dt.days
```

### Prediction-time consideration

Delivery time should only be included if it is available at the intended prediction point. If the model is intended to predict a return at checkout, delivery time should not be used because the delivery has not yet occurred.

## Code and usage

The relevant code is:
`feature_engineering/feature_engineering.py`

Run from the repository root:
```bash
python feature_engineering/feature_engineering.py
```

Expected input:
`data/raw_dataset.csv`

Expected output:
`data/engineered_data.csv`

The script reports which optional feature-engineering columns were created and which were skipped because the required source columns were not available.
