"""
CAP182 SS2 Part B
Preprocessing

Input:
    data/engineered_data.csv
Outputs:
    data/processed/
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT_PATH = Path("data/engineered_data.csv")
OUTPUT_DIR = Path("data/processed")
TARGET_COLUMN = "returned"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def build_preprocessor(numeric_columns, categorical_columns, scale_numeric):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",)

def save_processed_data(processor, X_train, X_test, output_prefix, output_dir):
    X_train_processed = processor.transform(X_train)
    X_test_processed = processor.transform(X_test)

    feature_names = processor.get_feature_names_out()

    pd.DataFrame(
        X_train_processed,
        columns=feature_names
    ).to_csv(
        output_dir / f"X_train_{output_prefix}.csv",
        index=False)

    pd.DataFrame(
        X_test_processed,
        columns=feature_names
    ).to_csv(
        output_dir / f"X_test_{output_prefix}.csv",
        index=False)
    return feature_names


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}\n"
            "Run feature_engineering.py first.")

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Remove duplicate records.
    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate records found: {duplicate_count}")

    if duplicate_count:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Shape after removing duplicates: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns include: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Ensure the binary target is numeric where possible.
    if not pd.api.types.is_numeric_dtype(y):
        mapping = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
            "not returned": 0,
            "returned": 1,}
        
        y_clean = y.astype(str).str.strip().str.lower().map(mapping)
        if y_clean.isna().any():
            raise ValueError(
                "The target column is not numeric and contains values "
                "that could not be mapped to 0/1.")
        y = y_clean.astype(int)

    # Stratified 80/20 train/test split before fitting transformations.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,)

    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in X_train.columns
        if column not in numeric_columns]

    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Logistic Regression pipeline: median imputation + scaling, with categorical imputation and one-hot encoding.
    lr_preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
        scale_numeric=True,)

    lr_preprocessor.fit(X_train)
    lr_features = save_processed_data(
        lr_preprocessor,
        X_train,
        X_test,
        "lr",
        OUTPUT_DIR,)

    joblib.dump(
        lr_preprocessor,
        OUTPUT_DIR / "lr_preprocessor.joblib")

    # Random Forest pipeline: no numerical scaling.
    rf_preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
        scale_numeric=False,)

    rf_preprocessor.fit(X_train)
    rf_features = save_processed_data(
        rf_preprocessor,
        X_train,
        X_test,
        "rf",
        OUTPUT_DIR,)

    joblib.dump(
        rf_preprocessor,
        OUTPUT_DIR / "rf_preprocessor.joblib")

    pd.DataFrame({"returned": y_train}).to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False)
    pd.DataFrame({"returned": y_test}).to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False)

    # LR and RF should have the same encoded feature names.
    pd.Series(lr_features).to_csv(
        OUTPUT_DIR / "feature_names.txt",
        index=False,
        header=False,)

    print("\nPreprocessing completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Encoded feature count: {len(lr_features)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
