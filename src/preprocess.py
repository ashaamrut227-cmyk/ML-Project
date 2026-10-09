
"""Data loading, cleaning, splitting and scaling for the Personal Loan project."""
from pathlib import Path
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
 
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = "Personal Loan"
RANDOM_STATE = 42
DROP_ZIP = True  # ZIP code is a location ID, not a real number; set False to keep it
 
 
def load_data(path=None) -> pd.DataFrame:
    """Load the Kaggle CSV. If no path is given, use the first CSV found in data/."""
    if path is None:
        csvs = sorted(DATA_DIR.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV found in {DATA_DIR}. Download the Kaggle dataset first (see README)."
            )
        path = csvs[0]
    print(f"Loading data from {path}")
    return pd.read_csv(path)
 
 
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the ID column, fix negative Experience values, optionally drop ZIP code."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
 
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
 
    # Some rows have negative years of experience, which is impossible (data entry error).
    # We take the absolute value, the usual fix for this dataset.
    if "Experience" in df.columns:
        n_neg = int((df["Experience"] < 0).sum())
        if n_neg:
            print(f"Fixing {n_neg} negative Experience values with abs()")
        df["Experience"] = df["Experience"].abs()
 
    if DROP_ZIP and "ZIP Code" in df.columns:
        df = df.drop(columns=["ZIP Code"])
 
    df = df.dropna().drop_duplicates()
    return df
 
 
def prepare_data(path=None, test_size=0.2):
    """Return scaled X_train, X_test and y_train, y_test (scaler fit on train only)."""
    df = clean_data(load_data(path))
    print(f"Dataset shape after cleaning: {df.shape}")
    print("Class balance:\n", df[TARGET].value_counts(normalize=True).round(3).to_string())
 
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
 
    # Fit the scaler on training data only, so no test information leaks in.
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train_s, X_test_s, y_train, y_test
 
