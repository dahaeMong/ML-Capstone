# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_and_preprocess(csv_path="data/match_sample_preprocessed.csv"):

    # -----------------------
    # 1. Load dataset
    # -----------------------
    df = pd.read_csv(csv_path)

    # Target
    y = df["win"]

    # -----------------------
    # 2. Identify feature columns
    # -----------------------
    numeric_features = ["kills", "deaths", "assists", "damage", "headshot_pct", "KDA"]
    categorical_onehot = ["map"]           # small cardinality → one-hot
    categorical_label = ["agent"]          # large cardinality → label encoding

    # Drop unnecessary columns
    drop_cols = ["match_id", "player"]     # Not needed for ML model
    df = df.drop(columns=drop_cols)

    # -----------------------
    # 3. Apply label encoding to agent
    # -----------------------
    label_encoder = LabelEncoder()
    df["agent_encoded"] = label_encoder.fit_transform(df["agent"])

    # -----------------------
    # 4. Feature list
    # -----------------------
    feature_cols = numeric_features + categorical_onehot + ["agent_encoded"]

    X = df[feature_cols].copy()

    # -----------------------
    # 5. Numeric preprocessing
    #    - KDA log transform
    #    - Scaling
    # -----------------------
    X["KDA"] = np.log1p(X["KDA"])

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # -----------------------
    # 6. Categorical transformer
    # -----------------------
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # -----------------------
    # 7. Column transformer
    # -----------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_onehot),
            # agent_encoded stays numeric → no transform
        ],
        remainder="passthrough"
    )

    # -----------------------
    # 8. Fit-transform
    # -----------------------
    X_processed = preprocessor.fit_transform(X)

    # -----------------------
    # 9. Train/test split
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor, label_encoder
