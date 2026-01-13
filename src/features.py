# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# Dans src/features.py, vous pouvez enrichir le pipeline:
# (exemple: no-op pédagogique via FunctionTransformer)
from sklearn.preprocessing import FunctionTransformer
def _clip(X):
    return X.clip(-3, 3)
def build_numeric_preprocess():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(_clip)),
    ])