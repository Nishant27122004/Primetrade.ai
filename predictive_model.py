import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=UserWarning)


merged = pd.read_csv("daily_with_sentiment.csv", parse_dates=["date"])

# --- Step 4: Predictive Modeling ---
# Define target: whether total_closed_pnl > 0 (profitable day)
merged["pnl_positive"] = (merged["total_closed_pnl"] > 0).astype(int)

# Features: sentiment and trading metrics
features = ["sentiment_value","sentiment_lag_1","sentiment_lag_3","sentiment_lag_7",
            "total_trades","total_volume_usd","avg_leverage","win_rate"]
X = merged[features].fillna(0)
y = merged["pnl_positive"]

# Since dataset is very small, just train/test split (but results will be unstable)
if len(merged) > 4:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    feat_importance = dict(zip(features, model.feature_importances_))
else:
    report, cm, feat_importance = "Not enough rows to train/test", None, None

report, cm, feat_importance