# train_model.py — Fixed for your actual dataset
#
# FIXES:
#  1. Labels are already integers (0,1,2) — no LabelEncoder needed, caused the crash.
#  2. target_names passed as plain strings to classification_report.
#  3. Timestamp column dropped — it's not a feature.
#  4. n_estimators=100, max_depth=10 — better accuracy than 50/5.
#  5. class_weight="balanced" handles imbalance (0:19229 vs 2:3060).
#
# Label mapping (from your data):
#   0 = Normal road
#   1 = Pothole
#   2 = Speed breaker

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ── Load ───────────────────────────────────────────────────────
data = pd.read_csv("data.csv")
print(f"✅ Dataset loaded: {len(data)} rows")
print(f"   Columns: {list(data.columns)}")

# Drop Timestamp — not a feature
if 'Timestamp' in data.columns:
    data = data.drop(columns=['Timestamp'])

# ── Features & Labels ──────────────────────────────────────────
X = data[['X', 'Y', 'Z', 'Magnitude']].values.astype(np.float32)
y = data['Label'].values.astype(int)   # already 0/1/2

CLASS_NAMES = {0: "normal", 1: "pothole", 2: "speed_breaker"}
# FIX: plain strings — numpy.int64 has no len(), caused the crash
target_names = [CLASS_NAMES[i] for i in sorted(np.unique(y))]

print(f"\n   Label distribution:")
for label, name in CLASS_NAMES.items():
    print(f"     {label} ({name:14s}): {np.sum(y == label)}")

# ── Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model ──────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n⏳ Training...")
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)

print(f"\n📊 Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
header = f"{'':>18}" + "".join(f"{n:>14}" for n in target_names)
print(header)
for i, row in enumerate(cm):
    print(f"  actual {target_names[i]:>10}  " + "".join(f"{v:>14}" for v in row))

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Feature Importances:")
for name, imp in zip(['X', 'Y', 'Z', 'Magnitude'], model.feature_importances_):
    bar = '█' * int(imp * 40)
    print(f"  {name:10s} {imp:.4f}  {bar}")

# ── Save ───────────────────────────────────────────────────────
joblib.dump(model,       "pothole_model.pkl")
joblib.dump(CLASS_NAMES, "class_names.pkl")

print("\n✅ pothole_model.pkl saved")
print("✅ class_names.pkl  saved")
print("\nNext → run: python convert_to_onnx.py")