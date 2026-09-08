# predict.py — Fixed for 3-class model (no LabelEncoder)

import joblib
import numpy as np

model       = joblib.load("pothole_model.pkl")
class_names = joblib.load("class_names.pkl")   # {0:'normal', 1:'pothole', 2:'speed_breaker'}

def predict(x, y, z, magnitude):
    data  = np.array([[x, y, z, magnitude]], dtype=np.float32)
    probs = model.predict_proba(data)[0]   # shape: (3,) for 3 classes
    return {class_names[i]: round(float(p), 4) for i, p in enumerate(probs)}

if __name__ == "__main__":
    tests = [
        ("Smooth road",    -1.0,  0.2,  9.8, 9.87),
        ("Pothole hit",     3.5,  4.2, 14.1, 15.2),
        ("Speed breaker",   1.2,  0.5, 12.5, 12.6),
    ]
    for label, x, y, z, mag in tests:
        probs = predict(x, y, z, mag)
        top   = max(probs, key=probs.get)
        print(f"{label:>14}: {probs}  → {top}")