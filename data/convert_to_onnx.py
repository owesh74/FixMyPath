# convert_to_onnx.py — Fixed for 3-class model
#
# FIXES:
#  1. No LabelEncoder — uses class_names.pkl instead.
#  2. zipmap=False — outputs float array not dict (required for Android ONNX Runtime).
#  3. Verifies all 3 class probabilities in test output.
#  4. Pothole index is 1 (confirmed from class_names mapping).

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# ── Load ───────────────────────────────────────────────────────
model       = joblib.load("pothole_model.pkl")
class_names = joblib.load("class_names.pkl")   # {0:'normal', 1:'pothole', 2:'speed_breaker'}
print(f"✅ Model loaded")
print(f"   Class map: {class_names}")
print(f"   Pothole index: {[k for k,v in class_names.items() if v=='pothole'][0]}")

# ── Convert ────────────────────────────────────────────────────
initial_type = [('float_input', FloatTensorType([None, 4]))]

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={type(model): {'zipmap': False}},  # plain float array, not dict
    target_opset=12
)

with open("pothole_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ pothole_model.onnx saved")

# ── Verify ─────────────────────────────────────────────────────
sess = rt.InferenceSession("pothole_model.onnx")
input_name   = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

print(f"\n📋 Input  : '{input_name}'  shape={sess.get_inputs()[0].shape}")
for i, o in enumerate(sess.get_outputs()):
    print(f"   Output[{i}]: '{o.name}'  shape={o.shape}")

# Test: smooth road
test = np.array([[-1.0, 0.2, 9.8, 9.87]], dtype=np.float32)
label, probs = sess.run(output_names, {input_name: test})
print(f"\n🧪 Smooth road test:")
for i, p in enumerate(probs[0]):
    print(f"   {class_names[i]:>14}: {p:.4f}")
print(f"   Predicted class: {class_names[int(label[0])]} (index {label[0]})")

print(f"""
────────────────────────────────────────
✅ ONNX export complete!

Copy pothole_model.onnx to Android:
  app/src/main/assets/pothole_model.onnx

In Android, output[1] = probabilities array
  probs[0][0] = normal prob
  probs[0][1] = pothole prob      ← use this
  probs[0][2] = speed_breaker prob
────────────────────────────────────────
""")