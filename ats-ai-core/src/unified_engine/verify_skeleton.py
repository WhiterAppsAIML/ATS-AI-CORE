import sys
sys.path.insert(0, ".")
import tensorflow as tf
from src.unified_engine.unified_model import build_unified_model, RSG_NUM_CLASSES

print("Building unified model skeleton...")
model = build_unified_model()

print("\n=== OUTPUT SHAPE CHECK ===")
dummy_resume = tf.constant(["Experienced Python developer 5 years Django REST APIs"])
dummy_jd     = tf.constant(["Looking for senior backend software engineer"])
ats_out, dom_out, rsg_out = model([dummy_resume, dummy_jd], training=False)

print(f"ATS  output shape: {ats_out.shape}  -- expected (1, 1)")
print(f"DOM  output shape: {dom_out.shape}  -- expected (1, 7)")
print(f"RSG  output shape: {rsg_out.shape}  -- expected (1, {RSG_NUM_CLASSES})")

assert ats_out.shape == (1, 1),               "FAIL: ATS shape wrong"
assert dom_out.shape == (1, 7),               "FAIL: Domain shape wrong"
assert rsg_out.shape == (1, RSG_NUM_CLASSES), "FAIL: RSG shape wrong"
print("All shapes: CORRECT")

print("\n=== ENCODER FROZEN CHECK ===")
frozen_names = [l.name for l in model.layers if not l.trainable]
assert "mobile_use_encoder" in frozen_names, "FAIL: USE encoder is not frozen"
print(f"Frozen layers: {frozen_names}")
print("Encoder frozen: CONFIRMED")

print("\n=== PARAMETER COUNT ===")
total    = model.count_params()
n_frozen = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
print(f"Total params    : {total:,}")
print(f"Frozen (encoder): {n_frozen:,}")
print(f"Trainable heads : {total - n_frozen:,}")

print("\n=== SKELETON VERIFICATION: PASS ===")
print("Send this output to Sai. Ready for INJECTION-1B.")
