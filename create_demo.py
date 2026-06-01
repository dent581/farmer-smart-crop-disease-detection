"""
create_demo.py — Creates demo CNN model
Works with TensorFlow 2.15+ and Keras 3.x
"""
import os, json, sys

print(f"Python: {sys.version}")
os.makedirs("models", exist_ok=True)

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

with open("models/class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)
print(f"✅ {len(CLASS_NAMES)} class names saved → models/class_names.json")

print("Building CNN model (this takes ~30 seconds)...")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

n = len(CLASS_NAMES)

model = keras.Sequential([
    keras.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(n, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Save in both formats for compatibility
try:
    model.save("models/plant_disease_model.h5")
    print("✅ Model saved → models/plant_disease_model.h5")
except Exception as e:
    print(f"h5 save failed ({e}), trying keras format...")
    model.save("models/plant_disease_model.keras")
    print("✅ Model saved → models/plant_disease_model.keras")

print("\n⚠️  DEMO MODEL: predictions are random (no real training data)")
print("🚀 Now run: python train_text_model.py")
print("   Then run: python app.py")
