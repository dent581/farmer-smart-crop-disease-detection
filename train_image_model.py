"""
train_image_model.py  —  Train CNN Plant Disease Detection Model
=================================================================
Usage:
  python train_image_model.py --demo              # quick demo model (no dataset needed)
  python train_image_model.py --data_dir ./dataset --epochs 25

Dataset structure (PlantVillage style):
  dataset/
    Apple___Apple_scab/    img1.jpg ...
    Apple___healthy/       img1.jpg ...
    Tomato___Late_blight/  img1.jpg ...
    ...
"""

import os, json, argparse, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def create_demo_model():
    """
    Build a small lightweight CNN with random weights.
    Predictions will be random — for UI testing only.
    Replace by training with a real dataset for production use.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    log.info("Building DEMO model (random weights — predictions are not accurate).")
    log.info("For accurate results download the PlantVillage dataset and run with --data_dir.")

    n = len(CLASS_NAMES)
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
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
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    os.makedirs("models", exist_ok=True)
    model.save("models/plant_disease_model.h5"C)
    with open("models/class_names.json", "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    log.info(f"✅  Demo model saved  →  models/plant_disease_model.h5")
    log.info(f"✅  {n} class names  →  models/class_names.json")
    log.warning("⚠️  DEMO MODEL: predictions are random. Train with real data for production.")
    return model


def train_with_dataset(data_dir: str, epochs: int = 20, batch: int = 32, img: int = 224):
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                             ReduceLROnPlateau)
    from tensorflow.keras.models import load_model

    log.info(f"Loading dataset from: {data_dir}")

    aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25, width_shift_range=0.15, height_shift_range=0.15,
        horizontal_flip=True, zoom_range=0.2, shear_range=0.12,
        brightness_range=[0.8, 1.2], validation_split=0.2
    )

    train_gen = aug.flow_from_directory(
        data_dir, target_size=(img, img), batch_size=batch,
        class_mode="categorical", subset="training", shuffle=True
    )
    val_gen = aug.flow_from_directory(
        data_dir, target_size=(img, img), batch_size=batch,
        class_mode="categorical", subset="validation", shuffle=False
    )

    num_classes = train_gen.num_classes
    idx_to_name = {v: k for k, v in train_gen.class_indices.items()}
    names = [idx_to_name[i] for i in range(num_classes)]

    log.info(f"  {num_classes} classes, {train_gen.samples} training images")

    model_path = "models/plant_disease_model.h5"

    # ✅ LOAD EXISTING MODEL (skip Phase 1)
    if os.path.exists(model_path):
        log.info("✅ Loading existing trained model for Phase 2...")
        model = load_model(model_path)
        base = model.layers[0]

    else:
        log.info("⚡ Creating new model (Phase 1 will run)...")

        base = MobileNetV2(input_shape=(img, img, 3), include_top=False, weights="imagenet")
        base.trainable = False

        model = models.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy", metrics=["accuracy"])

        log.info("Phase 1 — training classifier head …")
        model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # ✅ PHASE 2 — FINE TUNING
    log.info("Phase 2 — fine-tuning top layers …")
    base.trainable = True

    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint("models/plant_disease_model.h5", save_best_only=True,
                        monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    model.fit(train_gen, validation_data=val_gen,
              epochs=max(5, epochs // 4), callbacks=cbs)

    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w") as f:
        json.dump(names, f, indent=2)

    log.info("✅  Model  →  models/plant_disease_model.h5")
    log.info("✅  Names  →  models/class_names.json")

    return model, names

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="Path to dataset directory")
    ap.add_argument("--epochs",   type=int, default=20)
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--demo",     action="store_true",
                    help="Create a quick demo model without a dataset")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)

    if args.demo or args.data_dir is None:
        if args.data_dir is None:
            log.warning("No --data_dir provided → creating demo model.")
        create_demo_model()
    else:
        if not os.path.isdir(args.data_dir):
            log.error(f"Dataset directory not found: {args.data_dir}")
            exit(1)
        train_with_dataset(args.data_dir, args.epochs, args.batch)

    print("\n🚀  Done! Now start the app:  python app.py\n")
