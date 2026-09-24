"""
Skin Lesion Classifier - Training Script
Uses MobileNetV2 pretrained on ImageNet, fine-tuned on ISIC dataset.
Run this ONCE on your dev machine. Outputs: model_weights.h5
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 20
DATA_DIR   = "data"
MODEL_PATH = "model_weights.keras"

AUTOTUNE = tf.data.AUTOTUNE

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label

def load_dataset(split, augment_data=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, split),
        labels="inferred",
        label_mode="binary",
        class_names=["benign", "malignant"],
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=42,
    )
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

print("Loading datasets...")
train_ds = load_dataset("train", augment_data=True)
val_ds   = load_dataset("val",   augment_data=False)

# Collect labels for class weights
all_labels = np.concatenate([y.numpy() for _, y in train_ds])
cw = compute_class_weight("balanced", classes=np.array([0., 1.]), y=all_labels.flatten())
class_weights = {0: cw[0], 1: cw[1]}
print(f"Class weights: {class_weights}")

# ── Model ─────────────────────────────────────────────────────────────────────
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max"),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_auc", mode="max"),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_auc", mode="max"),
]

# Phase 1: head only
print("\n=== Phase 1: Training head ===")
model.fit(train_ds, epochs=EPOCHS // 2, validation_data=val_ds,
          class_weight=class_weights, callbacks=callbacks)

# Phase 2: fine-tune top 30 layers
print("\n=== Phase 2: Fine-tuning ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
          class_weight=class_weights, callbacks=callbacks)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== Final evaluation ===")
results = model.evaluate(val_ds)
metrics = dict(zip(model.metrics_names, results))
print(metrics)

meta = {
    "accuracy": round(metrics["accuracy"] * 100, 2),
    "auc": round(metrics["auc"] * 100, 2),
    "img_size": IMG_SIZE,
    "classes": ["benign", "malignant"],
    "model_file": MODEL_PATH,
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✓ Model saved to {MODEL_PATH}")
print(f"✓ Accuracy: {meta['accuracy']}%  |  AUC: {meta['auc']}%")
