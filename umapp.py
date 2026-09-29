# ============================================================
# IMPORTS
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import kagglehub

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import umap

# ============================================================
# 1. DOWNLOAD DATASET
# ============================================================

path = kagglehub.dataset_download("imsparsh/flowers-dataset")

print("Dataset path:", path)

flower_path = os.path.join(path, "train")

print("Flower path:", flower_path)

# ============================================================
# 2. LOAD DATASET
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(flower_path)
    if os.path.isdir(os.path.join(flower_path, folder))
])

print("Classes:", classes)

IMG_SIZE = (224, 224)

# Keep this limited because of RAM
MAX_IMAGES_PER_CLASS = 200

images = []
labels = []

for class_name in classes:
    class_path = os.path.join(flower_path, class_name)
    count = 0

    for filename in os.listdir(class_path):
        if count >= MAX_IMAGES_PER_CLASS:
            break

        file_path = os.path.join(class_path, filename)

        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(IMG_SIZE)

            images.append(np.array(img))
            labels.append(class_name)
            count += 1

        except Exception:
            pass

    print(f"{class_name}: {count} images")

images = np.array(images, dtype="float32")
labels = np.array(labels)

print("\nImages:", images.shape)
print("Labels:", labels.shape)

# ============================================================
# 3. NUMERICAL LABELS
# ============================================================

class_to_index = {
    name: i
    for i, name in enumerate(classes)
}

y = np.array([
    class_to_index[label]
    for label in labels
])

print("Numerical labels:", y.shape)

# ============================================================
# 4. PREPROCESS
# ============================================================

X = preprocess_input(images)

print("Preprocessed X:", X.shape)

# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)

# ============================================================
# 6. MOBILE NET V2
# ============================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)

base_model.trainable = False

x = base_model.output

# Flower-specific feature representation
x = Dense(
    128,
    activation="relu",
    name="flower_features"
)(x)

x = Dropout(0.3)(x)

output = Dense(
    len(classes),
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ============================================================
# 7. COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 8. TRAIN CLASSIFICATION HEAD
# ============================================================

print("\nTraining...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1
)

feature_model = Model(
    inputs=model.input,
    outputs=model.get_layer("flower_features").output
)

print("\nExtracting learned flower features...")

features = feature_model.predict(
    X,
    batch_size=16,
    verbose=0
)

print("Feature shape:", features.shape)

# ============================================================
# 11. STANDARD SCALING
# ============================================================

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

print("Scaled features:", scaled_features.shape)

print("\nRunning UMAP...")

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=50,  # 15
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

features_2d = umap_model.fit_transform(scaled_features)

print("UMAP feature shape:", features_2d.shape)

plt.figure(figsize=(12, 8))

for class_name in classes:
    mask = labels == class_name

    plt.scatter(
        features_2d[mask, 0],
        features_2d[mask, 1],
        label=class_name,
        s=40,
        alpha=0.7
    )

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.title("UMAP")

plt.legend()
plt.tight_layout()

plt.savefig(
    "umap_out_n=50_d=0.1.png",
    dpi=300
)