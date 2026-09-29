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
import umap.umap_ as umap

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# ============================================================
# 1. DOWNLOAD DATASET
# ============================================================
path = kagglehub.dataset_download("imsparsh/flowers-dataset")
print("Dataset path:", path)
flower_path = os.path.join(path, "train")
print("Flower path:", flower_path)

# ============================================================
# 2. FIND CLASSES
# ============================================================
classes = sorted([
    folder for folder in os.listdir(flower_path)
    if os.path.isdir(os.path.join(flower_path, folder))
])
print("\nClasses:", classes)

# ============================================================
# 3. LOAD DATASET
# ============================================================
IMG_SIZE = (224, 224)
MAX_IMAGES_PER_CLASS = 200
images, labels = [], []

for class_name in classes:
    class_path = os.path.join(flower_path, class_name)
    count = 0

    for filename in os.listdir(class_path):
        if count >= MAX_IMAGES_PER_CLASS:
            break

        file_path = os.path.join(class_path, filename)
        try:
            img = Image.open(file_path).convert("RGB").resize(IMG_SIZE)
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
# 4. NUMERICAL LABELS
# ============================================================
class_to_index = {name: i for i, name in enumerate(classes)}
y = np.array([class_to_index[label] for label in labels])

print("\nNumerical labels:", y.shape)
print("Class mapping:", class_to_index)

# ============================================================
# 5. MOBILENETV2 PREPROCESSING
# ============================================================
X = preprocess_input(images)
print("\nPreprocessed X:", X.shape)

# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)

# ============================================================
# 7. MOBILENETV2
# ============================================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
    pooling="avg"
)
base_model.trainable=False

# ============================================================
# 8. FLOWER-SPECIFIC FEATURE LAYER
# ============================================================
x = base_model.output
x = Dense(128, activation="relu", name="flower_features")(x)
x = Dropout(0.3)(x)
output = Dense(len(classes), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# ============================================================
# 9. COMPILE MODEL
# ============================================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 10. TRAIN CLASSIFICATION HEAD
# ============================================================
print("\nTraining classification head...")
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1
)

# ============================================================
# 11. VALIDATION ACCURACY
# ============================================================
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print("\nValidation loss:", val_loss)
print("Validation accuracy:", val_accuracy)

# ============================================================
# 12. EXTRACT 128-D FEATURES
# ============================================================
feature_model = Model(
    inputs=model.input,
    outputs=model.get_layer("flower_features").output
)

print("\nExtracting learned flower features...")
features = feature_model.predict(X, batch_size=16, verbose=1)
print("Feature shape:", features.shape)

# ============================================================
# 13. STANDARD SCALING
# ============================================================
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print("\nScaled feature shape:", scaled_features.shape)

# ============================================================
# 14. FIND EPS AUTOMATICALLY
# ============================================================
MIN_SAMPLES = 10
print("\nCalculating k-distances...")

neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES)
neighbors.fit(scaled_features)
distances, indices = neighbors.kneighbors(scaled_features)
k_distances = np.sort(distances[:, -1])

# ============================================================
# 15. AUTOMATIC KNEE DETECTION
# ============================================================
x_values = np.arange(len(k_distances))
knee_locator = KneeLocator(
    x_values,
    k_distances,
    curve="convex",
    direction="increasing"
)
knee_index = knee_locator.knee

# ============================================================
# 16. AUTOMATIC EPS SELECTION
# ============================================================
if knee_index is not None:
    EPS = k_distances[knee_index]

    print("\n====================================")
    print("AUTOMATIC EPS SELECTION")
    print("====================================")
    print("Knee index:", knee_index)
    print("Automatically selected EPS:", round(EPS, 4))
else:
    EPS = np.percentile(k_distances, 90)

    print("\n====================================")
    print("AUTOMATIC EPS SELECTION")
    print("====================================")
    print("No clear knee detected.")
    print(
        "Using 90th percentile as fallback EPS:",
        round(EPS, 4)
    )

# ============================================================
# 17. K-DISTANCE PLOT
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(k_distances, label="K-distance")

if knee_index is not None:
    plt.axvline(
        knee_index,
        linestyle="--",
        label=f"Knee → EPS = {EPS:.3f}"
    )
    plt.axhline(EPS, linestyle="--")

plt.xlabel("Points sorted by distance")
plt.ylabel(f"{MIN_SAMPLES}th nearest-neighbor distance")
plt.title("K-Distance Plot with Automatic EPS Selection")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 18. DBSCAN ON 128-D FEATURES
# ============================================================
print("\nRunning DBSCAN...")
print("Automatically selected EPS:", round(EPS, 4))
print("Min samples:", MIN_SAMPLES)

dbscan = DBSCAN(
    eps=EPS,
    min_samples=MIN_SAMPLES,
    metric="euclidean"
)
cluster_labels = dbscan.fit_predict(scaled_features)

# ============================================================
# 19. DBSCAN RESULTS
# ============================================================
unique_clusters = sorted(np.unique(cluster_labels))
n_clusters = len(set(cluster_labels) - {-1})
n_noise = np.sum(cluster_labels == -1)

print("\n====================================")
print("DBSCAN RESULTS")
print("====================================")
print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
print("Total points:", len(cluster_labels))
print(
    "Noise percentage:",
    round(100 * n_noise / len(cluster_labels), 2),
    "%"
)

# ============================================================
# 20. CLUSTER SIZES
# ============================================================
print("\nCluster sizes:")

for cluster in unique_clusters:
    count = np.sum(cluster_labels == cluster)

    if cluster == -1:
        print(f"Noise (-1): {count}")
    else:
        print(f"Cluster {cluster}: {count}")

# ============================================================
# 21. CLASS COMPOSITION
# ============================================================
print("\n====================================")
print("CLASS COMPOSITION OF DBSCAN CLUSTERS")
print("====================================")

for cluster in unique_clusters:
    mask = cluster_labels == cluster
    cluster_classes = labels[mask]

    if cluster == -1:
        print("\nNoise:")
    else:
        print(f"\nCluster {cluster}:")

    for class_name in classes:
        count = np.sum(cluster_classes == class_name)

        if count > 0:
            print(f"  {class_name}: {count}")

# ============================================================
# 22. UMAP FOR VISUALIZATION ONLY
# ============================================================
print("\nCreating UMAP representation for visualization...")

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=50,
    min_dist=0.05,
    metric="euclidean",
    random_state=42
)

umap_features = umap_model.fit_transform(scaled_features)

print("Original feature shape:", scaled_features.shape)
print("UMAP feature shape:", umap_features.shape)

# ============================================================
# 23. UMAP + DBSCAN VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 8))

clusters_only = [c for c in unique_clusters if c != -1]
cmap = plt.cm.get_cmap("tab10", max(len(clusters_only), 1))

for color_index, cluster in enumerate(clusters_only):
    mask = cluster_labels == cluster

    plt.scatter(
        umap_features[mask, 0],
        umap_features[mask, 1],
        color=cmap(color_index),
        label=f"Cluster {cluster}",
        s=45,
        alpha=0.75,
        edgecolors="none"
    )

noise_mask = cluster_labels == -1

if np.sum(noise_mask) > 0:
    plt.scatter(
        umap_features[noise_mask, 0],
        umap_features[noise_mask, 1],
        color="lightgray",
        label="Noise",
        s=35,
        alpha=0.5,
        edgecolors="none"
    )

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title(
    "UMAP Visualization of DBSCAN Clusters "
    "from 128-D MobileNetV2 Features"
)
plt.legend(loc="best")
plt.tight_layout()

plt.savefig(
    "umap_dbscan_feature_space.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()