from sklearn.datasets import load_iris
import umap
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)


reducer = umap.UMAP(
    n_neighbors=50,
    n_components=2,
    random_state=42,
    min_dist=0.9
)

X_umap = reducer.fit_transform(X)

print(X_umap.shape)

classes = iris.target_names

plt.figure(figsize=(8, 6))

handles = []

for i, class_name in enumerate(classes):
    mask = y == i

    points = plt.scatter(
        X_umap[mask, 0],
        X_umap[mask, 1]
    )

    handles.append(points)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP - Iris Dataset")

plt.legend(
    handles=handles,
    labels=classes,
    title="Iris Class"
)

plt.tight_layout()
plt.savefig("umap_iris_n50 m0.9.png", dpi=300)