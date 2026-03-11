import matplotlib.pyplot as plt
import mlflow
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def find_optimal_k(df, features, k_range=range(2, 10)):

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        inertias.append(model.inertia_)

    # plot elbow curve
    plt.figure(figsize=(8,5))
    plt.plot(list(k_range), inertias, marker="o")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method For Optimal k")

    plt.grid(True)
    plt.tight_layout()

    # automatic k detection
    drops = []
    for i in range(1, len(inertias)):
        drops.append(inertias[i-1] - inertias[i])

    best_k = list(k_range)[drops.index(max(drops)) + 1]

    print(f"\nSuggested optimal k: {best_k}")

    # MLflow logging
    with mlflow.start_run(run_name="kmeans_elbow_method"):

        mlflow.log_figure(plt.gcf(), "elbow_plot.png")
        mlflow.log_param("auto_selected_k", best_k)

    plt.close()

    return best_k