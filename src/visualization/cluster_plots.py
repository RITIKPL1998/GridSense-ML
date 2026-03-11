import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_cluster_pca(df, features):

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = df.copy()
    df_plot["pca_1"] = X_pca[:,0]
    df_plot["pca_2"] = X_pca[:,1]

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x="pca_1",
        y="pca_2",
        hue="cluster",
        palette="tab10",
        data=df_plot,
        s=8
    )

    plt.title("KMeans Cluster Visualization (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.tight_layout()

    mlflow.log_figure(plt.gcf(), "cluster_pca_visualization.png")

    plt.close()