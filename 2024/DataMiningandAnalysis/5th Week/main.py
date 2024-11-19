import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from fcmeans import FCM

# 1. 加载 Iris 数据集
iris = load_iris()
X = iris.data  # 数据
y = iris.target  # 真实标签

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-means 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 3. DBSCAN 聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# 4. 模糊C均值聚类
fcm = FCM(n_clusters=3)
fcm.fit(X_scaled)
fcm_labels = np.argmax(fcm.u, axis=1)

# 5. 评估聚类算法性能
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
fcm_silhouette = silhouette_score(X_scaled, fcm_labels)

print("\nSilhouette Score:")
print(f"K-Means: {kmeans_silhouette:.4f}")
print(f"DBSCAN: {dbscan_silhouette:.4f}")
print(f"FCM: {fcm_silhouette:.4f}")

# 6. 可视化聚类结果
# 创建 4 个单独的图形，分别显示每个聚类算法的结果
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 原始数据
axs[0, 0].scatter(X[:, 0], X[:, 1], c='gray', s=30, cmap='viridis')
axs[0, 0].set_title("Original Data")

# K-means 聚类结果
axs[0, 1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=30, cmap='viridis')
axs[0, 1].set_title("K-means Clustering")

# DBSCAN 聚类结果
axs[1, 0].scatter(X[:, 0], X[:, 1], c=dbscan_labels, s=30, cmap='viridis')
axs[1, 0].set_title("DBSCAN Clustering")

# FCM 聚类结果
axs[1, 1].scatter(X[:, 0], X[:, 1], c=fcm_labels, s=30, cmap='viridis')
axs[1, 1].set_title("FCM Clustering")

# 设置图形间距
plt.tight_layout()
plt.show()
