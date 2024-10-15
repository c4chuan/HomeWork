# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from numpy.linalg import inv

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 将数据集转换为DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].apply(lambda i: iris.target_names[i])

# 可视化描述统计
## 绘制特征的直方图
for feature in feature_names:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=feature, hue='species', kde=True, element='step')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

## 绘制特征之间的散点图矩阵
sns.pairplot(df, hue='species', vars=feature_names)
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

## 绘制箱线图
for feature in feature_names:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Species')
    plt.xlabel('Species')
    plt.ylabel(feature)
    plt.show()

# 定义不同的距离度量方式
V = np.cov(X.T)
VI = inv(V)  # 计算协方差矩阵的逆矩阵

metrics = {
    'Euclidean (p=2)': {'metric': 'minkowski', 'p': 2},
    'Manhattan (p=1)': {'metric': 'minkowski', 'p': 1},
    'Chebyshev': {'metric': 'chebyshev'},
    'Minkowski (p=3)': {'metric': 'minkowski', 'p': 3},
    'Mahalanobis': {'metric': 'mahalanobis', 'metric_params': {'VI': VI}},
}

# 测试不同的k值和距离度量方式
results = {}
k_values = range(1, 21)

for metric_name, metric_params in metrics.items():
    accuracies = []
    for k in k_values:
        if 'p' in metric_params:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric_params['metric'], p=metric_params['p'])
        elif 'metric_params' in metric_params:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric_params['metric'],
                                       metric_params=metric_params['metric_params'])
        else:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric_params['metric'])

        scores = cross_val_score(clf, X, y, cv=5)
        accuracies.append(scores.mean())
    results[metric_name] = accuracies

# 将结果可视化
results_df = pd.DataFrame(results, index=k_values)
results_df.index.name = 'k'

plt.figure(figsize=(12, 8))
for metric_name in results:
    plt.plot(k_values, results[metric_name], label=metric_name)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('k-NN Accuracy with Different Distance Metrics')
plt.legend(title='Distance Metrics')
plt.grid(True)
plt.show()
