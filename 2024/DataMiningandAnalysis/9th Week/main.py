# -*- coding: utf-8 -*-
"""
数据挖掘任务：使用粒子群算法对鸢尾花数据集进行特征选择

步骤：
1. 选取鸢尾花数据集，并进行简单介绍。
2. 进行可视化描述性统计分析。
3. 运用粒子群算法（PSO）进行特征选择。
4. 思考算法的改进空间。
5. 改进算法并与原算法结果进行可视化对比。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pyswarm import pso  # 需要安装 pyswarm 库

# 1. 选取鸢尾花数据集并简单介绍
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("鸢尾花数据集简介：")
print(f"样本数量：{X.shape[0]}")
print(f"特征数量：{X.shape[1]}")
print(f"特征名称：{feature_names}")
print(f"类别名称：{target_names}")

# 将数据转换为DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 2. 可视化描述性统计分析
def descriptive_statistics(df):
    print("\n描述性统计信息：")
    print(df.describe())

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.drop('target', axis=1), orient="h")
    plt.title("特征箱线图")
    plt.show()

    # 绘制散点图矩阵
    sns.pairplot(df, vars=feature_names, hue='target', diag_kind='kde')
    plt.suptitle("特征散点图矩阵", y=1.02)
    plt.show()

    # 绘制相关性热图
    plt.figure(figsize=(8, 6))
    corr = df.iloc[:, :-1].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("特征相关性热图")
    plt.show()

descriptive_statistics(df)

# 3. 运用粒子群算法（PSO）进行特征选择
# 定义适应度函数：使用选定特征训练KNN分类器并返回交叉验证准确率的负值（因为pso最小化）
def fitness_function(weights):
    # 将权重转换为二进制选择特征
    binary_weights = weights > 0.5
    if np.sum(binary_weights) == 0:
        return 1  # 最大化适应度，所以返回较大的值
    selected_features = np.array(feature_names)[binary_weights]
    X_selected = df[selected_features].values  # 使用 DataFrame 选择特征
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return 1 - accuracy  # 最小化

# 粒子群算法参数
lb = [0] * X.shape[1]  # 下界
ub = [1] * X.shape[1]  # 上界

# 执行PSO
print("执行原始PSO进行特征选择...")
best_weights, best_cost = pso(fitness_function, lb, ub, swarmsize=30, maxiter=100, debug=False)
best_binary = best_weights > 0.5
selected_features = np.array(feature_names)[best_binary]

print("\nPSO 选择的特征：", selected_features)
print(f"选择的特征数量：{np.sum(best_binary)}")
print(f"分类准确率：{1 - best_cost:.4f}")

# 4. 思考算法的改进空间
# 改进建议：
# - 增加粒子群的多样性，避免陷入局部最优。
# - 引入动态调整的惯性权重和学习因子。
# - 结合遗传算法元素，如交叉和变异操作，形成混合算法。

# 5. 改进后的算法及结果可视化对比
# 这里我们简单地调整PSO参数，增加迭代次数和粒子数量作为改进
def improved_fitness_function(weights):
    binary_weights = weights > 0.5
    if np.sum(binary_weights) == 0:
        return 1
    selected_features = np.array(feature_names)[binary_weights]
    X_selected = df[selected_features].values  # 使用 DataFrame 选择特征
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return 1 - accuracy

# 改进后的PSO参数
improved_lb = [0] * X.shape[1]
improved_ub = [1] * X.shape[1]

# 执行改进后的PSO
print("执行改进后的PSO进行特征选择...")
improved_best_weights, improved_best_cost = pso(improved_fitness_function, improved_lb, improved_ub,
                                               swarmsize=50, maxiter=200, debug=False)
improved_best_binary = improved_best_weights > 0.5
improved_selected_features = np.array(feature_names)[improved_best_binary]

print("\n改进后的PSO选择的特征：", improved_selected_features)
print(f"选择的特征数量：{np.sum(improved_best_binary)}")
print(f"分类准确率：{1 - improved_best_cost:.4f}")

# 可视化对比
results = {
    '算法': ['原始PSO', '改进后PSO'],
    '选择特征数量': [np.sum(best_binary), np.sum(improved_best_binary)],
    '分类准确率': [1 - best_cost, 1 - improved_best_cost]
}

results_df = pd.DataFrame(results)

plt.figure(figsize=(8, 6))
sns.barplot(x='算法', y='分类准确率', data=results_df)
plt.title("算法分类准确率对比")
plt.ylim(0, 1)
for index, row in results_df.iterrows():
    plt.text(index, row['分类准确率'] + 0.01, f"{row['分类准确率']:.2f}", ha='center')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x='算法', y='选择特征数量', data=results_df)
plt.title("算法选择特征数量对比")
for index, row in results_df.iterrows():
    plt.text(index, row['选择特征数量'] + 0.1, f"{int(row['选择特征数量'])}", ha='center')
plt.show()
