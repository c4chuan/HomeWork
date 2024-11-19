# 导入所需的库
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据集并描述性分析
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 创建 DataFrame 来查看数据
df = pd.DataFrame(X, columns=feature_names)
df['species'] = pd.Series(y).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 显示数据集的基本统计信息
print("数据集描述性统计：")
print(df.describe())

# 显示数据集前几行
print("\n数据集的前5行：")
print(df.head())

# 2. 可视化数据分布
sns.pairplot(df, hue='species', palette='Set1')
plt.suptitle("鸢尾花数据集特征分布", y=1.02)
plt.show()

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 使用三种不同的朴素贝叶斯分类器进行分类
# (1) Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)

# (2) Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)
accuracy_bnb = accuracy_score(y_test, y_pred_bnb)

# (3) Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)

# 5. 比较分类算法的性能
print("\n三种朴素贝叶斯分类器的准确度：")
print(f"Gaussian Naive Bayes: {accuracy_gnb:.4f}")
print(f"Bernoulli Naive Bayes: {accuracy_bnb:.4f}")
print(f"Multinomial Naive Bayes: {accuracy_mnb:.4f}")

# 显示分类报告和混淆矩阵
print("\nGaussian Naive Bayes 分类报告：")
print(classification_report(y_test, y_pred_gnb, target_names=target_names))

print("\nGaussian Naive Bayes 混淆矩阵：")
print(confusion_matrix(y_test, y_pred_gnb))

print("\nBernoulli Naive Bayes 分类报告：")
print(classification_report(y_test, y_pred_bnb, target_names=target_names))

print("\nMultinomial Naive Bayes 分类报告：")
print(classification_report(y_test, y_pred_mnb, target_names=target_names))

# 6. 可视化各算法的分类准确度
accuracies = [accuracy_gnb, accuracy_bnb, accuracy_mnb]
labels = ['Gaussian Naive Bayes', 'Bernoulli Naive Bayes', 'Multinomial Naive Bayes']

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=accuracies, palette='Blues_d')
plt.title("不同朴素贝叶斯分类器准确度比较")
plt.ylabel('准确度')
plt.show()
