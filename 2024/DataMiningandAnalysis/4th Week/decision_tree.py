import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier


# 1. 简单介绍一下分类任务中常用的肿瘤数据集
def load_data():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data


# 2. 使用最基本的决策树对数据集进行分类
def basic_decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf


# 3. 预剪枝决策树
def pre_pruned_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)  # Pre-pruning with max depth
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf


# 4. 后剪枝决策树
def post_pruned_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    # Perform post-pruning
    # We can use a cost-complexity pruning technique here
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alphas[-2], random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred), clf


# 5. 混合决策树
def hybrid_tree(X_train, X_test, y_train, y_test):
    # Train Decision Tree
    base_tree = DecisionTreeClassifier(random_state=0)
    base_tree.fit(X_train, y_train)

    # Get predictions from the Decision Tree
    train_preds = base_tree.predict(X_train).reshape(-1, 1)
    test_preds = base_tree.predict(X_test).reshape(-1, 1)

    # Concatenate original features with predictions
    X_train_hybrid = np.hstack((X_train, train_preds))
    X_test_hybrid = np.hstack((X_test, test_preds))

    # Train a Neural Network on the new feature set
    nn = MLPClassifier(random_state=0, max_iter=500)
    nn.fit(X_train_hybrid, y_train)

    # Predict with the Neural Network
    y_pred = nn.predict(X_test_hybrid)
    return accuracy_score(y_test, y_pred), (base_tree, nn)


# 6. 在每个节点选择最优的划分属性
def train_test_split_data(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    df, data = load_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    results = {}

    # Basic Decision Tree
    accuracy, clf_basic = basic_decision_tree(X_train, X_test, y_train, y_test)
    results['Basic DT'] = accuracy

    # Pre-pruned Decision Tree
    accuracy, clf_pre_pruned = pre_pruned_tree(X_train, X_test, y_train, y_test)
    results['Pre-pruned DT'] = accuracy

    # Post-pruned Decision Tree
    accuracy, clf_post_pruned = post_pruned_tree(X_train, X_test, y_train, y_test)
    results['Post-pruned DT'] = accuracy

    # Hybrid Decision Tree
    accuracy, clf_hybrid = hybrid_tree(X_train, X_test, y_train, y_test)
    results['Hybrid DT'] = accuracy

    # 7. 将所有方法的结果对比可视化出来
    methods = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.barh(methods, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Comparison of Decision Tree Methods')
    plt.xlim(0, 1)
    plt.grid(axis='x')
    plt.show()


if __name__ == "__main__":
    main()
