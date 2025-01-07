import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def jieba_tokenizer(text):
    return list(jieba.cut(text))


def main():
    # 读取数据
    pd_all = pd.read_csv('./ChnSentiCorp_htl_all.csv', encoding='utf-8')

    # 特征和标签
    X = pd_all['review'].astype(str).tolist()
    y = pd_all['label'].astype(int).tolist()

    # 特征提取：TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer, stop_words=None, max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    # 定义朴素贝叶斯分类器
    clf = MultinomialNB()

    # 十折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 存储每折的评估指标
    accuracy_macro = []
    precision_macro = []
    recall_macro = []
    f1_macro = []

    accuracy_micro = []
    precision_micro = []
    recall_micro = []
    f1_micro = []

    # 初始化总体混淆矩阵
    total_cm = None

    fold = 1
    for train_index, test_index in skf.split(X_tfidf, y):
        X_train, X_test = X_tfidf[train_index], X_tfidf[test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        # 训练模型
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)

        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1m_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        prec_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
        rec_micro = recall_score(y_test, y_pred, average='micro', zero_division=0)
        f1m_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)

        # 存储结果
        accuracy_macro.append(acc)
        precision_macro.append(prec_macro)
        recall_macro.append(rec_macro)
        f1_macro.append(f1m_macro)

        accuracy_micro.append(acc)  # Accuracy 对于macro和micro相同
        precision_micro.append(prec_micro)
        recall_micro.append(rec_micro)
        f1_micro.append(f1m_micro)  # 修正此处，原先为 f1_micro.append(f1_micro)

        # 计算并累加混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        if total_cm is None:
            total_cm = cm
        else:
            total_cm += cm

        # 打印每折的混淆矩阵
        print(f"\nFold {fold} 混淆矩阵:")
        print(cm)

        fold += 1

    # 计算平均值
    print("\n十折交叉验证结果（平均值）：")
    print("宏平均:")
    print(f"Accuracy: {np.mean(accuracy_macro):.4f}")
    print(f"Precision: {np.mean(precision_macro):.4f}")
    print(f"Recall: {np.mean(recall_macro):.4f}")
    print(f"F1-Score: {np.mean(f1_macro):.4f}")

    print("\n微平均:")
    print(f"Accuracy: {np.mean(accuracy_micro):.4f}")
    print(f"Precision: {np.mean(precision_micro):.4f}")
    print(f"Recall: {np.mean(recall_micro):.4f}")
    print(f"F1-Score: {np.mean(f1_micro):.4f}")

    # 打印总体混淆矩阵
    print("\n总体混淆矩阵（所有折的累加）:")
    print(total_cm)

    # 可选：将总体混淆矩阵可视化（需要matplotlib）
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(6, 5))
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('总体混淆矩阵')
        plt.show()
    except ImportError:
        print("matplotlib 或 seaborn 未安装，无法绘制混淆矩阵图。")


if __name__ == '__main__':
    main()
