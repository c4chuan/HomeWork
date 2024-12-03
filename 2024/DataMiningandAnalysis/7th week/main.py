import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 加载Fashion MNIST数据集
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

# 定义不同的网络配置
networks = {
    'shallow_relu': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=100, random_state=42),
    'deep_relu': MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', max_iter=100, random_state=42),
    'shallow_tanh': MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=100, random_state=42),
    'deep_tanh': MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='tanh', max_iter=100, random_state=42),
    'shallow_logistic': MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=100, random_state=42),
    'deep_logistic': MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='logistic', max_iter=100, random_state=42),
}

# 训练并评估每个神经网络
for name, net in networks.items():
    net.fit(X_train, y_train)
    y_pred = net.predict(X_test)
    print(f"{name} - Accuracy: {net.score(X_test, y_test):.4f}")
    print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
fig, axes = plt.subplots(3, 2, figsize=(15, 20))
for ax, (name, net) in zip(axes.flatten(), networks.items()):
    cm = confusion_matrix(y_test, net.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(ax=ax)
    ax.set_title(name)
plt.tight_layout()
plt.show()