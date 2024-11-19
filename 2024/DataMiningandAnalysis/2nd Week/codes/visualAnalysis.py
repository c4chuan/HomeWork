import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# 加载数据集
california = fetch_california_housing()

# 将数据转换为 DataFrame
data = pd.DataFrame(california.data, columns=california.feature_names)
data['Target'] = california.target

# 数据基本信息
print("Basic Information of the Dataset:")
print(data.info())
print("\nDescriptive Statistics of the Dataset:")
print(data.describe())

# 设置绘图风格
sns.set(style="whitegrid")

# 1. 绘制各特征的直方图
data.hist(bins=30, figsize=(20,15))
plt.suptitle("Histograms of Features", fontsize=20)
plt.show()

# 2. 相关性热力图
corr_matrix = data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. 目标变量与各特征的散点图
features = data.columns[:-1]  # 除去目标变量
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.scatterplot(data=data, x=feature, y='Target', alpha=0.5)
    plt.title(f"{feature} vs Target")
plt.tight_layout()
plt.show()

# 4. 地理位置与房价的关系
plt.figure(figsize=(10,8))
plt.scatter(data['Longitude'], data['Latitude'], c=data['Target'], cmap='coolwarm', alpha=0.5)
plt.colorbar(label='Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Distribution of California Housing Prices')
plt.show()

# 5. 对数变换后的房价分布
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(data['Target'], bins=30, kde=True)
plt.title('Original Distribution of House Prices')
plt.subplot(1,2,2)
sns.histplot(np.log1p(data['Target']), bins=30, kde=True)
plt.title('Log-transformed Distribution of House Prices')
plt.show()

# 6. 人口与平均房间数的关系
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='Population', y='AveRooms', alpha=0.5)
plt.title('Population vs Average Rooms')
plt.show()

# 7. 房屋年龄与房价的关系
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='HouseAge', y='Target', alpha=0.5)
plt.title('House Age vs Median House Value')
plt.show()

# 8. 平均收入与房价的关系
plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='MedInc', y='Target', alpha=0.5)
plt.title('Median Income vs Median House Value')
plt.show()

# 9. 箱线图分析
plt.figure(figsize=(15,10))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=data[feature])
    plt.title(f"Boxplot of {feature}")
plt.tight_layout()
plt.show()
