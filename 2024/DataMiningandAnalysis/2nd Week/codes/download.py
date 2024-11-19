import pandas as pd
from sklearn.datasets import fetch_california_housing

# 加载加利福尼亚州房价数据集
california = fetch_california_housing()

# 将数据集转换为 DataFrame
data = pd.DataFrame(california.data, columns=california.feature_names)
data['Target'] = california.target

# 将 DataFrame 保存为 Excel 文件
data.to_excel('california_housing.xlsx', index=False)

print("数据集已成功保存为 'california_housing.xlsx'")
