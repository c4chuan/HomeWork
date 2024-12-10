# association_rule_mining_groceries.py

"""
Association Rule Mining on the Groceries Dataset

This script performs association rule mining on the Groceries dataset using the mlxtend library.
It includes data loading, descriptive statistical analysis with visualizations, applying the Apriori
algorithm, suggesting improvements, and comparing the original and improved results through visualizations.

Author: [Your Name]
Date: [Current Date]
"""

# -----------------------------
# 1. 导入必要的库
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 设置绘图风格
sns.set(style='whitegrid')

# -----------------------------
# 2. 选取mlxtend库中的内置groceries数据集，进行简单介绍
# -----------------------------

# **数据集介绍**
# Groceries 数据集包含了一组购物篮交易记录，每笔交易记录了顾客购买的商品。
# 该数据集常用于关联规则挖掘的示例，尤其是在市场篮子分析中。

# **加载数据**
# 假设 mlxtend 库中有内置的 groceries 数据集。实际情况下，您可能需要从外部源加载数据。
# 这里我们使用 mlxtend 提供的示例数据。

# 请确保您已经安装了 mlxtend 库。如果没有安装，请取消下行注释并运行。
# !pip install mlxtend

# 使用 mlxtend 提供的 sample data
# 注意：mlxtend 并不直接提供 groceries 数据集，因此我们将从外部来源加载一个类似的数据集。
# 这里我们使用从 GitHub 获取的 Groceries 数据集。

import requests
import os


def download_groceries_dataset(url, filename):
    """
    下载 Groceries 数据集并保存为本地文件
    """
    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"Dataset downloaded and saved as {filename}.")
    else:
        print(f"Dataset {filename} already exists.")


# 下载 Groceries 数据集（每行一个事务，商品以逗号分隔）
dataset_url = 'https://git.zlong.eu.org/raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv'
dataset_filename = 'groceries.csv'
download_groceries_dataset(dataset_url, dataset_filename)

# 读取数据
with open(dataset_filename, 'r') as file:
    groceries = file.read().splitlines()

# 将每行转换为列表形式
transactions = [transaction.strip().split(',') for transaction in groceries]

print(f"Number of transactions: {len(transactions)}")
print(f"First 5 transactions:\n{transactions[:5]}")

# -----------------------------
# 3. 对其进行可视化描述性统计分析
# -----------------------------

# **描述性统计分析**
# 分析商品的频率分布，找出最常见的商品。

# 统计每个商品出现的次数
from collections import Counter

all_items = [item for sublist in transactions for item in sublist]
item_counts = Counter(all_items)

# 转换为 DataFrame
item_counts_df = pd.DataFrame.from_dict(item_counts, orient='index', columns=['Count'])
item_counts_df = item_counts_df.reset_index().rename(columns={'index': 'Item'})
item_counts_df = item_counts_df.sort_values(by='Count', ascending=False)

# **可视化最常见的前15个商品**
top_n = 15
plt.figure(figsize=(12, 8))
sns.barplot(x='Count', y='Item', data=item_counts_df.head(top_n), palette='viridis')
plt.title(f'Top {top_n} Most Common Items')
plt.xlabel('Count')
plt.ylabel('Item')
plt.tight_layout()
plt.show()

# **商品数量分布**
plt.figure(figsize=(10, 6))
sns.histplot(item_counts_df['Count'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Item Counts')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# -----------------------------
# 4. 将数据关联规则挖掘算法运用在数据集上
# -----------------------------

# **数据预处理**
# 将交易数据转换为适合 Apriori 算法的格式（One-Hot Encoding）

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Encoded DataFrame shape: {df_encoded.shape}")
print(f"First 5 rows of the encoded DataFrame:\n{df_encoded.head()}")

# **应用 Apriori 算法**
# 设置最小支持度为 0.01（即至少出现在1%的交易中）
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

print(f"Number of frequent itemsets: {len(frequent_itemsets)}")
print(f"First 5 frequent itemsets:\n{frequent_itemsets.head()}")

# **生成关联规则**
# 设置最小置信度为 0.3
rules = association_rules(frequent_itemsets, num_itemsets=len(frequent_itemsets),metric="confidence", min_threshold=0.3)

print(f"Number of association rules: {len(rules)}")
print(f"First 5 association rules:\n{rules.head()}")

# -----------------------------
# 5. 思考是否有可改进的空间，并解释原因
# -----------------------------

"""
**改进空间分析**

1. **参数调整**：
   - **支持度（Support）**：当前设置为0.01，可能导致生成大量规则。可以适当提高支持度以减少规则数量，提高规则的显著性。
   - **置信度（Confidence）**：当前设置为0.3，较低的置信度可能导致规则质量下降。提高置信度可以筛选出更可靠的规则。

2. **最小提升度（Lift）**：
   - 在生成关联规则时，增加对提升度的过滤可以进一步确保规则的有意义性。提升度大于1表示规则有正相关关系。

3. **规则简化**：
   - 只保留单个前件和后件的规则，避免复杂的多项规则，便于解释和应用。

4. **数据清洗**：
   - 检查并移除潜在的噪音数据或不相关的商品，提升挖掘效果。

5. **使用其他算法**：
   - 除了 Apriori，还可以尝试 FP-Growth 算法，通常在处理大规模数据时更高效。

在本次改进中，我们将调整支持度和置信度参数，并增加提升度的过滤，以获得更有意义的关联规则。
"""

# -----------------------------
# 6. 将算法结果和改进后的算法结果可视化并进行对比
# -----------------------------

# **原始规则可视化**

# 支持度 vs 置信度
plt.figure(figsize=(10, 6))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules, legend=False, sizes=(20, 200), hue='lift',
                palette='viridis')
plt.title('Association Rules (Original)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.legend(title='Lift')
plt.tight_layout()
plt.show()

# **应用改进的 Apriori 算法**

# **改进1**：提高支持度到0.02，置信度到0.4，提升度过滤为1.2
min_support_improved = 0.02
min_confidence_improved = 0.4

frequent_itemsets_improved = apriori(df_encoded, min_support=min_support_improved, use_colnames=True)
rules_improved = association_rules(frequent_itemsets_improved,num_itemsets=len(frequent_itemsets), metric="confidence",
                                   min_threshold=min_confidence_improved)
rules_improved = rules_improved[rules_improved['lift'] > 1.2]

print(f"Number of association rules after improvement: {len(rules_improved)}")
print(f"First 5 improved association rules:\n{rules_improved.head()}")

# **改进后规则可视化**

# 支持度 vs 置信度
plt.figure(figsize=(10, 6))
sns.scatterplot(x='support', y='confidence', size='lift', data=rules_improved, legend=False, sizes=(20, 200),
                hue='lift', palette='viridis')
plt.title('Association Rules (Improved)')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.legend(title='Lift')
plt.tight_layout()
plt.show()

# **对比可视化**

# 创建一个比较 DataFrame
comparison_df = pd.DataFrame({
    'Original Rules': len(rules),
    'Improved Rules': len(rules_improved)
}, index=['Number of Rules'])

print(comparison_df)

# 条形图比较
comparison_df.plot(kind='bar', figsize=(6, 4), color=['skyblue', 'salmon'])
plt.title('Comparison of Association Rules')
plt.ylabel('Number of Rules')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# **频繁项集的对比**

print(f"Number of frequent itemsets (Original): {len(frequent_itemsets)}")
print(f"Number of frequent itemsets (Improved): {len(frequent_itemsets_improved)}")

# 对比支持度分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(frequent_itemsets['support'], bins=30, kde=True, color='skyblue')
plt.title('Support Distribution (Original)')
plt.xlabel('Support')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(frequent_itemsets_improved['support'], bins=30, kde=True, color='salmon')
plt.title('Support Distribution (Improved)')
plt.xlabel('Support')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()