import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("merged_data.csv")

# 设置图形样式
sns.set_style("whitegrid")

# 定义颜色方案
custom_palette = {
    'GPCR': "#D23838",        # 红色
    'Kinase': "#FFBE0C",      # 黄色
    'Ion channel': "#457DD1", # 蓝色
    'Enzyme': "#56EAA5",      # 绿色
    'Transporter': "#B142F0"  # 紫色
}

plt.figure(figsize=(12, 8))

# 创建散点图 - pIC50为因变量，LogP为自变量
scatter = sns.scatterplot(data=df, x='LogP', y='pIC50', hue='target_class', 
                        palette=custom_palette, s=40, alpha=0.8, 
                        edgecolor='black', linewidth=0.5)

plt.title('Activity (pIC50) vs LogP by Target Class', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('LogP (Lipophilicity)', fontsize=16, fontweight='bold')
plt.ylabel('Activity (pIC50)', fontsize=16, fontweight='bold')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 调整图例
legend = plt.legend(title='Target Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.setp(legend.get_title(), fontsize=14)
plt.setp(legend.get_texts(), fontsize=12)

# 添加网格
plt.grid(True, alpha=0.3)

# 添加总体相关系数
overall_corr = df['LogP'].corr(df['pIC50'])
plt.text(0.05, 0.95, f'Overall Correlation: r = {overall_corr:.3f}\nTotal Molecules: {len(df)}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()