import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("merged_data.csv")

# 设置图形样式
sns.set_style("whitegrid")

plt.figure(figsize=(12, 8))

custom_palette = {
    'GPCR': "#D23838",        # 红色
    'Kinase': "#FFBE0C",      # 黄色
    'Ion channel': "#457DD1", # 蓝色
    'Enzyme': "#56EAA5",      # 绿色
    'Transporter': "#B142F0"  # 紫色
}

# 创建箱线图+散点图组合
ax = sns.boxplot(data=df, x='HBD', y='pIC50', color='lightgray', width=0.6, fliersize=0)
sns.stripplot(data=df, x='HBD', y='pIC50', hue='target_class', 
             palette=custom_palette, size=6, alpha=0.8, jitter=True, dodge=True,
             edgecolor='black', linewidth=0.5)

plt.title('pIC50 vs Hydrogen Bond Donors (HBD)', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Hydrogen Bond Donors (HBD)', fontsize=16, fontweight='bold')
plt.ylabel('pIC50 (Activity)', fontsize=16, fontweight='bold')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 处理图例
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, title='Target Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.setp(plt.gca().get_legend().get_title(), fontsize=14)
plt.setp(plt.gca().get_legend().get_texts(), fontsize=12)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()