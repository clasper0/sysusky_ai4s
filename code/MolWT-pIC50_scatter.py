import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("merged_data.csv")

# 设置图形样式
sns.set_style("whitegrid")

plt.figure(figsize=(10, 8))

# 自定义颜色方案
custom_palette = {
    'GPCR': "#D23838",        # 红色
    'Kinase': "#FFBE0C",      # 青色
    'Ion channel': "#457DD1", # 蓝色
    'Enzyme': "#56EAA5",      # 绿色
    'Transporter': "#B142F0"  # 黄色
}

scatter = sns.scatterplot(data=df, x='MolWt', y='pIC50', hue='target_class', 
                        palette=custom_palette, s=30, alpha=0.8, 
                        edgecolor='black', linewidth=0.5)

plt.title('pIC50 vs Molecular Weight (Custom Colors)', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Molecular Weight (MolWt)', fontsize=16, fontweight='bold')
plt.ylabel('pIC50 (Activity)', fontsize=16, fontweight='bold')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

legend = plt.legend(title='Target Class', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.setp(legend.get_title(), fontsize=14)
plt.setp(legend.get_texts(), fontsize=12)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()