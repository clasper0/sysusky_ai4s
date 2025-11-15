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

# 创建多面板图形
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2行3列，最后一个位置空出来
fig.suptitle('Molecular Weight vs Activity (pIC50) by Target Class', 
             fontsize=22, fontweight='bold', y=0.98)

# 获取所有靶点类别
target_classes = df['target_class'].unique()

# 为每个靶点类别创建散点图
for i, target_class in enumerate(target_classes):
    if i < 5:  # 确保不超过5个靶点
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 筛选当前靶点的数据
        target_data = df[df['target_class'] == target_class]
        
        # 绘制散点图
        scatter = ax.scatter(target_data['MolWt'], target_data['pIC50'], 
                           color=custom_palette[target_class], s=40, alpha=0.8,
                           edgecolor='black', linewidth=0.5)
        
        # 设置子图标题和标签
        ax.set_title(f'{target_class} Target', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('Molecular Weight (Da)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Activity (pIC50)', fontsize=12, fontweight='bold')
        
        # 设置刻度文字大小
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加一些统计信息
        mean_pIC50 = target_data['pIC50'].mean()
        ax.axhline(y=mean_pIC50, color='gray', linestyle='--', alpha=0.7, 
                  label=f'Mean pIC50: {mean_pIC50:.2f}')
        ax.legend(loc='lower right', fontsize=9)

# 隐藏最后一个空子图（第6个位置）
axes[1, 2].set_visible(False)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()