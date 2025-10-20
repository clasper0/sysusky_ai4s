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

# 第二个多面板图：pIC50 vs LogP
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Activity (pIC50) vs LogP by Target Class', 
             fontsize=22, fontweight='bold', y=0.98)

target_classes = df['target_class'].unique()

# 为每个靶点类别创建散点图
for i, target_class in enumerate(target_classes):
    if i < 5:
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 筛选当前靶点的数据
        target_data = df[df['target_class'] == target_class]
        
        # 绘制散点图
        scatter = ax.scatter(target_data['LogP'], target_data['pIC50'], 
                           color=custom_palette[target_class], s=30, alpha=0.8,
                           edgecolor='black', linewidth=0.5)
        
        # 设置子图标题和标签
        ax.set_title(f'{target_class} Target', fontsize=16, fontweight='bold', pad=10)
        ax.set_xlabel('LogP', fontsize=14, fontweight='bold')
        ax.set_ylabel('Activity (pIC50)', fontsize=14, fontweight='bold')
        
        # 设置刻度文字大小
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 计算并显示相关系数和数据点数量
        correlation = target_data['LogP'].corr(target_data['pIC50'])
        n_points = len(target_data)
        ax.text(0.05, 0.95, f'r = {correlation:.3f}\nn = {n_points}', 
                transform=ax.transAxes, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# 隐藏最后一个空子图
axes[1, 2].set_visible(False)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()