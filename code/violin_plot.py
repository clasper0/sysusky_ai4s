import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("../data/merged_data.csv")

custom_palette = {
    'GPCR': "#D23838",       
    'Kinase': "#FFBE0C",     
    'Ion channel': "#457DD1",
    'Enzyme': "#56EAA5",     
    'Transporter': "#B142F0" 
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Molecular Properties Distribution by Target Class (Violin Plots)', 
             fontsize=22, fontweight='bold', y=0.98)

# MolWt
sns.violinplot(data=df, x='target_class', y='MolWt', palette=custom_palette,
              inner='quartile', ax=axes[0, 0])
axes[0, 0].set_title('Molecular Weight', fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel('Target Class', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Molecular Weight (Da)', fontsize=14, fontweight='bold')

# LogP
sns.violinplot(data=df, x='target_class', y='LogP', palette=custom_palette,
              inner='quartile', ax=axes[0, 1])
axes[0, 1].set_title('LogP', fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('Target Class', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('LogP', fontsize=14, fontweight='bold')

# HBA
sns.violinplot(data=df, x='target_class', y='HBA', palette=custom_palette,
              inner='quartile', ax=axes[1, 0])
axes[1, 0].set_title('Hydrogen Bond Acceptors (HBA)', fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Target Class', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('HBA', fontsize=14, fontweight='bold')

# HBD
sns.violinplot(data=df, x='target_class', y='HBD', palette=custom_palette,
              inner='quartile', ax=axes[1, 1])
axes[1, 1].set_title('Hydrogen Bond Donors (HBD)', fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Target Class', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('HBD', fontsize=14, fontweight='bold')

for ax in axes.flat:
    ax.tick_params(axis='x', labelsize=12)  # 移除rotation=45
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()