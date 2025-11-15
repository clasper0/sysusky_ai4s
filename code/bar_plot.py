import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

try:

    df_props = pd.read_csv('../data/property.csv')
except FileNotFoundError:
    print("错误：未找到 'property.csv' 文件。")
    exit()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

fig.suptitle('Overall Distribution of Molecular Properties in the Dataset (n=200)', fontsize=20, fontweight='bold')

sns.histplot(data=df_props, x='MolWt', kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Molecular Weight (MolWt) Distribution', fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel('Molecular Weight (Da)', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)

sns.histplot(data=df_props, x='LogP', kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('LogP Distribution', fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('LogP', fontsize=12)
axes[0, 1].set_ylabel('Count', fontsize=12)

sns.histplot(data=df_props, x='HBA', discrete=True, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Hydrogen Bond Acceptors (HBA) Distribution', fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Number of HBAs', fontsize=12)
axes[1, 0].set_ylabel('Count', fontsize=12)

sns.histplot(data=df_props, x='HBD', discrete=True, ax=axes[1, 1], color='plum')
axes[1, 1].set_title('Hydrogen Bond Donors (HBD) Distribution', fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Number of HBDs', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()