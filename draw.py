import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_predictions_vs_actual(y_true, y_pred, title="预测值 vs 实际值", save_path=None):
    """
    绘制预测值与实际值的散点图
    
    参数:
    y_true: 实际值数组
    y_pred: 预测值数组
    title: 图表标题
    save_path: 保存路径（可选）
    """
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=20)
    
    # 绘制理想预测线（y=x）
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label=f'理想预测线')
    
    # 设置图表属性
    plt.xlabel('实际值 (Actual Values)', fontsize=12)
    plt.ylabel('预测值 (Predicted Values)', fontsize=12)
    plt.title(f'{title}\nRMSE: {rmse:.4f}, R²: {r2:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 确保坐标轴比例一致
    plt.axis('equal')
    
    # 添加文本信息
    plt.text(0.05, 0.95, f'样本数: {len(y_true)}', 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_predictions_by_target(y_true, y_pred, targets, title="各靶点预测值 vs 实际值", save_path=None):
    """
    按靶点分别绘制预测值与实际值的散点图
    
    参数:
    y_true: 实际值数组
    y_pred: 预测值数组
    targets: 靶点标识数组
    title: 图表标题
    save_path: 保存路径（可选）
    """
    
    # 获取唯一靶点
    unique_targets = np.unique(targets)
    n_targets = len(unique_targets)
    
    # 计算子图布局
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_targets > 1 else [axes]
    
    # 为每个靶点绘制散点图
    for i, target in enumerate(unique_targets):
        # 筛选当前靶点的数据
        mask = targets == target
        target_y_true = y_true[mask]
        target_y_pred = y_pred[mask]
        
        if len(target_y_true) > 0:
            # 计算评估指标
            rmse = np.sqrt(mean_squared_error(target_y_true, target_y_pred))
            r2 = r2_score(target_y_true, target_y_pred)
            
            # 绘制散点图
            axes[i].scatter(target_y_true, target_y_pred, alpha=0.6, color='blue', s=20)
            
            # 绘制理想预测线
            min_val = min(min(target_y_true), min(target_y_pred))
            max_val = max(max(target_y_true), max(target_y_pred))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
            
            # 设置子图属性
            axes[i].set_xlabel('实际值')
            axes[i].set_ylabel('预测值')
            axes[i].set_title(f'{target}\nRMSE: {rmse:.4f}, R²: {r2:.4f}')
            axes[i].grid(True, alpha=0.3)
            
            # 确保坐标轴比例一致
            axes[i].axis('equal')
    
    # 隐藏多余的子图
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

def load_prediction_data():
    """
    加载预测数据用于绘图
    """
    # 这里假设您已经有了预测结果文件
    # 如果没有，可以修改为从模型直接获取预测值
    
    try:
        # 尝试从现有预测文件加载数据
        predictions_df = pd.read_csv('output/final_optimized_candidate_predictions.csv')
        print("已加载预测结果文件")
        return predictions_df
    except FileNotFoundError:
        print("未找到预测结果文件，请先运行模型预测")
        return None

def main():
    """
    主函数 - 演示如何使用绘图功能
    """
    
    # 如果您想使用真实数据，请取消下面的注释并根据实际情况修改
    
    # 加载真实数据
    # 这需要您在模型训练过程中保存预测值和实际值
    data = pd.read_csv('./output/model_predictions_for_plotting.csv')  # 或其他包含真实值的文件
    y_true_real = data['actual'].values
    pred = pd.read_csv('./output/model_predictions_for_plotting.csv')
    y_pred_real = pred['predicted'].values  # 需要从模型获取
    
    plot_predictions_vs_actual(y_true_real, y_pred_real, title="真实数据预测效果")
    

if __name__ == "__main__":
    main()