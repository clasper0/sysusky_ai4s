#!/usr/bin/env python3
"""
简化版训练脚本，使用显式定义的参数。
"""

import os
import sys
import torch
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import create_data_splits
from gcn_model import create_model
from trainer import MolecularTrainer


def main():
    """主训练函数，使用显式定义的参数。"""
    print("简化版分子GCN训练")
    print("=" * 30)
    
    # 显式定义参数
    DATA_PATH = "data/candidate_hybrid.csv"
    SMILES_COL = "SMILES"
    TARGET_COL = None  # 让数据加载器自动检测目标列
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    BATCH_SIZE = 16
    EPOCHS = 30
    
    # 模型参数
    MODEL_TYPE = "multi_task"
    INPUT_DIM = 36
    HIDDEN_DIMS = [128, 256, 512]
    DROPOUT_RATE = 0.2
    USE_BATCH_NORM = True
    USE_RESIDUAL = True
    
    # 训练参数
    OPTIMIZER = "adam"
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    SCHEDULER = "plateau"
    PATIENCE = 20
    MIN_DELTA = 1e-4
    SAVE_EVERY = 10
    
    # 实验参数
    DEVICE = "cpu"
    EXPERIMENT_DIR = "experiments"
    EXPERIMENT_NAME = None  # 自动生成
    
    print(f"数据文件: {DATA_PATH}")
    print(f"设备: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print()

    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到数据文件 {DATA_PATH}!")
        print("请确保数据文件存在。")
        return

    # 加载数据
    print("正在加载数据...")
    try:
        train_loader, val_loader, test_loader, dataset = create_data_splits(
            DATA_PATH,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            batch_size=BATCH_SIZE,
            smiles_col=SMILES_COL,
            target_col=TARGET_COL
        )
        print("数据加载成功!")
    except Exception as e:
        print(f"数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 显示数据统计
    print(f"数据集加载成功!")
    print(f"  训练样本: {len(train_loader.dataset)}")
    print(f"  验证样本: {len(val_loader.dataset)}")
    print(f"  测试样本: {len(test_loader.dataset)}")
    print()

    # 确定输出维度
    try:
        sample_batch = next(iter(train_loader))
        actual_output_dim = sample_batch.y.shape[1] if len(sample_batch.y.shape) > 1 else 1
        print(f"检测到输出维度: {actual_output_dim}")
    except Exception as e:
        print(f"检测输出维度失败: {e}")
        actual_output_dim = 5  # 默认使用5个目标
        print(f"使用默认输出维度: {actual_output_dim}")

    # 创建模型
    print("正在创建模型...")
    model_params = {
        "input_dim": INPUT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "output_dim": actual_output_dim,
        "dropout_rate": DROPOUT_RATE,
        "use_batch_norm": USE_BATCH_NORM,
        "use_residual": USE_RESIDUAL
    }

    try:
        model = create_model(MODEL_TYPE, **model_params)
        print("模型创建成功!")
    except Exception as e:
        print(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 显示模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型类型: {type(model).__name__}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print()

    # 创建训练器
    try:
        trainer = MolecularTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            experiment_dir=EXPERIMENT_DIR,
            experiment_name=EXPERIMENT_NAME
        )
        print("训练器创建成功!")
    except Exception as e:
        print(f"训练器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 设置优化器和调度器
    print("正在设置优化器和调度器...")
    try:
        trainer.setup_optimizer(
            OPTIMIZER,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        trainer.setup_scheduler(SCHEDULER)
        print("优化器和调度器设置成功!")
    except Exception as e:
        print(f"优化器和调度器设置失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 开始训练
    print("\n开始训练...")
    print(f"早停耐心值: {PATIENCE}")
    print(f"最小改进值: {MIN_DELTA}")
    print(f"每 {SAVE_EVERY} 轮保存检查点")
    print()

    try:
        trainer.train(
            num_epochs=EPOCHS,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            save_every=SAVE_EVERY
        )

        print("\n训练完成!")
        print(f"实验保存至: {trainer.experiment_path}")

        # 最终测试集评估
        if test_loader is not None:
            print("\n最终测试集评估:")
            test_metrics = trainer.validate_epoch(test_loader)
            print(f"  测试损失: {test_metrics['loss']:.4f}")
            print(f"  测试RMSE: {test_metrics['rmse']:.4f}")
            print(f"  测试MAE: {test_metrics['mae']:.4f}")
            if 'r2' in test_metrics:
                print(f"  测试R²: {test_metrics['r2']:.3f}")

        # 保存最终模型
        final_model_path = os.path.join(trainer.experiment_path, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"\n最终模型保存至: {final_model_path}")

    except KeyboardInterrupt:
        print("\n用户中断训练")
        print(f"保存当前进度至 {trainer.experiment_path}")
        trainer.save_checkpoint(len(trainer.train_history['train_loss']), save_optimizer=True)

    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n训练脚本执行完成。")


if __name__ == "__main__":
    main()