import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_multiattention import UNetWithAttention  # 导入正确的模型类
from dataloader import get_loader
import os

def train():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150) #训练150轮
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_path', type=str,
                       default=r'C:\Users\Administrator\TGCAM\data\BUSI1',  # 使用绝对路径
                       help='Path to training dataset')
    parser.add_argument('--model_save_path', type=str,
                       default='unet_multiattention_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--bilinear', action='store_true',
                       help='Use bilinear upsampling')
    parser.add_argument('--layer2_attention', action='store_true', default=True,
                       help='Enable attention in layer 2')
    parser.add_argument('--layer3_attention', action='store_true', default=True,
                       help='Enable attention in layer 3')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                       help='MLP expansion ratio in attention blocks')
    parser.add_argument('--reduction_ratio', type=int, default=4,
                       help='Spatial reduction ratio for attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for attention layers')
    args = parser.parse_args()

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型 - 使用正确的参数
    model = UNetWithAttention(
        n_channels=3, 
        n_classes=1, 
        bilinear=args.bilinear,
        layer2_attention=args.layer2_attention,
        layer3_attention=args.layer3_attention,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        reduction_ratio=args.reduction_ratio,
        dropout=args.dropout
    ).to(device)
    
    # 打印模型信息
    print(f"Model initialized with:")
    print(f"  - Layer 2 Attention: {args.layer2_attention}")
    print(f"  - Layer 3 Attention: {args.layer3_attention}")
    print(f"  - Number of Heads: {args.num_heads}")
    print(f"  - MLP Ratio: {args.mlp_ratio}")
    print(f"  - Reduction Ratio: {args.reduction_ratio}")
    print(f"  - Dropout: {args.dropout}")
    print(f"  - Bilinear Upsampling: {args.bilinear}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=5)

    # 数据加载 - 使用绝对路径
    image_root = os.path.join(args.train_path, 'image')
    gt_root = os.path.join(args.train_path, 'mask')
    
    # 验证路径是否存在
    print(f"\n=== 路径验证 ===")
    print(f"训练数据根目录: {args.train_path}")
    print(f"图像目录: {image_root}")
    print(f"掩码目录: {gt_root}")
    
    # 检查路径是否存在
    if not os.path.exists(args.train_path):
        print(f"❌ 错误: 训练数据根目录不存在: {args.train_path}")
        print("请检查 --train_path 参数是否正确")
        return
    
    if not os.path.exists(image_root):
        print(f"❌ 错误: 图像目录不存在: {image_root}")
        print("请确保 'image' 文件夹存在于训练数据目录中")
        return
        
    if not os.path.exists(gt_root):
        print(f"❌ 错误: 掩码目录不存在: {gt_root}")
        print("请确保 'mask' 文件夹存在于训练数据目录中")
        return
    
    # 检查文件数量
    image_files = [f for f in os.listdir(image_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(gt_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"✅ 找到 {len(image_files)} 个图像文件")
    print(f"✅ 找到 {len(mask_files)} 个掩码文件")
    
    if len(image_files) == 0:
        print("❌ 错误: 图像目录中没有找到任何图像文件")
        return
        
    if len(mask_files) == 0:
        print("❌ 错误: 掩码目录中没有找到任何掩码文件")
        return
    
    # 显示前几个文件作为示例
    print(f"\n图像文件示例: {image_files[:5]}")
    print(f"掩码文件示例: {mask_files[:5]}")
    
    # 确保路径格式正确（添加路径分隔符）
    image_root = os.path.normpath(image_root) + os.sep
    gt_root = os.path.normpath(gt_root) + os.sep
    
    print(f"\n=== 最终使用的路径 ===")
    print(f"图像路径: {image_root}")
    print(f"掩码路径: {gt_root}")

    # 创建数据加载器
    try:
        train_loader = get_loader(image_root, gt_root, 
                                 batchsize=args.batch_size, 
                                 trainsize=384)
        print(f"✅ 数据加载器创建成功!")
        print(f"训练样本数量: {len(train_loader.dataset)}")
        print(f"批次数量: {len(train_loader)}")
    except Exception as e:
        print(f"❌ 创建数据加载器时出错: {e}")
        print("请检查 dataloader.py 中的路径处理逻辑")
        return

    # 训练循环
    best_loss = float('inf')
    train_losses = []
    
    print(f"\n=== 开始训练 ===")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for i, (images, gts, names) in enumerate(train_loader):
            images = images.to(device)
            gts = gts.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, gts)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 打印批次信息
            if (i + 1) % 10 == 0:
                print(f'轮次 [{epoch+1}/{args.epochs}], 步骤 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 手动打印学习率变化信息
        if new_lr < old_lr:
            print(f"学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
        
        # 打印epoch统计信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f'轮次 [{epoch+1}/{args.epochs}], 平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, args.model_save_path)
            print(f"✅ 最佳模型已保存，损失: {best_loss:.4f}")

    # 保存训练损失记录
    try:
        with open("unet_multiattention_train_losses.txt", "w") as f:
            f.write("Epoch,Average Loss\n")
            for epoch, loss in enumerate(train_losses):
                f.write(f"{epoch+1},{loss:.6f}\n")
        print(f"✅ 训练损失记录已保存到 'unet_multiattention_train_losses.txt'")
    except Exception as e:
        print(f"❌ 保存训练损失记录时出错: {e}")
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳模型已保存为: '{args.model_save_path}'")
    print(f"最佳损失: {best_loss:.4f}")

if __name__ == '__main__':
    train()
