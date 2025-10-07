import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_multiattention import UNetWithAttention
from dataloader import get_loader
import os
import torch.backends.cudnn as cudnn

def train():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)  # 增加批次大小
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_path', type=str,
                       default=r'C:\Users\Administrator\TGCAM\data\BUSI1',
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
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                       help='GPU IDs to use (e.g., "0,1" for two GPUs)')
    args = parser.parse_args()

    # 多GPU设置
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device_ids = [int(id) for id in args.gpu_ids.split(',')]
    
    # 检查可用GPU数量
    available_gpus = torch.cuda.device_count()
    print(f"✅ 系统检测到 {available_gpus} 个GPU")
    print(f"✅ 使用GPU: {device_ids}")
    
    if available_gpus < len(device_ids):
        print(f"⚠️  警告: 请求使用 {len(device_ids)} 个GPU，但只有 {available_gpus} 个可用")
        device_ids = list(range(available_gpus))
    
    # 设置主设备
    main_device = f'cuda:{device_ids[0]}'
    device = torch.device(main_device)
    
    # 启用cudnn加速
    cudnn.benchmark = True
    cudnn.enabled = True

    # 初始化模型
    print("🚀 初始化模型...")
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
    )
    
    # 使用DataParallel包装模型
    if len(device_ids) > 1:
        print(f"🔄 使用DataParallel在 {len(device_ids)} 个GPU上进行数据并行训练")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    
    # 打印模型信息
    print(f"📊 模型信息:")
    print(f"   - 使用设备: {device}")
    print(f"   - GPU数量: {len(device_ids)}")
    print(f"   - 批次大小: {args.batch_size}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - 总参数: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")

    # 优化器 - 可以根据GPU数量调整学习率
    base_lr = args.lr
    effective_batch_size = args.batch_size * len(device_ids)
    adjusted_lr = base_lr * len(device_ids)  # 线性缩放学习率
    
    print(f"   - 基础学习率: {base_lr}")
    print(f"   - 调整后学习率: {adjusted_lr}")
    print(f"   - 有效批次大小: {effective_batch_size} (batch_size * GPU数量)")
    
    optimizer = optim.Adam(model.parameters(), lr=adjusted_lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=5)

    # 数据加载
    image_root = os.path.join(args.train_path, 'image')
    gt_root = os.path.join(args.train_path, 'mask')
    
    # 验证路径
    if not all([os.path.exists(args.train_path), os.path.exists(image_root), os.path.exists(gt_root)]):
        print("❌ 路径验证失败，请检查数据路径")
        return

    # 创建数据加载器
    try:
        # 根据GPU数量调整num_workers
        num_workers = min(4, os.cpu_count() // 2)  # 根据CPU核心数调整
        
        train_loader = get_loader(
            image_root, 
            gt_root, 
            batchsize=args.batch_size, 
            trainsize=384,
            pin_memory=True,  # 多GPU时启用pin_memory
            num_workers=num_workers
        )
        print(f"✅ 数据加载器创建成功!")
        print(f"   - 训练样本: {len(train_loader.dataset)}")
        print(f"   - 批次数量: {len(train_loader)}")
        print(f"   - DataLoader workers: {num_workers}")
    except Exception as e:
        print(f"❌ 创建数据加载器时出错: {e}")
        return

    # 训练循环
    best_loss = float('inf')
    train_losses = []
    
    print(f"\n🎯 开始训练:")
    print(f"   - 训练轮数: {args.epochs}")
    print(f"   - 批次大小: {args.batch_size}")
    print(f"   - 学习率: {adjusted_lr}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for i, (images, gts, names) in enumerate(train_loader):
            # 将数据移动到主设备，DataParallel会自动分发到其他GPU
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            
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
                current_lr = optimizer.param_groups[0]['lr']
                print(f'📈 轮次 [{epoch+1}/{args.epochs}], 步骤 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 学习率变化信息
        if new_lr < old_lr:
            print(f"📉 学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
        
        # 打印epoch统计信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f'🎯 轮次 [{epoch+1}/{args.epochs}], 平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存时去掉DataParallel包装
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': vars(args)
            }, args.model_save_path)
            print(f"💾 最佳模型已保存，损失: {best_loss:.4f}")

    # 保存训练损失记录
    try:
        with open("unet_multiattention_train_losses.txt", "w") as f:
            f.write("Epoch,Average Loss\n")
            for epoch, loss in enumerate(train_losses):
                f.write(f"{epoch+1},{loss:.6f}\n")
        print(f"✅ 训练损失记录已保存")
    except Exception as e:
        print(f"❌ 保存训练损失记录时出错: {e}")
    
    print(f"\n🎉 训练完成!")
    print(f"   - 最佳模型: {args.model_save_path}")
    print(f"   - 最佳损失: {best_loss:.4f}")
    print(f"   - 总训练轮次: {args.epochs}")

if __name__ == '__main__':
    train()
