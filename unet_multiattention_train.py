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
                       default='TGCAM/data/BUSI1',
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

    # 数据加载
    image_root = '{}/image/'.format(args.train_path)
    gt_root = '{}/mask/'.format(args.train_path)
    train_loader = get_loader(image_root, gt_root, 
                             batchsize=args.batch_size, 
                             trainsize=384)

    # 训练循环
    best_loss = float('inf')
    train_losses = []
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    
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
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 手动打印学习率变化信息
        if new_lr < old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # 打印epoch统计信息
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
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
            print(f"Best model saved with loss: {best_loss:.4f}")

    # 保存训练损失记录
    with open("unet_multiattention_train_losses.txt", "w") as f:
        f.write("Epoch,Average Loss\n")
        for epoch, loss in enumerate(train_losses):
            f.write(f"{epoch+1},{loss:.6f}\n")
    
    print(f"\nTraining completed!")
    print(f"Best model saved as '{args.model_save_path}' with loss: {best_loss:.4f}")
    print(f"Training losses saved to 'unet_multiattention_train_losses.txt'")

if __name__ == '__main__':
    train()
