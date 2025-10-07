# unet_multiattention_train_fixed.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet_multiattention import UNetWithAttention
from dataloader import get_loader
import os
import torch.backends.cudnn as cudnn

def setup_device():
    """设置设备并检查GPU"""
    # 设置环境变量
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    print("🔧 设备初始化...")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"✅ CUDA可用: {cuda_available}")
    print(f"✅ 检测到GPU数量: {gpu_count}")
    
    if cuda_available:
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    内存: {props.total_memory / 1024**3:.1f} GB")
        
        # 启用cudnn加速
        cudnn.benchmark = True
        cudnn.enabled = True
        
        device_ids = list(range(min(2, gpu_count)))  # 最多使用2个GPU
        main_device = f'cuda:{device_ids[0]}'
        device = torch.device(main_device)
        
        print(f"🚀 使用GPU: {device_ids}")
        return device, device_ids, True
    else:
        print("⚠️  CUDA不可用，使用CPU训练")
        device = torch.device('cpu')
        return device, [], False

def validate_data_paths(train_path):
    """验证数据路径并修复格式"""
    print("\n📁 数据路径验证...")
    
    # 可能的路径组合
    possible_image_paths = [
        os.path.join(train_path, 'image'),
        os.path.join(train_path, 'images'),
        train_path + '/image/',
        train_path + '/images/',
        os.path.normpath(train_path) + '\\image\\',
        os.path.normpath(train_path) + '\\images\\'
    ]
    
    possible_gt_paths = [
        os.path.join(train_path, 'mask'),
        os.path.join(train_path, 'masks'),
        os.path.join(train_path, 'label'),
        os.path.join(train_path, 'labels'),
        train_path + '/mask/',
        train_path + '/masks/',
        os.path.normpath(train_path) + '\\mask\\',
        os.path.normpath(train_path) + '\\masks\\'
    ]
    
    # 查找图像路径
    image_root = None
    for path in possible_image_paths:
        if os.path.exists(path):
            image_root = path
            print(f"✅ 找到图像路径: {image_root}")
            break
    
    if image_root is None:
        print("❌ 未找到图像目录，请检查以下路径:")
        for path in possible_image_paths:
            print(f"  - {path}")
        return None, None
    
    # 查找掩码路径
    gt_root = None
    for path in possible_gt_paths:
        if os.path.exists(path):
            gt_root = path
            print(f"✅ 找到掩码路径: {gt_root}")
            break
    
    if gt_root is None:
        print("❌ 未找到掩码目录，请检查以下路径:")
        for path in possible_gt_paths:
            print(f"  - {path}")
        return None, None
    
    # 确保路径格式正确（以分隔符结尾）
    image_root = os.path.normpath(image_root) + os.sep
    gt_root = os.path.normpath(gt_root) + os.sep
    
    print(f"🔄 格式化后的路径:")
    print(f"  图像路径: {image_root}")
    print(f"  掩码路径: {gt_root}")
    
    # 验证文件数量
    try:
        image_files = [f for f in os.listdir(image_root.rstrip(os.sep)) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        gt_files = [f for f in os.listdir(gt_root.rstrip(os.sep)) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"📊 文件统计:")
        print(f"  图像文件: {len(image_files)} 个")
        print(f"  掩码文件: {len(gt_files)} 个")
        
        if len(image_files) == 0:
            print("❌ 图像目录中没有找到任何图像文件")
            return None, None
        if len(gt_files) == 0:
            print("❌ 掩码目录中没有找到任何掩码文件")
            return None, None
            
        # 显示示例文件
        if image_files:
            print(f"  图像示例: {image_files[:3]}")
        if gt_files:
            print(f"  掩码示例: {gt_files[:3]}")
            
    except Exception as e:
        print(f"❌ 读取文件列表时出错: {e}")
        return None, None
    
    return image_root, gt_root

def train():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_path', type=str,
                       default=r'C:\Users\Administrator\TGCAM\data\BUSI1',
                       help='Path to training dataset')
    parser.add_argument('--model_save_path', type=str,
                       default='unet_multiattention_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--bilinear', action='store_true')
    parser.add_argument('--layer2_attention', action='store_true', default=True)
    parser.add_argument('--layer3_attention', action='store_true', default=True)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--reduction_ratio', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    # 设置设备
    device, device_ids, use_gpu = setup_device()
    
    # 验证数据路径
    image_root, gt_root = validate_data_paths(args.train_path)
    if image_root is None or gt_root is None:
        print("❌ 数据路径验证失败，无法继续训练")
        return

    # 根据GPU情况调整批次大小
    if use_gpu and len(device_ids) > 1:
        batch_size = args.batch_size
        print(f"🚀 多GPU模式: 使用批次大小 {batch_size}")
    else:
        batch_size = max(2, args.batch_size // 2)  # CPU或单GPU减小批次
        print(f"⚡ 单设备模式: 使用批次大小 {batch_size}")

    # 初始化模型
    print("\n🚀 初始化模型...")
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
    
    # 多GPU数据并行
    if use_gpu and len(device_ids) > 1:
        print(f"🔄 启用数据并行，使用 {len(device_ids)} 个GPU")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    
    # 打印模型信息
    print(f"📊 模型配置:")
    print(f"  设备: {device}")
    print(f"  GPU数量: {len(device_ids)}")
    print(f"  批次大小: {batch_size}")
    
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 优化器配置
    base_lr = args.lr
    if use_gpu and len(device_ids) > 1:
        adjusted_lr = base_lr * len(device_ids)  # 线性缩放
    else:
        adjusted_lr = base_lr
        
    print(f"  学习率: {adjusted_lr}")
    print(f"  有效批次大小: {batch_size * max(1, len(device_ids))}")
    
    optimizer = optim.Adam(model.parameters(), lr=adjusted_lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=5)

    # 创建数据加载器
    print("\n📦 创建数据加载器...")
    try:
        train_loader = get_loader(
            image_root, 
            gt_root, 
            batchsize=batch_size, 
            trainsize=384,
            pin_memory=use_gpu,  # 只在有GPU时启用
            num_workers=2 if use_gpu else 0  # GPU模式下使用更多workers
        )
        print(f"✅ 数据加载器创建成功!")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  批次数量: {len(train_loader)}")
    except Exception as e:
        print(f"❌ 创建数据加载器失败: {e}")
        print("💡 请检查 dataloader.py 中的路径处理逻辑")
        return

    # 训练循环
    best_loss = float('inf')
    train_losses = []
    
    print(f"\n🎯 开始训练...")
    print(f"  训练轮数: {args.epochs}")
    print(f"  总批次: {len(train_loader)}")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for i, (images, gts, names) in enumerate(train_loader):
            # 移动数据到设备
            images = images.to(device, non_blocking=use_gpu)
            gts = gts.to(device, non_blocking=use_gpu)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, gts)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if (i + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'📈 [{epoch+1}/{args.epochs}] 步骤 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}, LR: {current_lr:.6f}')
        
        # 计算平均损失
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_loss)
        
        # 更新学习率
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 学习率变化提示
        if new_lr < old_lr:
            print(f"📉 学习率从 {old_lr:.6f} 降低到 {new_lr:.6f}")
        
        # 打印epoch统计
        current_lr = optimizer.param_groups[0]['lr']
        print(f'🎯 轮次 [{epoch+1}/{args.epochs}], 平均损失: {avg_loss:.4f}, 学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 处理DataParallel包装
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
            print(f"💾 保存最佳模型，损失: {best_loss:.4f}")

    # 保存训练记录
    try:
        with open("unet_multiattention_train_losses.txt", "w") as f:
            f.write("Epoch,Average Loss\n")
            for epoch, loss in enumerate(train_losses):
                f.write(f"{epoch+1},{loss:.6f}\n")
        print(f"✅ 训练损失记录已保存")
    except Exception as e:
        print(f"⚠️  保存训练记录失败: {e}")
    
    print(f"\n🎉 训练完成!")
    print(f"  最佳模型: {args.model_save_path}")
    print(f"  最佳损失: {best_loss:.4f}")
    print(f"  总训练轮次: {args.epochs}")

if __name__ == '__main__':
    train()
