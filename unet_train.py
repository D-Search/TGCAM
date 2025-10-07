import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from unet import UNet as UNet
from dataloader import get_loader  # 导入数据加载器
import os

def train():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_path', type=str, 
                       default='TGCAM\data\BUSI1',
                       help='Path to training dataset')
    args = parser.parse_args()

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失

    # 数据加载
    opt = parser.parse_args()
    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)
    train_loader = get_loader(image_root, gt_root, 
                             batchsize=args.batch_size, 
                             trainsize=384)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, gts in train_loader:
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
        
        # 打印统计信息
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Training completed! Model saved as 'unet_model.pth'")

if __name__ == '__main__':
    train()