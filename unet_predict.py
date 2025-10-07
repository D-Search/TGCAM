import torch
import argparse
from unet import UNet
from dataloader import get_test_loader
import numpy as np
import cv2
from tqdm import tqdm
import os

def evaluate(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, gts, names in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            gts = gts.to(device)
            # 预测并处理输出
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            # 计算Dice系数
            intersection = (preds * gts).sum()
            dice = (2. * intersection) / (preds.sum() + gts.sum() + 1e-8)
            dice_scores.append(dice.item())
            # 可视化保存结果（可选）
            save_results(images, gts, preds, names)
    
    print(f"\nMean Dice Score: {np.mean(dice_scores):.4f}")
    return dice_scores

def save_results(images, gts, preds, names):
    """保存测试结果可视化"""
    for i in range(images.shape[0]):
        # 反归一化图像
        img = images[i].cpu().numpy().transpose(1,2,0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255).astype(np.uint8)
        
        # 获取mask和预测
        gt = (gts[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred = (preds[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # 水平拼接显示
        vis = np.hstack([img, cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB), 
                        cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)])
        cv2.imwrite(f"results/{names[i]}", vis)

if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, 
                       default='TGCAM/data/smaller_testset',
                       help='path to test dataset')
    parser.add_argument('--model_path', type=str,
                       default='unet_model.pth',
                       help='path to trained model')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    
    # 准备测试集
    test_loader = get_test_loader(
        image_root=f"{args.test_path}/image/",
        gt_root=f"{args.test_path}/mask/",
        batchsize=args.batch_size,
        testsize=384
    )
    
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 运行评估
    dice_scores = evaluate(model, test_loader, device)
    
    # 保存评估结果
    with open("test_metrics.txt", "w") as f:
        f.write(f"Dice Scores: {dice_scores}\n")
        f.write(f"Average Dice: {np.mean(dice_scores):.4f}\n")