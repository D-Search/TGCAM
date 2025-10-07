import torch
import argparse
from unet_multiattention import UNetWithAttention  # 导入正确的模型类
from dataloader import get_test_loader
import numpy as np
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def evaluate(model, test_loader, device, save_visualization=True):
    """评估模型性能"""
    model.eval()
    dice_scores = []
    iou_scores = []
    
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
            
            # 计算IoU
            union = (preds | gts).sum()
            iou = intersection / (union + 1e-8)
            iou_scores.append(iou.item())
            
            # 可视化保存结果
            if save_visualization:
                save_results(images, gts, preds, names)
    
    # 打印评估结果
    print(f"\n=== Evaluation Results ===")
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Mean IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"Max Dice: {np.max(dice_scores):.4f}")
    print(f"Min Dice: {np.min(dice_scores):.4f}")
    
    return dice_scores, iou_scores

def save_results(images, gts, preds, names):
    """保存测试结果可视化"""
    for i in range(images.shape[0]):
        # 反归一化图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        # 获取mask和预测
        gt = (gts[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        pred = (preds[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # 创建彩色可视化
        pred_color = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        gt_color = cv2.applyColorMap(gt, cv2.COLORMAP_JET)
        
        # 叠加原图和预测
        pred_overlay = cv2.addWeighted(img, 0.7, pred_color, 0.3, 0)
        gt_overlay = cv2.addWeighted(img, 0.7, gt_color, 0.3, 0)
        
        # 水平拼接显示
        vis = np.hstack([img, gt_overlay, pred_overlay])
        
        # 保存结果
        cv2.imwrite(f"unet_multiattention_results/{names[i]}", vis)

def plot_metrics(dice_scores, iou_scores, save_path="unet_multiattention_metrics.png"):
    """绘制评估指标图表"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(dice_scores, 'b-', alpha=0.7, label='Dice Score')
    plt.axhline(y=np.mean(dice_scores), color='r', linestyle='--', label=f'Mean: {np.mean(dice_scores):.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('Dice Score')
    plt.title('Dice Scores Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(iou_scores, 'g-', alpha=0.7, label='IoU Score')
    plt.axhline(y=np.mean(iou_scores), color='r', linestyle='--', label=f'Mean: {np.mean(iou_scores):.4f}')
    plt.xlabel('Sample Index')
    plt.ylabel('IoU Score')
    plt.title('IoU Scores Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, 
                       default='TGCAM/data/smaller_testset',
                       help='path to test dataset')
    parser.add_argument('--model_path', type=str,
                       default='unet_multiattention_model.pth',
                       help='path to trained model')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--no_visualization', action='store_true',
                       help='disable result visualization')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='number of attention heads')
    parser.add_argument('--bilinear', action='store_true',
                       help='use bilinear upsampling')
    parser.add_argument('--layer2_attention', action='store_true', default=True,
                       help='enable attention in layer 2')
    parser.add_argument('--layer3_attention', action='store_true', default=True,
                       help='enable attention in layer 3')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                       help='MLP expansion ratio')
    parser.add_argument('--reduction_ratio', type=int, default=4,
                       help='spatial reduction ratio')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='dropout rate')
    args = parser.parse_args()

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型 - 使用正确的参数
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
    
    try:
        # 加载模型权重
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 处理不同的模型保存格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {args.model_path} not found!")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # 准备测试集
    test_loader = get_test_loader(
        image_root=f"{args.test_path}/image/",
        gt_root=f"{args.test_path}/mask/",
        batchsize=args.batch_size,
        testsize=384
    )
    
    # 创建结果目录
    os.makedirs("unet_multiattention_results", exist_ok=True)
    
    # 运行评估
    dice_scores, iou_scores = evaluate(
        model, test_loader, device, 
        save_visualization=not args.no_visualization
    )
    
    # 绘制指标图表
    plot_metrics(dice_scores, iou_scores)
    
    # 保存评估结果
    with open("unet_multiattention_test_metrics.txt", "w") as f:
        f.write("=== UNet Multi-Attention Evaluation Results ===\n")
        f.write(f"Test Configuration:\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Path: {args.test_path}\n")
        f.write(f"Number of Heads: {args.num_heads}\n")
        f.write(f"Bilinear Upsampling: {args.bilinear}\n")
        f.write(f"Layer 2 Attention: {args.layer2_attention}\n")
        f.write(f"Layer 3 Attention: {args.layer3_attention}\n")
        f.write(f"Number of Samples: {len(dice_scores)}\n\n")
        f.write(f"Dice Scores: {dice_scores}\n")
        f.write(f"IoU Scores: {iou_scores}\n")
        f.write(f"Average Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}\n")
        f.write(f"Average IoU: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}\n")
        f.write(f"Max Dice: {np.max(dice_scores):.4f}\n")
        f.write(f"Min Dice: {np.min(dice_scores):.4f}\n")
    
    print("Evaluation completed! Results saved to 'unet_multiattention_test_metrics.txt' and 'unet_multiattention_metrics.png'")