import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def optimized_srad(image, num_iter=15, delta_t=0.02, q0=0.1):
    """
    改进版SRAD算法（解决模糊问题关键修改）
    - 减少迭代次数至15次
    - 增大q0至0.1保留边缘
    - 添加自适应扩散控制
    """
    I = image.astype(np.float32) / 255.0
    for _ in range(num_iter):
        # 梯度计算（使用Scharr算子增强边缘敏感度）
        I_x = cv2.Scharr(I, cv2.CV_32F, 1, 0)
        I_y = cv2.Scharr(I, cv2.CV_32F, 0, 1)
        grad_mag = np.sqrt(I_x**2 + I_y**2)
        
        # 自适应扩散系数（核心改进）
        edge_mask = np.clip(grad_mag * 10, 0, 1)  # 边缘区域标识
        c = np.exp(-(grad_mag**2) / (q0**2 + 1e-6)) 
        c = c * (1 - edge_mask) + 0.1 * edge_mask  # 边缘处减少扩散
        
        # 各向异性扩散
        cI_x = c * I_x
        cI_y = c * I_y
        div = cv2.Scharr(cI_x, cv2.CV_32F, 1, 0) + cv2.Scharr(cI_y, cv2.CV_32F, 0, 1)
        I += delta_t * div
        I = np.clip(I, 0, 1)
    
    return (I * 255).astype(np.uint8)

def adaptive_sharpen(img):
    """自适应锐化（根据局部对比度调整强度）"""
    # 获取局部对比度
    laplacian = cv2.Laplacian(img, cv2.CV_32F)
    contrast = cv2.convertScaleAbs(laplacian)
    
    # 创建自适应锐化核
    kernel_base = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    weights = 1 + contrast / 255.0  # 动态权重
    kernel = np.clip(kernel_base * weights[:,:,None], -2, 2)
    
    # 应用非均匀锐化
    result = np.zeros_like(img)
    for i in range(3):
        for j in range(3):
            result += cv2.filter2D(img, -1, kernel[:,:,i,j]) * (1/9)
    return np.clip(result, 0, 255).astype(np.uint8)

def process_images(input_dir, output_dir):
    """完整的优化处理流程"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for img_file in tqdm(list(input_path.glob('*.*'))):
        try:
            # 1. 读取并预处理
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            
            # 2. 优化SRAD去噪（关键修改）
            denoised = optimized_srad(img)
            
            # 3. 分区域处理（ROI增强）
            _, mask = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            roi = cv2.bitwise_and(denoised, mask)
            non_roi = cv2.bitwise_and(denoised, cv2.bitwise_not(mask))
            
            # 4. 自适应锐化（仅强化ROI区域）
            roi_sharp = adaptive_sharpen(roi)
            final = cv2.addWeighted(roi_sharp, 0.7, non_roi, 0.3, 0)
            
            # 5. 保存结果
            cv2.imwrite(str(output_path / f"enhanced_{img_file.name}"), final)
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")

if __name__ == "__main__":
    # 配置路径（建议使用绝对路径）
    INPUT_DIR = "TGCAM/data/smaller_testset/original_image"
    OUTPUT_DIR = "TGCAM/data/smaller_testset/true_testset"
    
    # 运行优化后的处理流程
    process_images(INPUT_DIR, OUTPUT_DIR)