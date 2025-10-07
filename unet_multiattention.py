import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MultiHeadAttention(nn.Module):
    """简化的多头注意力机制，移除相对位置偏置"""
    
    def __init__(self, channels, num_heads=8, reduction_ratio=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert self.head_dim * num_heads == channels, "channels必须能被num_heads整除"
        
        self.scale = self.head_dim ** -0.5
        
        # 使用1x1卷积生成QKV
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        # 空间缩减以减少计算量
        self.sr_ratio = reduction_ratio
        if reduction_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, 
                               stride=reduction_ratio)
        
        # 输出投影
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv_conv(x)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 对K和V进行空间缩减
        if hasattr(self, 'sr') and self.sr_ratio > 1:
            k_v = self.sr(x)
            _, _, H_red, W_red = k_v.shape
            k_v = k_v.reshape(batch_size, self.num_heads, self.head_dim, H_red * W_red)
            k_v = k_v.permute(0, 1, 3, 2)
            k, v = k_v, k_v
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力权重
        x = (attn @ v).transpose(2, 3).reshape(batch_size, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class EnhancedAttentionBlock(nn.Module):
    """增强的注意力块"""
    
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0, dropout=0.1, reduction_ratio=4):
        super(EnhancedAttentionBlock, self).__init__()
        
        # 层归一化
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # 多头注意力
        self.attn = MultiHeadAttention(channels, num_heads, reduction_ratio, dropout)
        
        # MLP
        mlp_hidden_dim = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_hidden_dim, channels, 1),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        identity = x
        
        # 第一个残差块：注意力
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)
        x_attn = self.attn(x_norm)
        x_attn = self.dropout(x_attn)
        x = x + x_attn
        
        # 第二个残差块：MLP
        identity2 = x
        x_norm2 = x.permute(0, 2, 3, 1)
        x_norm2 = self.norm2(x_norm2)
        x_norm2 = x_norm2.permute(0, 3, 1, 2)
        x_mlp = self.mlp(x_norm2)
        x_mlp = self.dropout(x_mlp)
        x = x + x_mlp
        
        return x

class UNetWithAttention(nn.Module):
    """完整的UNet模型，在第2、3层集成注意力机制"""
    
    def __init__(self, n_channels=3, n_classes=1, bilinear=False, 
                 layer2_attention=True, layer3_attention=True,
                 num_heads=8, mlp_ratio=4.0, reduction_ratio=4, dropout=0.1):
        super(UNetWithAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 编码器部分
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
         
        # 第2层
        self.down2 = Down(128, 256)
        self.layer2_attention = EnhancedAttentionBlock(
            256, num_heads, mlp_ratio, dropout, reduction_ratio
        ) if layer2_attention else None  
          
        # 第3层
        self.down3 = Down(256, 512)
        self.layer3_attention = EnhancedAttentionBlock(
            512, num_heads, mlp_ratio, dropout, reduction_ratio
        ) if layer3_attention else None
        
        # 瓶颈层
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # 解码器部分
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        
        x3 = self.down2(x2)
        if self.layer2_attention is not None:
            x3 = self.layer2_attention(x3)
        
        x4 = self.down3(x3)
        if self.layer3_attention is not None:
            x4 = self.layer3_attention(x4)
        
        x5 = self.down4(x4)
        
        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits