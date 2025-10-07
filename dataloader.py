import os
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

import argparse
# from boundary_loss_tools.sometool import one_hot2dist
# from boundary_loss_tools.get_dist import dist_map_transform
import numpy as np
class MyDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize):
        '''
        trainsize:resize
        '''
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.filter_files()
        self.size = len(self.images)

        #img数据处理：归一化
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        # gt数据处理
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        

    # 原始
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.img_transform(image)
        gt = self.binary_loader(self.gts[index])
        gt = self.gt_transform(gt)
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, gt, name


    def filter_files(self):
        assert len(self.images) == len(self.gts)
        # print(f"原始图像数量: {len(self.images)}")
        images = []
        gts = []

        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts
        # print(f"过滤后图像数量: {len(self.images)}")

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    #?没有用过
    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root,batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    '''生成训练集DataLoader'''
    dataset = MyDataset(image_root, gt_root,trainsize)
    # print(len(dataset))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_test_loader(image_root, gt_root, batchsize, testsize, shuffle=False, num_workers=4):
    """生成测试集DataLoader"""
    test_set = test_dataset(image_root=image_root, gt_root=gt_root, testsize=testsize)
    data_loader = data.DataLoader(dataset=test_set,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True)
    return data_loader
    
class test_dataset(data.Dataset):
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.transform(image)
        gt = self.binary_loader(self.gts[index])
        gt = self.gt_transform(gt)
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default='TGCAM\data\smaller_dataset', help='path to train dataset')
    # TODO
    parser.add_argument('--train_save', type=str,
                        default='UNet') #默认保存在UNet中
    # parser.add_argument('--train_save', type=str,
    #                     default='UNet++')
    # parser.add_argument('--train_save', type=str,
    #                     default='AtUNet')
    opt = parser.parse_args()

    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/mask/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=4, trainsize=384)
    #检查 train_loader 是否成功加载了数据
    #print("Checking train_loader length: ", len(train_loader))
    #print("First batch data: ", next(iter(train_loader), "No data loaded"))