import torch
import os, sys
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torchvision.io import read_image
sys.path.append("/home/zeke/ProjectUbuntu/Projects/torch-ngp")
from nerf_sem.UNet import PatchFeaUNet
import torch.optim as optim
import torch.nn.functional as F

# model = PatchFeaUNet(1, 3, act='tanh').cuda()
model = PatchFeaUNet(1, 3, act='none').cuda()
optimizer = optim.Adam(model.parameters())

class CustomImageDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_path, 'normal')))

    def __getitem__(self, img_idx):
        f_path_depth = os.path.join(self.root_path, 'depth', f"{img_idx}.npy")
        depth_data = np.load(f_path_depth)
        depth_data[depth_data==0] = 20000 
        depth_data = depth_data / 1000 

        f_path_norm = os.path.join(self.root_path, 'normal', f"{img_idx}.npy")
        normal_data = np.load(f_path_norm, allow_pickle=True)
        normal = normal_data.item().get('normal')
        msk = ~normal_data.item().get('msk')

        return depth_data, normal, msk


from torch.utils.data import DataLoader
training_data = CustomImageDataset('../../data/replica_dinning_room')
test_data = CustomImageDataset('../../data/replica_dinning_room')

train_loader = DataLoader(training_data, batch_size=12, shuffle=True)
test_loader = DataLoader(test_data, batch_size=12, shuffle=True)

epochs = 100
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target, msk) in enumerate(train_loader):
        data, target, msk = data.float().cuda(), target.float().cuda(), msk.float().cuda()
        data.unsqueeze_(1)
        target = target.permute(0,3,1,2)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# Test evaluation
import cv2
os.makedirs('outs', exist_ok=True)
def draw_out(preds, gts, msk, i):
    for j, (pred, gt, ms) in enumerate(zip(preds, gts, msk)):
        ms = ms.cpu().numpy() > 0
        gt_normal = ((gt + 1)/2 * 255).cpu().numpy().astype(np.uint8)
        gt_normal[~ms] = 0
        cv2.imwrite(os.path.join('outs', f'{i}_{j}_normal_gt.png'), cv2.cvtColor(gt_normal, cv2.COLOR_RGB2BGR))

        pred = np.clip(pred.permute(1,2,0).cpu().numpy(), -1, 1)
        pred = ((pred + 1)/2 * 255).astype(np.uint8)
        pred[~ms] = 0
        cv2.imwrite(os.path.join('outs', f'{i}_{j}_normal_pred.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))


model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for i, (data, target, msk) in enumerate(test_loader):
        data, target, msk = data.float().cuda(), target.float().cuda(), msk.float().cuda()
        data.unsqueeze_(1)
        output = model(data)

        draw_out(output, target, msk, i)
