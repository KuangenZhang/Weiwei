
from torch.autograd import Variable
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np 
from losses import *
import torchvision.transforms as transforms
import torch.nn.functional as F

def train_rg(model, train_loader, optimizer, loss_fn):
    model.train()
    label_list = []
    pred_list = []
    loss_t = []
    for data, label, class_label, _ in train_loader:
        # print(label.shape)
        optimizer.zero_grad()
        data = Variable(data.float().to(device))
        pred_1 = model(data)
        data = randomly_transform_imgs(data)

        label = Variable((label).float().to(device))
        pred = model(data)
        # loss = loss_fn(pred, label)
        loss = torch.sum(torch.relu(torch.abs((pred - label) / label) - 0.1)) + \
               torch.sum(torch.relu(torch.abs((pred_1 - label) / label) - 0.1))
        # loss = -torch.mean(torch.float(torch.abs((pred - label) / label) <= 0.2))
        # loss = -np.mean(np.abs((pred - label) / label) <= 0.2)
        loss.backward()
        optimizer.step()

        label_list.append(label)
        pred_list.append(pred)
        loss_t.append(loss)

    y = torch.cat(label_list, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(pred_list, dim=0).cpu().detach().numpy()
    loss_avg = np.mean(torch.stack(loss_t, dim=0).cpu().detach().numpy())
    return loss_avg, y, y_pred


    # return loss_p

def randomly_transform_imgs(imgs):
    imgs = imgs * (1 - 5e-2 + Variable(1e-1 * torch.rand(imgs.shape).float().to(device)))
    img_trans = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomVerticalFlip(p=0.5))
    return img_trans(imgs)

