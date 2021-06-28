from __future__ import print_function
from torchsummary import summary
import torch.utils.data as data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import sys
import cv2 

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150
from matplotlib import pyplot as plt

from util import define_model
from util import get_scheduler
from prepare_data import split_data
from train import train_rg
from test import test_rg, test_rg_ens
from test_on_dicts import test_dict

import os
import time
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for ros label prediction")
    parser.add_argument('--re_split', type=bool, default= False, help="whether resplit the dataset")
    parser.add_argument('--phase', type=str, default='test', help='train, test, etc')
    parser.add_argument('--dict_files', nargs='+', help='list of files for testing')
    #data_related options
    parser.add_argument("--dataset_path", type=str, default= 'data/Data_mask_r7.npy',help="path to the training dataset")
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument("--phase_norm", type=str, default ='min_max', help="normalization on the phase data [min_max| self_min_max]")
    parser.add_argument("--ros_norm", type=str, default ='None', help="normalization on the phase data [min_max| self_min_max]")
    parser.add_argument("--std_transform", type=bool, default =True, help="whether perform std mean norm on the phase data")

    # training options 
    # parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--checkpoint_model", type=str, default='checkpoints/ke', help=" path to checkpoint model")
    parser.add_argument('--resume_from', type=bool, default=None, help = "whether retrain from a certain old model")
    parser.add_argument('--old_dict', required='--resume_from' in sys.argv)
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--loss', type=str, default='mse', help='loss function [mse | l1 | huber]')
    parser.add_argument('--delta', type=float, default=1, help='huber delta')

    # extraction model parameters
    parser.add_argument('--model_name', type=str, default='SimpleModel', help='chooses which model to use. [resnet18 | resnet34 | resnet50| resnet101 | resnet152 | SimpleModel]')
    parser.add_argument('--norm_type', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')

    # fc model parameters
    parser.add_argument('--fc_filters', type=int, default=512, help='# fc filters in the first layer')
    parser.add_argument('--fc_layers', type=int, default=3, help='# fc layers')
    parser.add_argument('--fc_norm_type', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--fc_drop_out', type = float, default = 0.2, help='dropout in the fc_layers')
    parser.add_argument('--fc_relu_slope', type = float, default = 0.01, help='relu slope in the fc_layers')

    #model init
    parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    # Parse and return arguments
    opt = parser.parse_args()


    best_loss = 1e3
    best_train_loss = 1e3
    best_loss_c = 1e3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_list = [device]


    # creat discinminator
    model = define_model(opt, device_list)

    # transformation 
    img_trans = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=45),
        transforms.RandomVerticalFlip(p=0.5),
    )
    scripted_transforms = torch.jit.script(img_trans)

    if opt.std_transform == True:
        img_trans_i = torch.nn.Sequential( transforms.Normalize(mean=[0.5], std=[0.2]),
        )
        scripted_transforms_i = torch.jit.script(img_trans_i)
    else:
        scripted_transforms_i = None


    #data_loader
    if opt.re_split == True:
        data_prepared = split_data(opt.dataset_path, opt.batch_size, opt.phase_norm, opt.ros_norm, train_phase=opt.phase,
                                   trans=None, trans_i=None)
    else:
        data_prepared = np.load('data/data_prepared.npy', allow_pickle=True).item()

    # data_prepared = split_data(opt.dataset_path, opt.batch_size, opt.phase_norm, opt.ros_norm, train_phase= opt.phase, trans=scripted_transforms, trans_i =scripted_transforms_i)

    #loss
    if opt.loss == 'mse':
        loss_fn = nn.MSELoss()
    if opt.loss == 'l1':
        loss_fn = nn.L1Loss()
    if opt.loss == 'huber':
        loss_fn = nn.HuberLoss(delta=opt.delta)
    
    if opt.resume_from is not None:
        print("Loading weights from %s" % opt.old_dict)
        model.load_state_dict(torch.load(opt.old_dict + ".pt"))
        # discriminator.load_state_dict(torch.load(args.old_dict + "_d.pt"))
        test_loader = data_prepared['data_loader_test']
        loss, loss_c, X, y, y_pred, class_label, condition_tag = test_rg(model, test_loader, opt)
        best_loss_c = loss_c
        best_loss = loss
        print('Best loss from pretrained model: {:.4f}'.format(best_loss))
        

    optimizer = torch.optim.Adam(model.parameters(), betas=(opt.beta1, 0.999), lr=opt.lr, weight_decay=0.001)
    lr_scheduler = get_scheduler(optimizer, opt)

    loss_te_t = []
    loss_tr_t = []

    if opt.phase == 'train':
        # create a logging file to store training losses
        log_name = os.path.join(opt.checkpoints_dir, opt.checkpoint_model, 'loss_log.txt')
        Path(os.path.join(opt.checkpoints_dir, opt.checkpoint_model)).mkdir(parents=True, exist_ok=True)
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        
        train_loader =  data_prepared['data_loader_train']
        test_loader =  data_prepared['data_loader_test']

        for epoch in tqdm(range(opt.n_epochs + opt.n_epochs_decay)):
            loss_tr, y_tr, y_trp = train_rg(model, train_loader, optimizer, loss_fn)
            # loss_tr, _, _, _, _, _, _ = test_rg(model, train_loader, opt)
            loss_te, loss_c, X_te, y_te, y_tep, class_label, condition_tag = test_rg(model, test_loader, opt)

            # loss_te, loss_c, X_te, y_te, y_tep, class_label, condition_tag = test_rg(model, test_loader, opt)
            
            if opt.loss == 'huber':
                print('Train loss: {:.4f}, Test loss: {:.4f}, True loss: {:.4f}'.format(loss_tr, loss_c, loss_te))
                with open(log_name, "a") as log_file:
                    log_file.write('Train loss: {:.4f}, Test loss: {:.4f}, True loss: {:.4f}'.format(loss_tr, loss_c, loss_te))  # save the message

            else: 
                print('Train loss: {:.4f}, Test loss: {:.4f}'.format(loss_tr, loss_te))
                with open(log_name, "a") as log_file:
                    log_file.write('Train loss: {:.4f}, Test loss: {:.4f}'.format(loss_tr, loss_te))  # save the message

            #update learning parameters
            old_lr = optimizer.param_groups[0]['lr']
            if opt.lr_policy == 'plateau':
                lr_scheduler.step(loss_te)
            else:
                lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            
            print('learning rate %.7f -> %.7f' % (old_lr, lr))

            loss_tr_t.append(loss_tr)
            loss_te_t.append(loss_te)

            if  loss_te < best_loss:
                torch.save(model.state_dict(), opt.checkpoint_model + ".pt")
                best_loss = loss_te
                print('Best loss: {:.4f}'.format(best_loss))

            
            if opt.loss == 'huber':
                if  loss_c < best_loss_c:
                    torch.save(model.state_dict(), opt.checkpoint_model + "_c.pt")
                    best_loss_c = loss_c
                    print('Best loss_c: {:.4f}'.format(best_loss_c))

            if  loss_tr < best_train_loss:
                torch.save(model.state_dict(), opt.checkpoint_model + "_tr.pt")
                best_train_loss = loss_tr
                print('Best train loss: {:.4f}'.format(best_train_loss))

            #visuaize images

            # a = np.random.choice(range(len(y_te)), 40)
            # fig, axs = plt.subplots(5, 8, figsize=(12, 10))
            # ax = axs.ravel()

            # for r, k in enumerate(a) :
            #
            #     ax[r].imshow(X_te[k, 0], cmap='gray')
            #     textstr = 'PD:%0.3f\nGT:%0.3f'%(y_tep[k], y_te[k])
            #     ax[r].text(0.0, 1.1, textstr, fontsize=10, transform=ax[r].transAxes)
            #     ax[r].axis('off')
            # plt.show()

            #visulize statistics
            # in_lier_20 = [e for e in range(len(y_te)) if abs(y_tep[e] - y_te[e])/y_te[e] <0.2]
            # ratio_20 = len(in_lier_20)/len(y_te)
            # plt.figure()
            # plt.figure(figsize=(5, 5))
            # plt.scatter(y_te, y_tep, s=0.1)
            # plt.plot(np.linspace(0, 3.5, 100), np.linspace(0, 3.5, 100), c= "red", linestyle=':')
            # plt.text(2.5, 2.5, '%.2f'%ratio_20, fontsize = 14)
            # plt.ylim((1, 3.5))
            # plt.xlim((1, 3.5))
            # plt.show()
        
        loss_1 = np.stack(loss_tr_t)
        loss_2 = np.stack(loss_te_t)
        plt.figure()
        plt.plot(range(len(loss_1)), loss_1)
        plt.plot(range(len(loss_1)), loss_2)
        plt.legend(['Train', 'Val'])
        plt.savefig('../Imgs/loss_train.png', dpi=300)
        plt.show()

    elif opt.phase == 'test':
        dataset_names = ['data_loader_train', 'data_loader_test']
        model_names = ['_tr', '']
        plt.figure(figsize=(10, 5))
        for i in range(2):
            test_loader =  data_prepared[dataset_names[i]]
            model_list = []
            for m in range(8):
                model = define_model(opt, device_list)
                model.load_state_dict(torch.load(opt.checkpoint_model + model_names[i] + "_{}.pt".format(m)))
                model_list.append(model)
            loss_te, loss_c, X_te, y_te, y_tep, class_label, condition_tag = test_rg_ens(model_list, test_loader, opt)
            # model.load_state_dict(torch.load(opt.checkpoint_model + model_names[i] + ".pt"))
            # loss_te, loss_c, X_te, y_te, y_tep, class_label, condition_tag = test_rg(model, test_loader, opt)
            print(dataset_names[i], ', accuracy: ', -loss_te)
            # visulize statistics
            in_lier_20 = [e for e in range(len(y_te)) if abs(y_tep[e] - y_te[e])/y_te[e] < 0.25]
            ratio_20 = len(in_lier_20)/len(y_te)
            plt.subplot(1, 2, i+1)
            plt.scatter(y_te, y_tep, s=0.1)
            plt.xlabel('Label\n{}'.format(dataset_names[i]))
            plt.ylabel('Predicted')
            plt.plot(np.linspace(0, 3.5, 100), np.linspace(0, 3.5, 100), c= "red", linestyle=':')
            plt.text(2.5, 2.5, '%.2f'%ratio_20, fontsize = 14)
            plt.ylim((1, 3.5))
            plt.xlim((1, 3.5))
        plt.savefig('../Imgs/scatter.png', dpi = 300)
        plt.show()


