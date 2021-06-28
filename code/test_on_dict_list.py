from __future__ import print_function
import torch 
from torchvision import transforms
import argparse
import cv2 
import sys
import numpy as np 
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150
from matplotlib import pyplot as plt
from util import define_model
from prepare_data import split_data
from test import test_rg

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_list = [device]

    parser = argparse.ArgumentParser(description="Parser for ros label prediction")
    parser.add_argument('--phase', type=str, default='train', help='train, test, etc')
    parser.add_argument('--dict_files', nargs='+', help='list of files for testing')

    #data_related options
    parser.add_argument("--dataset_path", type=str, help="path to the training dataset")
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument("--phase_norm", type=str, default ='min_max', help="normalization on the phase data [min_max| self_min_max]")
    parser.add_argument("--ros_norm", type=str, default ='None', help="normalization on the phase data [min_max| self_min_max]")
    parser.add_argument("--std_transform", type=bool, default =False, help="whether perform std mean norm on the phase data")
    

    # training options 
    # parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--checkpoint_model", type=str, help=" path to checkpoint model")
    parser.add_argument('--resume_from', type=bool, required=False, help = "whether retrain from a certain old model")
    parser.add_argument('--old_dict', required='--resume_from' in sys.argv)
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--loss', type=str, default='mse', help='loss function [mse | l1 | huber]')
    parser.add_argument('--delta', type=float, default=1, help='huber delta')

    # extraction model parameters
    parser.add_argument('--model_name', type=str, default='resnet34', help='chooses which model to use. [resnet18 | resnet34 | resnet50| resnet101 | resnet152]')
    parser.add_argument('--norm_type', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')

    # fc model parameters
    parser.add_argument('--fc_filters', type=int, default=512, help='# fc filters in the first layer')
    parser.add_argument('--fc_layers', type=int, default=3, help='# fc layers')
    parser.add_argument('--fc_norm_type', type=str, default='none', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--fc_drop_out', type = float, default = 0.2, help='dropout in the fc_layers')
    parser.add_argument('--fc_relu_slope', type = float, default = 0.01, help='relu slope in the fc_layers')

    #model init
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    # Parse and return arguments
    opt = parser.parse_known_args()[0]
    print(len(opt.dict_files))
    fig, axs = plt.subplots(len(opt.dict_files), 3, figsize=(12, 3*len(opt.dict_files)))
    ax = axs.ravel()

    for i, dict_file in enumerate(opt.dict_files):
        if dict_file.find('resnet_18') != -1:
            print('resnet_18')
            opt = parser.parse_args(['--model_name', 'resnet18'])
            opt = parser.parse_args(['--fc_filters', '512'])
            opt = parser.parse_args(['--fc_layers', '3'])

        elif dict_file.find('resnet_34') != -1:
            print('resnet_34s')
            opt = parser.parse_args(['--model_name', 'resnet34'])
            opt = parser.parse_args(['--fc_filters', '512'])
            opt = parser.parse_args(['--fc_layers', '3'])

        elif dict_file.find('resnet_50') != -1:
            print('resnet_50')
            opt = parser.parse_args(['--model_name', 'resnet50'])
            opt = parser.parse_args(['--fc_filters', '2048'])
            opt = parser.parse_args(['--fc_layers', '2'])

        elif dict_file.find('resnet_101') != -1:
            opt = parser.parse_args(['--model_name', 'resnet101'])
        elif dict_file.find('resnet_152') != -1:
            opt = parser.parse_args(['--model_name', 'resnet152'])

        if dict_file.find('bs16') != -1:
            opt = parser.parse_args(['--batch_size', '16'])
        elif dict_file.find('bs128') != -1:
            opt = parser.parse_args(['--batch_size', '128'])
        elif dict_file.find('bs8') != -1:
            opt = parser.parse_args(['--batch_size', '8'])

        if dict_file.find('instance') != -1:
            opt = parser.parse_args(['--norm_type', 'instance'])
        else:
            opt = parser.parse_args(['--norm_type', 'batch'])

    # creat discinminator
        model = define_model(opt, device_list)
        model.load_state_dict(torch.load(dict_file + ".pt"))

        img_trans = torch.nn.Sequential(
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(45, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None),
                                transforms.RandomVerticalFlip(p=0.5),)
        scripted_transforms = torch.jit.script(img_trans)

        if opt.std_transform == True:
            img_trans_i = torch.nn.Sequential( transforms.Normalize(mean=[0.5], std=[0.2]),
            )
            scripted_transforms_i = torch.jit.script(img_trans_i)
        else:
            scripted_transforms_i = None

        data_prepared = split_data('Data_mask_ponly_r7.npy', opt.batch_size, opt.phase_norm, opt.ros_norm, train_phase= opt.phase, trans=scripted_transforms, trans_i =scripted_transforms_i)
        data_prepared_1 = split_data('Data_mask_test_r7.npy', opt.batch_size, opt.phase_norm, opt.ros_norm, train_phase= 'test', trans=scripted_transforms, trans_i =scripted_transforms_i)
        test_loader =  data_prepared['data_loader_test']
        train_loader =  data_prepared['data_loader_train']
        test_loader_1 =  data_prepared_1['data_loader_test']

        loader_list = [train_loader, test_loader, test_loader_1]
        for j in range(3): 
            loss_te, loss_c, X_te, y_te, y_tep, class_label, condition_tag = test_rg(model, loader_list[j], opt)
            mae = np.mean(np.abs(y_te - y_tep))
            mape = np.mean(np.abs(y_te - y_tep)/y_te)
            ax[i*3+j].scatter(y_te, y_tep, s=0.1)
            ax[i*3+j].plot(np.linspace(0, 5, 30), np.linspace(0, 5, 30), c= "red", linestyle=':')
            ax[i*3+j].set_xlim([1, 3.5])
            ax[i*3+j].set_ylim([1, 3.5])
        # plt.title('Test-PMA')
            ax[i*3+j].set_xlabel('Ground truth', fontsize=14)
            #plt.ylim((0, 6000))
            ax[i*3+j].set_ylabel('Predition', fontsize=14)
        # plt.legend(handles = legend_vec)
            ax[i*3+j].grid(which='both', axis='y', alpha=0.4)
            ax[i*3+j].text(1.1, 2.8, 'MAE: %.4f\nMAPE:%.2f%% \nMSE: %.4f'%(mae, mape*100, loss_te), fontsize = 14)
    plt.show()