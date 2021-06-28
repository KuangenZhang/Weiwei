import torch.nn.functional as F
from torch.autograd import Variable
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np 
from losses import * 
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 150
from matplotlib import pyplot as plt

def test_dict(model, dict_files, test_loader):

    for modeldict in dict_files: 
        model.load_state_dict(torch.load(modeldict + ".pt"))
        model.eval()

        y_list = []
        y_pred_list = []
        class_label_list = []
        condition_list = []
        image_list = []
        with torch.no_grad():  # validation does not require gradient
            for X, y, class_label, condition in test_loader:

                X = Variable(torch.FloatTensor(X)).to(device)
                y = Variable(torch.FloatTensor(y)).to(device)
                y_pred = model(X)

                y_list.append(y)
                y_pred_list.append(y_pred)
                class_label_list.append(class_label)
                condition_list.append (condition)
                image_list.append(X)

        X = torch.cat(image_list, dim=0).cpu().detach().numpy()
        y = torch.cat(y_list, dim=0).cpu().detach().numpy()
        y_pred = torch.cat(y_pred_list, dim=0).cpu().detach().numpy()
        class_label = torch.cat(class_label_list, dim=0).cpu().detach().numpy()
        condition_tag = torch.cat(condition_list, dim=0).cpu().detach().numpy()
        loss = np.mean(np.abs(y - y_pred))

        in_lier_20 = [e for e in range(len(y)) if abs(y[e] - y_pred[e])/y[e] <0.2]
        ratio_20 = len(in_lier_20)/len(y)

        plt.figure()
        plt.figure(figsize=(5, 5))
        plt.scatter(y, y_pred, s=0.1)
        plt.plot(np.linspace(0, 3.5, 100), np.linspace(0, 3.5, 100), c= "red", linestyle=':')
        plt.text(2.5, 2.5, '%.2f'%ratio_20, fontsize = 14)
        plt.ylim((1, 3.5))
        plt.xlim((1, 3.5))
        plt.show()

  

