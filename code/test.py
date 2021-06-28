import torch.nn.functional as F
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np 
from losses import * 
from torch.autograd import Variable

def test_rg(model, test_loader, opt):
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
    if opt.loss == 'mse':
        # loss_te = np.mean(np.square(y - y_pred))
        loss_te = -np.mean(np.abs((y-y_pred)/y) <= 0.2)
        loss_true = loss_te
    if opt.loss == 'l1':
        loss_te = np.mean(np.abs(y - y_pred))
        loss_true = np.mean(np.square(y - y_pred))
    if opt.loss == 'huber':
        loss_te_list = np.where(np.abs(y - y_pred) < opt.delta , 0.5*((y - y_pred)**2), opt.delta*np.abs(y - y_pred) - 0.5*(opt.delta**2))
        loss_te = np.mean(loss_te_list)
        loss_true = np.mean(np.square(y - y_pred))
    return loss_true, loss_te, X, y, y_pred, class_label, condition_tag