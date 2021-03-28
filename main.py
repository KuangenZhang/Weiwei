import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)


class Net(nn.Module):
    def __init__(self, class_num=3, hidden_size = 100):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv1_bn = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.conv2_bn = nn.BatchNorm2d(hidden_size)
        self.conv3 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.conv3_bn = nn.BatchNorm2d(hidden_size)
        self.fc1 = nn.Linear(in_features=5 * 5 * hidden_size, out_features=hidden_size)
        self.fc1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc2_bn = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=class_num)
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        # x [1, 28, 28]
        x = x.view((-1, 1, 60, 60))
        x = F.relu(self.conv1_bn(self.conv1(x)))  # [100, 58, 58]
        x = self.dropout(x)
        x = F.max_pool2d(x, 2, 2)  # [100, 29, 29]
        x = F.relu(self.conv2_bn(self.conv2(x)))  # [100, 27, 27]
        x = self.dropout(x)
        x = F.max_pool2d(x, 2, 2)  # [100, 13, 13]
        x = F.relu(self.conv3_bn(self.conv3(x)))  # [100, 11, 11]
        x = self.dropout(x)
        x = F.max_pool2d(x, 2, 2)  # [100, 5, 5]
        x = x.view(-1, 5 * 5 * self.hidden_size)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for data, label in train_loader:
        optimizer.zero_grad()
        data = Variable(data.float().to(device))
        label = Variable(label.long().to(device))
        pred = model(data)
        loss = F.nll_loss(pred, label)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = 0
    y_list = []
    y_pred_list = []
    with torch.no_grad():  # validation does not require gradient
        for X, y in test_loader:
            X = Variable(torch.FloatTensor(X)).to(device)
            y = Variable(torch.LongTensor(y)).to(device)
            y_pred = model(X)
            y_pred = y_pred.argmax(dim=1, keepdim=True)
            correct += y_pred.eq(y.view_as(y_pred)).sum().item()
            y_list.append(y)
            y_pred_list.append(y_pred)
    y = torch.cat(y_list, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred_list, dim=0).cpu().detach().numpy().squeeze()
    # print('y_size: {}, {}, {}'.format(len(y), y.shape, y_pred.shape))
    # print('accuracy: {}'.format(np.mean(y_pred == y)))
    acc = correct / len(y)
    return acc, y, y_pred


def split_data(batch_size=128):
    data = np.load('Data_PCF.npy', allow_pickle=True).item()
    X = data['Images']
    y = data['classes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print('Initial y_size: {}'.format(len(y_test)))
    data_loader_train = torch.utils.data.DataLoader(
        Dataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    data_loader_test = torch.utils.data.DataLoader(
        Dataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    return data_loader_train, data_loader_test, (X_train, y_train), (X_test, y_test)


def main():
    lr = 1e-3
    epochs = 50
    train_loader, test_loader, train_dataset, test_dataset = split_data()
    model = Net().to(device)

    is_train = True
    if is_train:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_acc_test = 0
        for epoch in tqdm(range(epochs)):
            train(model, train_loader, optimizer, epoch)
            acc_train, _, _ = test(model, train_loader)
            acc_test, _, _ = test(model, test_loader)
            print('Training accuracy : {:.2f}%, test accuracy: {:.2f}%'.format(acc_train * 100, acc_test * 100))
            if acc_test > best_acc_test:
                best_acc_test = acc_test
                torch.save(model.state_dict(), "net.pt")

    model.load_state_dict(torch.load("net.pt"))
    acc_test, y, y_pred = test(model, test_loader)
    plot_comfusion_matrix(y, y_pred)
    print('Final test accuracy: {:.2f}%'.format(acc_test * 100))


def plot_comfusion_matrix(y, y_pred):
    cf_mat = confusion_matrix(y, y_pred)
    cf_mat_sum = np.sum(cf_mat, axis=-1, keepdims=True)
    cf_mat = cf_mat/cf_mat_sum
    print(cf_mat)


if __name__ == '__main__':
    main()

