import torch.utils.data as data
import numpy as np 
import torch 
import torch.nn.functional as F
import os

class Dataset(data.Dataset):
    """Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, data, label, classes, condition, 
                 transform_p=None, transform_i=None):
        self.transform = transform_p
        self.target_transform = transform_i
        
        self.data = data
        self.labels = label
        self.classes = classes
        self.condition = condition 



    def __getitem__(self, index):
        """
         Args:
             index (int): Index
         Returns:
             tuple: (image, target) where target is index of the target class.
         """
 
        img, ros_target, class_target, condition = self.data[index], self.labels[index], self.classes[index], self.condition[index]

        img = np.float32(img)
        img = torch.from_numpy(img)

        p2d = (2, 2, 2, 2)


        img= F.pad(img, p2d, "constant", 0)


        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            img = self.target_transform(img)
        
        img = img.type(torch.FloatTensor)

        ros_target = np.float32(ros_target)
        class_target = np.long(class_target)

        return img, ros_target, class_target, condition

    def __len__(self):
        return len(self.data)


def split_data(datafile, batch_size, phase_norm, ros_norm, train_phase, trans=None, trans_i=None):

    img_data = np.load(datafile, allow_pickle=True).item()
    phase_img = img_data['Images']
    ros_level = img_data['ros_level']
    classes = img_data['classes']
    condition =img_data['condition']

    phase_data = []
    print('phase data: data max %.5f, data min %.5f' %(np.amin(phase_img), np.amax(phase_img)))

    if phase_norm == 'min_max':
        phase_img = (phase_img - 0.29) / (10-  0.29)
    if phase_norm == 'self_min_max':
        for i in range(len(ros_level)):
            phase_data.append((phase_img[i] - np.amin(phase_img[i]))/(np.amax(phase_img[i]) - np.amin(phase_img[i])))
        phase_img = np.stack(phase_data, axis=0)

    if ros_norm == 'min_max' :
        rmin = np.amin(ros_level)
        rmax = np.amax(ros_level)
        ros_level = (ros_level - rmin) / (rmax - rmin)

    if ros_norm == 'tanh':
        r_mean = np.mean(ros_level)
        r_std = np.std(ros_level)
        ros_level= np.tanh((0.1 * (ros_level - r_mean) / r_std ) - (0.1 * (rmin - r_mean) / r_std))

    ros_level = np.expand_dims(ros_level, axis=-1)
    phase_img = np.expand_dims(phase_img, axis = 1)

    np.random.seed(0)
    index_vec = np.arange(len(condition))
    np.random.shuffle(index_vec)
    tr_idx = int(0.8 * len(condition))
    index_tr = index_vec[:tr_idx]
    index_te = index_vec[tr_idx:]
    # index_tr = np.random.choice(len(condition), int(0.8 * len(condition)), replace=False)
    # index_te = [e for e in range(len(condition)) if e not in index_tr]
    train_dataset = Dataset(phase_img[index_tr], ros_level[index_tr], classes[index_tr], condition[index_tr],
                            transform_p=trans, transform_i=trans_i)
    test_dataset = Dataset(phase_img[index_te], ros_level[index_te], classes[index_te], condition[index_te],
                           transform_i=trans_i)

    print('Train size: {:d}% Test size: {:d}%'.format(len(index_tr), len(index_te)))
    data_loader_train = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_loader_test = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    data_prepared = {'data_loader_train': data_loader_train, 'data_loader_test': data_loader_test,
                     'train_dataset': train_dataset, 'test_dataset': test_dataset}
    np.save('data/data_prepared.npy', data_prepared)
    # if train_phase == 'train':
    #
    # else:
    #     train_dataset = Dataset(phase_img, ros_level, classes, condition, transform_i = trans_i)
    #     data_loader_train = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     data_prepared = {'data_loader_test': data_loader_train,'test_dataset': train_dataset}
    return data_prepared
