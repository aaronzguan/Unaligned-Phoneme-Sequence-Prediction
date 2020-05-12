import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
import pickle
import os

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tensor.mean(dim=0)
    return tensor_minusmean


class load_dataset(Dataset):
    def __init__(self, args, name='train'):
        self.name = name
        if name == 'train':
            data_path = args._data_root + args._train_data
            label_path = args._data_root + args._train_label
        elif name == 'val':
            data_path = args._data_root + args._val_data
            label_path = args._data_root + args._val_label
        elif name == 'test':
            data_path = args._data_root + args._test_data

        self.data = np.load(data_path, encoding='bytes', allow_pickle=True)
        self.data_aug = args.data_aug
        self.usa_pca = args.use_pca
        if self.usa_pca:
            if name == 'train' and not os.path.exists('pca_{0}_features.pkl'.format(args.pca_components)):
                self.pca = PCA(args.pca_components).fit(np.concatenate(self.data))
                pickle.dump(self.pca, open('pca_{0}_features.pkl'.format(self.pca.n_components), 'wb'))
            else:
                self.pca = pickle.load(open('pca_{0}_features.pkl'.format(args.pca_components), 'rb'))
        if name != 'test':
            self.label = np.load(label_path, allow_pickle=True)
            # Add label by 1 to make sure the blank label is the first
            # Make sure the blank_id is consistent between CTCLoss, CTCdecode and the model.
            # If the blank_id is 0, add 1 to all the labels.
            self.label += 1

    def __getitem__(self, index):
        if self.name != 'test':
            if self.usa_pca:
                return torch.tensor(self.pca.transform(self.data[index])), torch.LongTensor(self.label[index])
            elif self.data_aug:
                return torch.cat((torch.tensor(self.data[index]), torch.tensor(self.data[index])), dim=1), \
                       torch.LongTensor(self.label[index])
            else:
                return normalize(torch.tensor(self.data[index])), torch.LongTensor(self.label[index])
        else:
            if self.usa_pca:
                return torch.tensor(self.pca.transform(self.data[index]))
            elif self.data_aug:
                return torch.cat((torch.tensor(self.data[index]), torch.tensor(self.data[index])), dim=1)
            else:
                return normalize(torch.tensor(self.data[index]))

    def __len__(self):
        return len(self.data)


def pad_collate_train(batch):
    (xx, yy) = zip(*batch)
    # Get the length before padding
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    # Pad the sequence of x and y with 0
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, x_lens, yy_pad, y_lens


def pad_collate_test(batch):
    x_lens = [len(x) for x in batch]
    xx_pad = pad_sequence(batch, batch_first=True, padding_value=0)
    return xx_pad, x_lens


def get_loader(args, name, shuffle=True, drop_last=False):
        dataset = load_dataset(args, name)
        if name == 'test':
            dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4, collate_fn=pad_collate_test, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        else:
            dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4, collate_fn=pad_collate_train, pin_memory=True, shuffle=shuffle, drop_last=drop_last)
        return dataloader

