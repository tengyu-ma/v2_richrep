import os
import shutil
import conf
import torch
import pandas as pd


class HyperP:
    def __init__(self, lr, batch_size, num_workers, epochs, preload):
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.preload = preload


def get_mean_std(data_loader):
    """ Calculate dataset mean and std
    """
    mean = 0.
    var = 0.
    nb_samples = 0.
    for batch_idx, (x, y_true, file) in enumerate(data_loader):
        batch_samples = x.size(0)
        x = x.view(batch_samples, x.size(1), -1)
        mean += x.mean(2).sum(0)
        var += x.var(2).sum(0)
        nb_samples += batch_samples
        print(f'\rCalculate mean and std: {batch_idx + 1}/{len(data_loader)}', end='')
    print('')

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)

    print('mean:', mean)
    print('std:', std)
    return mean, std


def sort_v2_file(path):
    segs = path.split('/')
    ca = segs[-5]
    se = segs[-4]
    no = segs[-3]
    tr_x, tr_y, tr_z = list(map(int, segs[-2].split('_')))
    v2_conf = int(segs[-1].split('_')[0])
    return se, ca, no, tr_x, tr_y, tr_z, v2_conf


def dir_naming(row):
    v2_dict = {
        1: '1_1_128_128_2|128',
        2: '2_2_64_64_2|128',
        4: '4_4_32_32_2|128',
        8: '8_8_16_16_2|128',
        16: '16_16_8_8_2|128',
        32: '32_32_4_4_2|128',
        64: '64_64_2_2_2|128',
        128: '128_128_1_1_2|128',
    }
    return f'{row.root}/{row.ca}/{row.se}/{int(row.no):04d}/{row.tr_x}_{row.tr_y}_{row.tr_z}_{v2_dict[row.v2_conf]}'


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
#
#
# class NpResize:
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, img):
#         return img.