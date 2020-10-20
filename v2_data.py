import csv
import glob
import os
import re
import conf
import util
import pickle
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import trimesh
import logging

from PIL import Image
from itertools import product


class V2Data(torch.utils.data.Dataset):
    def __init__(self, root, dataset, tr, v2_conf, mode='sv', preload=True, transform=None, nview=12, init_df=True):
        assert mode in ['sv', 'rich'], ValueError(f'Invalid mode {mode}')

        self.root = root
        self.dataset = dataset
        self.all_df = self._get_df(init_df)
        self.index_df = self.all_df[['se', 'ca', 'no']].drop_duplicates().reset_index(drop=True)

        self.tr = tr
        self.v2_conf = v2_conf
        self.df = self._filter()

        self.mode = mode
        self.transform = transform
        self.nview = nview

        self.preload = preload
        self.loaded_data = self._preload() if preload else None

    def _get_df(self, init_df):
        if init_df:
            all_files_pickle = pickle.load(open(conf.InitFile, 'rb'))
            all_files = list(map(lambda x: x.replace('{root_placeholder}', self.root), all_files_pickle))
            all_df = pd.read_csv(conf.InitDf)
            all_df['dir'] = all_files
            all_df = all_df[all_df.se == self.dataset]
        else:
            all_files = sorted(glob.glob(f'{self.root}/*/{self.dataset}/*/*/*.pickle'), key=util.sort_v2_file)
            all_df = pd.DataFrame(
                list(map(util.sort_v2_file, all_files)),
                columns=['se', 'ca', 'no', 'tr_x', 'tr_y', 'tr_z', 'v2_conf'],
            )
            all_df['tr'] = all_df.apply(lambda row: f'{row.tr_x}_{row.tr_y}_{row.tr_z}', axis=1)
            all_df.to_csv(conf.InitDf, index=False)
        return all_df

    def _filter(self):
        df = self.all_df
        df = df[df['tr'].isin(self.tr)] if self.tr is not None else df
        df = df[df['v2_conf'].isin(self.v2_conf)] if self.v2_conf is not None else df
        return df.reset_index(drop=True)

    def _preload(self):
        loaded_data = []
        for index in range(len(self)):
            print(f'\rLoading data... {index + 1} / {len(self)}', end='')
            loaded_data.append(self._getitem(index))
        print('')
        return loaded_data

    def _getitem(self, index):
        if self.mode == 'sv':
            info = self.df.loc[index]
            label = conf.ModelNet40Categories.index(info.ca)

            img = Image.fromarray((pickle.load(open(info.dir, 'rb'))[:, :, :3] / 2 * 255).astype(np.uint8))
            if self.transform is not None:
                img = self.transform(img)
            path = info.dir
        elif self.mode == 'rich':
            info = self.index_df.loc[index]
            label = conf.ModelNet40Categories.index(info.ca)

            rich_views = pd.DataFrame(
                product(self.tr, self.v2_conf),
                columns=['tr', 'v2_conf']
            )
            rich_views['ca'] = info.ca
            rich_views['no'] = info.no
            rich_df = pd.merge(self.df, rich_views, on=['ca', 'no', 'tr', 'v2_conf'])
            imgs = [Image.fromarray((pickle.load(open(d, 'rb'))[:, :, :3] / 2 * 255).astype(np.uint8)) for d in rich_df.dir]
            if self.transform is not None:
                imgs = [self.transform(img) for img in imgs]

            img = torch.stack(imgs).float()
            path = '>'.join(rich_df.dir)
        else:
            raise ValueError(f'invalid mode {self.mode}')

        return img, label, path

    def __len__(self):
        return len(self.index_df) if self.mode == 'rich' else len(self.df)

    def __getitem__(self, index):
        if self.preload:
            return self.loaded_data[index]
        return self._getitem(index)
