import gc
import os
import util
import conf
import torch
import pickle
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

from torch.autograd import Variable
from datetime import datetime
from v2_data import V2Data


class V2Trainer:
    def __init__(self, tr, v2_conf, img_size, net, net_name, mode, optimizer, loss_func, hyper_p):
        self.tr = tr
        self.v2_conf = v2_conf
        self.img_size = img_size
        self.net = net
        self.net_name = net_name
        self.mode = mode
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.hyper_p = hyper_p

        self.exp_v2data = f'{tr}-{v2_conf}'
        self.exp_name = f'{net_name}-{tr}-{v2_conf}'
        self.log_dir = f'{conf.V2LogDir}/{self.exp_name}'
        self.train_loader, self.test_loader = self.get_dataloader()

    def get_dataloader(self):
        Data = V2Data

        if os.path.exists(conf.V2DataMeanStdCacheFile):
            mean_std_cache = pickle.load(open(conf.V2DataMeanStdCacheFile, 'rb'))
        else:
            mean_std_cache = {}

        if self.exp_v2data in mean_std_cache:
            print('=== Loading mean std from cache ===')
            mean, std = mean_std_cache[self.exp_v2data]
        else:
            print('=== Calculating mean std... ===')
            data = Data(
                root=conf.V2DataDir,
                dataset='train',
                tr=self.tr,
                v2_conf=self.v2_conf,
                mode='sv',
                preload=self.hyper_p.preload,
                transform=transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                ]),
            )
            data_loader = torch.utils.data.DataLoader(
                data,
                batch_size=self.hyper_p.batch_size,
                shuffle=False,
                num_workers=self.hyper_p.num_workers,
                pin_memory=True
            )
            mean, std = util.get_mean_std(
                data_loader=data_loader
            )

            del data
            del data_loader
            gc.collect()  # free memory

            mean_std_cache[self.exp_v2data] = mean, std
            pickle.dump(mean_std_cache, open(conf.V2DataMeanStdCacheFile, 'wb'))

        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_loader = torch.utils.data.DataLoader(
            Data(
                root=conf.V2DataDir,
                dataset='train',
                tr=self.tr,
                v2_conf=self.v2_conf,
                mode=self.mode,
                preload=self.hyper_p.preload,
                transform=transform,
            ),
            batch_size=self.hyper_p.batch_size,
            shuffle=True,
            num_workers=self.hyper_p.num_workers,
            pin_memory=True,
            drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(
            Data(
                root=conf.V2DataDir,
                dataset='test',
                tr=self.tr,
                v2_conf=self.v2_conf,
                mode=self.mode,
                preload=self.hyper_p.preload,
                transform=transform,
            ),
            batch_size=self.hyper_p.batch_size,
            shuffle=False,
            num_workers=self.hyper_p.num_workers,
            pin_memory=True,
            drop_last=False
        )
        return train_loader, test_loader

    def update_lr(self, epoch):
        for p in self.optimizer.param_groups:
            if epoch > 0 and (epoch + 1) % 10 == 0:
                lr = p['lr'] * 0.5
            else:
                lr = p['lr']
            p['lr'] = lr

    def train_test_save(self):
        start_time = datetime.now()
        for epoch in range(self.hyper_p.epochs):
            total_loss, total_acc, res_dfs = self.run_one_epoch(epoch, train=True)
            self.print_res(epoch, total_loss, total_acc, start_time, train=True)
            self.save(epoch, total_loss, total_acc, res_dfs, train=True)
            torch.cuda.empty_cache()
            gc.collect()

            if epoch == 0 or (epoch + 1) % 10 == 0:
                total_loss, total_acc, res_dfs = self.run_one_epoch(epoch, train=False)
                self.print_res(epoch, total_loss, total_acc, start_time, train=False)
                self.save(epoch, total_loss, total_acc, res_dfs, train=False)
                torch.cuda.empty_cache()
                gc.collect()

    def print_res(self, epoch, total_loss, total_acc, start_time, train=True):
        stage = 'Train' if train else 'Test'
        print(
            f'{stage}: [Epoch {epoch + 1}/{self.hyper_p.epochs} {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] '
            f'Loss {total_loss:.06} '
            f'Acc {total_acc:.06} '
            f'Spent {(datetime.now() - start_time)} '
            f'ETA {(datetime.now() - start_time) / (epoch + 1) * self.hyper_p.epochs}'
        )

    def run_one_epoch(self, epoch, train=True):
        dataloader = self.train_loader if train else self.test_loader
        self.update_lr(epoch)

        total_loss = 0
        res_dfs = []
        for batch_idx, (x, y_true, file) in enumerate(dataloader):
            print(f'\rBatch: {batch_idx + 1}/{len(dataloader)} ', end='')
            self.net.train() if train else self.net.eval()

            if self.mode == 'rich':
                N, V, C, H, W = x.size()
                x = Variable(x).view(-1, C, H, W)
            x, y_true = x.float().cuda(), Variable(y_true).cuda()

            y_pred = self.net(x)
            loss = self.loss_func(y_pred, y_true)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

            y_preda_df = pd.DataFrame(y_pred.data.cpu().numpy(), columns=conf.ModelNet40Categories)
            y_pred_df = pd.DataFrame(y_pred.data.max(1)[1].cpu(), columns=['y_pred'])
            y_true_df = pd.DataFrame(y_true.data.cpu(), columns=['y_true'])
            file_df = pd.DataFrame(file, columns=['file'])

            res_df = pd.concat([y_preda_df, y_pred_df, y_true_df, file_df], axis=1)
            res_dfs.append(res_df)

        res_dfs = pd.concat(res_dfs)
        total_loss /= len(dataloader)
        total_acc = np.sum(res_dfs.y_pred == res_dfs.y_true) / len(res_dfs)
        return total_loss, total_acc, res_dfs

    def save(self, epoch, total_loss, total_acc, res_dfs, train=True):
        net_state_dir = os.path.join(self.log_dir, 'state')
        log_dir = os.path.join(self.log_dir, 'train' if train else 'test')

        util.create_folder(net_state_dir)
        util.create_folder(log_dir)

        if not train:
            torch.save(self.net.state_dict(), os.path.join(net_state_dir, f'{epoch:03d}-{self.exp_name}-state.pkl'))

        res_dfs.to_csv(os.path.join(
            log_dir, f'{epoch:03d}-{self.exp_name}-loss_{total_loss:0.6f}-acc_{total_acc:0.6f}.csv'
        ))
