import gc
import torch
import torch.nn as nn
import conf
import util

from nets.rich_net import RichNet
from exps.v2trainer import V2Trainer

torch.backends.cudnn.benchmark = True


def exp_main(tr, v2_conf, nview_all, net_name, pretrained, batch_size, mode):
    net = RichNet(nview_all=nview_all, net_name=net_name, pretrained=pretrained, mode=mode)
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-05, weight_decay=0.0)
    loss_func = nn.CrossEntropyLoss()
    hyper_p = util.HyperP(
        lr=0.5,
        batch_size=batch_size,
        num_workers=0,
        epochs=300,
        preload=False,
    )

    tb_trainer = V2Trainer(
        tr=tr,
        v2_conf=v2_conf,
        mode='rich',
        img_size=(299, 299),
        net=net,
        net_name=net_name,
        optimizer=optimizer,
        loss_func=loss_func,
        hyper_p=hyper_p
    )
    print(f'=== {tb_trainer.exp_name} ===')
    tb_trainer.train_test_save()


def main():
    tr = ['0_0_0']
    v2_conf = [1, 2]

    # tr = [f'{i}_0_0' for i in range(12)]
    # v2_conf = [int(i.split('_')[0]) for i in conf.V2Config]

    nview_all = len(tr) * len(v2_conf)
    net_name = 'resnet34'
    pretrained = True
    mode = 'rich_flatten'
    batch_size = 1

    exp_main(tr, v2_conf, nview_all, net_name, pretrained, batch_size, mode)


if __name__ == '__main__':
    main()
