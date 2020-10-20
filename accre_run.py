import conf
import argparse

from exps import rich_exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toybox Experiments')
    parser.add_argument('-tr', type=int, help='How many rotations you want to use')
    parser.add_argument('-v2n', type=int, help='How many v2 configurations you want to use')
    parser.add_argument('-net_name', type=str, help='The backbone network for the view fusing')
    parser.add_argument('-batch_size', type=int, help='The batch size for training')
    parser.add_argument('-mode', type=str, help='How to fuse V2 representations')

    args = parser.parse_args()

    tr = [f'{i}_0_0' for i in range(args.tr)]
    v2_conf = [int(i.split('_')[0]) for i in conf.V2Config[:args.v2n]]
    nview_all = len(tr) * len(v2_conf)
    net_name = args.net_name
    pretrained = True
    batch_size = args.batchsize
    mode = args.mode

    rich_exp.exp_main(tr, v2_conf, nview_all, net_name, pretrained, batch_size, mode)
