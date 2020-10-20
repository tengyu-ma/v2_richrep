import conf
import argparse

from exps import rich_exp


if __name__ == '__main__':
    tr = [f'{i}_0_0' for i in range(1)]
    v2_conf = [int(i.split('_')[0]) for i in conf.V2Config]

    parser = argparse.ArgumentParser(description='Toybox Experiments')
    parser.add_argument('-net_name', type=str, help='The backbone network for the view fusing')

    args = parser.parse_args()

    nview_all = len(tr) * len(v2_conf)
    net_name = args.net_name
    pretrained = True
    mode = 'rich_flatten_extra'

    rich_exp.exp_main(tr, v2_conf, nview_all, net_name, pretrained, mode)
