import yaml
import argparse
import os
from easydict import EasyDict
from interfaces.super_resolution import TextSR
from setup import Logger
import warnings

def main(config, args):
    Mission = TextSR(config, args)
    Mission.test()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    parser.add_argument('--STN', action='store_true', default=True, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--n_eca', type=int, default=5, help='')
    parser.add_argument('--n_mlp', type=int, default=8, help='')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--general', action='store_true', default=False, help='')
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--test_model', type=str, default='CRNN', choices=['ASTER', "CRNN", "MORAN",'ABINet','MATRN','PARSeq'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--lr_position', type=float, default=1e-5, help='fine tune for mae module')
    parser.add_argument('--tpg', type=str, default="PARSeq", choices=["CRNN",'PARSeq'])
    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'rb'), Loader=yaml.Loader)
    Logger.init('logs', 'StyleSRN', 'test')
    Logger.enable_file()
    config = EasyDict(config)
    main(config, args)
