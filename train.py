import argparse

from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='')

    return parser.parse_args()


def debug():
    import json
    with open('data/didemo/train_data.json') as fp:
        data = json.load(fp)
        for k, v in data.items():
            print(v.keys())
        exit(0)


def main(args):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    # debug()
    # exit(0)

    seed = 8
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    # logging.info('base seed {}'.format(seed))
    args = load_json(args.config_path)
    print(args)

    runner = MainRunner(args)

    runner.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
