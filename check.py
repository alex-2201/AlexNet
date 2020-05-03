import argparse

from utils.parser import get_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_alex_net", type=str, default="./configs/alex_net.yaml")
    return parser.parse_args()


def main():
    # create model
    pass
    # define loss function (criterion) and optimizer

    # optionally resume from a checkpoint

    # data loading code


def train():
    pass


def validate():
    pass


def save_checkpoint():
    pass


class AverageMeter(object):
    pass


def adjust_learning_rate():
    pass


def accuracy():
    pass



if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_alex_net)
    print(cfg.EPOCHS)
