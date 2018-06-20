import torch

from args import ArgParser
from data_loader import PBTDataLoader


def train(args):

    data_loader = PBTDataLoader(args.dataset, is_training=True,
                                batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == '__main__':
    parser = ArgParser()

    train(parser.parse_args())
