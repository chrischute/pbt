import argparse


class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Population-Based Training')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='Comma-separated list of GPUs to use.')
        self.parser.add_argument('--num_epochs', type=int, default=0,
                                 help='Number of epochs to train for. If 0, train forever.')
        self.parser.add_argument('--population_size', type=int, default=10,
                                 help='Number of models in a population.')
        self.parser.add_argument('--dataset', type=str, default='CIFAR10', choices=('CIFAR10',),
                                 help='Dataset to train on.')
        self.parser.add_argument('--ckpt_dir', type=str, default='ckpts/',
                                 help='Directory to save checkpoints and population info.')

    def parse_args(self):
        args = self.parser.parse_args()
        args.gpu_ids = [int(i) for i in str(args.gpu_ids).split(',') if int(i) >= 0]

        return args
