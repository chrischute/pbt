import argparse


class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Population-Based Training')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='Comma-separated list of GPUs to use.')
        self.parser.add_argument('--population_size', type=int, default=10,
                                 help='Number of models in a population.')
        self.parser.add_argument('--dataset', type=str, default='CIFAR10', choices=('CIFAR10',),
                                 help='Dataset to train on.')

    def parse_args(self):
        return self.parser.parse_args()
