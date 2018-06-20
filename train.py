import multiprocessing as mp
import re
import queue as Queue
import torch

from args import ArgParser
from data_loader import PBTDataLoader
from trainer import PBTrainer


def main(args):
    # Construct queue of jobs and pool to read from the queue
    population_queue = mp.Queue()
    num_trainers = len(args.gpu_ids)
    trainer_pool = mp.Pool(num_trainers, trainer_loop, (args, population_queue))

    epoch = 1
    while epoch != args.num_epochs:
        # Construct arguments to pass to each trainer
        for member_idx in range(args.population_size):
            population_member = (args, member_idx, epoch)
            population_queue.put(population_member)

        # Train each member of the population for one epoch
        epoch += 1

    # Terminate training by sending None to each trainer
    for _ in range(num_trainers):
        population_queue.put(None)
    trainer_pool.close()
    trainer_pool.join()


def trainer_loop(args, population_queue):
    """Initialize a trainer tied to a single GPU.
    Loop over the queue training .

    Args:
        args: Command-line arguments.
        population_queue: Queue of members in the population.
    """
    # Each worker gets its own GPU
    worker_id = int(re.search('\d+', mp.current_process().name).group(0)) - 1
    gpu_id = args.gpu_ids[worker_id % len(args.gpu_ids)]
    print('[{}] Initialized worker on GPU: {}'.format(worker_id, gpu_id))

    num_trained = 0
    while True:
        member = population_queue.get(block=True)
        if member is None:
            break

        num_trained += 1

    print('[{}] Trained {} models'.format(worker_id, num_trained))


def train(args, member_idx, epoch):
    """Train a member of the population for one epoch.
    Update population file with results.

    Args:
        args: Command-line arguments.
        member_idx: Index of population to train.
        epoch: Epoch to train.
    """
    gpu_id = args.gpu_ids[member_idx % len(args.gpu_ids)]
    data_loader = PBTDataLoader(args.dataset, is_training=True,
                                batch_size=args.batch_size, num_workers=args.num_workers)

    # Get most recent checkpoint for this member

    # Truncation selection, exploit, and explore

    # Step for one epoch


if __name__ == '__main__':
    parser = ArgParser()
    main(parser.parse_args())
