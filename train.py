import multiprocessing as mp
import os
import pickle
import re
import time
import util

from args import ArgParser
from data_loader import PBTDataLoader
from member import Member


def main(args):
    # Construct queue of jobs and pool to read from the queue
    population_queue = mp.Queue()
    num_trainers = len(args.gpu_ids)
    trainer_pool = mp.Pool(num_trainers, trainer_loop, (args, population_queue))

    epoch = 1
    while epoch != args.num_epochs:
        # Construct arguments to pass to each trainer
        for member_id in range(args.population_size):
            population_member = (args, member_id, epoch)
            population_queue.put(population_member)

        # Train each member of the population for one epoch
        epoch += 1

    # Stop the trainers by sending None
    for _ in range(num_trainers):
        population_queue.put(None)
    trainer_pool.close()
    trainer_pool.join()


def trainer_loop(args, population_queue):
    """Initialize a trainer tied to a single GPU.
    Loop over the queue training members of the population for one epoch at a time.

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
        # Get a new member to train
        population_member = population_queue.get(block=True)
        if population_member is None:
            break
        train(*population_member, gpu_id)

        num_trained += 1

    print('[{}] Trained {} models'.format(worker_id, num_trained))


def train(args, member_id, epoch, gpu_id):
    """Train a member of the population for one epoch.
    Update population file with results.

    Args:
        args: Command-line arguments.
        member_id: ID for member to train.
        epoch: Epoch to train.
        gpu_id: ID of GPU to use during training.
    """

    # Read population from disk
    population_lock = os.path.join(args.ckpt_dir, 'population.lock')
    with util.FileLock(population_lock):
        population_path = os.path.join(args.ckpt_dir, 'population.pkl')
        if os.path.exists(population_path):
            with open(population_path, 'rb') as pkl_fh:
                population = pickle.load(pkl_fh)
        else:
            population = []
        # Find member in population
        member = None
        for m in population:
            if m.member_id == member_id:
                member = m
                break
        # If member not found, add to population
        if member is None:
            member = Member(member_id)
            population.append(member)
            with open(population_path, 'wb') as pkl_fh:
                pickle.dump(population, pkl_fh)

    # Truncation selection, exploit, and explore
    print('[{}] At epoch {} saw population size {}'.format(gpu_id, epoch, len(population)))

    # Step for one epoch
    time.sleep(0.1)

    # Write to population
    population_lock = os.path.join(args.ckpt_dir, 'population.lock')
    with util.FileLock(population_lock):
        population_path = os.path.join(args.ckpt_dir, 'population.pkl')
        assert os.path.exists(population_path), 'Population should exist by now.'
        with open(population_path, 'rb') as pkl_fh:
            population = pickle.load(pkl_fh)
        # Find member in population
        member = None
        for m in population:
            if m.member_id == member_id:
                member = m
                break
        assert member is not None, '{}: Failed to find member {} in population.'.format(gpu_id, member_id)
        # TODO: Actual member
        member.add_epoch('/random/path', {'lr': 0.0001}, 1.0)
        with open(population_path, 'wb') as pkl_fh:
            pickle.dump(population, pkl_fh)


if __name__ == '__main__':
    parser = ArgParser()
    main(parser.parse_args())
