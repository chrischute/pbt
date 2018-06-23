import models
import multiprocessing as mp
import os
import pickle
import random
import re
import torch
import torch.nn as nn
import util

from args import ArgParser
from data_loader import PBTDataLoader
from evaluator import ModelEvaluator
from logger import TrainLogger
from member import Member
from saver import ModelSaver


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
    epoch_info, do_explore = None, False
    prev_epochs = [m.get_epoch(epoch - 1) for m in population if epoch > 1 and m.num_epochs() == epoch - 1]
    if len(prev_epochs) > 0:
        prev_epochs.sort(key=lambda x: x['metric_val'], reverse=args.maximize_metric)
        member_idx = population.index(member)
        if member_idx <= int(0.2 * len(population)):
            # Exploit from top 20%
            exploit_idx = random.randint(int(0.8 * len(population)), len(population) - 1)
            member = population[exploit_idx]

            # Explore
            do_explore = True

        epoch_info = member.get_epoch(epoch - 1)
        if do_explore:
            for h in epoch_info['hyperparameters'].values():
                # Multiply by random factor between 0.8 and 1.2
                h *= random.uniform(0.8, 1.2)

    if epoch_info is not None:
        model, ckpt_info = ModelSaver.load_model(epoch_info['ckpt_path'], gpu_ids=[gpu_id])
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Get optimizer
    optimizer = util.get_optimizer(model.parameters(), args)
    if epoch_info is not None:
        ModelSaver.load_optimizer(epoch_info['ckpt_path'], optimizer)
    if do_explore:
        # Update optimizer hyperparameters to explored values
        hyperparameters = epoch_info['hyperparameters']
        for h in hyperparameters:
            for param_group in optimizer.param_groups:
                param_group[h] *= hyperparameters[h]

    # Get logger, evaluator, saver
    train_loader = PBTDataLoader(args.dataset, phase='train', is_training=False,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    loss_fn = nn.BCEWithLogitsLoss()
    logger = TrainLogger(args, epoch, len(train_loader.dataset))
    eval_loaders = [PBTDataLoader(args.dataset, phase='train', is_training=False,
                                  batch_size=args.batch_size, num_workers=args.num_workers),
                    PBTDataLoader(args.dataset, phase='val', is_training=False,
                                  batch_size=args.batch_size, num_workers=args.num_workers)]
    evaluator = ModelEvaluator(eval_loaders, logger, args.num_visuals, args.max_eval)
    saver = ModelSaver(args.save_dir, args.max_ckpts, args.metric_name, args.maximize_metric)

    # Train for one epoch
    logger.start_epoch()

    for inputs, targets in train_loader:
        logger.start_iter()

        with torch.set_grad_enabled(True):
            logits = model.forward(inputs.to(args.device))
            loss = loss_fn(logits, targets.to(args.device))

            logger.log_iter(inputs, logits, targets, loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.end_iter()

    # Evaluate and save parameters
    metrics, curves = evaluator.evaluate(model, args.device, logger.epoch)
    metric_val = metrics.get(args.metric_name, None)
    ckpt_path = saver.save(epoch, model, optimizer, args.device, metric_val=metric_val)
    logger.end_epoch(metrics, curves)

    # Update population
    population_lock = os.path.join(args.ckpt_dir, 'population.lock')
    with util.FileLock(population_lock):
        population_path = os.path.join(args.ckpt_dir, 'population.pkl')
        assert os.path.exists(population_path), '{}: Failed to find popluation at {}.'.format(gpu_id, population_path)
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
        member.add_epoch(ckpt_path, epoch_info['hyperparameters'], metric_val)
        with open(population_path, 'wb') as pkl_fh:
            pickle.dump(population, pkl_fh)


if __name__ == '__main__':
    parser = ArgParser()
    main(parser.parse_args())
