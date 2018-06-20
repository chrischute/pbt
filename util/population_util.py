import os
import pickle

from .file_lock import FileLock


def read_population(ckpt_dir):
    """Read population from checkpoint dir with locking."""
    population_lock = os.path.join(ckpt_dir, 'population.lock')
    with FileLock(population_lock):
        population_path = os.path.join(ckpt_dir, 'population.pkl')
        if os.path.exists(population_path):
            with open(population_path, 'rb') as pkl_fh:
                population = pickle.load(pkl_fh)
        else:
            population = []

    return population


def update_population(population, ckpt_dir):
    """Write population to checkpoint dir with locking."""
    population_lock = os.path.join(ckpt_dir, 'population.lock')
    with FileLock(population_lock):
        population_path = os.path.join(ckpt_dir, 'population.pkl')
        with open(population_path, 'wb') as pkl_fh:
            pickle.dump(population, pkl_fh)
