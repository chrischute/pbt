import json
import math
import pandas as pd
import random
import time

from ast import literal_eval
from checkpoint import PBTCheckpoint
from multiprocessing.managers import SyncManager


class PBTClient(object):
    """Client module for a member of the population to communicate with the server."""
    def __init__(self, client_id, server_ip, server_port, auth_key, config_path):
        auth_key = auth_key.encode('UTF-8')

        # Create a manager to communicate with the PBTServer
        class PBTClientManager(SyncManager):
            pass
        PBTClientManager.register('save')
        PBTClientManager.register('should_exploit')
        PBTClientManager.register('exploit')
        self._client = PBTClientManager(address=(server_ip, server_port), authkey=auth_key)
        self._client.connect()

        self.client_id = client_id
        self.hyperparameters = self._read_config(config_path)
        print(json.dumps(self.hyperparameters, indent=2))

    def train_epoch(self):
        """Train for an epoch (Randomly generate a checkpoint)."""
        checkpoint = PBTCheckpoint(self.client_id, random.random(), {'lr': 0.01}, 'ckpts/best.pth.tar')
        self._client.save(checkpoint)

        time.sleep(10.)

        return checkpoint

    def exploit(self):
        """Possibly exploit another member of the population.

        Returns:
            True if the client exploited another member, otherwise False.
        """
        should_exploit = literal_eval(str(self._client.should_exploit(self.client_id)))
        if should_exploit:
            checkpoint = self._client.exploit()
            if checkpoint.member_id() != self.client_id:
                print('{}: EXPLOIT({})'.format(self.client_id, checkpoint.member_id()))
                self.hyperparameters = {k: v for k, v in checkpoint.hyperparameters()}
                print(json.dumps(self.hyperparameters, indent=2))
                return True

        return False

    def explore(self):
        print('{}: EXPLORE'.format(self.client_id))
        for k, v in self.hyperparameters.items():
            mutation = random.choice([0.8, 1.2])
            print('Mutating {} from {} by {}'.format(k, v, mutation))
            self.hyperparameters[k] = mutation * v

        print(json.dumps(self.hyperparameters, indent=2))

    @staticmethod
    def _read_config(config_path):
        """Read a configuration file of hyperparameters.

        Args:
            config_path: Path to CSV configuration file.

        Returns:
            Dictionary of hyperparameters, randomly initialized in the search space.
        """
        hyperparameters = {}

        config_df = pd.read_csv(config_path)
        for _, row in config_df.iterrows():
            # Randomly initialize a hyperparameter using the search space from the config file
            hyperparameter_name = str(row['hyperparameter'])
            min_value = float(row['min_value'])
            max_value = float(row['max_value'])
            search_scale = str(row['search_scale'])

            if search_scale == 'log':
                # Sample randomly along a logarithm search scale
                min_exp = math.log(min_value, 10)
                max_exp = math.log(max_value, 10)
                random_exp = min_exp + random.random() * (max_exp - min_exp)
                hyperparameter_value = 10 ** random_exp
            elif search_scale == 'linear':
                # Sample randomly along a linear search scale
                hyperparameter_value = min_value + random.random() * (max_value - min_value)
            else:
                raise ValueError('Expected "log" or "linear" search scale, got "{}"'.format(search_scale))

            hyperparameters[hyperparameter_name] = hyperparameter_value

        return hyperparameters
