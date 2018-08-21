import math
import pandas as pd
import random
import time

from ast import literal_eval
from pbt.checkpoint import PBTCheckpoint
from multiprocessing.managers import SyncManager


class PBTClient(object):
    """Client module for a member of the population to communicate with the server."""
    def __init__(self, server_ip, server_port, auth_key, config_path):
        auth_key = auth_key.encode('UTF-8')

        # Create a manager to communicate with the PBTServer
        class PBTClientManager(SyncManager):
            pass
        PBTClientManager.register('save')
        PBTClientManager.register('should_exploit')
        PBTClientManager.register('exploit')
        PBTClientManager.register('get_id')

        self._client = PBTClientManager(address=(server_ip, server_port), authkey=auth_key)
        self._client.connect()

        self._client_id = int(str(self._client.get_id()))
        self._hyperparameters = self._read_config(config_path)
        self._parameters_path = None

    def exploit(self):
        """Exploit another member of the population, i.e. copy their parameters and hyperparameters."""
        checkpoint = self._client.exploit()
        self._hyperparameters = checkpoint.hyperparameters().copy()
        self._parameters_path = checkpoint.parameters_path()

    def explore(self):
        """Explore the hyperparameter space, i.e. randomly mutate each hyperparameter."""
        for k, v in self._hyperparameters.items():
            mutation = random.choice([0.8, 1.2])
            self._hyperparameters[k] = mutation * v

    def save(self, parameters_path, metric_value):
        """Save a checkpoint by sending information to the server.

        Note that weights must be saved to disk outside of the client.

        Args:
            parameters_path: Path to parameters that have already been saved on disk.
            metric_value: Value of performance metric for sending to the server.
        """
        self._parameters_path = parameters_path
        checkpoint = PBTCheckpoint(self._client_id, metric_value, self._hyperparameters, self._parameters_path)

        self._client.save(checkpoint)

        return checkpoint

    def should_exploit(self):
        """Check whether this client is under-performing and should exploit another member."""
        should_exploit = literal_eval(str(self._client.should_exploit(self._client_id)))

        return should_exploit

    def parameters_path(self):
        """Get the client's current checkpoint path."""
        return self._parameters_path

    def hyperparameters(self):
        """Get the client's current hyperparameters."""
        return self._hyperparameters

    def set_hyperparameter(self, hyperparameter_name, hyperparameter_value):
        """Set a hyperparameter."""
        self._hyperparameters[hyperparameter_name] = hyperparameter_value

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
            hyperparameter_name = str(row['name'])
            min_value = float(row['min'])
            max_value = float(row['max'])
            search_scale = str(row['scale'])

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
