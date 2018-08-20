import random
import time

from ast import literal_eval
from checkpoint import PBTCheckpoint
from multiprocessing.managers import SyncManager


class PBTClient(object):
    """Client module for a member of the population to communicate with the server."""
    def __init__(self, client_id, server_ip, server_port, auth_key):
        # Create a manager to communicate with the PBTServer
        class PBTClientManager(SyncManager):
            pass
        PBTClientManager.register('save')
        PBTClientManager.register('should_exploit')
        PBTClientManager.register('exploit')
        self._client = PBTClientManager(address=(server_ip, server_port), authkey=auth_key)
        self._client.connect()

        self.client_id = client_id

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
            print('{}: EXPLOIT({})'.format(self.client_id, checkpoint.member_id()))
            return True

        return False

    def explore(self):
        print('{}: EXPLORE'.format(self.client_id))
