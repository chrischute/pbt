import random
import time

from ast import literal_eval
from checkpoint import PBTCheckpoint
from multiprocessing.managers import SyncManager


class PBTClient(object):
    """Client module for a member of the population to communicate with the server."""
    def __init__(self, member_id, manager_ip, port, auth_key):
        # Create a manager to communicate with the PBTServer
        class PBTClientManager(SyncManager):
            pass
        PBTClientManager.register('save')
        PBTClientManager.register('should_exploit')
        PBTClientManager.register('exploit')
        self._client = PBTClientManager(address=(manager_ip, port), authkey=auth_key)
        self._client.connect()

        self.member_id = member_id

    def train_epoch(self):
        """Train for an epoch (Randomly generate a checkpoint)."""
        checkpoint = PBTCheckpoint(self.member_id, random.random(), {'lr': 0.01}, 'ckpts/best.pth.tar')
        self._client.save(checkpoint)

        time.sleep(0.1)

        return checkpoint

    def exploit(self):
        """Possibly exploit another member of the population."""
        should_exploit = literal_eval(str(self._client.should_exploit(self.member_id)))
        if should_exploit:
            checkpoint = self._client.exploit()
            print('{}: EXPLOIT({})'.format(self.member_id, checkpoint.member_id()))
            return True

        return False

    def explore(self):
        print('{}: EXPLORE'.format(self.member_id))

    def shut_down(self):
        """Shut down the connection to the server."""
        self._client.shutdown()
