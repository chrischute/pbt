import random
import time

from ast import literal_eval
from checkpoint import PBTCheckpoint
from multiprocessing.managers import SyncManager


class PBTMember(object):
    """Member of the population in a population-based training session."""
    def __init__(self, member_id, manager_ip, port, auth_key):

        self.member_id = member_id

        # Create a surrogate to communicate with the PBTManager
        class PBTClientManager(SyncManager):
            pass
        PBTClientManager.register('save')
        PBTClientManager.register('should_exploit')
        PBTClientManager.register('exploit')
        self.manager = PBTClientManager(address=(manager_ip, port), authkey=auth_key)
        self.manager.connect()

    def train_epoch(self):
        """Train for an epoch (Randomly generate a checkpoint)."""
        checkpoint = PBTCheckpoint(self.member_id, random.random(), {'lr': 0.01}, 'ckpts/best.pth.tar')
        self.manager.save(checkpoint)

        time.sleep(0.1)

        return checkpoint

    def exploit(self):
        """Possibly exploit another member of the population."""
        should_exploit = literal_eval(str(self.manager.should_exploit(self.member_id)))
        if should_exploit:
            checkpoint = self.manager.exploit()
            print('{}: EXPLOIT({})'.format(self.member_id, checkpoint.member_id()))
            return True

        return False

    def explore(self):
        print('{}: EXPLORE'.format(self.member_id))
