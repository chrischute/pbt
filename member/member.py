from copy import deepcopy


class Member(object):
    """Member of the population. Used for saving info during training."""
    def __init__(self, member_id):
        self.member_id = member_id
        self._history = []

    def add_epoch(self, ckpt_path, hyperparameters, metric_val):
        """Add an epoch to the member's history.

        Args:
            ckpt_path: Path to checkpoint containing parameters after this epoch.
            hyperparameters: Dictionary of hyperparameters for the model during this epoch.
            metric_val: Metric as evaluated after this epoch.
        """
        epoch = {
            'ckpt_path': ckpt_path,
            'hyperparameters': hyperparameters,
            'metric_val': metric_val
        }
        self._history.append(epoch)

    def num_epochs(self):
        """Get the number of epochs completed by this member."""
        return len(self._history)

    def get_epoch(self, epoch_num):
        """Get an info dict for population member after the given `epoch_num`.
        Returned dict has keys 'ckpt_path', 'hyperparameters', and 'metric_val'.

        Args:
            epoch_num: Epoch (indexed from 1) to get info for.
        """
        assert 0 < epoch_num <= self.num_epochs(), 'epoch_num is out of bounds.'

        # Deep copy to avoid changing the population during explore
        epoch_info = deepcopy(self._history[epoch_num - 1])

        return epoch_info
