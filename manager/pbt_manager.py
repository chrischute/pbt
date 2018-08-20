import random

from checkpoint import PBTCheckpoint


class PBTManager(object):
    """Manager for a population based training session."""
    def __init__(self, port, auth_key='', maximize_metric=True):
        """
        Args:
            port: Port on which to run the manager server.
            auth_key: Authorization key for the manager server.
            maximize_metric: Whether the manager should maximize the metric values,
                as opposed to minimizing them.
        """
        self._port = port
        self._auth_key = auth_key
        self._maximize_metric = maximize_metric

        self._checkpoints = {}  # Maps member ID -> list of PBTCheckpoints
        self._truncation_ratio = 0.2   # Ratio of population for truncation selection

    def save(self, checkpoint):
        """Save a checkpoint with model performance.

        Args:
            checkpoint: PBTCheckpoint containing population member's performance values.
        """
        if checkpoint.member_id not in self._checkpoints:
            self._checkpoints[checkpoint.member_id] = []
        self._checkpoints[checkpoint.member_id].append(checkpoint)

    def should_exploit(self, member_id):
        """Check whether a member should exploit another member of the population

        Args:
            member_id: ID of member asking whether it should exploit another member.

        Returns:
            True if member is under-performing and should exploit another member.
        """
        first_surviving_idx = max(1, int(self._truncation_ratio * len(self._checkpoints)))
        checkpoints = self._sorted_best_checkpoints(best_first=False)
        for checkpoint in checkpoints[:first_surviving_idx]:
            if checkpoint.member_id == member_id:
                return True

        return False

    def exploit(self):
        """Get a checkpoint to exploit.

        Returns:
            A checkpoint randomly sampled from the top performers of the population.
        """
        first_ineligible_idx = max(1, int(self._truncation_ratio * len(self._checkpoints)))
        checkpoints = self._sorted_best_checkpoints(best_first=True)
        exploited_checkpoint = random.choice(checkpoints[:first_ineligible_idx])

        return exploited_checkpoint

    def _get_best_checkpoint(self, member_id):
        """Get the best checkpoint for a member of the population,
        as rated by checkpoint metric values.

        Args:
            member_id: ID of the member whose checkpoints will be considered.

        Returns:
            Best PBTCheckpoint for this
        """
        if member_id not in self._checkpoints[member_id] or len(self._checkpoints) == 0:
            raise ValueError('_get_best_checkpoint called on a member with no registered checkpoints.')

        if self._maximize_metric:
            best_checkpoint = max(self._checkpoints[member_id], lambda checkpoint: checkpoint.metric_value)
        else:
            best_checkpoint = min(self._checkpoints[member_id], lambda checkpoint: checkpoint.metric_value)

        return best_checkpoint

    def _sorted_best_checkpoints(self, best_first=True):
        """Get best checkpoints (i.e., best one per member) in sorted order.

        Args:
            best_first: Sort such that the best members of the population come first.

        Returns:
            List of best checkpoints for all members.
        """
        best_checkpoints = [self._get_best_checkpoint(member_id) for member_id in self._checkpoints]
        sorted_best_checkpoints = list(sorted(best_checkpoints, reverse=(best_first == self._maximize_metric)))

        return sorted_best_checkpoints
