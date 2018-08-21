import json
import random

from multiprocessing.managers import SyncManager


class PBTServer(object):
    """Manager for a population based training session."""
    def __init__(self, port, auth_key='', maximize_metric=True, verbose=True):
        """
        Args:
            port: Port on which to run the manager server.
            auth_key: Authorization key for the manager server.
            maximize_metric: Whether the manager should maximize the metric values,
                as opposed to minimizing them.
            verbose: Log to console if verbose.
        """
        auth_key = auth_key.encode('UTF-8')

        # Define a manager server to communicate with worker nodes
        class PBTServerManager(SyncManager):
            pass
        PBTServerManager.register('save', callable=lambda c: self.save(c))
        PBTServerManager.register('should_exploit', callable=lambda m: self.should_exploit(m))
        PBTServerManager.register('exploit', callable=lambda: self.exploit())
        PBTServerManager.register('get_id', callable=lambda: self.get_id())
        self._server = PBTServerManager(address=('', port), authkey=auth_key)

        self._port = port
        self._auth_key = auth_key
        self._maximize_metric = maximize_metric
        self._verbose = verbose

        self._checkpoints = {}         # Maps member ID -> list of PBTCheckpoints
        self._truncation_ratio = 0.2   # Ratio of population for truncation selection
        self._min_population_size = 5  # Minimum population size before members can exploit
        self._num_clients = 0

        self._server.start()

    def get_id(self):
        """Get the next available ID for a client."""
        client_id = self._num_clients
        self._num_clients += 1

        self._write('New client: {}'.format(client_id))

        return client_id

    def save(self, checkpoint):
        """Save a checkpoint with model performance.

        Args:
            checkpoint: PBCheckpoint containing population member's performance values.
        """
        if checkpoint.client_id() not in self._checkpoints:
            self._checkpoints[checkpoint.client_id()] = []

        checkpoint = checkpoint.copy()
        self._checkpoints[checkpoint.client_id()].append(checkpoint)

        self._write('{}: Saved checkpoint (performance: {})'.format(checkpoint.client_id(), checkpoint.metric_value()))
        self._write(json.dumps(checkpoint.hyperparameters(), indent=2))

    def should_exploit(self, client_id):
        """Check whether a client should exploit another member of the population.

        A member should exploit another when the member's most recent performance is
        in the bottom 20% of most recent performance of the whole population.

        Args:
            client_id: ID of client asking whether it should exploit another member.

        Returns:
            True if client is under-performing and should exploit another member.
        """
        if len(self._checkpoints) < self._min_population_size:
            return False

        first_surviving_idx = max(1, int(self._truncation_ratio * len(self._checkpoints)))
        checkpoints = self._sorted_checkpoints(best_first=False)
        for checkpoint in checkpoints[:first_surviving_idx]:
            if checkpoint.client_id() == client_id:
                self._write('{}: Should exploit'.format(client_id))
                return True

        return False

    def exploit(self):
        """Get a checkpoint to exploit.

        Returns:
            Dict of checkpoint randomly sampled from the top performers of the population.
        """
        first_ineligible_idx = max(1, int(self._truncation_ratio * len(self._checkpoints)))
        checkpoints = self._sorted_checkpoints(best_first=True)
        exploited_checkpoint = random.choice(checkpoints[:first_ineligible_idx])

        self._write('{}: Exploited'.format(exploited_checkpoint.client_id()))

        return exploited_checkpoint

    def port(self):
        """Get the server's listening port."""
        return self._port

    def shut_down(self):
        """Shut down the server."""
        self._server.shutdown()

    def _get_most_recent_checkpoint(self, client_id):
        """Get the most recent checkpoint for a member of the population.

        Args:
            client_id: ID of the client whose checkpoints will be considered.

        Returns:
            Best PBTCheckpoint for the specified client.
        """
        if client_id not in self._checkpoints or len(self._checkpoints[client_id]) == 0:
            raise ValueError('_get_best_checkpoint called on a member with no registered checkpoints.')

        checkpoint = self._checkpoints[client_id][-1]

        return checkpoint

    def _sorted_checkpoints(self, best_first=True):
        """Get a sorted list of the most recent checkpoint per member. List sorted in order of performance.

        Args:
            best_first: Sort such that the best come first.

        Returns:
            List of best checkpoints for all members.
        """
        most_recent_checkpoints = [self._get_most_recent_checkpoint(client_id) for client_id in self._checkpoints]
        sorted_checkpoints = list(sorted(most_recent_checkpoints, reverse=(best_first == self._maximize_metric)))

        return sorted_checkpoints

    def _write(self, s):
        """Handle a write to the console. No write if not in verbose mode."""
        if self._verbose:
            print(s)
