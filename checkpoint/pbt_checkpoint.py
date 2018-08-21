class PBTCheckpoint(object):
    """Checkpoint for saving model performance during training."""

    def __init__(self, client_id, metric_value, hyperparameters, parameters_path):
        """
        Args:
            client_id (int): ID for population member.
            metric_value (float): Value of metric for determining which checkpoints are best.
            hyperparameters (dict): Dictionary of hyperparameters.
            parameters_path (str): Path to saved network parameters.
        """
        self._client_id = client_id
        self._metric_value = metric_value
        self._hyperparameters = hyperparameters.copy()
        self._parameters_path = parameters_path

    def client_id(self):
        return self._client_id

    def metric_value(self):
        return self._metric_value

    def hyperparameters(self):
        return self._hyperparameters

    def parameters_path(self):
        return self._parameters_path

    def __gt__(self, other):
        """Order checkpoints by their metric value."""
        return self._metric_value > other.metric_value()
