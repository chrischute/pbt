
class PBTCheckpoint(object):
    """Checkpoint for saving model performance during training."""

    def __init__(self, member_id, metric_value, hyperparameters, parameters_path):
        """
        Args:
            member_id (int): ID for population member.
            metric_value (float): Value of metric for determining which checkpoints are best.
            hyperparameters (dict): Dictionary of hyperparameters.
            parameters_path (str): Path to saved network parameters.
        """
        self.member_id = member_id
        self.metric_value = metric_value
        self.hyperparameters = {k: v for k, v in hyperparameters.items()}
        self.parameters_path = parameters_path
