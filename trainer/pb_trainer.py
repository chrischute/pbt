import torch


class PBTrainer(object):
    """Class for training a model using population-based training."""
    def __init__(self, loss_fn, gpu_id):
        """
        Args:
            loss_fn: Loss function for model.
        """
        self.loss_fn = loss_fn
        self.device = 'cuda:{}'.format(gpu_id)

    def step(self, model, data_loader, optimizer):
        """Train for one epoch.

        Args:
            model: Model to train.
            data_loader: Data loader to sample from.
            optimizer: Optimizer to update model parameters.
        """
        for inputs, targets in data_loader:
            with torch.set_grad_enabled(True):
                logits = model.forward(inputs.to(self.device))
                loss = self.loss_fn(logits, targets.to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
