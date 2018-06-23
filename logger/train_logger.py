import util

from time import time
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""
    def __init__(self, args, dataset_len, pixel_dict):
        super(TrainLogger, self).__init__(args, dataset_len, pixel_dict)

        assert args.is_training
        assert args.iters_per_print % args.batch_size == 0, "iters_per_print must be divisible by batch_size"
        assert args.iters_per_visual % args.batch_size == 0, "iters_per_visual must be divisible by batch_size"

        self.iters_per_print = args.iters_per_print
        self.iters_per_visual = args.iters_per_visual
        self.experiment_name = args.name
        self.max_eval = args.max_eval
        self.num_epochs = args.num_epochs
        self.loss_meter = util.AverageMeter()

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def log_iter(self, inputs, logits, targets, loss):
        """Log results from a training iteration."""
        loss = loss.item()
        self.loss_meter.update(loss, inputs.size(0))

        # Periodically write to the log and TensorBoard
        if self.iter % self.iters_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}, loss: {:.3g}]' \
                .format(self.epoch, self.iter, self.dataset_len, avg_time, self.loss_meter.avg)

            # Write all errors as scalars to the graph
            self._log_scalars({'batch_loss': self.loss_meter.avg}, print_to_stdout=False)
            self.loss_meter.reset()

            self.write(message)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.iter % self.iters_per_visual == 0:
            self.visualize(inputs, logits, targets, phase='train')

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch.

        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            curves: Dictionary of curves. Items have format '{phase}_{curve}: value.
        """
        self.write('[end of epoch {}, epoch time: {:.2g}]'.format(self.epoch, time() - self.epoch_start_time))
        self._log_scalars(metrics)

        self._plot_curves(curves)

        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
