"""Example train loop for Population-Based Training."""

# These imports are not expected to work as-is.
# This is just meant to show how pbt might look in a train loop.
import models
import optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import TrainArgParser
from data_loader import DataLoader
from evaluator import ModelEvaluator
from pbt.client import PBTClient
from saver import ModelSaver


def train(args):

    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()

    # Set up population-based training client
    pbt_client = PBTClient(args.pbt_server_url, args.pbt_server_port, args.pbt_server_key, args.pbt_config_path)

    # Get optimizer and scheduler
    parameters = model.module.parameters()
    optimizer = optim.get_optimizer(parameters, args, pbt_client)
    ModelSaver.load_optimizer(args.ckpt_path, args.gpu_ids, optimizer)

    # Get logger, evaluator, saver
    train_loader = DataLoader(args, 'train', is_training_set=True)
    eval_loaders = [DataLoader(args, 'valid', is_training_set=False)]
    evaluator = ModelEvaluator(eval_loaders, args.epochs_per_eval,
                               args.max_eval, args.num_visuals, use_ten_crop=args.use_ten_crop)
    saver = ModelSaver(**vars(args))

    for _ in range(args.num_epochs):
        optim.update_hyperparameters(model.module, optimizer, pbt_client.hyperparameters())

        for inputs, targets in train_loader:
            with torch.set_grad_enabled(True):
                logits = model.forward(inputs.to(args.device))
                loss = F.binary_cross_entropy_with_logits(logits, targets.to(args.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = evaluator.evaluate(model, args.device)
        metric_val = metrics.get(args.metric_name, None)
        ckpt_path = saver.save(model, args.model, optimizer, args.device, metric_val)

        pbt_client.save(ckpt_path, metric_val)
        if pbt_client.should_exploit():
            # Exploit
            pbt_client.exploit()

            # Load model and optimizer parameters from exploited network
            model, ckpt_info = ModelSaver.load_model(pbt_client.parameters_path(), args.gpu_ids)
            model = model.to(args.device)
            model.train()
            ModelSaver.load_optimizer(pbt_client.parameters_path(), args.gpu_ids, optimizer)

            # Explore
            pbt_client.explore()


if __name__ == '__main__':
    parser = TrainArgParser()
    train(parser.parse_args())
