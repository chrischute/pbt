import argparse
import random

from client import PBTClient


def main(args):
    client = PBTClient(args.server_url, args.server_port, args.auth_key, args.config_path)

    for _ in range(args.num_epochs):
        client.step()

        client.save('checkpoints/placeholder', metric_value=random.random())

        if client.should_exploit():
            # Exploit another member's parameters and hyperparameters
            client.exploit()
            load_checkpoint(client.checkpoint_path())

            # Explore hyperparameter space
            client.explore()
            update_hyperparameters(client.hyperparameters())


def load_checkpoint(checkpoint_path):
    pass


def update_hyperparameters(hyperparameters):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # PBT Connection Settings
    parser.add_argument('--client_id', type=int, required=True,
                        help='ID of PBT client. Must be unique for each member of the population.')
    parser.add_argument('--server_url', type=str, required=True,
                        help='URL or IP address of PBT server.')
    parser.add_argument('--server_port', type=int, default=7777,
                        help='Port on which the server listens for clients.')
    parser.add_argument('--auth_key', type=str, default='insecure',
                        help='Key for clients to authenticate with server.')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to configuration file defining hyperparameter search space (see templates).')

    # Training Settings
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train.')

    main(parser.parse_args())
