import random
import time

from server import PBTServer
from client import PBTClient

AUTH_KEY = 'secure'.encode('UTF-8')
IP = '127.0.0.1'
PORT = 5678

NUM_MEMBERS = 20
NUM_EPOCHS = 5


def main():
    # Create manager and members of the population
    server = PBTServer(PORT, auth_key=AUTH_KEY)
    clients = [PBTClient(member_id, IP, PORT, AUTH_KEY) for member_id in range(NUM_MEMBERS)]

    try:
        for pbt_client in clients:

            # TODO: TRAIN FOR AN EPOCH
            time.sleep(10)

            # TODO: EVALUATE MODEL AND SAVE PARAMETERS TO SHARED DATA STORE
            ckpt_path = 'checkpoint/placeholder'
            metric_val = random.random()

            pbt_client.save(ckpt_path, metric_val)
            if pbt_client.should_exploit():
                # Copy parameters from another network
                pbt_client.exploit()

                # TODO: LOAD MODEL PARAMETERS FROM SHARED DATA STORE

                # Explore and update hyperparameters
                pbt_client.explore()
                hyperparameters = pbt_client.hyperparameters()

                # TODO: UPDATE YOUR OPTIMIZER AND MODEL WITH HYPERPARAMETERS

    finally:
        server.shut_down()


if __name__ == '__main__':
    main()
