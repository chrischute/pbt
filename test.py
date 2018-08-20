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

    # Simulate some epochs
    for _ in range(NUM_EPOCHS):
        # Train for an epoch
        for member in clients:
            member.train_epoch()

        # Exploit and explore
        for member in clients:
            if member.exploit():
                member.explore()


if __name__ == '__main__':
    main()
