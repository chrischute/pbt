import argparse
import util

from server import PBTServer


def main(args):
    server = PBTServer(args.port, args.auth_key, args.maximize_metric)
    try:
        print('Server running on port {}...'.format(server.port()))
        _ = input('Press any key to shut down...')
    finally:
        server.shut_down()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--port', type=int, default=7777,
                        help='Port on which to listen for clients.')
    parser.add_argument('--auth_key', type=str, default='insecure',
                        help='Key for clients to authenticate with server.')
    parser.add_argument('--maximize_metric', type=util.str_to_bool, default=True,
                        help='If true, higher metric value means better performing member.')

    main(parser.parse_args())
