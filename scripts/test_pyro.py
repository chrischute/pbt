import argparse
import Pyro4
import Pyro4.naming
import Pyro4.util
import sys


def main(args):

    if args.name_server:
        if not args.host:
            raise ValueError('Must specify host when running name server.')
        Pyro4.naming.startNS(args.host)
    elif args.warehouse:
        # Start Pyro4 daemon
        if not args.host:
            raise ValueError('Must specify host when running warehouse.')
        Pyro4.Daemon.serveSimple({Warehouse: "example.warehouse"},
                                 host=args.host,
                                 ns=True)
    else:
        sys.excepthook = Pyro4.util.excepthook

        # Visit the warehouse
        warehouse = Pyro4.Proxy('PYRONAME:example.warehouse')

        janet = Person('Janet')
        henry = Person('Henry')

        janet.visit(warehouse)
        henry.visit(warehouse)


@Pyro4.expose
@Pyro4.behavior(instance_mode="single")
class Warehouse(object):
    """Warehouse class from official Pyro4 tutorial."""
    def __init__(self):
        self.contents = ['chair', 'bike', 'flashlight', 'laptop', 'couch']

    def list_contents(self):
        return self.contents

    def take(self, name, item):
        self.contents.remove(item)
        print(f'{name} took the {item}.')

    def store(self, name, item):
        self.contents.append(item)
        print(f'{name} stored the {item}.')


class Person(object):
    """Person class form official Pyro4 tutorial."""
    def __init__(self, name):
        self.name = name

    def visit(self, warehouse):
        print(f'This is {self.name}')
        self.deposit(warehouse)
        self.retrieve(warehouse)
        print('Thank you, come again!')

    def deposit(self, warehouse):
        item = input('Type a thing to store (or empty): ').strip()
        if item:
            warehouse.store(self.name, item)

    def retrieve(self, warehouse):
        print(f'The warehouse contains: {warehouse.list_contents()}')
        item = input('Type something you want to take (or empty): ').strip()
        if item:
            warehouse.take(self.name, item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name_server', action='store_true', default='Run the name server.')
    parser.add_argument('--warehouse', action='store_true', default='Run the warehouse rather than visiting.')
    parser.add_argument('--host', type=str, help='Host/IP of the warehouse daemon.')

    main(parser.parse_args())
