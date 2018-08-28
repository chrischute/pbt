import argparse
import Pyro4 as pyro


def main(args):
    warehouse = Warehouse()
    janet = Person('Janet')
    henry = Person('Henry')

    janet.visit(warehouse)
    henry.visit(warehouse)


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
        print(f'The warehouse contains: {warehouse.contents}')
        item = input('Type something you want to take (or empty): ').strip()
        if item:
            warehouse.take(self.name, item)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    main(parser.parse_args())