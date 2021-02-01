import itertools
import math

num_buckets = 4
num_bits = int(math.log2(num_buckets))
max_depth = 3


class Node:
    def __init__(self, data, identifier="", level=0):
        self.level = level
        self.data = data
        self.children = {}
        self.identifier = identifier

        self.done = False
        self.out_of_bits = False
        self.already_split = False

    def print(self):
        print("\t"*self.level + self.identifier + ": " + str(len(self.data)) + ", " + str(self.done))
        for child in self.children.values():
            if child is not None:
                child.print()

    def populate(self, data):

        if self.level == max_depth:
            self.data += data

        else:
            buckets = {}

            # Generate the buckets in the right order (ascending).
            # for bucket in [''.join(x) for x in itertools.product("01", repeat=num_bits)]:
            #     buckets[bucket] = []

            # Sort the data into the buckets based on MSB.
            for row in data:
                msb = "{:0>{}b}".format(row[1], num_bits * max_depth)[(self.level * num_bits):((self.level + 1) * num_bits)]

                if msb not in buckets.keys():
                    buckets[msb] = []

                buckets[msb].append(row)

            self.data = []

            for child in buckets:
                self.children[child].populate(buckets[child])

    def query(self, q=None):

        resultids = []

        if self.level != max_depth:
            msb = "{:0>{}b}".format(int(q.right), num_bits * 3)[(self.level * num_bits):((self.level + 1) * num_bits)]

            keylist = list(self.children.keys())

            if (q.operator == "<=") | (q.operator == "<"):
                keylist = keylist[:keylist.index(msb)]

            elif (q.operator == ">=") | (q.operator == ">"):
                keylist = keylist[keylist.index(msb)+1:]

            if self.children[msb] is not None:
                resultids += self.children[msb].query(q)

            for key in keylist:
                if self.children[key] is not None:
                    data = [x[0] for x in self.children[key].get_data()]
                    resultids += data

            return resultids

        else:

            querystring = "row[1] " + q.operator + " " + q.right

            result = [row[0] for row in self.data if eval(querystring)]

            return result

    def get_data(self):
        if self.level == max_depth:
            return self.data
        else:
            data = []
            for child in self.children.values():
                if child is not None:
                    data += child.get_data()
            return data

    def generate_children(self):

        buckets = {}

        # Generate the buckets in the right order (ascending).
        for bucket in [''.join(x) for x in itertools.product("01", repeat=num_bits)]:
            buckets[bucket] = []

            # Initialize the children.
            if self.level != max_depth:
                self.children[bucket] = Node(data=[], identifier=bucket, level=self.level + 1)
                self.children[bucket].generate_children()

class Tree(Node):
    def __init__(self, data, identifier="", level=0):
        super().__init__(data, identifier, level)

        # Generate the WHOLE tree structure.
        self.generate_children()

        # On creation, populate self with the given data.
        self.populate(self.data)