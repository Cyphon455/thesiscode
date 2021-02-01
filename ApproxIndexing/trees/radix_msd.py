import itertools
import math
import traceback

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

    def split(self):

        # A child node will be done if it cannot split anymore and is sorted.
        # Obviously, do not attempt to split this node.
        if not self.done:

            # If the node was split previously and has children..
            if self.already_split:

                # Assume all children are done.
                all_done = True
                for child in self.children.values():
                    if child is not None:

                        # If a child is not done, try splitting it.
                        if not child.done:
                            child.split()

                            # A child might be done after it attempts splitting and
                            # realizes that it cannot split anymore.
                            # If it is STILL not done, the parent is not done either.
                            if not child.done:
                                all_done = False

                # If ALL children are done (read: unable to split and sorted),
                # retrieve all data in the right order and delete the children.
                if all_done:
                    # for child in self.children.values():
                    #     if child is not None:
                    #         self.data.extend(child.data)
                    #
                    # self.children = {}

                    # Set this node to done.
                    self.done = True

            # If the node is NOT split yet.
            else:

                # See if it is even worth splitting.
                # For very small datasets, just sort and set to done.
                if len(self.data) <= 3:
                    self.data.sort(key=lambda x: x[1])
                    self.done = True

                else:
                    buckets = {}

                    # Generate the buckets in the right order (ascending).
                    for bucket in [''.join(x) for x in itertools.product("01", repeat=num_bits)]:
                        buckets[bucket] = []
                        self.children[bucket] = None

                    # Populate the buckets based on MSB.
                    for row in self.data:

                        # Bits taken from the entry are dependant on the depth of the tree.
                        # UU msb = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', row[1]))[
                        # UU        (self.level * num_bits):((self.level + 1) * num_bits)]
                        msb = "{:0>{}b}".format(row[1], num_bits * max_depth)[(self.level * num_bits):((self.level + 1) * num_bits)]

                        # If there are NO bits left to sort with, stop trying.
                        if msb == '':
                            self.out_of_bits = True
                            break

                        buckets[msb].append(row)

                    # If we are out of bits to split with, just sort the data and set this node to done.
                    if self.out_of_bits:
                        self.data.sort(key=lambda x: x[1])
                        self.done = True

                    else:
                        # Generate a new child node for each bucket with data.
                        for child in buckets:
                            if len(buckets[child]) > 0:
                                self.children[child] = Node(buckets[child], child, self.level + 1)

                        # Delete the data in the current node as to not retain duplicate data.
                        self.data = []
                        self.already_split = True

    def print(self):
        print("\t"*self.level + self.identifier + ": " + str(len(self.data)) + ", " + str(self.done))
        for child in self.children.values():
            if child is not None:
                child.print()

    def query(self, q):

        resultids = []

        if self.level != max_depth:
            msb = "{:0>{}b}".format(int(q.right), num_bits * max_depth)[(self.level * num_bits):((self.level + 1) * num_bits)]

            keylist = list(self.children.keys())
            try:
                if (q.operator == "<=") | (q.operator == "<"):
                    keylist = keylist[:keylist.index(msb)]

                elif (q.operator == ">=") | (q.operator == ">"):
                    keylist = keylist[keylist.index(msb):]
            except:
                return resultids

            for key in keylist:
                if self.children[key] is not None:

                    if key == msb:
                        data = [x[0] for x in self.children[key].get_data() if eval(str(x[1]) + q.operator + q.right)]
                        resultids += data
                    else:
                        data = [x[0] for x in self.children[key].get_data()]
                        resultids += data

            return resultids

        else:

            querystring = "row[1] " + q.operator + " " + q.right

            result = [row[0] for row in self.data if eval(querystring)]

            return result

    def get_data(self):
        # if len(self.children) == 0:
        #     return self.data
        all_none = True
        for child in self.children.values():
            if child is not None:
                all_none = False
                break

        if all_none:
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
        # self.generate_children()

        # On creation, populate self with data and delete the data entry since it is now present in the children.
        self.populate(self.data)
        self.data = []

    def populate(self, data):
        """
        This will sort data into the first set of buckets, based on MSB.
        :param data: The sample rows
        """

        buckets = {}

        # Generate the buckets in the right order (ascending).
        for bucket in [''.join(x) for x in itertools.product("01", repeat=num_bits)]:
            buckets[bucket] = []

            # If the children have not been initialized yet, do that.
            # This is done now to keep the right order of children.
            if not self.already_split:
                self.children[bucket] = None

        # Sort the data into the buckets based on MSB.
        for row in data:
            msb = "{:0>{}b}".format(row[1], num_bits * max_depth)[(self.level * num_bits):((self.level + 1) * num_bits)]
            buckets[msb].append(row)

        for child in buckets:

            # If there is actual data in the bucket (some might remain empty)..
            if len(buckets[child]) > 0:

                # Check if a child node for this bucket already exists.
                # If so, just append the data to that child.
                if self.children[child] is not None:
                    self.children[child].data.extend(buckets[child])

                # If it does not, create it.
                else:
                    self.children[child] = Node(data=buckets[child], identifier=child, level=self.level + 1)

        # Since we are splitting here, set "already_split" to True.
        self.already_split = True

    def export_tree(self):
        pass