import itertools
import math
import os
import pickle
import time
from collections import defaultdict
from enum import Enum

num_buckets = 4
num_bits = int(math.log2(num_buckets))
max_depth = 3

trees = {}
indexed = []

fraction_to_index = 0.1


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
                        msb = "{:0>{}b}".format(row[1], num_bits * max_depth)[
                              (self.level * num_bits):((self.level + 1) * num_bits)]

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
        print("\t" * self.level + self.identifier + ": " + str(len(self.data)) + ", " + str(self.done))
        for child in self.children.values():
            if child is not None:
                child.print()

    def query(self, q):

        resultids = []

        if self.level != max_depth:
            msb = "{:0>{}b}".format(int(q.right), num_bits * max_depth)[
                  (self.level * num_bits):((self.level + 1) * num_bits)]

            keylist = list(self.children.keys())
            try:
                if (q.operator == "<=") | (q.operator == "<"):
                    keylist = keylist[:keylist.index(msb)]

                elif (q.operator == ">=") | (q.operator == ">"):
                    keylist = keylist[keylist.index(msb):]
            except:
                return resultids

            if msb in self.children.keys():
                if self.children[msb] is not None:
                    resultids += self.children[msb].query(q)

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
                # self.children[bucket].generate_children()


class Tree(Node):
    def __init__(self, data, identifier="", level=0):
        super().__init__(data, identifier, level)

        # Generate the WHOLE tree structure.
        self.generate_children()

        self.already_split = True

        # On creation, populate self with data and delete the data entry since it is now present in the children.
        # self.populate(self.data)
        self.data = []

    def populate(self, data, predicates=None):
        """
        This will sort data into the first set of buckets, based on MSB.
        :param data: The sample rows
        """

        buckets = {}
        resultset = []

        # Generate the buckets in the right order (ascending).
        for bucket in [''.join(x) for x in itertools.product("01", repeat=num_bits)]:
            buckets[bucket] = []

            # If the children have not been initialized yet, do that.
            # This is done now to keep the right order of children.
            if not self.already_split:
                self.children[bucket] = None

        # Sort the data into the buckets based on MSB.
        for row in data:

            if predicates is not None:
                if eval(construct_preds(predicates)):
                    resultset.append(row)

            stripped_row = (row['id'], row[attribute])

            msb = "{:0>{}b}".format(stripped_row[1], num_bits * max_depth)[
                  (self.level * num_bits):((self.level + 1) * num_bits)]
            buckets[msb].append(stripped_row)

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

        indexed.extend(data)

        return resultset

    def export_tree(self):
        pass


class AttrType(Enum):
    CAT = 1
    NUM = 2


class Predicate:

    def __init__(self, query, attr_type: AttrType, connector=None):
        query = query.split(" ")
        self.type = attr_type
        self.connector = connector
        self.left = query[0]
        self.operator = query[1]
        self.right = query[2]


def construct_preds(predicates):
    '''
    This simply turns a Predicate object into a evaluable String.
    :param predicates:
    :return:
    '''
    preds = ""

    if len(predicates) > 0:
        for pred in predicates:
            if pred.connector is not None:
                preds += pred.connector + " "

            preds += "(row['" + pred.left + "'] " + pred.operator + " " + pred.right + ") "

    else:
        preds += "True"

    return preds


def query(attributes, predicates):
    results = {}
    global data
    global trees

    pred = predicates[0]

    for attr in attributes:
        results[attr] = defaultdict(int)

    t0 = time.time()
    if len(data) > 0:
        indexed_amount = int(fraction_to_index * len(data))
        if indexed_amount < 5000:
            to_be_indexed = data
            data = []
        else:
            to_be_indexed = data[:indexed_amount]
            data = data[indexed_amount:]

        # Query the existing index for data.
        idset = set(trees[pred.left].query(pred))
        indexdata = [x for x in indexed if x['id'] in idset]

        # Index the data and simultaneously get the query result from the newly indexed data.
        indexed_result = trees[attribute].populate(to_be_indexed, predicates)

        # Merge those two datasets and count them.
        if len(indexdata) > 1:
            indexed_result.extend(indexdata)

        for row in indexed_result:
            # If so, add to the respective counters.
            for attr in attributes:
                results[attr][row[attr]] += 1

        # For each row in the data sample..
        for row in data:

            # Check if it satisfies the predicates.
            if eval(construct_preds(predicates)):

                # If so, add to the respective counters.
                for attr in attributes:
                    results[attr][row[attr]] += 1

    else:

        if not trees[attribute].done:
            trees[attribute].split()

        # Query the existing index for data.
        idset = set(trees[pred.left].query(pred))
        indexdata = [x for x in indexed if x['id'] in idset]

        for row in indexdata:
            # If so, add to the respective counters.
            for attr in attributes:
                results[attr][row[attr]] += 1

    querytime = time.time() - t0
    print("Query time: {}".format(str(querytime)))
    return results


def print_result(result):
    print("")
    for key in result.keys():

        # Sort the result by key
        result[key] = dict(sorted(result[key].items()))

        for subkey in result[key].keys():
            print("{},{}:{}".format(key, subkey, str(result[key][subkey])))


# IMPORTANT: Set this value to the column you want to index!
# Only supports one column indexing!
attribute = 'age'

trees[attribute] = Tree([], identifier=attribute)

t0 = time.time()
print("Loading..")
with open(os.environ['DATA_PATH'], 'rb') as f:
    data = pickle.load(f)
print("Done loading!")
print("Time for Pickle fetch: {}".format(str(time.time() - t0)))

# Predicates are defined here.
pred1 = Predicate("age >= 32", AttrType.NUM)

t0 = time.time()
querycounter = 1
while not trees[attribute].done:
    result = query(attributes=[attribute], predicates=[pred1])
    print("Result {}".format(str(querycounter)))
    querycounter += 1
    print_result(result)

fulltime = time.time() - t0

result = query(attributes=[attribute], predicates=[pred1])
print_result(result)
result = query(attributes=[attribute], predicates=[pred1])
print_result(result)
result = query(attributes=[attribute], predicates=[pred1])
print_result(result)
