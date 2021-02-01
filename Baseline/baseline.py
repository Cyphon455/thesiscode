import os
import pickle
import time
from collections import defaultdict
from enum import Enum


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
    preds = ""

    if len(predicates) > 0:
        for pred in predicates:
            if pred.connector is not None:
                preds += pred.connector + " "

            preds += "(row['" + pred.left + "'] " + pred.operator + " " + pred.right + ") "

    else:
        preds += "True"

    return preds


def query(mode, attributes, predicates):
    global data

    results = {}

    for attr in attributes:
        results[attr] = defaultdict(int)

    if mode == 'count':

        for row in data:
            if eval(construct_preds(predicates)):
                for attr in attributes:
                    results[attr][row[attr]] += 1

    return results


def print_results(results):
    results = dict(sorted(results.items()))

    for key in results.keys():

        results[key] = dict(sorted(results[key].items()))

        print("Results for: {}".format(str(key)))

        for subkey in results[key].keys():
            print("\t {}: {}".format(str(subkey), str(results[key][subkey])))

        print("")


t0 = time.time()
with open(os.environ['DATA_PATH'], 'rb') as f:
    data = pickle.load(f)
print("Time for Pickle fetch: {}".format(str(time.time() - t0)))

# Predicates are defined here.
pred1 = Predicate("sex == 'Male'", AttrType.CAT)
pred2 = Predicate("age == 27", AttrType.NUM, '&')
pred3 = Predicate("tip < 16", AttrType.NUM, '&')
pred4 = Predicate("age >= 29", AttrType.NUM)

# t0 = time.time()
# results = query(mode='count', attributes=['tip'], predicates=[pred1, pred2, pred3])
# print("Time for query, three predicates: {}".format(str(time.time()-t0)))
#
# t0 = time.time()
# results = query(mode='count', attributes=['tip'], predicates=[pred1, pred2])
# print("Time for query, two predicates: {}".format(str(time.time()-t0)))
#
# t0 = time.time()
# results = query(mode='count', attributes=['tip'], predicates=[pred1])
# print("Time for query, one predicate: {}".format(str(time.time()-t0)))
#
# t0 = time.time()
# results = query(mode='count', attributes=['tip'], predicates=[])
# print("Time for simple query: {}".format(str(time.time()-t0)))

for i in range(5):
    t0 = time.time()
    results = query(mode='count', attributes=['age'], predicates=[pred4])
    print_results(results)
    query_time = time.time() - t0
    print("Time for query, one predicate: {}".format(str(query_time)))

print_results(results)
