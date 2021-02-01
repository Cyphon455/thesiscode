import math
import os
import pickle
import queue
import time
import traceback
from collections import defaultdict
from enum import Enum

import numpy as np

from ApproxIndexing.trees import radix_msd, radix_msd_splitearly, bplus

reservoir = {}
rare_pops = {}
indexed = []
data = []
groups = {}
counts = {}

stop_sampler = False
sampler_stopped = False
sampler_done = False

trees = None
q = queue.Queue()


class QuerySimulator:

    # Fire ONE query.
    def query(self, mode, attributes, predicates):
        '''
        Sends a query to the sample or trees, depending on if the trees exist.
        :param mode: What type of query is run?
        :param attributes: The groupby attributes.
        :param predicates: A list of predicates the query needs to satisfy.
        :return: None
        '''

        global stop_sampler
        global sampler_stopped
        global reservoir
        global sampler_done

        # Tell the sampler to stop since we want to query.
        stop_sampler = True

        # Only wait for the sampler to stop if the sampler is not completely done already!
        if not sampler_done:
            while not sampler_stopped:
                time.sleep(0.01)

        results = {}

        # Initialize the result dictionary.
        # This is going to have the format of 'attribute value: count'.
        for attr in attributes:
            results[attr] = defaultdict(int)

        # If NO INDEX exists yet, that means we are still in the initial streaming sample run.
        if trees is None:

            # Generate the sample to query on.
            generated_sample = self.gen_reservoir_sample(self.attr)

            # Currently, only 'count' is supported.
            if mode == 'count':

                # For each row in the data sample..
                for row in generated_sample:

                    # Check if it satisfies the predicates.
                    if eval(self.construct_preds(predicates)):

                        # If so, add to the respective counters.
                        for attr in attributes:
                            results[attr][row[attr]] += 1

                # Signal the sampler that it can keep going.
                stop_sampler = False

                return results

        else:

            # If the sampler is fully done, exit the program.
            if (sampler_done) & (not self.sequential):
                return None

            # In this case, the trees exist! Exclusively query those then!
            else:
                id_sets = []

                # Find the IDs for all attributes that satisfy the predicates.
                for pred in predicates:
                    id_sets.append(set(trees[pred.left].query(pred)))

                # Query ONLY THE DATA that is mentioned in those id sets.
                if len(predicates) != 0:

                    # Find the intersection of all id sets.
                    idset = id_sets[0]
                    for s in id_sets[1:]:
                        idset = idset.intersection(s)

                    querydata = [x for x in indexed if x['id'] in idset]
                else:
                    querydata = indexed

                # Currently, only 'count' is supported.
                if mode == 'count':

                    for row in querydata:
                        for attr in attributes:
                            results[attr][row[attr]] += 1

                    stop_sampler = False

                    return results

                stop_sampler = False

                return 1

    def gen_reservoir_sample(self, attr):
        '''
        Appends rare sample data to a reservoir sample.
        :param attr: The attribute we have the distribution information for.
        :return: The modified reservoir sample.
        '''

        global reservoir
        global counts
        global rare_pops

        # Initialize the random number generator.
        rng = np.random.default_rng()

        # Make a copy of the current reservoir sample so it does not change during execution.
        reservoir_snapshot = reservoir.copy()

        # Turn the sample into a list.
        sample_out = [value for key in reservoir.keys() for value in reservoir[key]]

        # Calculate the expected distribution of the sample.
        distribution = {x: y for x, y in counts[attr].items()}
        sample_distribution = {x: math.ceil(sum([len(entry) for entry in reservoir_snapshot.values()]) /
                                            sum(distribution.values()) * y) for x, y in distribution.items()}

        # Fill the sample with rare values if necessary.
        for key, value in rare_pops.items():
            missing_entries = sample_distribution[key] - len(reservoir_snapshot[key])

            if missing_entries > 0:
                sample_out.extend(list(rng.choice(value, missing_entries)))

        return sample_out

    def construct_preds(self, predicates):
        '''
        This simply turns several Predicate objects into an evaluable String.
        :param predicates: A list of predicate objects.
        :return: The same predicates in one String.
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

    def simulate(self, attr, sequential):
        '''
        Fire off multiple queries in 1 second intervals.
        :param attr: The attribute to get the distribution from.
        :return: None
        '''

        self.sequential = sequential

        # This is the attibute that will be used for distribution calculation and stratified sample generation!
        self.attr = attr

        while True:

            try:
                # The delay between each query.
                time.sleep(1)

                # Establish all predicates.
                pred1 = Predicate("age >= 29", AttrType.NUM)
                pred2 = Predicate("salary > 1000", AttrType.NUM, '&')
                pred3 = Predicate("tip < 12", AttrType.NUM, '&')
                pred4 = Predicate("age >= 32", AttrType.NUM)

                # Query the trees and time it.
                t0 = time.time()
                # result = self.query(mode='count', attributes=['age'], predicates=[])
                result = self.query(mode='count', attributes=['age'], predicates=[pred1])
                # result = self.query(mode='count', attributes=['age'], predicates=[pred1, pred2])
                # result = self.query(mode='count', attributes=['age'], predicates=[pred1, pred2, pred3])
                # result = self.query(mode='count', attributes=['age'], predicates=[pred4])
                query_time = time.time() - t0

                if result is None:
                    break

                print("Time for query: {:.5f}".format(query_time))

                if type(result) is not int:
                    self.print_result(result)
            except Exception as e:
                traceback.print_exc()

            if self.sequential:
                break

    def print_result(self, result):
        print("")

        for key in result.keys():

            # Sort the result by key
            result[key] = dict(sorted(result[key].items()))

            for subkey in result[key].keys():
                print("{},{}:{}".format(key, subkey, str(result[key][subkey])))


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


class Sampler:

    def __init__(self, sample_size):

        # DEBUG: Usually we do not know the rowcount prior to first scan!
        # DEBUG: This WILL be reset if the sampler runs with dynamic sampling as the first step!
        self.rowcount = 5000000
        self.sample_size = sample_size

    # Runs the sequence of reservoir first, then static sampling.
    def run(self, style, attrs, first_run_dynamic=True):
        global trees
        global reservoir
        global indexed
        global counts

        t0 = time.time()
        try:

            # First, do a streaming sample, since we do not know about the size of the dataset.
            if first_run_dynamic:

                # Reset the rowcount since we do not know any metadata.
                self.rowcount = 0

                # Initiate reservoir sampling.
                self.sample_reservoir(attrs[0])

                # Generate the trees for each given attribute, based on that reservoir sample
                try:
                    trees = self.generate_trees(reservoir, attrs, style)
                except Exception:
                    traceback.print_exc()

                # Let the program know that we have indexed that sample.
                indexed.extend([value for key in reservoir.keys() for value in reservoir[key]])

                # Start static sampling.
                # self.sample_static_uniform(attrs, style, first_run_dynamic)
                self.sample_static_stratified(attrs, style, first_run_dynamic)

            else:

                # Start static sampling.
                self.sample_static_uniform(attrs, style, first_run_dynamic)
        except Exception as e:
            traceback.print_exc()

    # Reservoir sampling for first pass.
    def sample_reservoir(self, attr):
        '''
        Use modified reservoir sampling to return a sample of the data.
        :param attr: The attribute to record the distribution from.
        :return: None
        '''

        global reservoir
        global data
        global rare_pops
        global sampler_stopped
        global stop_sampler

        rng = np.random.default_rng()

        t0 = time.time()

        reservoir = defaultdict(list)
        counts[attr] = defaultdict(int)
        groups[attr] = defaultdict(list)
        rare_pops = defaultdict(list)

        # Do reservoir sampling for each row in the returned sample.
        for row in data:

            while stop_sampler:
                sampler_stopped = True
                time.sleep(0.1)

            sampler_stopped = False

            # Count the occurrence of attribute variations for the given column.
            # We do not just count but also append the row directly which allows for some easier data manipulation later.
            # Real systems would not have this ability without pulling all data into RAM.
            counts[attr][row[attr]] += 1

            # If the reservoir sample size is smaller than the size we want it to have,
            # just append the row to the sample.
            if sum([len(entry) for entry in reservoir.values()]) < self.sample_size:
                reservoir[row[attr]].append(row)

            else:
                # Otherwise, calculate how likely it is to be replaced,
                # depending on how many rows we have already sampled.
                probability_for_replacement = self.sample_size / (self.rowcount + 1)

                # Find out if we actually replace this row.
                diceroll = rng.random()

                # If so, find a random entry of our current reservoir and replace it with that row.
                if diceroll < probability_for_replacement:
                    replacement_key = rng.choice(list(reservoir.keys()))
                    replacement_position = rng.integers(0, len(reservoir[replacement_key]))

                    reservoir[row[attr]].append(row)

                    replaced_row = reservoir[replacement_key].pop(replacement_position)
                    groups[attr][row[attr]].append(replaced_row)

                    if len(reservoir[replacement_key]) == 0:
                        del reservoir[replacement_key]

                    # Test the replaced row for a potential rare sample.
                    self.sample_rare(replaced_row, attr, rng)
                else:
                    groups[attr][row[attr]].append(row)

                    # Potentially still sample the row because it is a rare value.
                    self.sample_rare(row, attr, rng)

            self.rowcount += 1
            self.rare_threshold = self.rowcount * 0.01

        print("Done reservoir sampling!")

        print("Time taken: {}".format(time.time() - t0))

    def sample_rare(self, row, attr, rng):
        global rare_pops

        # If the sample does not get pulled into the sample by random chance but is
        # a rare entry, we save it separately.
        if counts[attr][row[attr]] < self.rare_threshold:

            # We essentially do reservoir sampling again, but group specific and with a smaller sample!
            if len(rare_pops[row[attr]]) < (self.sample_size * 0.01):
                rare_pops[row[attr]].append(row)
            else:
                probability_for_replacement = (self.sample_size * 0.01) / counts[attr][row[attr]]

                # Find out if we actually replace this row.
                diceroll = rng.random()

                # If so, find a random entry of our current reservoir and replace it with that row.
                if diceroll < probability_for_replacement:
                    replacement_position = rng.integers(0, len(rare_pops[row[attr]]))
                    rare_pops[row[attr]][replacement_position] = row

        # If the supposed rare population exceeds the rare_threshold by 100%, kick it out.
        elif counts[attr][row[attr]] > (self.rare_threshold * 2):
            if row[attr] in rare_pops:
                del rare_pops[row[attr]]

    # Static random sampling when more info (i.e. rowcount) is known.
    def sample_static_uniform(self, attrs, style, first_run_dynamic):
        global trees
        global indexed
        global data
        global reservoir
        global stop_sampler
        global sampler_stopped

        # Exclude the already sampled data from the given data, so that we do not sample it again.
        if len(reservoir) > 0:
            used_ids = set([x['id'] for x in reservoir])
            filtered_data = [x for x in data if x['id'] not in used_ids]
        else:
            filtered_data = data

        rng = np.random.default_rng()

        rng.shuffle(filtered_data)

        samples = []
        # Generate random samples from the given data.
        for i in range(0, len(filtered_data), self.sample_size):

            if i + self.sample_size > len(filtered_data):
                samples.append(filtered_data[i:])
            else:
                samples.append(filtered_data[i:i + self.sample_size])

        # If the first run was NOT DYNAMIC, the trees DO NOT EXIST yet! Generate those first!
        if not first_run_dynamic:
            # Make sure that the sample we use to generate the trees now
            # is not used later.
            sample = samples.pop(0)

            trees = self.generate_trees(sample.tolist(), attrs, style)

            indexed += sample

        # While there are still samples available for processing..
        t0 = time.time()
        for counter, sample in enumerate(samples):

            if stop_sampler:
                sampler_stopped = True

            while stop_sampler:
                time.sleep(0.1)

            sampler_stopped = False

            # Populate all trees.
            self.populate_trees(sample, attrs)

            print("Sample {} done!".format(str(counter)))

            # Add the data to the array of indexed data.
            indexed += sample
        print("Time for full population: {}".format(str(time.time() - t0)))

        print("Done!")

    def sample_static_stratified(self, attrs, style, first_run_dynamic):
        '''
        Provides several stratified samples based on the distribution data collected in the dynamic sampling step.
        :param attrs: The attributes that are indexed.
        :param style: Style of the tree to be generated.
        :param first_run_dynamic:
        :return:
        '''

        global trees
        global indexed
        global counts
        global stop_sampler
        global sampler_stopped
        global sampler_done

        # For ease of use, we only consider the FIRST attribute for stratified sampling.
        attr = attrs[0]

        # Get and generate the distribution of the samples.
        distribution = {x: y for x, y in counts[attr].items()}
        sample_distribution = {x: math.ceil(self.sample_size / sum(distribution.values()) * y) for x, y in
                               distribution.items()}

        rng = np.random.default_rng()

        samples = list()

        # For every group of attribute values that is in the sample distribution..
        for key in sample_distribution:

            # Shuffle the entries belonging to that group.
            rng.shuffle(groups[attr][key])

            sample_count = 0

            # Now, take samples sequentially based on the distribution, until none are left.
            for i in range(0, distribution[key], sample_distribution[key]):

                if i + sample_distribution[key] > distribution[key]:
                    try:
                        samples[sample_count].extend(groups[attr][key][i:])
                    except IndexError:
                        samples.append(groups[attr][key][i:])
                else:
                    try:
                        samples[sample_count].extend(groups[attr][key][i:i + sample_distribution[key]])
                    except IndexError:
                        samples.append(groups[attr][key][i:i + sample_distribution[key]])

                sample_count += 1

        # If the first run was NOT DYNAMIC, the trees DO NOT EXIST yet! Generate those first!
        if not first_run_dynamic:
            # Make sure that the sample we use to generate the trees now
            # is not used later.
            sample = samples.pop(0)

            trees = self.generate_trees(sample, attrs, style)

            indexed += sample

        # For every sample that we generated..
        for sample in samples:

            # Check if we need to stop because a query wants to run.
            if stop_sampler:
                sampler_stopped = True

            while stop_sampler:
                time.sleep(0.1)

            sampler_stopped = False

            # Populate all trees with a new sample.
            self.populate_trees(sample, attrs)

            # Add the data to the array of indexed data.
            indexed += sample

        sampler_done = True

    # For each given attribute, generate a Tree for the given sample.
    def generate_trees(self, sample, attrs, style):
        print("Generating trees..")
        trees = {}

        sample = [value for key in sample.keys() for value in sample[key]]

        # We require one tree per attribute as to not mix the data.
        for attr in attrs:

            # Generate a tree for this attribute, depending on the passed style.
            if style == 'MSD_splitearly':
                trees[attr] = radix_msd_splitearly.Tree([(x['id'], x[attr]) for x in sample])
            elif style == 'MSD':
                trees[attr] = radix_msd.Tree([(x['id'], x[attr]) for x in sample])
            elif style == 'BPLUS':
                trees[attr] = bplus.Tree([(x['id'], x[attr]) for x in sample])
            else:
                raise Exception

        # static_trees = pickle.loads(pickle.dumps(trees))

        print("Trees generated!")
        return trees

    def populate_trees(self, sample, attrs):
        """
        Adds data of a sample to all trees.
        :param sample: The sample containing the data.
        :param attrs: The trees we want to populate.
        :return: None
        """

        global trees

        for attr in attrs:
            trees[attr].populate([(x['id'], x[attr]) for x in sample])

    def get_trees(self):
        global trees
        return trees


# Load data from pickle.
t0 = time.time()
print("Loading..")
with open(os.environ['DATA_PATH'], 'rb') as f:
    data = pickle.load(f)
print("Done loading!")
print("Time for Pickle fetch: {}".format(str(time.time() - t0)))
