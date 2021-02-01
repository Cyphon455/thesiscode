import concurrent.futures
import time

from ApproxIndexing.dev import QuerySimulator
from ApproxIndexing.dev import Sampler


def run_sequential(sampler, qs, type, columns):
    print("Sampling initiated")

    # What type of tree | What columns to index | Use dynamic sampling for first sample or not
    if type == 'MSD':
        sampler.run('MSD', columns, True)
    elif type == 'MSD_splitearly':
        sampler.run('MSD_splitearly', columns, True)
    elif type == 'BPLUS':
        sampler.run('BPLUS', columns, True)

    print("Querying initiated")
    qs.simulate('age', True)

    if type == 'MSD':
        # Since sampling is done and tree base level is fully populated, start refining.
        for tree in sampler.get_trees().values():
            while not tree.done:
                tree.split()

    print("Done!")


def run_multithreaded(sampler, qs, type, columns):
    print("Sampling initiated")

    # What type of tree | What columns to index | Use dynamic sampling for first sample or not
    if type == 'MSD':
        samplerThread = executor.submit(sampler.run, 'MSD', columns, True)
    elif type == 'MSD_splitearly':
        samplerThread = executor.submit(sampler.run, 'MSD_splitearly', columns, True)
    elif type == 'BPLUS':
        samplerThread = executor.submit(sampler.run, 'BPLUS', columns, True)

    print("Querying initiated")
    qsThread = executor.submit(qs.simulate, 'age', False)

    # Wait until ALL sampling is done.
    concurrent.futures.wait([samplerThread])

    if type == 'MSD':
        # Since sampling is done and tree base level is fully populated, start refining.
        for tree in sampler.get_trees().values():
            while not tree.done:
                tree.split()

    print("Done with all!")


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    print("Starting..")

    sample_size = 5000

    # Instantiate sampler and query-engine.
    sampler = Sampler(sample_size)
    qs = QuerySimulator()

    t0 = time.time()
    # Run the system.
    # run_sequential(sampler, qs, 'BPLUS', ['age'])
    # run_sequential(sampler, qs, 'MSD_splitearly', ['age'])
    # run_sequential(sampler, qs, 'MSD', ['age'])

    run_multithreaded(sampler, qs, 'BPLUS', ['age'])
    # run_multithreaded(sampler, qs, 'MSD_splitearly', ['age'])
    # run_multithreaded(sampler, qs, 'MSD', ['age'])

    total_time = time.time() - t0
    print("Total time: " + str(total_time))
