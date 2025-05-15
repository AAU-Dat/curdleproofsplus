import random
import numpy as np


def honest_shuffle(ciphertexts, water, indices, active_set):
    """Fair shuffle and average water among active cups only."""
    active_indices = [i for i in indices if i in active_set]
    if not active_indices:
        return
    average = sum(water[i] for i in active_indices) / len(active_indices)
    for i in active_indices:
        water[i] = average

    # waterclone = water.copy()

    # Shuffle ciphertexts
    sublist = [ciphertexts[i] for i in indices]
    random.shuffle(sublist)
    for idx, val in zip(indices, sublist):
        ciphertexts[idx] = val
        # water[idx] = waterclone[val]


def adversarial_shuffle(ciphertexts, water, indices):
    """Adversarial shuffler does nothing."""
    pass


def distributed_shuffle_with_water(n, k, T, alpha):
    ciphertexts = list(range(0, n))
    water = [0.0] * n
    water[0] = 1.0  # Initial water in cup 0

    active_set = set(range(n - alpha))
    threshold = 2 / (n - alpha)

    res = []
    while True:

        indices = random.sample(range(n), k)
        honest_shuffle(ciphertexts, water, indices, active_set)

        not_passing_threshold = not any(w > threshold for i, w in enumerate(water))
        if not_passing_threshold == True:
            res.append("success")
            return res
        res.append("fail")

    return res

    # Return True if successful


# Run 100 trials
def run_experiments(num_trials=1000, n=16384, k=128, T=8192, alpha=8192):
    successes = 0
    result = [[] for i in range(num_trials)]
    for i in range(num_trials):
        res = distributed_shuffle_with_water(n, k, T, alpha)
        for j, elem in enumerate(res):
            result[i].append(elem)

    first_success_indices = [lst.index("success") for lst in result]

    f = open("results.txt", "a")
    for value in first_success_indices:
        f.write(f"{k},{alpha},{value}\n")
    f.close()


def benchmark_run():
    f = open("results.txt", "a")
    f.write("k,alpha,value\n")
    f.close()
    for k in range(32, 513):
        for alpha in [4096, 5462, 8192]:
            run_experiments(k=k, alpha=alpha)


if __name__ == "__main__":
    # run_experiments()
    benchmark_run()
