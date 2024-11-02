import random as r
import dataset as ds


def generate_eqn_true(seed: int, max_int: int = 100):
    r.seed(seed)
    first_int = r.randint(1, max_int)
    second_int = r.randint(1, max_int)
    sum = first_int + second_int
    return f"{first_int} + {second_int} = {sum}"


def generate_eqn_false(seed: int, max_int: int = 100, max_diff: int = 5):
    r.seed(seed)
    first_int = r.randint(1, max_int)
    second_int = r.randint(1, max_int)
    random_sign = r.choice([+1, -1])
    sum = first_int + second_int + random_sign * r.randint(1, max_diff)
    return f"{first_int} + {second_int} = {sum}"


def gen_sum_ds(N_total_each: int, max_int: int = 100, max_diff: int = 5):
    seeds = range(N_total_each)
    eqns_true = [generate_eqn_true(seed, max_int) for seed in seeds]
    eqns_false = [generate_eqn_false(seed, max_int, max_diff) for seed in seeds]
    dataset_sums = ds.Dataset(
        desc=f"sums_maxint_{max_int}_maxdiff_{max_diff}_N_{N_total_each}",
        pos_examples=eqns_true,
        neg_examples=eqns_false,
    )
    dataset_sums.save()


def load_sum_ds(N_total_each: int, max_int: int = 100, max_diff: int = 5):
    return ds.Dataset.load(f"sums_maxint_{max_int}_maxdiff_{max_diff}_N_{N_total_each}")


def calculate_primes_up_to(n: int = 200):
    """use sieve of eratosthenes to calculate primes up to n"""
    primes = []
    sieve = [True] * n
    for p in range(2, n):
        if sieve[p]:
            primes.append(p)
            for i in range(p * p, n, p):
                sieve[i] = False
    return primes


def get_number_rule_dataset(
    N_total_each: int,
    seed: int,
    max_int: int = 100,
    rule: str = "even",
):
    r.seed(seed)

    desc = f"{rule}_maxint_{max_int}_N_{N_total_each}"

    if rule in ["even", "odd"]:
        evens = [r.randint(0, max_int) * 2 for _ in range(N_total_each)]
        odds = [r.randint(0, max_int) * 2 + 1 for _ in range(N_total_each)]
        if rule == "even":
            dataset = ds.Dataset(desc, evens, odds)
        else:
            dataset = ds.Dataset(desc, odds, evens)

    elif rule in ["div3", "not_div3"]:
        div3s = [r.randint(0, max_int) * 3 for _ in range(N_total_each)]
        not_div3s = [
            r.randint(0, max_int) * 3 + r.choice([1, 2]) for _ in range(N_total_each)
        ]
        if rule == "div3":
            dataset = ds.Dataset(desc, div3s, not_div3s)
        else:
            dataset = ds.Dataset(desc, not_div3s, div3s)

    elif rule in ["prime", "not_prime"]:
        primes = calculate_primes_up_to(max_int)
        non_primes = [n for n in range(max_int) if n not in primes]
        # subsample N_total_of each
        prime_samples = r.choices(primes, k=N_total_each)
        non_prime_samples = r.choices(non_primes, k=N_total_each)

        if rule == "prime":
            dataset = ds.Dataset(desc, prime_samples, non_prime_samples)
        else:
            dataset = ds.Dataset(desc, non_prime_samples, prime_samples)

    else:
        raise ValueError(f"rule {rule} not recognized")

    # want to return the dataset, only save it for easy analysis / error catching
    dataset.save()
    return dataset


if __name__ == "__main__":
    pass
