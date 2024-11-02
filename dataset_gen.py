import random as r
import dataset as ds

def generate_eqn_true(seed: int, max_int: int=100):
        r.seed(seed)
        first_int = r.randint(1, max_int)
        second_int = r.randint(1, max_int)
        sum = first_int + second_int
        return f"{first_int} + {second_int} = {sum}"

def generate_eqn_false(seed: int, max_int: int=100, max_diff: int=5):
    r.seed(seed)
    first_int = r.randint(1, max_int)
    second_int = r.randint(1, max_int)
    random_sign = r.choice([+1, -1])
    sum = first_int + second_int + random_sign * r.randint(1, max_diff)
    return f"{first_int} + {second_int} = {sum}"

def gen_sum_ds(N_total_each: int, max_int: int=100, max_diff: int=5):
    seeds = range(N_total_each)
    eqns_true = [generate_eqn_true(seed, max_int) for seed in seeds]
    eqns_false = [generate_eqn_false(seed, max_int, max_diff) for seed in seeds]
    dataset_sums = ds.Dataset(
        desc=f'sums_maxint_{max_int}_maxdiff_{max_diff}_N_{N_total_each}',
        pos_examples=eqns_true,
        neg_examples=eqns_false,
    )
    dataset_sums.save()

def load_sum_ds(N_total_each: int, max_int: int=100, max_diff: int=5):
    return ds.Dataset.load(f'sums_maxint_{max_int}_maxdiff_{max_diff}_N_{N_total_each}')

if __name__ == "__main__":
     pass