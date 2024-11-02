import argparse
import pandas as pd

import evaluation as ev
import dataset_gen as dg


def initial_rule_testing(rule: str):
    """For relatively simple versions of each rule, see how many datapoints needed to learn"""
    N_total_each = 200
    seed = 0
    max_int = 40
    ds = dg.get_number_rule_dataset(N_total_each, seed, max_int, rule)

    outdicts = ev.evaluate_tf_completion(
        ds.desc,
        n_eg_options=[4, 8, 16, 32],
        num_seeds=50,
        max_tokens=10,
        json_format=True,
        model="gpt-4o",
    )
    return outdicts


def mcq_rule_testing(rule: str):
    """A larger run of the mcq rules"""
    ev.evaluate_num_rule_mcq(
        rule,
        n_eg_options=[4, 8, 16, 32],
        num_seeds=50,
        max_tokens=10,
        json_format=True,
        model="gpt-4o",
    )


if __name__ == "__main__":
    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-r", "--rule", type=str, help="rule to test")
    # args = parser.parse_args()

    # _ = initial_rule_testing(args.rule)
    #
    for rule in dg.RULES:
        print("Now testing rule:", rule)
        mcq_rule_testing(rule)

    print("Finished successfully!")
