from typing import List
import json
import pandas as pd

import dataset as ds
import oai_utils as ou
import dataset_gen as dg


def evaluate_tf_completion(
    dataset_desc: str,
    n_eg_options: List[int],
    num_seeds: int,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1,
    json_format: bool = True,
):
    """For each seed evaluate the API"""
    seeds = range(num_seeds)
    dataset = ds.Dataset.load(dataset_desc)
    response_format = ou.BOOLEAN_RESPONSE_FORMAT if json_format else None
    extractor = (
        ou.extract_bool_from_completion_json_format
        if json_format
        else ou.extract_bool_response
    )
    out_dicts = []
    for n_egs in n_eg_options:
        for seed in seeds:
            tm_dict = dataset.generate_true_false_prompt(n_egs, seed)
            for label, tm in tm_dict.items():
                response = ou.call_api(
                    tm, seed, model, max_tokens, response_format=response_format
                )
                bool_response = extractor(response)
                out_dict = {
                    "seed": seed,
                    "n_egs": n_egs,
                    "label": label,
                    "response": bool_response,
                }
                out_dicts.append(out_dict)

    # Cache this output to a json file (in case want to run full analysis)
    # have a descriptive file name using all input arguments
    out_file = f"{dataset_desc}_n_egs_{n_eg_options}_seeds_{num_seeds}_{model}_maxtokens{max_tokens}.json"
    out_path = "cache/" + out_file
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(out_dicts, f)

    # some further analysis to get summary statistics
    perfs = pd.DataFrame(out_dicts)
    perfs["correct"] = perfs["label"] == perfs["response"]
    sum_stats = perfs.groupby(["label", "n_egs"])["correct"].mean()
    summary_out_file = f"{dataset_desc}_n_egs_{n_eg_options}_seeds_{num_seeds}_{model}_maxtokens{max_tokens}_summary.csv"
    sum_stats.to_csv("cache/" + summary_out_file)

    return out_dicts, sum_stats


# specifically for the numerical rules defined in dataset_gen.py
NUM_CHOICE_SYSTEM_PROMPT = """
You are a helpful AI assistant that will solve the following simple maths problem...
The user will give you a set of pairs of the form (number, boolean)
where the boolean is True if and only if the number satisfies a certain simple rule.

The rule can take one of the following options:
- even: the number is even
- odd: the number is odd
- div3: the number is divisible by 3
- not_div3: the number is not divisible by 3
- prime: the number is prime
- not_prime: the number is not prime

Reply with the rule that describes the relationship between the number and the boolean.
"""


def generate_mcq_prompt_messages(
    dataset: ds.Dataset, n_egs: int, seed: int
) -> ou.OAI_MSGS:
    """Generate messages to send to api for MCQ on dataset"""
    user_content = dataset.generate_example_body(n_egs, seed)
    return ou.construct_messages(NUM_CHOICE_SYSTEM_PROMPT, user_content)


def evaluate_num_rule_mcq(
    rule: str,
    n_eg_options: List[int],
    num_seeds: int,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1,
    json_format: bool = True,
):
    """Use MCQ format to see how good choosing between different options"""
    dataset = dg.get_number_rule_dataset(
        rule=rule, N_total_each=200, seed=0, max_int=40
    )
    response_format = ou.NUM_MCQ_RESPONSE_FORMAT if json_format else None
    extractor = (
        ou.extract_mcq_choice_from_completion_json_format if json_format else None
    )
    out_dicts = []
    for n_egs in n_eg_options:
        print("Moving to n_egs:", n_egs)
        for seed in range(num_seeds):
            msgs = generate_mcq_prompt_messages(dataset, n_egs, seed)
            response = ou.call_api(
                msgs, seed, model, max_tokens, response_format=response_format
            )
            choice = extractor(response)
            out_dict = {"seed": seed, "n_egs": n_egs, "choice": choice}
            out_dicts.append(out_dict)

    # Again cache the output for reference
    out_stem = f"{rule}_n_egs_{n_eg_options}_seeds_{num_seeds}_{model}_maxtokens{max_tokens}.json"
    out_file = "cache/mcq/" + out_stem
    with open(out_file, "w", encoding="utf8") as f:
        json.dump(out_dicts, f)

    # later want to return and test out summary statistic getting
    df = pd.DataFrame(out_dicts)
    df["correct"] = df["choice"] == rule
    summary = df.groupby("n_egs")["correct"].mean()
    summary_out_file = f"{rule}_n_egs_{n_eg_options}_seeds_{num_seeds}_{model}_maxtokens{max_tokens}_summary.csv"
    summary.to_csv("cache/mcq/" + summary_out_file)

    return out_dicts, summary


if __name__ == "__main__":
    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("desc", type=str, help="Description of the dataset")
    pass
