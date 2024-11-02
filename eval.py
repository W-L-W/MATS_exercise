from typing import List
import dataset as ds
import oai_utils as ou
import argparse
import json

def evaluate_api(
        dataset_desc: str,
        n_eg_options: List[int],
        num_seeds: int,
        model: str='gpt-4o-mini',
        max_tokens: int=1,
        json_format: bool=True,
):
    """For each seed evaluate the API"""
    seeds = range(num_seeds)
    dataset = ds.Dataset.load(dataset_desc)
    response_format = ou.BOOLEAN_RESPONSE_FORMAT if json_format else None
    extractor = ou.extract_bool_from_completion_json_format if json_format else ou.extract_bool_response
    out_dicts = []
    for n_egs in n_eg_options:
        for seed in seeds:
            tm_dict = dataset.generate_true_false_prompt(n_egs, seed)
            for label, tm in tm_dict.items():
                response = ou.call_api(tm, seed, model, max_tokens, response_format=response_format)
                bool_response = extractor(response)
                out_dict = {
                    'seed': seed,
                    'n_egs': n_egs,
                    'label': label,
                    'response': bool_response
                }
                out_dicts.append(out_dict)

    # Cache this output to a json file (in case want to run full analysis)
    # have a descriptive file name using all input arguments
    out_file = f"{dataset_desc}_n_egs_{n_eg_options}_seeds_{num_seeds}_{model}_maxtokens{max_tokens}.json"
    out_path = "cache/" + out_file
    with open(out_path, 'w') as f:
        json.dump(out_dicts, f)

    return out_dicts


    


if __name__ == "__main__":
    # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("desc", type=str, help="Description of the dataset")
    parser.add_argument("n_eg_options", )