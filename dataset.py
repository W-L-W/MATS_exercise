"""All things dataset"""

import json
from typing import Dict, List, Tuple
import random as r
from dataclasses import dataclass

from oai_utils import construct_messages, OAI_MSGS

# BalancedExamples = namedtuple("BalancedExamples", ["pos_examples", "neg_examples"])
# ClassifiedExample = namedtuple("ClassifiedExample", ["example", "bool"])

SYSTEM_CONTENT = (
"""
You are a helpful assistant given a classification challenge.
Learn the rule from the following True False examples, then output the correct classification for the test input.
You should return a single {'True', 'False'} value.
"""
)

# short utility functions
def format_as_example_body(examples_tuples: List[Tuple[str, bool]]):
    return "\n".join([f'Input: "{pair[0]}" Label: {pair[1]}' for pair in examples_tuples])

def get_user_content(example_pairs: List[Tuple[str, bool]], test_pairs: Dict[bool, str]) -> Dict[bool, str]:
    """Return a pair of prompts for each test pair, one for the positive label and one for the negative label."""
    head = "Example pairs to learn from:\n"
    body = format_as_example_body(example_pairs)
    tail = "\n\nTest pairs to evaluate:\n"
    shared_stem = head + body + tail

    def get_user_prompt(text: str):
        return shared_stem + f'Input: "{text}" Label: '
    
    return {label: get_user_prompt(text) for label, text in test_pairs.items()}

def construct_test_messages(example_pairs: List[Tuple[str, bool]], test_pairs: Dict[bool, str]) -> Dict[bool, OAI_MSGS]:
    user_content = get_user_content(example_pairs, test_pairs)
    return {
        label: construct_messages(SYSTEM_CONTENT, user_content[label]) 
        for label in test_pairs.keys()
    }



# Supporting data classes
@dataclass
class BalancedExamples():
    pos_examples: List[str]
    neg_examples: List[str]

@dataclass
class ClassifiedExample():
    example: str
    bool: bool

def get_path(desc: str):
    return f"datasets/{desc}.json"

# Main class
class Dataset:
    """Dataset containing large balanced number of examples of True/False for desired rule"""

    def __init__(self, desc: str, pos_examples: List[str], neg_examples: List[str]):
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.desc = desc

    def save(self):
        """Save the dataset in json format"""
        # construct dictionary for Dataset
        path = get_path(self.desc)
        data = {
            "pos_examples": self.pos_examples,
            "neg_examples": self.neg_examples
        }
        with open(path, 'w', encoding='utf8') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, desc: str):
        """Load the dataset from json file"""
        path = get_path(desc)
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        return cls(desc, data["pos_examples"], data["neg_examples"])

    def subsample(self, n: int, seed: int) -> BalancedExamples:
        """Return n random poss"""
        r.seed(seed)
        pos_sample = r.sample(self.pos_examples, n)
        neg_sample = r.sample(self.neg_examples, n)
        return BalancedExamples(pos_sample, neg_sample)

    def generate_true_false_prompt(self, n_egs: int, seed: int) -> Dict[bool, OAI_MSGS]:
        balanced_examples = self.subsample(n_egs, seed)
        test_pairs = {
            True: balanced_examples.pos_examples[-1],
            False: balanced_examples.neg_examples[-1]
        }

        pos_eg_pairs = [(ex, True) for ex in balanced_examples.pos_examples[:-1]]
        neg_eg_pairs = [(ex, False) for ex in balanced_examples.neg_examples[:-1]]

        # concatenate and shuffle the list of tuples
        example_pairs = pos_eg_pairs + neg_eg_pairs
        r.shuffle(example_pairs)

        test_messages = construct_test_messages(example_pairs, test_pairs)
        return test_messages
    


    
