#!/usr/bin/env python
"""
coding=utf-8

Code template courtesy https://github.com/bjherger/Python-starter-repo

"""
import pandas

TEST_RUN = True

def gen_word_pairs():
    global TEST_RUN

    # Reference variables
    vocabulary = set()

    # Read in data
    observations = pandas.read_pickle('../data/posts.pkl')
    if TEST_RUN:
        observations = observations.sample(1000)
        observations = observations.reset_index()
    print(observations.columns)
    # TODO Create vocabulary
    for document in observations['text_clean']:
        # print(document)
        pass
    # TODO Create word pairs
    # TODO Return
    pass


def train_model():
    pass


def extract_embedding():
    pass


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    # Reference variables
    gen_word_pairs_config = True
    train_model_config = True
    extract_embedding_config = True

    # TODO Create word pairs
    if gen_word_pairs_config:
        gen_word_pairs()

    # TODO Train model
    if train_model_config:
        train_model()

    # TODO Extract embedding
    if extract_embedding_config:
        extract_embedding()

    pass

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

# Main section
if __name__ == '__main__':
    main()
