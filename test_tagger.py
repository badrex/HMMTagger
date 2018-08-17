#!/usr/bin/python3

"""A program to use a trained POS tagger based on Hidden Markov Model."""

import argparse
import pickle

import nltk
from nltk.corpus.reader import ConllCorpusReader


def tag_testset():
    """
    Read options from user input, and tagger on test set.
    """

    DISCR = 'Test a POS tagger given a test data.'
    parser = argparse.ArgumentParser(description=DISCR)

    parser.add_argument('-test_dir', type=str,
        help='Directory of test file.', required=True)

    parser.add_argument('-test_fileid', type=str,
        help='Name of the test file.', required=True)

    parser.add_argument('-POS_tagger', type=str,
        help='POS Tagger object.', required=True)

    parser.add_argument('-tagged_output', type=str,
        help='Path to tagged words file.', required=True)

    args = parser.parse_args()

    # create a CoNLL corpus reader object
    test_corpus = nltk.corpus.ConllCorpusReader(
        args.test_dir, args.test_fileid, ['words'], tagset='universal', encoding='utf8'
    )

    # obtain tagger object from disk
    with open(args.POS_tagger, "rb") as f:
            POSTagger= pickle.load(f)

            with open(args.tagged_output, 'w') as op_file:
                for sent in test_corpus.sents():
                    predicted_tags = POSTagger.decode(sent)

                    for w, tag in zip(sent, predicted_tags):
                        op_file.write(w + '\t' + tag + '\n')

                    op_file.write('\n')


def main():
    tag_testset()


if __name__ == '__main__':
    main()
