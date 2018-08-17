#!/usr/bin/python3

"""A program to train a POS tagger based on Hidden Markov Model."""

import argparse
from HMMTagger import *
import pickle

import nltk
from nltk.corpus.reader import ConllCorpusReader


def build_tagger():
    """
    Read options from user input, and train tagger.
    """

    DISCR = 'Build an HMM-based POS tagger given a training data.'
    parser = argparse.ArgumentParser(description=DISCR)

    parser.add_argument('-train_dir', type=str,
        help='Directory of training file.', required=True)

    parser.add_argument('-train_fileid', type=str,
        help='Name of the training file.', required=True)

    parser.add_argument('-smoothing', type=str,
        help='Smoothing technique; add_one or abs_disc.', default = 'add_one')

    parser.add_argument('-d', type=float, default = 0.5,
        help='Discounting parameters; floating-point value between [0, 1].')

    parser.add_argument('-tagger_file', type=str,
        help='Path to tagger file.', required=True)

    args = parser.parse_args()

    columntypes  = ['words', 'pos']

    # create a CoNLL corpus reader object
    train_corpus = nltk.corpus.ConllCorpusReader(
        args.train_dir, args.train_fileid, columntypes, tagset='universal', encoding='utf8'
    )

    # build a tagger
    POSTagger = HMMTagger(train_corpus, args.smoothing, args.d)

    # save tagger object to disk
    pickle.dump(POSTagger, open( args.tagger_file, "wb" ) )


def main():
    build_tagger()


if __name__ == '__main__':
    main()
