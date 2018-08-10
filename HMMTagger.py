#!/usr/bin/python3

"""
This module contains classes to support parts-of-speech (POS) taggers based on
hidden Markov models.
"""

# basic imports
import math
from collections import Counter, defaultdict

class ProbDist():
    """A class to represent a discrete probability distribution."""

    def __init__(self, population, event_space, smoothing=None, d=0.5):
        """"Make a probability distribution, given evidence from a population
        with a predefined event space, and a smoothing technique."""

        # define the event space for the distribution
        self.event_space = event_space

        # make a counter object from the population & sum the event counts
        counter = Counter(population)
        N = sum(counter.values())

        # a dictionary to store the probability of each item in the event space
        self.P = dict.fromkeys(event_space)

        # apply smoothing, either add one (Laplace) or absolute discounting
        if smoothing == 'add_one':
            # add 1 to each event in the counter
            for x in event_space: counter[x] += 1

            # obtain the sum of the events after adding one to each event
            N = sum(counter.values())

            # populate probabilities dictionary
            for x in event_space:
                self.P[x] = counter[x]/N

        elif smoothing == 'abs_disc':
            # nPlus: number of events with non-zero counts
            nPlus = len(counter.keys())

            # size of the event space
            E = len(event_space)

            for x in event_space:
                self.P[x] = max(counter[x] - d, 0)/N + (d*nPlus/N) * (1/E)

        else: # no smoothing (not recommanded)
            for x in event_space:
                self.P[x] = counter[x]/N


    def test(self):
        """Test whether or not the probability mass sums up to one."""

        P_sum = sum(self.P[x] for x in self.event_space)
        precision = 10**-10

        assert abs(1.0 - P_sum) < precision, 'Probability mass does not sum up to one.'

        print('Test passed. Probability mass sums up to one.')


class HMMTagger():
    """A class to represent an HMM-based POS tagger."""

    def __init__(self, corpus, smoothing=None, d=0.5):
        """"Initialize HMM POS tagger given a training a corpus.

        :param corpus: training corpus
        :type object of nltk.corpus.ConllCorpusReader
        :param smoothing: smoothing technique, either 'add_one' or 'abs_disc'
        :type string, default None """

        # define smoothing method for tagger
        self.smoothing = smoothing

        # obtain tagset and word vocabulary from the training coprus
        self.tagset = set([t for (w, t) in corpus.tagged_words()])
        self.word_vocab = set([w for (w, t) in corpus.tagged_words()])

        # to handle unknow words, add <UNK> token to the vocab
        self.word_vocab.add('<UNK>')

        # train HMM
        self.train(corpus)

    def train(self, corpus):
        """Train the tagger and compute HMM components; that is,
        initial, transition, and emission probabilities. """

        ##### INITIAL PROBABILITIES
        # make initail tag population & make distribution for initial probabilites
        initial_tags = [t for s in corpus.tagged_sents() for (w, t) in s[:1]]
        self.initials = ProbDist(initial_tags, self.tagset, smoothing=self.smoothing)

        ##### TRANSITION PROBABILITIES
        # obtain next tag population for eahc tag from the training corpus
        next_tags = defaultdict(list)
        for sent in corpus.tagged_sents():
            for i in range(len(sent) - 1):
                tag, next_tag = sent[i][1], sent[i + 1][1]
                next_tags[tag].append(next_tag)

        # make a distribution for each tag in the tag set
        self.transitions = dict.fromkeys(self.tagset)

        for tag in self.tagset:
            self.transitions[tag] = ProbDist(next_tags[tag], self.tagset,
                 smoothing=self.smoothing)

        ##### EMISSION PROBABILITIES
        # words emissions per tag
        emitted_words_per_tag = defaultdict(list)

        for (word, tag) in corpus.tagged_words():
            emitted_words_per_tag[tag].append(word)

        self.emissions = dict.fromkeys(self.tagset)

        for tag in self.tagset:
            self.emissions[tag] = ProbDist(emitted_words_per_tag[tag],
                self.word_vocab, smoothing=self.smoothing)

    def decode(self, W):
        """A decoding procedure using Viterbi algorithm for POS tagging
        Returns the most probable tag sequence

        :param W: sequence of observations (words) [W]
        :rtype: list of the most probable tag sequence [T] """

        # data structures for viterbi matrix and backpointer
        viterbi = [{} for i in range(len(W))]
        backpointer = [{} for i in range(len(W))]

        # a list to store the most probable tag sequence
        tag_seq = []

        # Initialization step: at i = 1
        # loop through each state, update viterbi and backpointer at initial state
        for tag in self.tagset:
            # unknown word handling
            word = W[0] if W[0] in self.word_vocab else '<UNK>'

            viterbi[0][tag] = math.log(self.initials.P[tag]) + \
                math.log(self.emissions[tag].P[word])
            backpointer[0][tag] = None

        # Recursion step: for i > 1, loop through every word observation
        for i in range(1, len(W)):
            # unknown word handling
            word = W[i] if W[i] in self.word_vocab else '<UNK>'

            # loop through each state
            for tag in self.tagset:
                max_v = self.viterbi_max(viterbi[i-1], tag)
                viterbi[i][tag] = max_v + math.log(self.emissions[tag].P[word])
                backpointer[i][tag] = self.arg_max(viterbi[i-1], tag, max_v)

        # termination step
        max_prob = max(v for v in viterbi[-1].values())
        previous = None

        for tag, prob in viterbi[-1].items():
            if prob == max_prob:
                tag_seq.append(tag)
                previous = tag
                break

        # follow the backpointer to get into the first observation
        for i in range(len(W) - 2, -1, -1):
            tag_seq.append(backpointer[i + 1][previous])
            previous = backpointer[i + 1][previous]

        tag_seq.reverse()
        return tag_seq

    # Two helper functions to be used in the viterbi decoder
    def viterbi_max(self, viterbi, tag):
        """
        Return the max of viterbi[t-1]*transitions[prev_state, current_state].
        """
        v_values = [viterbi[prev] + \
            math.log(self.transitions[prev].P[tag]) for prev in self.tagset]

        return max(v_values)

    def arg_max(self, viterbi, tag, viterbi_max):
        """
        Return the state of value obtained by viterbi_max.
        """
        for prev in self.tagset:
            t_prob = self.transitions[prev].P[tag]

            if viterbi[prev] + math.log(t_prob) == viterbi_max:
                return prev
