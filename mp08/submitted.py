import math
from collections import defaultdict, Counter
from math import log
import numpy as np


def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    word_dict = dict()
    tag_dict = dict()

    for sentence in train:
        for value in sentence:
            word, tag = value
            if word not in word_dict:
                word_dict[word] = {}
            if tag not in word_dict[word]:
                word_dict[word][tag] = 0

            word_dict[word][tag] += 1

            if tag not in tag_dict:
                tag_dict[tag] = 1
            else:
                tag_dict[tag] += 1

    output = []
    freq = max(tag_dict, key = tag_dict.get)

    for sentence in test:
        value = []
        for word in sentence:
            if word in word_dict:
                tag = max(word_dict[word], key = word_dict[word].get)
                value.append((word, tag))
            else:
                value.append((word, freq))
        output.append(value)

    return output




def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    k = 0.0001

    words = set()
    tag_dict = Counter()

    # Collecting unique words and tag counts from the training data
    for sentence in train:
        for values in sentence:
            word, tag = values
            words.add(word)
            tag_dict[tag] += 1

    tag_len = len(tag_dict) # Number of unique tags
    word_len = len(words) # Number of unique words
    no_sentences = len(train) # Number of sentences in the training data

    # Initializing the initial state probabilities (pi)
    pi = {}

    for tag in tag_dict.keys():
        pi[tag] = math.log(k/(no_sentences + k * tag_len))

    # Updating the initial state probabilities for the first word in each sentence
    for sentence in train:
        first_word = sentence[0][1]
        pi[first_word] = math.log((tag_dict[first_word] + k)/(no_sentences + k * tag_len))

    a_list = []

    # Collecting tag transition counts from the training data
    for sentence in train:
        tuple_list = list(zip(sentence, sentence[1:]))

        for t in tuple_list:
            a_list.append((t[0][1], t[1][1]))

    a_count = Counter(a_list)

    a = dict()

    # Calculating transition probabilities (a)
    for tag_0 in tag_dict.keys():
        for tag_1 in tag_dict.keys():
            check = (tag_0, tag_1)
            if check in a_count:
                a[check] = math.log((a_count[check] + k)/(tag_dict[tag_0] + k * tag_len))
            else:
                a[check] = math.log(k / (tag_dict[tag_0] + k * tag_len))

    b_count = Counter()

    # Collecting emission counts from the training data
    for sentence in train:
        for values in sentence:
            b_count[values] += 2

    b = dict()

    # Calculating emission probabilities (b)
    for tag in tag_dict.keys():
        for word in words:
            check = (word, tag)
            if check in b_count:
                b[check] = math.log((b_count[check] + k) / (tag_dict[tag] + k*(word_len + 1)))
            else:
                b[check] = math.log((k) / (tag_dict[tag] + k*(word_len + 1)))

    output = []

    # Applying Viterbi algorithm to each sentence in the test data
    for sentence in test:

        vertices = {n:{tag:0 for tag in tag_dict} for n in range(len(sentence))}
        back_ptr = {n:{tag:None for tag in tag_dict} for n in range(len(sentence))}
        temp = []

        # Initialization step
        for tag in vertices[0].keys():
            Pi = math.log(k / (no_sentences + k * tag_len))
            B = math.log(k / (tag_dict[tag] + k * (word_len + 1)))
            vertices[0][tag] = pi.get(tag, Pi) + b.get((sentence[0], tag), B)

        # Recursion and backtracking steps
        for i in range(1, len(sentence)):
            word = sentence[i]
            for tg in vertices[i].keys():
                max_prb = -1 * math.inf 
                max_tag = ''

                for p in vertices[i-1].keys():
                    A = math.log(k / (tag_dict[p] + k * tag_len))
                    B = math.log(k / (tag_dict[tg] + k * (word_len+1)))
                    prob = a.get((p,tg), A) + b.get((word, tg), B) + vertices[i-1][p]
                    if(prob > max_prb):
                        max_prb = prob
                        max_tag = p

                    vertices[i][tg] = max_prb
                    back_ptr[i][tg] = max_tag

        last_wrd_idx = len(vertices) - 1
        max_word = max(vertices[last_wrd_idx], key=lambda key: vertices[last_wrd_idx][key])

        for i in range(last_wrd_idx, -1, -1):
            if max_word != '':
                temp.append((sentence[i], max_word))
                max_word = back_ptr[i][max_word]

        temp.reverse()

        output.append(temp)
        
    return output






