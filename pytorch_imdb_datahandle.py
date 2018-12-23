#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-19 
# function: 
import torch
import numpy as np
from tflearn.data_utils import pad_sequences
import pickle as pkl
import json
from torch.utils.data import Dataset, DataLoader


def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

def load_imdb_data(max_feature,ngram_range,sentence_len):
    def load_data(path='imdb.npz', num_words=None, skip_top=0, seed=113, start_char=1, oov_char=2, index_from=3):
        # 1. load data
        with np.load(path) as f:
            x_train, labels_train = f['x_train'], f['y_train']
            x_test, labels_test = f['x_test'], f['y_test']

        # 2. shuffle train/test
        np.random.seed(seed)
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        xs = np.concatenate([x_train, x_test])
        labels = np.concatenate([labels_train, labels_test])

        # 保留前3个index
        if start_char is not None:
            xs = [[start_char] + [w + index_from for w in x] for x in xs]
        elif index_from:
            xs = [[w + index_from for w in x] for x in xs]

        if not num_words:
            num_words = max([max(x) for x in xs])

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
        else:
            xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
        x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

        return (x_train, y_train), (x_test, y_test)

    print('MAX_FEATURE:', max_feature)

    # 1. load original data
    print('loading data...')
    (trainX, trainY), (testX, testY) = load_data(num_words=max_feature)
    print('train_data length:', len(trainX))
    print('test_data length:', len(testX))

    # 2. add n-gram
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in trainX:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        print('MAX_FEATURE:', max_feature)
        start_index = max_feature + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        MAX_FEATURE = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        trainX = add_ngram(trainX, token_indice, ngram_range)
        testX = add_ngram(testX, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, trainX)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, testX)), dtype=int)))

    # 3.Data preprocessing      Sequence padding
    print("start padding & transform to one hot...")
    trainX = pad_sequences(trainX, maxlen=sentence_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=sentence_len, value=0.)  # padding to max length
    print('x_train shape:', trainX.shape)
    print('x_test shape:', testX.shape)

    print("end padding & transform to one hot...")
    return (trainX, trainY), (testX, testY)
    # return (trainX, to_categorical(trainY)), (testX, to_categorical(testY))

def lazy_load_imdb_data(ngram_range=1, max_features=20000, sentence_len=400):
    filename = "-".join(["data", str(ngram_range), str(max_features), str(sentence_len)])
    filename += ".pkl"
    print(filename)

    try:
        with open(filename, "rb") as source:
            print('lazy loading...')
            data = pkl.load(source)
            print("Lazy load successful")
            return data
    except FileNotFoundError:
        #         data = fetch_imdb_data(ngram_range, max_features, maxlen)
        data = load_imdb_data(max_features,ngram_range,sentence_len)
        with open(filename, "wb") as target:
            pkl.dump(data, target)
        return data






# MAX_FEATURE = 10000
# SENTENCE_LEN = 250
# (x_train, y_train), (x_test, y_test) = lazy_load_imdb_data(ngram_range=1, max_features=MAX_FEATURE, sentence_len=SENTENCE_LEN)
