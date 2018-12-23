#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-18 
# function: 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD, Adagrad # Adam doesn't currently support autograd with embedding layers
import numpy as np
from tflearn.data_utils import pad_sequences
import pickle as pkl
from keras.utils import to_categorical
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pytorch_imdb_datahandle import lazy_load_imdb_data
from pytorch_fasttext_model import TorchFastText


MAX_FEATURE = 10000
SENTENCE_LEN = 300
NGRAME_RANGE = 1
EMBEDDING_DIMS = 100


# def create_ngram_set(input_list, ngram_value=2):
#     return set(zip(*[input_list[i:] for i in range(ngram_value)]))
#
# def add_ngram(sequences, token_indice, ngram_range=2):
#     new_sequences = []
#     for input_list in sequences:
#         new_list = input_list[:]
#         for ngram_value in range(2, ngram_range + 1):
#             for i in range(len(new_list) - ngram_value + 1):
#                 ngram = tuple(new_list[i:i + ngram_value])
#                 if ngram in token_indice:
#                     new_list.append(token_indice[ngram])
#         new_sequences.append(new_list)
#
#     return new_sequences
#
# def load_imdb_data():
#     def load_data(path='imdb.npz', num_words=None, skip_top=0, seed=113, start_char=1, oov_char=2, index_from=3):
#         # 1. load data
#         with np.load(path) as f:
#             x_train, labels_train = f['x_train'], f['y_train']
#             x_test, labels_test = f['x_test'], f['y_test']
#
#         # 2. shuffle train/test
#         np.random.seed(seed)
#         indices = np.arange(len(x_train))
#         np.random.shuffle(indices)
#         x_train = x_train[indices]
#         labels_train = labels_train[indices]
#
#         indices = np.arange(len(x_test))
#         np.random.shuffle(indices)
#         x_test = x_test[indices]
#         labels_test = labels_test[indices]
#
#         xs = np.concatenate([x_train, x_test])
#         labels = np.concatenate([labels_train, labels_test])
#
#         # 保留前3个index
#         if start_char is not None:
#             xs = [[start_char] + [w + index_from for w in x] for x in xs]
#         elif index_from:
#             xs = [[w + index_from for w in x] for x in xs]
#
#         if not num_words:
#             num_words = max([max(x) for x in xs])
#
#         # by convention, use 2 as OOV word
#         # reserve 'index_from' (=3 by default) characters:
#         # 0 (padding), 1 (start), 2 (OOV)
#         if oov_char is not None:
#             xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
#         else:
#             xs = [[w for w in x if skip_top <= w < num_words] for x in xs]
#
#         idx = len(x_train)
#         x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
#         x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
#
#         return (x_train, y_train), (x_test, y_test)
#
#     global MAX_FEATURE
#     print('MAX_FEATURE:', MAX_FEATURE)
#
#     # 1. load original data
#     print('loading data...')
#     (trainX, trainY), (testX, testY) = load_data(num_words=MAX_FEATURE)
#     print('train_data length:', len(trainX))
#     print('test_data length:', len(testX))
#
#     # 2. add n-gram
#     if NGRAME_RANGE > 1:
#         print('Adding {}-gram features'.format(NGRAME_RANGE))
#         # Create set of unique n-gram from the training set.
#         ngram_set = set()
#         for input_list in trainX:
#             for i in range(2, NGRAME_RANGE + 1):
#                 set_of_ngram = create_ngram_set(input_list, ngram_value=i)
#                 ngram_set.update(set_of_ngram)
#
#         # Dictionary mapping n-gram token to a unique integer.
#         # Integer values are greater than max_features in order
#         # to avoid collision with existing features.
#         print('MAX_FEATURE:', MAX_FEATURE)
#         start_index = MAX_FEATURE + 1
#         token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
#         indice_token = {token_indice[k]: k for k in token_indice}
#
#         # max_features is the highest integer that could be found in the dataset.
#         MAX_FEATURE = np.max(list(indice_token.keys())) + 1
#
#         # Augmenting x_train and x_test with n-grams features
#         trainX = add_ngram(trainX, token_indice, NGRAME_RANGE)
#         testX = add_ngram(testX, token_indice, NGRAME_RANGE)
#         print('Average train sequence length: {}'.format(np.mean(list(map(len, trainX)), dtype=int)))
#         print('Average test sequence length: {}'.format(np.mean(list(map(len, testX)), dtype=int)))
#
#     # 3.Data preprocessing      Sequence padding
#     print("start padding & transform to one hot...")
#     trainX = pad_sequences(trainX, maxlen=SENTENCE_LEN, value=0.)  # padding to max length
#     testX = pad_sequences(testX, maxlen=SENTENCE_LEN, value=0.)  # padding to max length
#     print('x_train shape:', trainX.shape)
#     print('x_test shape:', testX.shape)
#
#     print("end padding & transform to one hot...")
#     return (trainX, trainY), (testX, testY)
#     # return (trainX, to_categorical(trainY)), (testX, to_categorical(testY))
#
# def lazy_load_imdb_data(ngram_range=1, max_features=20000, maxlen=400):
#     filename = "-".join(["data", str(ngram_range), str(max_features), str(maxlen)])
#     filename += ".pkl"
#     print(filename)
#
#     try:
#         with open(filename, "rb") as source:
#             print('lazy loading...')
#             data = pkl.load(source)
#             print("Lazy load successful")
#             return data
#     except FileNotFoundError:
#         #         data = fetch_imdb_data(ngram_range, max_features, maxlen)
#         data = load_imdb_data()
#         with open(filename, "wb") as target:
#             pkl.dump(data, target)
#         return data

def create_glove_embeddings(embedding_dims, max_feature):
    def get_word2index_dict(path='imdb_word_index.json'):
        with open(path) as f:
            return json.load(f)

    # A dictionary mapping words to an integer index
    word_index = get_word2index_dict()  # {word:index}

    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    print('Pretrained embeddings GloVe is loading...')

    embeddings_index = {}
    f = open('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.%id.txt' % embedding_dims)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors in GloVe embedding' % len(embeddings_index))

    embedding_matrix = np.zeros((max_feature, embedding_dims))
    # embedding_matrix = torch.zeros(MAX_FEATURE, EMBEDDING_DIMS)

    for word, i in word_index.items():
        if i >= max_feature:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            # embedding_matrix[i,:] = torch.from_numpy(embedding_vector)

    embedding_matrix = torch.tensor(embedding_matrix)
    return embedding_matrix


class MyData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print(f"x.shape: {self.x.shape}")
        print(f"y.shape: {self.y.shape}")

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        #         y_i = torch.FloatTensor(self.y[idx, :])
        #         x_i = torch.LongTensor(self.x[idx].tolist())

        y_i = torch.LongTensor([self.y[idx]])
        x_i = torch.LongTensor(self.x[idx].tolist())

        return {"x": x_i, "y": y_i}


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch in tqdm(testing_loader):
        batch_x = Variable(batch["x"])
        outputs = model(batch_x)
        batch_y = Variable(batch['y'].reshape(1, -1).squeeze(0))
        test_loss += binary_loss(outputs, batch_y).data

        prediction = outputs.data.max(1, keepdim=True)[1]
        label = batch['y'].data
        correct += prediction.eq(torch.LongTensor(label)).sum()

    test_loss /= len(testing_loader.dataset)
    accuracy = 100. * correct / len(testing_loader.dataset)
    print(f'Average Test loss: {test_loss.data[0]}')
    print(f'Accuracy: {accuracy}')

def train(epoch):
    print('-' * 10)
    print(f'Epoch: {epoch+1}')

    batch_num = 0
    for batch in training_loader:
        batch_num += 1
        # Get the inputs and wrap as Variables
        batch_x = Variable(batch["x"])
        # batch_y = Variable(batch["y"])
        batch_y = Variable(batch['y'].reshape(1, -1).squeeze(0))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(batch_x)

        loss = binary_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # batch test
        if batch_num % 10 == 0:
            prediction = outputs.data.max(1, keepdim=True)[1]
            label = batch['y'].data
            correct = prediction.eq(torch.LongTensor(label)).sum()
            train_acc = correct.float() / len(batch_x)
            print(f'batch:{batch_num}\ttrain_loss:{loss.data}\ttrain_acc:{train_acc}')




(x_train, y_train), (x_test, y_test) = lazy_load_imdb_data()
training_data = MyData(x_train, y_train)
testing_data = MyData(x_test, y_test)
training_loader = DataLoader(training_data, batch_size=200)
testing_loader = DataLoader(testing_data, batch_size=200)


embedding_matrix = create_glove_embeddings(EMBEDDING_DIMS,MAX_FEATURE)
model = TorchFastText(MAX_FEATURE, EMBEDDING_DIMS, SENTENCE_LEN)
model.embeds.weight.data.copy_(embedding_matrix)

#binary_loss = nn.BCELoss()
binary_loss = nn.CrossEntropyLoss()
# optimizer = Adagrad(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters())

n_epochs = 3

for i in range(n_epochs):
    train(i)
    test()
# test()