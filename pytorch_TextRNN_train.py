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
from pytorch_TextRNN_model2 import TextRNN
# from pytorch_TextRNN_model import TextRNN


MAX_FEATURE = 10000
SENTENCE_LEN = 300
EMBEDDING_DIMS = 100
BATCH_SIZE = 400
NGRAME_RANGE = 1
N_FILTERS = 100
FILTER_SIZES = [3,4,5]
OUTPUT_DIM = 2
HIDDEN_DIM = 256
DROPOUT = 0.5
N_LAYERS = 2
BIDIRECTIONAL = True


USE_CUDA = torch.cuda.is_available()
# gpus = [0]
# torch.cuda.set_device(gpus[0])




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
        # test_loss += binary_loss(outputs, batch_y)
        test_loss += binary_loss(outputs, batch_y).data

        prediction = outputs.data.max(1, keepdim=True)[1]
        # label = batch_y.data.max(1, keepdim=True)[1]
        label = batch['y'].data
        correct += prediction.eq(torch.LongTensor(label)).sum()

    test_loss /= len(testing_loader.dataset)
    accuracy = 100. * correct / len(testing_loader.dataset)
    print(f'Average Test loss: {test_loss}')
    print(f'Accuracy: {accuracy}')

def train(epoch):
    model.train()
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

        # print("outputs.shape:",outputs.shape)
        # print("batch_y.shape:",batch_y.shape)
        # print("outputs",outputs)
        # print("batch_y",batch_y)

        loss = binary_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # batch test
        prediction = outputs.data.max(1, keepdim=True)[1]
        label = batch['y'].data
        correct = prediction.eq(torch.LongTensor(label)).sum()
        train_acc = correct / len(batch_x)
        print('batch:',batch_num,'\ttrain_loss:',loss.data,'\ttrain_acc:',train_acc)


(x_train, y_train), (x_test, y_test) = lazy_load_imdb_data(ngram_range=1, max_features=MAX_FEATURE, sentence_len=SENTENCE_LEN)
training_data = MyData(x_train, y_train)
testing_data = MyData(x_test, y_test)
training_loader = DataLoader(training_data, batch_size=BATCH_SIZE)
testing_loader = DataLoader(testing_data, batch_size=BATCH_SIZE)


embedding_matrix = create_glove_embeddings(EMBEDDING_DIMS,MAX_FEATURE)
model = TextRNN(MAX_FEATURE,EMBEDDING_DIMS,HIDDEN_DIM,OUTPUT_DIM,N_LAYERS,BIDIRECTIONAL,DROPOUT)
# model = TextRNN(BATCH_SIZE,OUTPUT_DIM,HIDDEN_DIM,MAX_FEATURE,EMBEDDING_DIMS)   # model2
model.embeds.weight.data.copy_(embedding_matrix)




# if USE_CUDA:
#     model = model.cuda()

binary_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

n_epochs = 1
for i in range(n_epochs):
    train(i)
    test()
# test()
