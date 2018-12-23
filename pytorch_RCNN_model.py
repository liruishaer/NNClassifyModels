#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-19 
# function: 

import numpy as np
from torch import nn
import torch
import torch as t
import time
import json

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))  # 默认名字

    def load(self, path, change_opt=True):
        print(path)
        data = t.load(path)
        if 'opt' in data:
            # old_opt_stats = self.opt.state_dict()
            if change_opt:
                self.opt.parse(data['opt'], print_=False)
                self.opt.embedding_path = None
                self.__init__(self.opt)
            # self.opt.parse(old_opt_stats,print_=False)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self.cuda()

    def save(self, name=None, new=False):
        prefix = 'checkpoints/' + self.model_name + '_' + self.opt.type_ + '_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix + name

        if new:
            data = {'opt': self.opt.state_dict(), 'd': self.state_dict()}
        else:
            data = self.state_dict()

        t.save(data, path)
        return path

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        ignored_params = list(map(id, self.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             self.parameters())
        if lr2 is None: lr2 = lr1 * 0.5
        optimizer = t.optim.Adam([
            dict(params=base_params, weight_decay=weight_decay, lr=lr1),
            {'params': self.encoder.parameters(), 'lr': lr2}
        ])
        return optimizer


def get_embedding_matrix(embedding_dim,vocab_size):
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
    f = open('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.%id.txt' % embedding_dim)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors in GloVe embedding' % len(embeddings_index))

    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix



KERNEL_SIZE = 3   # 单尺度卷积核
VOCAB_SIZE = 10000 # num of words
EMBEDDING_DIM = 100 # embedding大小
HIDDEN_SIZE = 256 #LSTM hidden size
NUM_LAYERS=2 #LSTM layers
SENTENCE_LEN = 300
KMAX_POOLING = 2# k
LINEAR_HIDDEN_SIZE = 2000 # 全连接层隐藏元数目
NUM_CLASSES = 2 # 类别
STATIC=False # 是否训练embedding
USE_GOLVE = True  # 是否使用预训练词向量

class RCNN():
    def __init__(self):
        super(RCNN, self).__init__()
        self.model_name = 'RCNN'

        # kernel_size = KERNEL_SIZE
        self.encoder = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

        self.content_lstm = nn.LSTM(input_size=EMBEDDING_DIM,
                                  hidden_size=HIDDEN_SIZE,
                                  num_layers=NUM_LAYERS,
                                  bias=True,
                                  batch_first=False,
                                  # dropout = 0.5,
                                  bidirectional=True
                                  )
        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels=HIDDEN_SIZE * 2 + EMBEDDING_DIM,
                      out_channels=SENTENCE_LEN,
                      kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(SENTENCE_LEN),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=SENTENCE_LEN,
                      out_channels=SENTENCE_LEN,
                      kernel_size=KERNEL_SIZE),
            nn.BatchNorm1d(SENTENCE_LEN),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size = (opt.title_seq_len - kernel_size + 1))
        )


        # self.dropout = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(KMAX_POOLING * SENTENCE_LEN, LINEAR_HIDDEN_SIZE),
            nn.BatchNorm1d(LINEAR_HIDDEN_SIZE),
            nn.ReLU(inplace=True),
            nn.Linear(LINEAR_HIDDEN_SIZE, NUM_CLASSES)
        )
        # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
        if USE_GOLVE:
            # ???self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))
            self.encoder.weight.data.copy_(t.from_numpy(get_embedding_matrix(EMBEDDING_DIM,VOCAB_SIZE)))

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)

        if STATIC:
            title.detach()
            content.detach()

        # title_out = self.title_lstm(title.permute(1, 0, 2))[0].permute(1, 2, 0)
        # title_em = title.permute(0, 2, 1)
        # title_out = t.cat((title_out, title_em), dim=1)

        content_out = self.content_lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)
        content_em = (content).permute(0, 2, 1)
        content_out = t.cat((content_out, content_em), dim=1)

        # title_conv_out = kmax_pooling(self.title_conv(title_out), 2, self.opt.kmax_pooling)
        content_conv_out = kmax_pooling(self.content_conv(content_out), 2, KMAX_POOLING)

        # conv_out = t.cat((title_conv_out, content_conv_out), dim=1)
        conv_out = content_conv_out
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits

    # def get_optimizer(self):
    #    return  t.optim.Adam([
    #             {'params': self.title_conv.parameters()},
    #             {'params': self.content_conv.parameters()},
    #             {'params': self.fc.parameters()},
    #             {'params': self.encoder.parameters(), 'lr': 5e-4}
    #         ], lr=self.opt.lr)
    # # end method forward


if __name__ == '__main__':
    m = RCNN()
    title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(title,content)
    print(o.size())
    print(o)