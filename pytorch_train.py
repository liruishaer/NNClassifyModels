#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-18
# function:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pytorch_imdb_datahandle import load_imdb_data
from pytorch_fasttext_model import TorchFastText
from pytorch_TextCNN_model import TextCNN
from pytorch_TextRNN_model import TextRNN
from pytorch_config import fastText_conf,textCNN_conf,textRNN_conf
import time
import csv
import os


MODEL = 'fasttext'   # value:  fasttext   textcnn   textrnn
TRAIN_DATA_SIZE = 10000   # value:  10000      20000     30000     40000

model_conf = None
if MODEL=='fasttext':
    model_conf = fastText_conf
    model = TorchFastText(model_conf.vocab_size, model_conf.embedding_dims, model_conf.sentence_len)
    file_epoch_result_csv = model_conf.csv_dir + f'fastText-pth-gpu-train-{TRAIN_DATA_SIZE}-epoch-{model_conf.num_epochs}-batch-{model_conf.batch_size}.csv'
elif MODEL=='textcnn':
    model_conf = textCNN_conf
    model = TextCNN(model_conf.vocab_size, model_conf.embedding_dims, model_conf.filter_num,
                    model_conf.filter_sizes, model_conf.label_size, model_conf.dropout_prob)
    file_epoch_result_csv = model_conf.csv_dir + f'TextCNN-pth-gpu-train-{TRAIN_DATA_SIZE}-epoch-{model_conf.num_epochs}-batch-{model_conf.batch_size}.csv'
elif MODEL=='textrnn':
    model_conf = textRNN_conf
    model = TextRNN(model_conf.batch_size, model_conf.label_size, model_conf.hidden_dim,
                    model_conf.vocab_size, model_conf.embedding_dims)
    file_epoch_result_csv = model_conf.csv_dir + f'TextRNN-pth-gpu-train-{TRAIN_DATA_SIZE}-epoch-{model_conf.num_epochs}-batch-{model_conf.batch_size}.csv'


def create_embedding_matrix(embedding_dims, vocab_size):
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

    print('Loading pretrained embeddings...')

    embeddings_index = {}
    f = open('/liruishaer/Work2/NLP_models/glove.6B/glove.6B.%id.txt' % embedding_dims)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors in GloVe embedding' % len(embeddings_index))

    embedding_matrix = np.zeros((vocab_size, embedding_dims))

    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

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
    loss = 0
    correct = 0
    counter = 0
    for batch in tqdm(testing_loader):
        batch_x = Variable(batch["x"])
        batch_y = Variable(batch['y'].reshape(1, -1).squeeze(0))
        outputs = model(batch_x)

        prediction = outputs.data.max(1, keepdim=True)[1]
        label = batch['y'].data

        loss += binary_loss(outputs, batch_y).item()
        correct += prediction.eq(torch.LongTensor(label)).sum().item()
        counter += 1

    test_loss = loss / counter
    test_acc = 1.0 * correct / len(testing_data)

    return test_loss, test_acc

def train(epoch):
    model.train()
    print('-' * 10)
    print(f'Epoch: {epoch+1}')

    batch_num = 0
    train_loss = 0
    train_correct = 0
    counter = 0

    for batch in training_loader:
        batch_num += 1
        batch_x = Variable(batch["x"])
        batch_y = Variable(batch['y'].reshape(1, -1).squeeze(0))

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = binary_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()

        prediction = outputs.data.max(1, keepdim=True)[1]
        label = batch['y'].data
        counter += 1
        train_loss += loss.item()
        # print('loss:',loss.item())
        train_correct += prediction.eq(torch.LongTensor(label)).sum().item()

        # batch test
        # if batch_num % 1 == 0:
        #     prediction = outputs.data.max(1, keepdim=True)[1]
        #     label = batch['y'].data
        #     correct = prediction.eq(torch.LongTensor(label)).sum()
        #     train_acc = correct.float() / len(batch_x)
        #     print(f'batch:{batch_num}\ttrain_loss:{loss.data}\ttrain_acc:{train_acc}')

    train_loss = train_loss / counter
    train_acc = train_correct / len(training_data)

    return train_loss, train_acc


# 加载数据
(x_train, y_train), (x_test, y_test) = load_imdb_data(TRAIN_DATA_SIZE, vocab_size=model_conf.vocab_size, sentence_len=model_conf.sentence_len)
training_data = MyData(x_train, y_train)
testing_data = MyData(x_test, y_test)
training_loader = DataLoader(training_data, batch_size=model_conf.batch_size)
testing_loader = DataLoader(testing_data, batch_size=model_conf.batch_size)

# 加载预训练词向量
embedding_matrix = create_embedding_matrix(model_conf.embedding_dims, model_conf.vocab_size)
# model = TorchFastText(MAX_FEATURE, EMBEDDING_DIMS, SENTENCE_LEN)
model.embeds.weight.data.copy_(embedding_matrix)

# 定义损失函数、优化器
binary_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=model_conf.learning_rate)

total_time = 0  # 记录训练时间
csv_content_list = [['loss', 'acc', 'time', 'val_loss', 'val_acc']]
best_test_acc = 0
for epoch_id in range(model_conf.num_epochs):
    start_time = time.time()
    # 训练
    train_loss, train_acc = train(epoch_id)
    total_time += time.time() - start_time
    # 测试
    val_loss, val_acc = test()
    # 结果记录
    csv_content_list.append([train_loss, train_acc, total_time, val_loss, val_acc])
    # 保存模型
    if val_acc > best_test_acc:
        if not os.path.isdir(model_conf.ckpt_dir):
            os.makedirs(model_conf.ckpt_dir)
        filename = model_conf.ckpt_dir  + f'model.ckpt-pytorch_train_size_{TRAIN_DATA_SIZE}' \
                  f'_gpu_epoch_{epoch_id}_batch_{model_conf.batch_size}_acc_{val_acc:.4f}.pth'
        torch.save(model, filename)
        best_test_acc = val_acc


if not os.path.exists(model_conf.csv_dir):
    os.mkdir(model_conf.csv_dir)

with open(file_epoch_result_csv, 'w') as f:
    w = csv.writer(f)
    for ls in csv_content_list:
        w.writerow(ls)