#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-19
# function:


import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def load_imdb_ori_data(path='imdb.npz', seed=113):
    # 1. load data
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    # 2. concatenate
    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    # 3. shuffle data
    np.random.seed(seed)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    xs = xs[indices]
    labels = labels[indices]

    return xs,labels


def split_dataset(xs,ys,train_size,test_size):
    sss = StratifiedShuffleSplit(n_splits=2,train_size=train_size,test_size=test_size,random_state=521)
    for train_idx,test_idx in sss.split(xs,ys):
        x_train,x_test = xs[train_idx],xs[test_idx]
        y_train,y_test = ys[train_idx],ys[test_idx]
        print('x_train:',len(x_train))
        print('x_test:',len(x_test))
        print('*******')

    return x_train,x_test,y_train,y_test


xs,ys = load_imdb_ori_data()


train_size=40000
test_size=10000
x_train,x_test,y_train,y_test = split_dataset(xs,ys,train_size=train_size,test_size=test_size)
np.savez(f'imdb_npz_data/imdb_train{train_size}_test{test_size}',
         x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test)


# path = f'imdb_npz_data/imdb_train{10000}_test{10000}.npz'
# with np.load(path) as f:
#     x_train, labels_train = f['x_train'], f['y_train']
#     x_test, labels_test = f['x_test'], f['y_test']
#     print(len(x_test))
#     print(len(x_train))
