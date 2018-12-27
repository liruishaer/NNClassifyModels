#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-25 
# function: 

class FastTextConfig:
    label_size = 2
    vocab_size = 10000
    sentence_len = 300
    ngrame_range = 1
    embedding_dims = 100
    num_epochs = 5
    batch_size = 200
    learning_rate = 0.01
    ckpt_dir = "pytorch/models/FastText_checkpoint/"
    csv_dir = 'pytorch/results/FastText/'




class TextCNNConfig:
    label_size = 2
    vocab_size = 10000
    sentence_len = 300
    embedding_dims = 100
    batch_size = 400
    filter_num = 100
    filter_sizes = [3,4,5]
    dropout_prob = 0.5
    num_epochs = 5
    learning_rate = 0.01
    ckpt_dir = "pytorch/models/TextCNN_checkpoint/"
    csv_dir = 'pytorch/results/TextCNN/'



class TextRNNConfig:
    label_size = 2
    vocab_size = 10000
    sentence_len = 300
    embedding_dims = 100
    batch_size = 400
    filter_num = 100
    filter_sizes = [3, 4, 5]
    hidden_dim = 256
    learning_rate = 0.01
    dropout_prob = 0.5
    layer_num = 2
    bidirectional = True
    num_epochs=5
    ckpt_dir = "pytorch/models/TextRNN_checkpoint/"
    csv_dir = 'pytorch/results/TextRNN/'




fastText_conf = FastTextConfig()
textCNN_conf = TextCNNConfig()
textRNN_conf = TextRNNConfig()