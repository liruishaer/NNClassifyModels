#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-24 
# function:


class FastTextConfig:
    label_size = 2
    vocab_size = 10000
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 20000
    decay_rate = 0.8
    num_sampled = 5
    sentence_len = 300
    embed_size = 100
    is_training = True
    num_epochs = 10
    validate_every = 1   # "Validate every validate_every epochs."
    use_embedding=True
    ckpt_dir = "models/tensor_FastText_checkpoint/"
    cache_path = "models/tensor_FastText_checkpoint/data_cache.pik"
    csv_dir = 'results/tensor_epoch_results/FastText/'


class TextCNNConfig:
    label_size = 2
    vocab_size = 10000
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 20000
    decay_rate = 0.8
    num_sampled = 5
    sentence_len = 300
    embed_size = 100
    is_training = True
    num_epochs = 10
    validate_every = 1
    use_embedding=True
    filter_sizes = [2,3,4]
    num_filters = 128
    dropout_keep_prob = 1
    cache_path = "TextCNN_tensor_checkpoint/data_cache.pik"
    ckpt_dir = "TextCNN_tensor_checkpoint/"
    csv_dir = 'results/tensor_epoch_results/TextCNN/'


class TextRNNConfig:
    label_size = 2
    vocab_size = 10000
    learning_rate = 0.01
    batch_size = 128
    decay_steps = 20000
    decay_rate = 0.8
    num_sampled = 5
    sentence_len = 300
    embed_size = 100
    is_training = True
    num_epochs = 10
    validate_every = 1
    use_embedding=True
    dropout_keep_prob = 0.5
    ckpt_dir = "TextRNN_tensor_checkpoint/"
    cache_path = "models/TextRNN_tensor_checkpoint/data_cache.pik"
    csv_dir = 'results/tensor_epoch_results/TextRNN/'


fastText_conf = FastTextConfig()
textCNN_conf = TextCNNConfig()
textRNN_conf = TextRNNConfig()