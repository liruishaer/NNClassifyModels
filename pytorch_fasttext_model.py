#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-17 
# function:

import torch.nn as nn
import torch.nn.functional as F

class TorchFastText(nn.Module):

    def __init__(self, max_features, embedding_dims, maxlen, num_classes=2):
        super(TorchFastText, self).__init__()
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.maxlen = maxlen
        self.num_classes = num_classes

        # self.embeds = nn.EmbeddingBag(max_features, embedding_dims)
        # self.linear = nn.Linear(self.embedding_dims, self.num_classes)

        self.embeds = nn.Embedding(max_features, embedding_dims)
        self.linear = nn.Linear(self.embedding_dims, self.num_classes)

    def forward(self, x):
        embedded_sentence = self.embeds(x)
        pooled = F.avg_pool2d(embedded_sentence, (embedded_sentence.shape[1], 1)).squeeze(1)
        predicted = self.linear(pooled)

        return predicted


