#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-26 
# function: 
import torch.nn as nn
import torch


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.embeds = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        # self.rnn = nn.RNN(embedding_length, hidden_size, num_layers=2, bidirectional=True, dropout=dropout )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0)
        # x = [sent len, batch size]
        embedded = self.dropout(self.embeds(x))

        # embedded = [sent len, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded)

        # output = [sent len, batch size, hid dim * num directions]
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden.squeeze(0))