#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author: lirui
# datetime:18-12-19 
# function: 

import numpy as np
from torch import nn
import torch as t
import time

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


class RCNN(BasicModule):
    def __init__(self, opt):
        super(RCNN, self).__init__()
        self.model_name = 'RCNN'
        self.opt = opt

        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)

        self.title_lstm = nn.LSTM(input_size=opt.embedding_dim, \
                                  hidden_size=opt.hidden_size,
                                  num_layers=opt.num_layers,

                                  bias=True,
                                  batch_first=False,
                                  # dropout = 0.5,
                                  bidirectional=True
                                  )
        self.title_conv = nn.Sequential(
            nn.Conv1d(in_channels=opt.hidden_size * 2 + opt.embedding_dim,
                      out_channels=opt.title_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(opt.title_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=opt.title_dim,
                      out_channels=opt.title_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(opt.title_dim),
            nn.ReLU(inplace=True),
            # nn.MaxPool1d(kernel_size = (opt.title_seq_len - kernel_size + 1))
        )

        self.content_lstm = nn.LSTM(input_size=opt.embedding_dim, \
                                    hidden_size=opt.hidden_size,
                                    num_layers=opt.num_layers,
                                    bias=True,
                                    batch_first=False,
                                    # dropout = 0.5,
                                    bidirectional=True
                                    )

        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels=opt.hidden_size * 2 + opt.embedding_dim,
                      out_channels=opt.content_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=opt.content_dim,
                      out_channels=opt.content_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(opt.content_dim),
            nn.ReLU(inplace=True),

            # nn.MaxPool1d(kernel_size = (opt.content_seq_len - opt.kernel_size + 1))
        )
        # self.dropout = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling * (opt.title_dim + opt.content_dim), opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size, opt.num_classes)
        )
        # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)

        if self.opt.static:
            title.detach()
            content.detach()

        title_out = self.title_lstm(title.permute(1, 0, 2))[0].permute(1, 2, 0)
        title_em = title.permute(0, 2, 1)
        title_out = t.cat((title_out, title_em), dim=1)

        content_out = self.content_lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)
        content_em = (content).permute(0, 2, 1)
        content_out = t.cat((content_out, content_em), dim=1)

        title_conv_out = kmax_pooling(self.title_conv(title_out), 2, self.opt.kmax_pooling)
        content_conv_out = kmax_pooling(self.content_conv(content_out), 2, self.opt.kmax_pooling)

        conv_out = t.cat((title_conv_out, content_conv_out), dim=1)
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


# if __name__ == '__main__':
#     from ..config import opt
#     m = CNNText(opt)
#     title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
#     content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
#     o = m(title,content)
#     print(o.size())