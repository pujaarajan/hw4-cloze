
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from torch import cuda

class BiRNNLMwithDropout(nn.Module):
    def __init__(self, vocab_size):
        super(BiRNNLMwithDropout, self).__init__()
        self.embedding_size = 32 # arbitrary dimension
        self.hidden_size = 16
        # self.embedding_size = 150 # arbitrary dimension
        # self.hidden_size = 30
        self.vocab_size = vocab_size

        self.W_ih_lr = nn.Parameter(torch.Tensor(self.embedding_size + self.hidden_size, self.hidden_size).cuda())
        self.b_ih_lr = nn.Parameter(torch.Tensor(1, self.hidden_size).cuda())
        self.W_ih_rl = nn.Parameter(torch.Tensor(self.embedding_size + self.hidden_size, self.hidden_size).cuda())
        self.b_ih_rl = nn.Parameter(torch.Tensor(1, self.hidden_size).cuda())

        self.embedding = nn.Parameter(torch.randn(self.vocab_size, self.embedding_size).cuda())  # random word embedding
        self.W_ho = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.vocab_size).cuda())
        self.b_ho = nn.Parameter(torch.Tensor(1, self.vocab_size).cuda())

        self.dropout_percent = 0.2

        self.softmax = nn.LogSoftmax().cuda()

        self.initial_hidden = nn.Parameter(torch.Tensor(1, self.hidden_size).cuda())

        self.init_params()

    def forward(self, input_batch, withDropout = True):
        seq_len, batch_size = input_batch.size()
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size).cuda(), requires_grad=False)
        hLR = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size).cuda(), requires_grad=False)
        hRL = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size).cuda(), requires_grad=False)

        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size).cuda())
        hLR[0,:,:] = hidden

        for t in xrange(seq_len):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            combined = torch.cat((w, hidden), 1)
            hidden = combined.matmul(self.W_ih_lr)
            hidden = hidden + self.b_ih_lr
            hidden = torch.tanh(hidden)
            if withDropout:
                mask = Variable(torch.Tensor(np.random.binomial(np.ones(hidden.size(), dtype='int64'), 1-self.dropout_percent)).cuda())
                hidden = torch.mul(hidden, mask)
                hidden = torch.mul(hidden, 1.0 / (1 - self.dropout_percent))
            hLR[t+1,:,:] = hidden

        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size).cuda())
        hRL[seq_len,:,:] = hidden

        for t in xrange(seq_len, 0, -1):
            word_ix = input_batch[t-1, :]
            w = self.embedding[word_ix.data, :]
            combined = torch.cat((w, hidden), 1)
            hidden = combined.matmul(self.W_ih_rl)
            hidden = hidden + self.b_ih_rl
            hidden = torch.tanh(hidden)
            if withDropout:
                mask = Variable(torch.Tensor(np.random.binomial(np.ones(hidden.size(), dtype='int64'), 1-self.dropout_percent)).cuda())
                hidden = torch.mul(hidden, mask)
                hidden = torch.mul(hidden, 1.0 / (1 - self.dropout_percent))
            hRL[t-1,:,:] = hidden

        for i in xrange(seq_len):
            j = i + 1
            concatHidden = torch.cat((hLR[i,:,:], hRL[j,:,:]), 1)
            output = concatHidden.matmul(self.W_ho) + self.b_ho
            output = self.softmax(output)
            predictions[i,:,:] = output

        return predictions

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
