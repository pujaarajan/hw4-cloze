import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class LSTMLM(nn.Module):
    def __init__(self, vocab_size):
        super(LSTMLM, self).__init__()

        self.embedding_size = 32 # arbitrary dimension
        self.hidden_size = 16
        self.vocab_size = vocab_size
        self.embedding = nn.Parameter(torch.randn(vocab_size, self.embedding_size))  # random word embeddin
        self.lstm = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()
        self.init_params()

    def forward(self, input_batch):
        ## input_batch of size (seq_len, batch_size)
        seq_len, batch_size = input_batch.size()
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)

        hidden = Variable(torch.randn(batch_size, self.hidden_size), requires_grad=True)
        C_prev = Variable(torch.randn(batch_size, self.hidden_size), requires_grad=True)
        for t in xrange(seq_len):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            hidden, C_prev = self.lstm(w, (hidden, C_prev))
            output = self.softmax(self.h2o(hidden))
            predictions[t,:,:] = output
        return predictions

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



class BiLSTMLM(nn.Module):
    def __init__(self, vocab_size):
        super(BiLSTMLM, self).__init__()

        self.embedding_size = 32 # arbitrary dimension
        self.hidden_size = 16
        self.vocab_size = vocab_size
        self.embedding = nn.Parameter(torch.randn(vocab_size, self.embedding_size))  # random word embeddin
        self.lstm = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size + self.hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax()
        self.init_params()

    def forward(self, input_batch):
        ## input_batch of size (seq_len, batch_size)
        seq_len, batch_size = input_batch.size()
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)

        C_prevRL = Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)
        C_prevLR = Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)
        hLR = [Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)] 
        hRL = [Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)]

        for t in xrange(seq_len-1):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            hidden, C_prevLR = self.lstm(w, (hLR[t], C_prevLR))
            hLR.append(hidden)

        for t in xrange(seq_len - 1, 0, -1):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            hidden, C_prevRL = self.lstm(w, (hRL[seq_len - t - 1], C_prevRL)) #
            hRL.append(hidden)

        for i in range(len(hLR)):
            j = len(hLR) - 1 - i
            concatHidden = Variable(torch.cat((hLR[i].data, hRL[j].data), 1))
            output = self.softmax(self.h2o(concatHidden))
            predictions[i,:,:] = output
        return predictions     

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)