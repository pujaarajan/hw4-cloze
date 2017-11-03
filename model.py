import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np


# create a RNN. this is not the final class
class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = 16
        self.output_size = output_size

        self.W_ih = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size + self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))
        self.b_ho = nn.Parameter(torch.Tensor(self.output_size))
        self.softmax = torch.nn.LogSoftmax()

        self.init_params()

    def forward(self, input, hidden):
        combined = Variable(torch.cat((input.data, hidden.data), 1), requires_grad=False) # concatenate
        hidden = combined.matmul(self.W_ih.t()) + self.b_ih
        hidden = torch.tanh(hidden)
        output = hidden.matmul(self.W_ho.t()) + self.b_ho
        output = self.softmax(output)
        return output, hidden

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

# this is the final class that will use RNN
class RNNLM(nn.Module):
    def __init__(self, vocab_size):
        super(RNNLM, self).__init__()

        embedding_size = 32 # arbitrary dimension

        self.hidden_size = 16
        self.vocab_size = vocab_size
        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_size))  # random word embedding
        self.rnn = RNN(embedding_size, vocab_size)

        self.init_params()

    def forward(self, input_batch):
        ## input_batch of size (seq_len, batch_size)
        seq_len, batch_size = input_batch.size()
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)

        hidden = Variable(torch.randn(batch_size, self.hidden_size), requires_grad=False)
        for t in xrange(seq_len):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            output, hidden = self.rnn(w, hidden)
            predictions[t,:,:] = output
        return predictions

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class BiRNNLM(nn.Module):
    def __init__(self, vocab_size):
        super(BiRNNLM, self).__init__()
        # self.embedding_size = 32 # arbitrary dimension
        # self.hidden_size = 16
        self.embedding_size = 150 # arbitrary dimension
        self.hidden_size = 30
        self.vocab_size = vocab_size

        self.W_ih_lr = nn.Parameter(torch.Tensor(self.embedding_size + self.hidden_size, self.hidden_size))
        self.b_ih_lr = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.W_ih_rl = nn.Parameter(torch.Tensor(self.embedding_size + self.hidden_size, self.hidden_size))
        self.b_ih_rl = nn.Parameter(torch.Tensor(1, self.hidden_size))

        self.embedding = nn.Parameter(torch.randn(self.vocab_size, self.embedding_size))  # random word embedding
        self.W_ho = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.vocab_size))
        self.b_ho = nn.Parameter(torch.Tensor(1, self.vocab_size))

        self.softmax = nn.LogSoftmax()

        self.initial_hidden = nn.Parameter(torch.Tensor(1, self.hidden_size))

        self.init_params()

    def forward(self, input_batch):
        seq_len, batch_size = input_batch.size()
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)
        hLR = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size), requires_grad=False)
        hRL = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size), requires_grad=False)

        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size))
        hLR[0,:,:] = hidden

        for t in xrange(seq_len):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            combined = torch.cat((w, hidden), 1)
            hidden = combined.matmul(self.W_ih_lr)
            hidden = hidden + self.b_ih_lr
            hidden = torch.tanh(hidden)
            hLR[t+1,:,:] = hidden

        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size))
        hRL[seq_len,:,:] = hidden

        for t in xrange(seq_len, 0, -1):
            word_ix = input_batch[t-1, :]
            w = self.embedding[word_ix.data, :]
            combined = torch.cat((w, hidden), 1)
            hidden = combined.matmul(self.W_ih_rl)
            hidden = hidden + self.b_ih_rl
            hidden = torch.tanh(hidden)
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

class BiRNNLMwithDropout(nn.Module):
    def __init__(self, vocab_size):
        super(BiRNNLMwithDropout, self).__init__()
        self.embedding_size = 32 # arbitrary dimension
        self.hidden_size = 16
        # self.embedding_size = 150 # arbitrary dimension
        # self.hidden_size = 30
        self.vocab_size = vocab_size

        self.W_ih_lr = nn.Parameter(torch.Tensor(self.embedding_size + self.hidden_size, self.hidden_size))
        self.b_ih_lr = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.W_ih_rl = nn.Parameter(torch.Tensor(self.embedding_size + self.hidden_size, self.hidden_size))
        self.b_ih_rl = nn.Parameter(torch.Tensor(1, self.hidden_size))

        self.embedding = nn.Parameter(torch.randn(self.vocab_size, self.embedding_size))  # random word embedding
        self.W_ho = nn.Parameter(torch.Tensor(self.hidden_size * 2, self.vocab_size))
        self.b_ho = nn.Parameter(torch.Tensor(1, self.vocab_size))

        self.dropout_percent = 0.4

        self.softmax = nn.LogSoftmax()

        self.initial_hidden = nn.Parameter(torch.Tensor(1, self.hidden_size))

        self.init_params()

    def forward(self, input_batch, withDropout = True):
        seq_len, batch_size = input_batch.size()
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)
        hLR = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size), requires_grad=False)
        hRL = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size), requires_grad=False)

        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size))
        hLR[0,:,:] = hidden

        for t in xrange(seq_len):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            combined = torch.cat((w, hidden), 1)
            hidden = combined.matmul(self.W_ih_lr)
            hidden = hidden + self.b_ih_lr
            hidden = torch.tanh(hidden)
            if withDropout:
                mask = Variable(torch.Tensor(np.random.binomial(np.ones(hidden.size(), dtype='int64'), 1-self.dropout_percent)))
                hidden = torch.mul(hidden, mask)
                hidden = torch.mul(hidden, 1.0 / (1 - self.dropout_percent))
            hLR[t+1,:,:] = hidden

        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size))
        hRL[seq_len,:,:] = hidden

        for t in xrange(seq_len, 0, -1):
            word_ix = input_batch[t-1, :]
            w = self.embedding[word_ix.data, :]
            combined = torch.cat((w, hidden), 1)
            hidden = combined.matmul(self.W_ih_rl)
            hidden = hidden + self.b_ih_rl
            hidden = torch.tanh(hidden)
            if withDropout:
                mask = Variable(torch.Tensor(np.random.binomial(np.ones(hidden.size(), dtype='int64'), 1-self.dropout_percent)))
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
