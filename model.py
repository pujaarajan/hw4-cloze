import torch
import torch.nn as nn
from torch.autograd import Variable
import math

import torch.nn.functional as functional
class OurSoftmax(nn.Module):
  def __init__(self):
    super(OurSoftmax, self).__init__()

  def forward(self, x):
    # x = torch.exp(x)
    # Z = torch.sum(x, 0)
    # x = torch.div(x, Z)
    # x = torch.log(x)

    return functional.log_softmax(x)

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
        combined = Variable(torch.cat((input.data, hidden.data), 1), requires_grad=True) # concatenate
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

        hidden = Variable(torch.randn(batch_size, self.hidden_size), requires_grad=True)
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

# TODO: Your implementation goes here
class BiRNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(BiRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = 16

        self.W_ih = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size + self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init_params()

    def forward(self, input, hidden):
        combined = Variable(torch.cat((input.data, hidden.data), 1), requires_grad=True) # concatenate
        hidden = combined.matmul(self.W_ih.t()) + self.b_ih
        hidden = torch.tanh(hidden)
        return hidden

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class BiRNNLM(nn.Module):
    def __init__(self, vocab_size):
        super(BiRNNLM, self).__init__()
        self.embedding_size = 32 # arbitrary dimension
        self.hidden_size = 16
        self.vocab_size = vocab_size

        self.embedding = nn.Parameter(torch.randn(self.vocab_size, self.embedding_size))  # random word embedding
        self.rnnLR = BiRNN(self.embedding_size, self.vocab_size)
        self.rnnRL = BiRNN(self.embedding_size, self.vocab_size)
        self.W_ho = nn.Parameter(torch.Tensor(self.vocab_size, self.hidden_size * 2))
        self.b_ho = nn.Parameter(torch.Tensor(self.vocab_size))

        self.softmax = nn.LogSoftmax()

        self.init_params()

    def forward(self, input_batch):
        seq_len, batch_size = input_batch.size()
        print('seq_len: ', seq_len)
        print('batch_size: ', batch_size)
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)
        hLR = [Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)]
        hRL = [Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)]

        for t in xrange(seq_len - 1):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            hidden = self.rnnLR(w, hLR[t]) #
            hLR.append(hidden)

        for t in xrange(seq_len - 1, 0, -1):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            hidden = self.rnnRL(w, hRL[seq_len - t - 1]) #
            hRL.append(hidden)

        for i in range(len(hLR)):
            j = len(hLR) - 1 - i
            concatHidden = Variable(torch.cat((hLR[i].data, hRL[j].data), 1))
            output = concatHidden.matmul(self.W_ho.t()) + self.b_ho
            output = self.softmax(output)
            predictions[i,:,:] = output

        return predictions

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
