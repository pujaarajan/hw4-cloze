import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class LSTM_Cell(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM_Cell, self).__init__()

        self.input_size = input_size
        self.hidden_size = 16
        self.output_size = output_size

        self.W_f = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size + self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))

        self.W_i = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size + self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))

        self.W_C = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size + self.hidden_size))
        self.b_C = nn.Parameter(torch.Tensor(self.hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(self.hidden_size, self.input_size + self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_size))

        self.W_h2o = nn.Parameter(torch.Tensor(self.output_size, self.hidden_size))
        self.b_h2o = nn.Parameter(torch.Tensor(self.output_size))        

        self.softmax = torch.nn.LogSoftmax()
        self.sigmoid = torch.nn.Sigmoid()

        self.init_params()

    def forward(self, input, hidden, C_previous_t):
        combined = Variable(torch.cat((input.data, hidden.data), 1), requires_grad=True) # concatenate

        f_t = self.sigmoid(combined.matmul(self.W_f.t()) + self.b_f)
        i_t = self.sigmoid(combined.matmul(self.W_i.t()) + self.b_i)

        C_tilde_t = torch.tanh(combined.matmul(self.W_C.t()) + self.b_C)
        C_t = f_t.matmul(C_previous_t.t()) + i_t.matmul(C_tilde_t.t())

        o_t = self.sigmoid(combined.matmul(self.W_o.t()) + self.b_o)
        h_t = o_t.matmul(torch.tanh(C_t))

        output = self.softmax(h_t.matmul(self.W_h2o.t()) + self.b_h2o)

        return output, h_t, C_t 

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

class LSTMLM(nn.Module):
    def __init__(self, vocab_size):
        super(LSTMLM, self).__init__()

        embedding_size = 24 # arbitrary dimension

        self.hidden_size = 16
        self.vocab_size = vocab_size
        self.embedding = nn.Parameter(torch.randn(vocab_size, embedding_size))  # random word embedding
        self.lstm = LSTM_Cell(embedding_size, vocab_size)

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
            output, hidden, C_prev = self.lstm(w, hidden, C_prev)
            predictions[t,:,:] = output
        return predictions

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)




