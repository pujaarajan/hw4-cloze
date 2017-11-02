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

        self.sigmoid = torch.nn.Sigmoid()

        self.init_params()

    def forward(self, input, ttuple):
        hidden, C_previous_t = ttuple
        combined = Variable(torch.cat((input.data, hidden.data), 1), requires_grad=True) # concatenate

        f_t = self.sigmoid(combined.matmul(self.W_f.t()) + self.b_f)
        i_t = self.sigmoid(combined.matmul(self.W_i.t()) + self.b_i)
        C_tilde_t = torch.tanh(combined.matmul(self.W_C.t()) + self.b_C)
        
        C_t = f_t * C_previous_t + i_t * C_tilde_t

        o_t = self.sigmoid(combined.matmul(self.W_o.t()) + self.b_o)

        h_t = o_t * torch.tanh(C_t)

        return h_t, C_t 

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)



# class LSTMLM(nn.Module):
#     def __init__(self, vocab_size):
#         super(LSTMLM, self).__init__()

#         self.embedding_size = 32 # arbitrary dimension
#         self.hidden_size = 16
#         self.vocab_size = vocab_size
#         self.embedding = nn.Parameter(torch.randn(vocab_size, self.embedding_size))  # random word embeddin
#         self.lstm = nn.LSTMCell(self.embedding_size, self.hidden_size)
#         self.h2o = nn.Linear(self.hidden_size, vocab_size)
#         self.softmax = nn.LogSoftmax()

#         self.initial_hidden = nn.Parameter(torch.Tensor(1, self.hidden_size))
#         self.initial_C = nn.Parameter(torch.Tensor(1, self.hidden_size))
#         self.init_params()

#     def forward(self, input_batch):
#         ## input_batch of size (seq_len, batch_size)
#         seq_len, batch_size = input_batch.size()
#         predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)

#         hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size))
#         C_prev = Variable(self.initial_C.data.expand(batch_size, self.hidden_size))
        

#         for t in xrange(seq_len):
#             word_ix = input_batch[t, :]
#             w = self.embedding[word_ix.data, :]
#             hidden, C_prev = self.lstm(w, (hidden, C_prev))
#             output = self.softmax(self.h2o(hidden))
#             predictions[t,:,:] = output
#         return predictions

#     def init_params(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)



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

        self.initial_hidden = nn.Parameter(torch.Tensor(1,self.hidden_size))
        self.initial_C =  nn.Parameter(torch.Tensor(1,self.hidden_size))

    def forward(self, input_batch):
        ## input_batch of size (seq_len, batch_size)
        seq_len, batch_size = input_batch.size()
        predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size), requires_grad=False)


        hLR = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size), requires_grad=False)
        hRL = Variable(torch.rand(seq_len + 1, batch_size, self.hidden_size), requires_grad=False)
        
        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size))        
        hLR[0,:,:] = hidden
        C_prevLR = Variable(self.initial_C.data.expand(batch_size, self.hidden_size))        
        
        for t in xrange(seq_len):
            word_ix = input_batch[t, :]
            w = self.embedding[word_ix.data, :]
            hidden, C_prevLR = self.lstm(w, (hidden, C_prevLR))
            hLR[t+1,:,:] = hidden

        hidden = Variable(self.initial_hidden.data.expand(batch_size, self.hidden_size))        
        hRL[seq_len,:,:] = hidden
        C_prevRL = Variable(self.initial_C.data.expand(batch_size, self.hidden_size))        

        for t in xrange(seq_len, 0, -1):
            word_ix = input_batch[t-1, :]
            w = self.embedding[word_ix.data, :]
            hidden, C_prevRL = self.lstm(w, (hidden, C_prevRL)) #
            hRL[t-1,:,:] = hidden
        
        for i in xrange(seq_len):
            j = i+1
            concatHidden = torch.cat((hLR[i,:,:], hRL[j,:,:]), 1)
            output = self.softmax(self.h2o(concatHidden))
            predictions[i,:,:] = output
        return predictions     

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)