import torch
import torch.nn as nn
from torch.autograd import Variable

# since we cannot use Linear we create our own class
class FCLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(FCLayer, self).__init__()
    self.W = nn.Parameter(torch.rand(input_size, output_size))
    self.b = nn.Parameter(torch.rand(output_size))
  def forward(self, x):
    return self.W * x + self.b

class Softie(nn.Module):
  def __init__(self, input_size):
    self.input_size = input_size

  def forward(self, x):
    pass

# create a RNN. this is not the final class
class RNN(nn.Module):
    def __init__(self, input_size, output_size):
      super(RNN, self).__init__()

      hidden_size = 16 # can be arbitrary
      self.hidden_size = hidden_size

      self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
      self.i2o = nn.Linear(input_size + hidden_size, output_size)
      self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
      #print type(input), type(hidden), type(input.data), type(hidden.data)
      combined = Variable(torch.cat((input.data, hidden.data), 1), requires_grad=True) # concatenate
      hidden = self.i2h(combined)
      output = self.i2o(combined)
      output = self.softmax(output)
      return output, hidden

# this is the final class that will use RNN
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()

    embedding_size = 32 # arbitrary dimension
    self.hidden_size = 16
    self.vocab_size = vocab_size
    self.embedding = torch.rand(vocab_size, embedding_size)  # random word embedding
    self.rnn = RNN(embedding_size, vocab_size)



  def forward(self, input_batch):
    ## input_batch of size (seq_len, batch_size)
    seq_len, batch_size = input_batch.size()
    predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size))

    hLR = [Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)]
    for t in xrange(seq_len):
      word_ix = input_batch[t, :]
      w = Variable(self.embedding[word_ix.data, :], requires_grad=True)
      output, hidden = self.rnn(w, hLR[t]) #
      hLR.append(hidden)
      predictions[t,:,:] = output

    return predictions




# TODO: Your implementation goes here


class NewRNN(nn.Module):
    def __init__(self, input_size, output_size):
      super(NewRNN, self).__init__()

      hidden_size = 16 # can be arbitrary
      self.hidden_size = hidden_size
      self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
      combined = Variable(torch.cat((input.data, hidden.data), 1), requires_grad=True) # concatenate
      hidden = self.i2h(combined)
      return hidden

class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    self.embedding_size = 32 # arbitrary dimension
    self.hidden_size = 16
    self.vocab_size = vocab_size
    self.embedding = torch.rand(self.vocab_size, self.embedding_size)  # random word embedding
    self.lay = nn.Linear(self.hidden_size + self.hidden_size, self.vocab_size)
    self.softmax = nn.LogSoftmax()
    self.rnnLR = NewRNN(self.embedding_size, self.vocab_size)
    self.rnnRL = NewRNN(self.embedding_size, self.vocab_size)

  def forward(self, input_batch):
      seq_len, batch_size = input_batch.size()
      predictions = Variable(torch.zeros(seq_len, batch_size, self.vocab_size))
      hLR = [Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)]
      hRL = [Variable(torch.rand(batch_size, self.hidden_size), requires_grad=True)]

      for t in xrange(seq_len - 1):
        word_ix = input_batch[t, :]
        w = Variable(self.embedding[word_ix.data, :], requires_grad=True)
        hidden = self.rnnLR(w, hLR[t]) #
        hLR.append(hidden)

      for t in xrange(seq_len - 1, 0, -1):
        word_ix = input_batch[t, :]
        w = Variable(self.embedding[word_ix.data, :], requires_grad=True)
        hidden = self.rnnRL(w, hRL[seq_len - t - 1]) #
        hRL.append(hidden)
        
      for i in range(len(hLR)):
          j = len(hLR) - 1 - i
          concatHidden = Variable(torch.cat((hLR[i].data, hRL[j].data), 1))
          output = self.softmax(self.lay(concatHidden))
          predictions[i,:,:] = output

      return predictions

