import torch
import torch.nn as nn
from torch.autograd import Variable

# since we cannot use Linear we create our own class
class DenseLayer(nn.Module):
  def __init__(self, input_size, output_size):
    super(DenseLayer, self).__init__()
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
    
    hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True)
    for t in xrange(seq_len):
      word_ix = input_batch[t, :]
      w = Variable(self.embedding[word_ix.data, :], requires_grad=True)
      output, hidden = self.rnn(w, hidden) #
      predictions[t,:,:] = output

    return predictions




# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
