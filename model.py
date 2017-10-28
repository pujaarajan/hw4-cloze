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

# create a RNN. this is not the final class
class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = 10 # can be arbitrary

        self.i2h = nn.FCLayer(input_size + hidden_size, hidden_size)
        self.i2o = nn.FCLayer(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax() # honky, sue me

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # concatenate
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

# this is the final class that will use RNN
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
     
    self.embedding_size = 10 # arbitrary dimension
    self.vocab_size = vocab_size

    self.embedding = torch.randn(vocab_size, self.embedding_size)  # random word embedding
    self.rnn = RNN(self.embedding_size, self.vocab_size)



  def forward(self, input_batch):
    ## input_batch of size (seq_len, batch_size)
    ## apply for loop
    hidden = Variable(torch.zeros(1, self.hidden_size))
    for t in xrange(seq_len -1):
      w = self.embedding[...] # need to work out batches and embedding
      output, hidden = rnn(w, hidden)
      # store output somewhere

    # process outputs so they turn into actual words (or indexes to be more precise)

    return predictions




# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    pass

  def forward(self, input_batch):
    pass
